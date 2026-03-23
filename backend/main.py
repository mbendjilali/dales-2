from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import glob
import json
import tempfile
from typing import List, Optional, Any, Dict

from backend.core.graph_manager import GraphManager, _groups_key
from backend.core.scene_builder import build_scene_data
from backend.core.graph_edges import to_frontend_group_relations
from backend.core.graph_io import is_graph_edges_file, load_merged_graph, save_split_graph

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

graph_manager = GraphManager(data_dir="data")


class GroupEditRequest(BaseModel):
    tile_id: str
    object_type: str
    ids: List[int]
    operation: str  # "wedge", "split", or "delete"


class GroupRelationUpdate(BaseModel):
    id: Optional[int] = None
    member_type: str
    group_id: int
    a_id: int
    b_id: int
    cls: str
    peer_type: Optional[str] = None


class GroupRelationBatchRequest(BaseModel):
    relations: List[GroupRelationUpdate]


@app.get("/api/tiles")
async def get_tiles() -> List[str]:
    graph_path = os.path.join("data", "graph")
    if not os.path.exists(graph_path):
        return []
    files = glob.glob(os.path.join(graph_path, "graph_*.json"))
    base_ids = set()
    for f in files:
        basename = os.path.basename(f)
        if is_graph_edges_file(f):
            continue
        if not (basename.startswith("graph_") and basename.endswith(".json")):
            continue
        tile_id = basename[6:-5]
        base_id = tile_id.split("_edit_")[0]
        base_ids.add(base_id)
    return sorted(base_ids)


@app.get("/api/tile_versions/{base_id}")
async def get_tile_versions(base_id: str):
    graph_path = os.path.join("data", "graph")
    if not os.path.exists(graph_path):
        raise HTTPException(status_code=404, detail="Graph directory not found")

    pattern = os.path.join(graph_path, f"graph_{base_id}*.json")
    files = glob.glob(pattern)
    if not files:
        raise HTTPException(status_code=404, detail="No versions found for base tile")

    versions = []
    for f in sorted(files):
        basename = os.path.basename(f)
        if is_graph_edges_file(f):
            continue
        if not (basename.startswith("graph_") and basename.endswith(".json")):
            continue
        tile_id = basename[6:-5]
        suffix = tile_id[len(base_id):]
        is_edit = suffix.startswith("_edit_")
        edit_index = None
        if is_edit:
            try:
                edit_index = int(suffix.split("_edit_")[1])
            except Exception:
                edit_index = None
        if not is_edit:
            label = f"{base_id} (original)"
        elif edit_index is not None:
            label = f"{base_id} (edit {edit_index})"
        else:
            label = tile_id
        versions.append({"id": tile_id, "label": label, "is_edit": is_edit, "edit_index": edit_index})

    originals = [v for v in versions if not v["is_edit"]]
    edits = [v for v in versions if v["is_edit"]]
    edits.sort(key=lambda v: (v["edit_index"] is None, v["edit_index"] or 0))
    return {"status": "success", "versions": originals + edits}


@app.get("/api/graph/{tile_id}")
async def get_graph(tile_id: str):
    try:
        data = graph_manager.load_tile(tile_id)
        graph_manager.recompute_auto_macros()
        scene_data = build_scene_data(data["graph"], data["geometry"])
        scene_data["connector_spans"] = graph_manager.graph_data.get("connector_spans", [])
        scene_data["electrical_grids"] = graph_manager.graph_data.get("electrical_grids", [])
        return scene_data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Tile not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/save/{tile_id}")
async def save_graph(tile_id: str):
    try:
        path = graph_manager.save_tile(tile_id)
        return {"status": "success", "path": path}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/edit/group")
async def edit_group(request: GroupEditRequest):
    try:
        if graph_manager.current_tile_id != request.tile_id:
            graph_manager.load_tile(request.tile_id)
        updated_graph = graph_manager.modify_group(
            request.object_type, request.ids, request.operation,
        )
        api_key = request.object_type
        groups_key = _groups_key(api_key)
        return {
            "status": "success",
            api_key: updated_graph.get(api_key, []),
            groups_key: updated_graph.get(groups_key, []),
            "group_relations": to_frontend_group_relations(updated_graph),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/api/group_relation/{tile_id}")
async def update_group_relations(tile_id: str, body: GroupRelationBatchRequest):
    try:
        if graph_manager.current_tile_id != tile_id:
            graph_manager.load_tile(tile_id)
        for rel in body.relations:
            graph_manager.upsert_group_relation(
                member_type=rel.member_type,
                group_id=rel.group_id,
                a_id=rel.a_id,
                b_id=rel.b_id,
                rel_class=rel.cls,
                peer_type=rel.peer_type,
            )
        all_rels = to_frontend_group_relations(graph_manager.graph_data)
        return {"status": "success", "group_relations": all_rels}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/load_laz")
async def load_laz(
    laz_file: UploadFile = File(...),
    network_file: Optional[UploadFile] = File(None),
):
    """Process a LAZ/LAS file (and optional network JSON) into a graph tile."""
    try:
        from pipeline.lib.geom_utils import load_laz_points
        from pipeline.lib.add_buildings import process_graph as process_buildings
        from pipeline.lib.add_vehicules import process_graph as process_vehicles
        from pipeline.lib.add_trees import process_graph as process_trees
        from pipeline.lib.add_conductor_instances import process_graph as process_conductor_instances
        from pipeline.lib.add_pole_instances import process_graph as process_pole_instances
        from pipeline.lib.generate_json_graph import adjust_instances
        from backend.core.graph_edges import strip_proximity_edges, rewire_graph_to_laz_instance_ids

        tile_id = os.path.splitext(laz_file.filename)[0]
        graph_dir = os.path.join("data", "graph")
        geom_dir = os.path.join("data", "geometry")
        os.makedirs(graph_dir, exist_ok=True)
        os.makedirs(geom_dir, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".laz", delete=False) as tmp:
            tmp.write(await laz_file.read())
            laz_path = tmp.name

        graph_data: Dict[str, Any] = {"poles": [], "conductors": [], "edges": []}
        geom_data: Dict[str, Any] = {"scale": 1.0, "poles": {}, "conductors": {}, "buildings": {}, "vehicles": {}, "trees": {}}

        if network_file is not None:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="wb") as tmp_net:
                tmp_net.write(await network_file.read())
                net_path = tmp_net.name
            out_path = os.path.join(graph_dir, f"graph_{tile_id}.json")
            adjust_instances(net_path, output_path=out_path)
            graph_data = load_merged_graph(out_path)
            geom_path_out = os.path.join(geom_dir, f"geom_{tile_id}.json")
            if os.path.exists(geom_path_out):
                with open(geom_path_out, "r") as f:
                    geom_data = json.load(f)
            os.unlink(net_path)

        laz_data = load_laz_points(laz_path)
        os.unlink(laz_path)

        process_buildings(graph_data, geom_data, 0.5, 8.0, laz_data)
        process_vehicles(graph_data, geom_data, 1000, 0.5, 5.0, laz_data)
        process_trees(graph_data, geom_data, 8.0, laz_data)
        process_conductor_instances(graph_data, geom_data, laz_data)
        process_pole_instances(graph_data, geom_data, laz_data)
        rewire_graph_to_laz_instance_ids(graph_data, geom_data)

        gm = GraphManager(data_dir="data")
        gm.graph_data = graph_data
        gm.geom_data = geom_data
        gm.current_tile_id = tile_id

        strip_proximity_edges(gm.graph_data)

        for object_type in ("buildings", "vehicles"):
            member_type = gm._member_type_for_object_type(object_type)
            if not member_type:
                continue
            gk = _groups_key(object_type)
            for g in gm.graph_data.get(gk, []):
                gid, members = g.get("id"), g.get("members", [])
                if gid is not None and len(members) >= 2:
                    gm._recompute_relations_for_group(member_type, gid, members)

        gm._recompute_all_tree_peer_relations()

        gm.recompute_auto_macros()

        graph_path = os.path.join(graph_dir, f"graph_{tile_id}.json")
        save_split_graph(graph_path, gm.graph_data)
        geom_path = os.path.join(geom_dir, f"geom_{tile_id}.json")
        with open(geom_path, "w") as f:
            json.dump(gm.geom_data, f, indent=2, allow_nan=False)

        return {"status": "success", "tile_id": tile_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
