import argparse
import glob
import json
import os
from typing import List, Dict, Any
from tqdm import tqdm

from pipeline.lib.add_buildings import process_graph as process_buildings
from pipeline.lib.add_vehicules import process_graph as process_vehicles
from pipeline.lib.add_trees import process_graph as process_trees
from pipeline.lib.add_conductor_instances import process_graph as process_conductor_instances
from pipeline.lib.add_pole_instances import process_graph as process_pole_instances
from pipeline.lib.instance_ids import mark_virtual_poles_from_geom
from pipeline.lib.generate_json_graph import adjust_instances
from pipeline.lib.geom_utils import load_laz_points
from backend.core.graph_manager import GraphManager, _groups_key
from backend.core.graph_edges import strip_proximity_edges, rewire_graph_to_laz_instance_ids
from backend.core.graph_io import (
    filter_graph_node_paths,
    load_merged_graph,
    save_split_graph,
)


def build_full_graph(
    input_path: str,
    laz_dir: str,
    graph_dir: str,
    geom_dir: str,
    pole_tol: float,
    max_vehicles: int,
    vehicle_spacing: float,
    veh_group_tol: float,
    building_group_tol: float,
    tree_group_tol: float,
) -> None:
    graph_files: List[str] = []

    if os.path.isfile(input_path):
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "poles" in data and "conductors" in data:
            os.makedirs(graph_dir, exist_ok=True)
            target = os.path.join(graph_dir, os.path.basename(input_path))
            if os.path.abspath(input_path) != os.path.abspath(target):
                gd = dict(data)
                gd.setdefault("edges", [])
                save_split_graph(target, gd)
                graph_files = [target]
            else:
                graph_files = [input_path]
        else:
            os.makedirs(graph_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(input_path))[0]
            out_path = os.path.join(graph_dir, f"graph_{base}.json")
            graph_path = adjust_instances(input_path, output_path=out_path)
            graph_files = [graph_path]
    else:
        graph_files = filter_graph_node_paths(
            glob.glob(os.path.join(input_path, "graph_*.json"))
        )
        if not graph_files:
            network_files = [f for f in glob.glob(os.path.join(input_path, "*.json"))
                             if not os.path.basename(f).startswith(("graph_", "web_", "geom_"))]
            os.makedirs(graph_dir, exist_ok=True)
            for nf in network_files:
                base = os.path.splitext(os.path.basename(nf))[0]
                out_path = os.path.join(graph_dir, f"graph_{base}.json")
                graph_files.append(adjust_instances(nf, output_path=out_path))

    graph_files = filter_graph_node_paths(graph_files)

    laz_files = glob.glob(os.path.join(laz_dir, "**", "*.laz"), recursive=True)
    existing_ids = {
        os.path.basename(f)[6:-5]
        for f in graph_files
        if os.path.basename(f).startswith("graph_")
    }

    for lf in laz_files:
        tid = os.path.splitext(os.path.basename(lf))[0]
        if tid not in existing_ids:
            path = os.path.join(graph_dir, f"graph_{tid}.json")
            save_split_graph(path, {"poles": [], "conductors": [], "edges": []})
            graph_files.append(path)
            geom_path = os.path.join(geom_dir, f"geom_{tid}.json")
            os.makedirs(geom_dir, exist_ok=True)
            with open(geom_path, "w") as f:
                json.dump({"scale": 1.0, "poles": {}, "conductors": {}, "buildings": {}, "vehicles": {}, "trees": {}}, f, indent=2)

    if not graph_files:
        print("No tiles found.")
        return

    print(f"Processing {len(graph_files)} tiles...")
    for graph_path in tqdm(sorted(graph_files), desc="Building Full Graphs"):
        basename = os.path.basename(graph_path)
        tid = basename[6:-5]

        graph_data = load_merged_graph(graph_path)
        geom_path = os.path.join(geom_dir, f"geom_{tid}.json")
        if os.path.exists(geom_path):
            with open(geom_path, 'r') as f:
                geom_data = json.load(f)
        else:
            geom_data = {"scale": 1.0, "poles": {}, "conductors": {}, "buildings": {}, "vehicles": {}, "trees": {}}

        laz_path = None
        for sub in ["test", "train", ""]:
            p = os.path.join(laz_dir, sub, f"{tid}.laz")
            if os.path.exists(p):
                laz_path = p
                break

        if laz_path:
            laz_data = load_laz_points(laz_path)
            process_buildings(graph_data, geom_data, pole_tol, building_group_tol, laz_data)
            process_vehicles(graph_data, geom_data, max_vehicles, vehicle_spacing, veh_group_tol, laz_data)
            process_trees(graph_data, geom_data, tree_group_tol, laz_data)
            process_conductor_instances(graph_data, geom_data, laz_data)
            mark_virtual_poles_from_geom(graph_data, geom_data)
            process_pole_instances(graph_data, geom_data, laz_data)
            rewire_graph_to_laz_instance_ids(graph_data, geom_data, laz_data)

            gm = GraphManager(data_dir=os.path.dirname(graph_dir))
            gm.graph_data = graph_data
            gm.geom_data = geom_data
            gm.current_tile_id = tid

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
            graph_data = gm.graph_data
            geom_data = gm.geom_data
        else:
            rewire_graph_to_laz_instance_ids(graph_data, geom_data, None)

        save_split_graph(graph_path, graph_data)
        os.makedirs(geom_dir, exist_ok=True)
        with open(geom_path, 'w') as f:
            json.dump(geom_data, f, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description="Build full network graphs and geometry.")
    parser.add_argument("--input", default="data/network")
    parser.add_argument("--laz_dir", default="dales_2")
    parser.add_argument("--graph_dir", default="data/graph")
    parser.add_argument("--geom_dir", default="data/geometry")
    parser.add_argument("--pole_tol", type=float, default=0.5)
    parser.add_argument("--max_vehicles", type=int, default=1e9)
    parser.add_argument("--vehicle_spacing", type=float, default=0.5)
    parser.add_argument("--veh_group_tol", type=float, default=3.0)
    parser.add_argument("--building_group_tol", type=float, default=8.0)
    parser.add_argument("--tree_group_tol", type=float, default=5.0)
    args = parser.parse_args()

    build_full_graph(args.input, args.laz_dir, args.graph_dir, args.geom_dir,
                     args.pole_tol, args.max_vehicles, args.vehicle_spacing,
                     args.veh_group_tol, args.building_group_tol, args.tree_group_tol)


if __name__ == "__main__":
    main()
