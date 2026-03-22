import numpy as np
from scipy.spatial import cKDTree
from typing import List, Dict, Any, Tuple

from pipeline.lib.geom_utils import dbscan_obb, compute_marching_cubes_mesh, compute_obb
from backend.core.graph_edges import upsert_unified_edge


def process_graph(
    graph_data: Dict[str, Any],
    geom_data: Dict[str, Any],
    pole_tol: float,
    building_group_tol: float,
    laz_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Dict[str, int]:
    xyz, sem, ins = laz_data
    buildings = []
    building_points = np.empty((0, 3))
    building_ins = np.empty((0,), dtype=int)

    if len(xyz) > 0:
        building_mask = (sem == 12) | (sem == 13) | (sem == 14)
        xyz_b, sem_b, ins_b = xyz[building_mask], sem[building_mask], ins[building_mask]
        keep = ins_b != -1
        xyz_b, sem_b, ins_b = xyz_b[keep], sem_b[keep], ins_b[keep]

        if len(xyz_b) > 0:
            unique_pairs = list(set(zip(sem_b.astype(int), ins_b.astype(int))))
            if len(unique_pairs) > 500:
                unique_pairs = unique_pairs[:500]

            keep_sampled = np.array([(int(s), int(i)) in set(unique_pairs) for s, i in zip(sem_b, ins_b)])
            building_points = xyz_b[keep_sampled]
            building_ins = ins_b[keep_sampled].astype(int)

            for (sem_class, ins_id) in unique_pairs:
                pts = xyz_b[(sem_b == sem_class) & (ins_b == ins_id)]
                if len(pts) == 0:
                    continue
                buildings.append({
                    "id": int(ins_id), "sem_class": int(sem_class),
                    "min": [float(x) for x in np.min(pts, axis=0)],
                    "max": [float(x) for x in np.max(pts, axis=0)],
                    "hull": compute_marching_cubes_mesh(pts, spacing=1.0),
                    "obb": compute_obb(pts),
                })

    # Cluster buildings using OBB distances
    clusters = []
    if buildings and building_group_tol > 0:
        obbs = [b.get("obb") for b in buildings]
        mins = [np.asarray(b["min"]) for b in buildings]
        maxs = [np.asarray(b["max"]) for b in buildings]
        ids = [b["id"] for b in buildings]
        clusters = dbscan_obb(obbs, mins, maxs, ids, eps=building_group_tol, min_samples=3)

    id_to_group: Dict[int, int] = {}
    building_groups = []
    for gid, cluster in enumerate(clusters, start=1):
        building_groups.append({"id": gid, "members": sorted(cluster)})
        for bid in cluster:
            id_to_group[bid] = gid

    graph_buildings, geom_buildings, hulls = [], {}, geom_data.setdefault("buildingHulls", {})
    for b in buildings:
        bid_str = str(b["id"])
        entry = {"id": b["id"], "sem_class": b["sem_class"]}
        if b["id"] in id_to_group:
            entry["group_id"] = id_to_group[b["id"]]
        graph_buildings.append(entry)
        geom_buildings[bid_str] = {"min": b["min"], "max": b["max"], "obb": b.get("obb")}
        hulls[bid_str] = b["hull"]

    graph_data['buildings'] = graph_buildings
    graph_data['building_groups'] = building_groups
    geom_data['buildings'] = geom_buildings

    poles, updated_poles, poles_to_delete, pole_id_to_building_id = graph_data.get('poles', []), 0, set(), {}
    if len(building_points) > 0:
        building_tree_2d = cKDTree(building_points[:, :2])
        geom_poles = geom_data.get('poles', {})
        for pole in poles:
            pid = pole.get('id')
            p_geom = geom_poles.get(str(pid), {})
            if p_geom.get('X') is None:
                continue
            dist, idx = building_tree_2d.query(np.array([p_geom['X'], p_geom['Y']]))
            if dist <= pole_tol:
                pole_id_to_building_id[pid] = int(building_ins[idx])
                fp = p_geom.get('footprint', {})
                if fp.get('is_virtual', False) or (not fp and all(p_geom.get(k) is not None for k in ('X', 'Y', 'Z'))):
                    poles_to_delete.add(pid)
                    updated_poles += 1

    edges = graph_data.setdefault("edges", [])
    graph_data["edges"] = [
        e
        for e in edges
        if not (
            str(e.get("class", "")).lower() == "support"
            and (
                (
                    str(e.get("b_type", "")).lower() == "pole"
                    and int(e.get("b_id", -1)) in poles_to_delete
                )
                or (
                    str(e.get("a_type", "")).lower() == "pole"
                    and int(e.get("a_id", -1)) in poles_to_delete
                )
            )
        )
    ]
    for cond in graph_data.get('conductors', []):
        support_bids: List[int] = []
        for pid in cond.get('poles', []):
            if pid in pole_id_to_building_id:
                bid = pole_id_to_building_id[pid]
                if bid not in support_bids:
                    support_bids.append(bid)
        cond['poles'] = [p for p in cond.get('poles', []) if p not in poles_to_delete]
        uid = cond.get("uid")
        if uid is not None:
            for bid in support_bids:
                upsert_unified_edge(
                    graph_data, "conductor", int(uid), "building", int(bid), "support_building"
                )
    graph_data['poles'] = [p for p in poles if p['id'] not in poles_to_delete]

    return {"count": len(buildings), "supports": len(pole_id_to_building_id), "snapped": updated_poles}
