import numpy as np
from typing import Dict, Any, Tuple, List

from pipeline.lib.geom_utils import dbscan_obb, compute_marching_cubes_mesh, compute_obb


def process_graph(
    graph_data: Dict[str, Any],
    geom_data: Dict[str, Any],
    max_vehicles: int,
    spacing: float,
    group_tol: float,
    laz_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> int:
    xyz, sem, ins = laz_data
    vehicles: List[Dict[str, Any]] = []

    if len(xyz) > 0:
        vehicle_mask = (sem == 2) | (sem == 6) | (sem == 7) | (sem == 8)
        xyz_v, sem_v, ins_v = xyz[vehicle_mask], sem[vehicle_mask], ins[vehicle_mask]
        keep = ins_v != -1
        xyz_v, sem_v, ins_v = xyz_v[keep], sem_v[keep], ins_v[keep]

        if len(xyz_v) > 0:
            unique_pairs = list(set(zip(sem_v.astype(int), ins_v.astype(int))))
            if len(unique_pairs) > max_vehicles:
                unique_pairs = unique_pairs[:max_vehicles]

            for (sem_class, ins_id) in unique_pairs:
                mask = (sem_v == sem_class) & (ins_v == ins_id)
                pts = xyz_v[mask]
                if len(pts) == 0:
                    continue
                vehicles.append({
                    "id": int(ins_id), "sem_class": int(sem_class),
                    "min": [float(x) for x in np.min(pts, axis=0)],
                    "max": [float(x) for x in np.max(pts, axis=0)],
                    "hull": compute_marching_cubes_mesh(pts, spacing=spacing),
                    "obb": compute_obb(pts),
                })

    clusters = []
    if vehicles and group_tol > 0:
        obbs = [v.get("obb") for v in vehicles]
        mins = [np.asarray(v["min"]) for v in vehicles]
        maxs = [np.asarray(v["max"]) for v in vehicles]
        ids = [v["id"] for v in vehicles]
        clusters = dbscan_obb(obbs, mins, maxs, ids, eps=group_tol, min_samples=3)

    id_to_group: Dict[int, int] = {}
    vehicle_groups: List[Dict[str, Any]] = []
    for gid, cluster in enumerate(clusters, start=1):
        vehicle_groups.append({"id": gid, "members": sorted(cluster)})
        for vid in cluster:
            id_to_group[vid] = gid

    graph_vehicles, geom_vehicles, hulls = [], {}, {}
    for v in vehicles:
        vid_str = str(v["id"])
        entry = {"id": v["id"], "sem_class": v["sem_class"]}
        if v["id"] in id_to_group:
            entry["group_id"] = id_to_group[v["id"]]
        graph_vehicles.append(entry)
        geom_vehicles[vid_str] = {"min": v["min"], "max": v["max"], "obb": v.get("obb")}
        hulls[vid_str] = v["hull"]

    graph_data["vehicles"] = graph_vehicles
    graph_data["vehicle_groups"] = vehicle_groups
    geom_data["vehicles"] = geom_vehicles
    geom_data["vehicleHulls"] = hulls

    return len(vehicles)
