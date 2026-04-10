"""
Assign LAZ instance id to each network pole by matching pole XY to pole-class points.

Uses semantic classes 9, 10, 11 (utility / light / traffic pole) per README taxonomy.
"""
from collections import Counter
from typing import Any, Dict, Tuple

import numpy as np
from scipy.spatial import cKDTree

# Utility pole, Light pole, Traffic pole — see README semantic taxonomy
POLE_SEMANTIC_CLASSES = (9, 10, 11)


def process_graph(
    graph_data: Dict[str, Any],
    geom_data: Dict[str, Any],
    laz_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    max_xy_distance: float = 3.0,
) -> int:
    """
    Set pole[\"instance_id\"] from dominant LAZ instance among labeled pole points
    near the pole footprint. Returns count of poles updated.
    """
    xyz, sem, ins = laz_data
    if len(xyz) == 0:
        return 0

    classes = np.array(POLE_SEMANTIC_CLASSES, dtype=np.int32)
    mask = np.isin(sem.astype(np.int32), classes) & (ins >= 0)
    if not np.any(mask):
        return 0

    pl_xy = xyz[mask][:, :2]
    pl_ins = ins[mask].astype(int)
    tree = cKDTree(pl_xy)

    gp = geom_data.get("poles") or {}
    n_set = 0
    for pole in graph_data.get("poles") or []:
        pid = pole.get("id")
        if pid is None:
            continue
        if pole.get("is_virtual_pole"):
            continue
        g = gp.get(str(int(pid)))
        if not g:
            continue
        x, y = g.get("X"), g.get("Y")
        if x is None or y is None:
            continue
        q = np.array([float(x), float(y)], dtype=np.float64)
        k = min(8, len(pl_xy))
        dist, idx = tree.query(q, k=k, distance_upper_bound=max_xy_distance)
        votes = []
        if np.isscalar(dist):
            if np.isfinite(float(dist)) and float(dist) <= max_xy_distance:
                votes.append(int(pl_ins[int(idx)]))
        else:
            for di, ii in zip(np.atleast_1d(dist), np.atleast_1d(idx)):
                if np.isfinite(float(di)) and float(di) <= max_xy_distance:
                    votes.append(int(pl_ins[int(ii)]))
        if not votes:
            continue
        pole["instance_id"] = int(Counter(votes).most_common(1)[0][0])
        n_set += 1

    return n_set
