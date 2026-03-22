"""
Assign LAZ instance id to each network conductor by matching geometry to powerline points (sem==3).
"""
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree


def _sample_segment(sp: List[float], ep: List[float], n: int = 8) -> List[np.ndarray]:
    a = np.array(sp[:3], dtype=np.float64)
    b = np.array(ep[:3], dtype=np.float64)
    out: List[np.ndarray] = []
    for t in np.linspace(0.0, 1.0, n):
        out.append(a + (b - a) * float(t))
    return out


def process_graph(
    graph_data: Dict[str, Any],
    geom_data: Dict[str, Any],
    laz_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    max_xy_distance: float = 4.0,
) -> int:
    """
    Set conductor[\"instance_id\"] to the dominant LAZ instance among powerline points
    near the conductor segment. Returns count of conductors updated.
    """
    xyz, sem, ins = laz_data
    if len(xyz) == 0:
        return 0

    pw = (sem == 3) & (ins >= 0)
    if not np.any(pw):
        return 0

    pl_xy = xyz[pw][:, :2]
    pl_ins = ins[pw].astype(int)
    tree = cKDTree(pl_xy)

    c_geoms = geom_data.get("conductors") or {}
    n_set = 0
    for c in graph_data.get("conductors") or []:
        li = c.get("link_idx")
        cid = c.get("conductor_id")
        if li is None or cid is None:
            continue
        key = f"{li}_{cid}"
        g = c_geoms.get(key)
        if not g:
            continue
        sp, ep = g.get("startpoint"), g.get("endpoint")
        if not sp or not ep:
            continue
        votes: List[int] = []
        for p in _sample_segment(sp, ep, n=10):
            q = np.array([p[0], p[1]], dtype=np.float64)
            dist, idx = tree.query(q, k=1, distance_upper_bound=max_xy_distance)
            if np.isfinite(dist) and dist <= max_xy_distance:
                votes.append(int(pl_ins[int(idx)]))
        if not votes:
            continue
        c["instance_id"] = int(Counter(votes).most_common(1)[0][0])
        n_set += 1

    return n_set
