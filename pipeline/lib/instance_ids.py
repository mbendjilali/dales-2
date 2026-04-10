"""
Tile-level instance id helpers: virtual poles, LAZ universe, synthetic allocation.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


def laz_used_instance_ids(laz_data: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Set[int]:
    """All non-negative instance labels appearing anywhere in the point cloud."""
    _, _, ins = laz_data
    if ins is None or len(ins) == 0:
        return set()
    arr = np.asarray(ins).astype(np.int64, copy=False)
    return {int(x) for x in arr[arr >= 0]}


def mark_virtual_poles_from_geom(graph_data: Dict[str, Any], geom_data: Dict[str, Any]) -> None:
    """
    Set pole[\"is_virtual_pole\"] from geometry: not represented as a real pole in PCL matching
    (virtual footprint, or centroid-only with no footprint).
    """
    gp = geom_data.get("poles") or {}
    for pole in graph_data.get("poles") or []:
        pid = pole.get("id")
        if pid is None:
            continue
        g = gp.get(str(int(pid)), {})
        fp = g.get("footprint")
        if isinstance(fp, dict) and fp.get("is_virtual") is True:
            pole["is_virtual_pole"] = True
        elif not fp and g.get("X") is not None and g.get("Y") is not None:
            pole["is_virtual_pole"] = True
        else:
            pole["is_virtual_pole"] = False


def next_free_tile_id(pcl_used: Set[int], occupied_graph: Set[int]) -> int:
    """
    Next integer usable for synthetic graph ids: not used in the PCL instance universe
    and not already assigned to another graph object (contiguous block above max when possible).
    """
    x = (max(pcl_used | occupied_graph) + 1) if (pcl_used or occupied_graph) else 1
    while x in pcl_used or x in occupied_graph:
        x += 1
    return x
