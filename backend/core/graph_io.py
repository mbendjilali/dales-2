"""
Split graph storage: graph_{tile}.json holds nodes, groups, spans, etc.; graph_{tile}_edges.json holds only {"edges": [...]}.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from backend.core.graph_edges import dedupe_proximity_edges


def is_graph_edges_file(path: str) -> bool:
    return os.path.basename(path).endswith("_edges.json")


def edges_path_for_graph(graph_json_path: str) -> str:
    base, ext = os.path.splitext(graph_json_path)
    return f"{base}_edges{ext}"


def load_merged_graph(graph_path: str) -> Dict[str, Any]:
    """
    Load nodes JSON and merge edges from graph_*_edges.json when present,
    else from legacy \"edges\" key inside the nodes file.
    """
    with open(graph_path, "r", encoding="utf-8") as f:
        nodes: Dict[str, Any] = json.load(f)

    ep = edges_path_for_graph(graph_path)
    if os.path.exists(ep):
        with open(ep, "r", encoding="utf-8") as f:
            blob = json.load(f)
        edges: List[Any] = blob["edges"] if isinstance(blob, dict) else blob
    else:
        edges = nodes.get("edges") or []

    if not isinstance(edges, list):
        edges = []

    out = dict(nodes)
    out["edges"] = edges
    dedupe_proximity_edges(out)
    return out


def save_split_graph(graph_path: str, graph_data: Dict[str, Any]) -> None:
    """Write nodes (no top-level \"edges\") and companion *_edges.json."""
    dedupe_proximity_edges(graph_data)
    edges = graph_data.get("edges") or []
    save_nodes = {
        k: v
        for k, v in graph_data.items()
        if k != "edges" and k != "macro_instances"
    }
    os.makedirs(os.path.dirname(graph_path) or ".", exist_ok=True)
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(save_nodes, f, indent=2, allow_nan=False)

    ep = edges_path_for_graph(graph_path)
    with open(ep, "w", encoding="utf-8") as f:
        json.dump({"edges": edges}, f, indent=2, allow_nan=False)


def filter_graph_node_paths(paths: List[str]) -> List[str]:
    """Exclude *_edges.json from glob results."""
    return [p for p in paths if not is_graph_edges_file(p)]
