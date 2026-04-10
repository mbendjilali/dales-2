import json
import argparse
import math
from typing import List, Dict, Any, Tuple, Optional

from pipeline.lib.find_extensions import dist3d

def get_node_attachments(nodes: List[Dict], links: List[Dict]) -> Tuple[Dict[int, List[List[float]]], Dict[int, List[Tuple[int, Any]]]]:
    """
    Map each node id to list of attachment points and to list of conductor (link_idx, conductor_id).
    Returns (node_attachments, node_conductor_ids).
    """
    node_map = {n['id']: n for n in nodes}
    node_attachments: Dict[int, List[List[float]]] = {}
    node_conductor_ids: Dict[int, List[Tuple[int, Any]]] = {}
    for link_idx, link in enumerate(links):
        source_id = link.get('source')
        target_id = link.get('target')
        for c in link.get('conductors', []):
            start = c.get('startpoint')
            end = c.get('endpoint')
            conductor_uid = (link_idx, c.get('id'))
            if not start or not end:
                continue
            if source_id is not None and source_id in node_map:
                s_node = node_map[source_id]
                s_pos = [s_node.get('X', 0), s_node.get('Y', 0), s_node.get('Z', 0)]
                pt = start if dist3d(start, s_pos) < dist3d(end, s_pos) else end
                node_attachments.setdefault(source_id, []).append(pt)
                node_conductor_ids.setdefault(source_id, []).append(conductor_uid)
            if target_id is not None and target_id in node_map:
                t_node = node_map[target_id]
                t_pos = [t_node.get('X', 0), t_node.get('Y', 0), t_node.get('Z', 0)]
                pt = start if dist3d(start, t_pos) < dist3d(end, t_pos) else end
                node_attachments.setdefault(target_id, []).append(pt)
                node_conductor_ids.setdefault(target_id, []).append(conductor_uid)

    return node_attachments, node_conductor_ids

def reconstruct_footprints(
    nodes: List[Dict],
    node_attachments: Dict[int, List[List[float]]],
    global_min_z: float = 0.0,
) -> Tuple[List[Dict], int]:
    """
    Ensure every node has a footprint. Use global_min_z as min Z for all poles (real and virtual).
    Real poles: keep size/rotation, set footprint.min[2] = global_min_z.
    Virtual (reconstructed): build from attachments, min Z = global_min_z.
    Returns (nodes, reconstructed_count).
    """
    reconstructed_count = 0

    for n in nodes:
        fp = n.get('footprint')
        if fp and isinstance(fp, dict):
            # Real pole: discard min Z, use global_min_z
            if isinstance(fp.get('min'), list) and len(fp['min']) >= 3:
                fp['min'] = [fp['min'][0], fp['min'][1], fp['min'][2]]
            continue

        attachments = node_attachments.get(n.get('id'), [])

        if attachments:
            xs = [p[0] for p in attachments]
            ys = [p[1] for p in attachments]
            zs = [p[2] for p in attachments]
        else:
            xs = [n.get('X', 0)]
            ys = [n.get('Y', 0)]
            zs = [n.get('Z', 0)]
        n['footprint'] = {
            'min': [min(xs) - 0.2, min(ys) - 0.2, global_min_z],
            'max': [max(xs) + 0.2, max(ys) + 0.2, max(zs)],
            'rotation': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            'is_virtual': True,
        }
        reconstructed_count += 1

    return nodes, reconstructed_count

def reconstruct_supports(json_file: str, global_min_z: Optional[float] = None) -> None:
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    nodes = data.get('nodes', [])
    links = data.get('links', [])
    if global_min_z is not None:
        min_z = global_min_z
    else:
        min_z = min(n.get('Z', 0) for n in nodes)
    node_attachments, node_conductor_ids = get_node_attachments(nodes, links)
    print("\n--- Pole Conductor IDs ---")
    for nid in sorted(node_conductor_ids.keys()):
        print(f"Pole {nid}: Conductor IDs {node_conductor_ids[nid]}")
    nodes, reconstructed_count = reconstruct_footprints(nodes, node_attachments, global_min_z=min_z)
    print(f"\nTotal poles reconstructed: {reconstructed_count} (global_min_z={min_z})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find supports and reconstruct pole footprints.")
    parser.add_argument("json_file", help="Path to the JSON file.")
    parser.add_argument(
        "--global-min-z",
        type=float,
        default=None,
        help="Global min Z for all poles (default: minimum pole Z from the file).",
    )
    args = parser.parse_args()
    reconstruct_supports(args.json_file, global_min_z=args.global_min_z)
