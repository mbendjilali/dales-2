import json
import argparse
import os
import glob
import sys
from typing import Dict, Any, List, Tuple, Optional

from pipeline.lib.find_extensions import get_all_conductors, find_extensions
from backend.core.graph_edges import upsert_unified_edge


def build_instance_graph(
    data: Dict[str, Any],
    tolerance: float = float("inf"),
) -> Dict[str, Any]:
    """
    Given a network JSON (with nodes / links), compute the extension graph
    and return a compact JSON-serializable structure where:

    - conductors carry a temporary numeric uid until rewire_graph_to_laz_instance_ids removes it.
    - For each connector we expose:
        * uid          (int, local to this graph)
        * link_id      (original link index in 'links')
        * conductor_id (original conductor id inside the link)
        * poles        (support pole ids)
        * component    (connected-component index in the extension graph)
    - Topological links are in graph_data[\"edges\"] as
      {id, a_type, b_type, a_id, b_id, class} (extension, bifurcation, cross, support).
    - All poles (nodes) are replicated as-is from the input JSON.
    """
    nodes: List[Dict[str, Any]] = data.get("nodes", [])
    links: List[Dict[str, Any]] = data.get("links", [])

    node_map: Dict[int, Dict[str, Any]] = {n["id"]: n for n in nodes if "id" in n}

    # Flatten conductors with geometric and pole information (source_id / target_id).
    all_conductors = get_all_conductors(links, node_map)

    # Build extension graph on uid_tuples and also get bifurcations and crosses.
    # IMPORTANT: use a finite tolerance so that both extensions/crosses/bifurcations
    # match what we see in the visualization.
    G, comp_map, bifurcations, crosses = find_extensions(all_conductors, tolerance=tolerance)

    # Map from (link_idx, conductor_id) -> purely numeric uid (0..N-1).
    uid_tuple_to_numeric: Dict[Tuple[int, Any], int] = {}
    for idx, c in enumerate(all_conductors):
        uid_tuple = c["uid_tuple"]
        uid_tuple_to_numeric[uid_tuple] = idx

    # Build connector entries.
    geom_poles = {}
    geom_conductors = {}
    
    conductors: List[Dict[str, Any]] = []

    for idx, c in enumerate(all_conductors):
        uid_tuple = c["uid_tuple"]
        numeric_uid = uid_tuple_to_numeric[uid_tuple]

        # Support poles for this connector (if any), as a small list of pole ids.
        poles_for_connector: List[Any] = []
        for pole_key in ("source_id", "target_id"):
            pole_id = c.get(pole_key)
            if pole_id is not None and pole_id not in poles_for_connector:
                poles_for_connector.append(pole_id)

        cid = f"{c.get('link_idx')}_{c.get('id')}"
        geom_conductors[cid] = {
            "model": c.get("model"),
            "startpoint": c.get("startpoint"),
            "endpoint": c.get("endpoint"),
        }

        entry: Dict[str, Any] = {
            "uid": numeric_uid,
            "link_idx": c.get("link_idx"),
            "conductor_id": c.get("id"),
            "poles": poles_for_connector,
            "component": comp_map.get(uid_tuple),
        }
        conductors.append(entry)

    drop_pole_keys = {
        "lean_angle",
        "pole_angle",
        "powerline_metrics",
        "tool_metrics",
        "source_file",
        "validated",
        "pole_footprint",
    }
    poles: List[Dict[str, Any]] = []
    for n in nodes:
        cleaned = {k: v for k, v in n.items() if k not in drop_pole_keys}
        
        # Move geometric data to geom_poles
        geom_p = {}
        for key in ["X", "Y", "Z", "footprint"]:
            if key in cleaned:
                geom_p[key] = cleaned.pop(key)
        
        if geom_p and "id" in cleaned:
            geom_poles[str(cleaned["id"])] = geom_p

        poles.append(cleaned)

    graph_data: Dict[str, Any] = {
        "poles": poles,
        "conductors": conductors,
        "edges": [],
    }
    for u, v in G.edges():
        nu = uid_tuple_to_numeric[u]
        nv = uid_tuple_to_numeric[v]
        upsert_unified_edge(graph_data, "conductor", nu, "conductor", nv, "extension")
    for uid_tuple, others in bifurcations.items():
        nu = uid_tuple_to_numeric[uid_tuple]
        for o in others:
            nv = uid_tuple_to_numeric[o]
            upsert_unified_edge(
                graph_data, "conductor", nu, "conductor", nv, "bifurcation"
            )
    for uid_tuple, others in crosses.items():
        if not others:
            continue
        nu = uid_tuple_to_numeric[uid_tuple]
        for o in others:
            nv = uid_tuple_to_numeric[o]
            upsert_unified_edge(graph_data, "conductor", nu, "conductor", nv, "cross")
    for c in all_conductors:
        numeric_uid = uid_tuple_to_numeric[c["uid_tuple"]]
        for pole_key in ("source_id", "target_id"):
            pole_id = c.get(pole_key)
            if pole_id is None:
                continue
            upsert_unified_edge(
                graph_data, "conductor", numeric_uid, "pole", int(pole_id), "support"
            )
    geom_data = {
        "poles": geom_poles,
        "conductors": geom_conductors,
    }
    return graph_data, geom_data

def adjust_instances(input_path: str, tolerance: float = float("inf"), output_path: Optional[str] = None) -> str:
    """
    Load a network JSON, compute its extension graph, and dump a companion
    JSON named 'graph_{filename}.json' (by default).
    Also writes the separated geometry data to a 'geom_{filename}.json' file.
    
    Returns the path to the written graph JSON.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    graph_data, geom_data = build_instance_graph(data, tolerance=tolerance)

    if output_path is None:
        base = os.path.basename(input_path)
        if base.lower().endswith(".json"):
            base = base[:-5]
        output_path = f"graph_{base}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2)

    # Automatically derive geom path from graph path
    geom_path = output_path.replace("graph_", "geom_").replace("graph", "geometry")
    os.makedirs(os.path.dirname(geom_path) or ".", exist_ok=True)
    with open(geom_path, "w", encoding="utf-8") as f:
        json.dump(geom_data, f, indent=2)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the extension graph for a network JSON or directory "
            "and dump to 'graph_{filename}.json'."
        )
    )
    parser.add_argument("--json", required=True, help="Path to the input network JSON file or directory.")
    parser.add_argument("--output", help="Output JSON file (only used if --json is a single file).")
    parser.add_argument(
        "--tol",
        type=float,
        default=1.0,
        help="Max distance at pole / along curves to consider extensions/bifurcations.",
    )
    args = parser.parse_args()

    if not args.json:
        parser.print_help()
        sys.exit(1)

    if os.path.isdir(args.json):
        # Process directory
        json_files = glob.glob(os.path.join(args.json, "*.json"))
        # Exclude those that are already output files
        json_files = [f for f in json_files if not (os.path.basename(f).startswith("web_") or os.path.basename(f).startswith("graph_"))]

        print(f"Processing {len(json_files)} JSON files in directory: {args.json}")
        for f in json_files:
            out_path = adjust_instances(f, tolerance=args.tol, output_path=None)
            print(f"Wrote extension graph to: {out_path}")
    else:
        # Process single file
        out_path = adjust_instances(args.json, tolerance=args.tol, output_path=args.output)
        print(f"Wrote extension graph to: {out_path}")


if __name__ == "__main__":
    main()

