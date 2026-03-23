import os
import glob
import csv
import sys
from collections import defaultdict

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backend.core.graph_io import filter_graph_node_paths, load_merged_graph


def count_edges():
    graph_files = filter_graph_node_paths(glob.glob("data/graph/graph_*.json"))

    results = []
    all_rel_classes = set()

    for gf in sorted(graph_files):
        data = load_merged_graph(gf)

        edges = data.get("edges") or []

        tile_id = os.path.basename(gf).replace("graph_", "").replace(".json", "")

        row = {"tile_id": tile_id}

        by_cls = defaultdict(int)
        for e in edges:
            if not isinstance(e, dict):
                continue
            cls = str(e.get("class", "unknown")).lower()
            by_cls[cls] += 1

        row["extensions"] = by_cls.get("extension", 0)
        row["bifurcations"] = by_cls.get("bifurcation", 0)
        row["crosses"] = by_cls.get("cross", 0)
        row["conductor_pole_attachments"] = by_cls.get("support", 0) + by_cls.get(
            "support_building", 0
        )

        rel_counts = defaultdict(int)
        for cls in ("adjacent", "near"):
            n = by_cls.get(cls, 0)
            if n:
                col_name = f"relation_{cls}"
                rel_counts[col_name] = n
                all_rel_classes.add(col_name)
            
        # 6. Group Memberships (implicit bipartite edges: object -> group)
        group_member_edges = 0
        # 7. Group Cliques (implicit pairwise edges within a group)
        group_clique_edges = 0
        
        group_types = ["trees_groups", "buildings_groups", "vehicles_groups", "poles_groups", "conductors_groups"]
        for g_type in group_types:
            groups = data.get(g_type, [])
            for g in groups:
                n = len(g.get("members", []))
                group_member_edges += n
                if n > 1:
                    group_clique_edges += n * (n - 1) // 2
                    
        row["group_member_edges"] = group_member_edges
        row["group_clique_edges"] = group_clique_edges

        for k, v in rel_counts.items():
            row[k] = v
            
        results.append(row)
        
    all_rel_classes = sorted(list(all_rel_classes))
    fieldnames = [
        "tile_id", 
        "extensions", 
        "bifurcations", 
        "crosses", 
        "conductor_pole_attachments",
        "group_member_edges",
        "group_clique_edges"
    ] + all_rel_classes
    
    with open("edge_counts.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            # Fill missing relations with 0
            out_row = {fn: row.get(fn, 0) for fn in fieldnames}
            out_row["tile_id"] = row["tile_id"]
            writer.writerow(out_row)
            
    print(f"Processed {len(results)} tiles.")
    print("Saved edge counts to edge_counts.csv")

if __name__ == "__main__":
    count_edges()
