#!/usr/bin/env python3
"""
Count graph edges per relation class and endpoint pair (a_type, b_type).

Expects unified ``edges``: {id, a_type, b_type, a_id, b_id, class}.

Examples:
  python scripts/dev/count_graph_relation_edges.py data/graph
  python scripts/dev/count_graph_relation_edges.py data/graph/graph_5105_54460.json --csv counts.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


EdgeKey = Tuple[str, str, str]  # (class, a_type, b_type)


def _count_edges_by_class_and_types(data: Dict[str, Any]) -> Dict[EdgeKey, int]:
    edges = data.get("edges") or []
    out: Dict[EdgeKey, int] = defaultdict(int)
    for e in edges:
        if not isinstance(e, dict):
            continue
        cls = str(e.get("class", "")).lower()
        at = str(e.get("a_type", "")).lower()
        bt = str(e.get("b_type", "")).lower()
        if not cls:
            continue
        if not at or not bt:
            at, bt = at or "?", bt or "?"
        out[(cls, at, bt)] += 1
    return dict(out)


def collect_graph_paths(target: Path) -> List[Path]:
    if target.is_file():
        return [target] if target.suffix.lower() == ".json" else []
    if target.is_dir():
        return sorted(target.glob("graph_*.json"))
    return []


def key_to_column(k: EdgeKey) -> str:
    cls, at, bt = k
    return f"{cls}:{at}:{bt}"


def process_file(path: Path) -> Dict[EdgeKey, int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _count_edges_by_class_and_types(data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "path",
        type=Path,
        help="graph_*.json file or directory containing them",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Write per-file counts to this CSV path",
    )
    args = parser.parse_args()

    paths = collect_graph_paths(args.path.resolve())
    if not paths:
        print(f"No graph JSON files found under {args.path}", file=sys.stderr)
        return 2

    per_file: List[Dict[EdgeKey, int]] = []
    all_keys: set[EdgeKey] = set()
    grand: Dict[EdgeKey, int] = defaultdict(int)

    for p in paths:
        counts = process_file(p)
        per_file.append(counts)
        for k, v in counts.items():
            all_keys.add(k)
            grand[k] += v

    sorted_keys = sorted(all_keys, key=lambda t: (t[0], t[1], t[2]))

    print(f"Files: {len(paths)}")
    current_cls: str | None = None
    for k in sorted_keys:
        cls, at, bt = k
        v = grand[k]
        if cls != current_cls:
            current_cls = cls
            print(f"  {cls}")
        print(f"    {at}-{bt}  {v}")
    print(f"  {'TOTAL(all edges)':24s} {sum(grand.values())}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["tile_id"] + [key_to_column(k) for k in sorted_keys]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for p, counts in zip(paths, per_file):
                row: Dict[str, Any] = {
                    "tile_id": p.stem.replace("graph_", "", 1) if p.name.startswith("graph_") else p.stem
                }
                for k in sorted_keys:
                    row[key_to_column(k)] = counts.get(k, 0)
                w.writerow(row)
            total_row: Dict[str, Any] = {"tile_id": "_TOTAL"}
            for k in sorted_keys:
                total_row[key_to_column(k)] = grand[k]
            w.writerow(total_row)
        print(f"Wrote {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
