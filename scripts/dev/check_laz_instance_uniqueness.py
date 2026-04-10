#!/usr/bin/env python3
"""
Verify that LAZ instance IDs are unique across semantic classes (one classification per instance id).

Run from the repository root:
  PYTHONPATH=. python scripts/dev/check_laz_instance_uniqueness.py path/to/tile.laz
  PYTHONPATH=. python scripts/dev/check_laz_instance_uniqueness.py path/to/tiles/

Exit codes: 0 = all OK, 1 = violations in at least one file, 2 = error (missing dims, I/O, etc.).
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# Repo root on sys.path when run as scripts/dev/...
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from pipeline.lib.geom_utils import load_laz_points


def collect_laz_files(root: Path, recursive: bool) -> list[Path]:
    root = root.resolve()
    if not root.is_dir():
        return []
    pattern = "**/*" if recursive else "*"
    out: list[Path] = []
    for p in root.glob(pattern):
        if not p.is_file():
            continue
        if p.suffix.lower() in (".laz", ".las"):
            out.append(p)
    return sorted(out)


def check_one(path: Path, ignore_instance: int) -> int:
    """Returns 0 = OK, 1 = violations, 2 = error."""
    xyz, sem, ins = load_laz_points(str(path))
    if len(xyz) == 0:
        print(f"Error: {path}: no points loaded (missing classification or empty file).", file=sys.stderr)
        return 2
    if np.all(ins == -1):
        print(f"Error: {path}: no 'instance' dimension or all points are -1.", file=sys.stderr)
        return 2

    by_instance: dict[int, set[int]] = defaultdict(set)
    for s, i in zip(sem.astype(int), ins.astype(int)):
        if i == ignore_instance:
            continue
        by_instance[int(i)].add(int(s))

    violations = sorted(
        (iid, sorted(classes)) for iid, classes in by_instance.items() if len(classes) > 1
    )

    n_inst = len(by_instance)
    n_pts = int(np.sum(ins != ignore_instance))
    print(f"File: {path}")
    print(f"Points with instance != {ignore_instance}: {n_pts}")
    print(f"Distinct instance ids (excluding {ignore_instance}): {n_inst}")

    if not violations:
        print("OK: each instance id maps to exactly one semantic class.")
        return 0

    print(f"FAIL: {len(violations)} instance id(s) appear under multiple semantic classes:")
    for iid, classes in violations[:50]:
        print(f"  instance_id={iid} -> classifications {classes}")
    if len(violations) > 50:
        print(f"  ... and {len(violations) - 50} more")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "path",
        type=Path,
        help="Path to .laz/.las file or directory of such files",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When input is a directory, include subdirectories",
    )
    parser.add_argument(
        "--ignore-instance",
        type=int,
        default=0,
        help="Skip points with this instance id (default: -1, typical 'no instance')",
    )
    args = parser.parse_args()
    inp = args.path

    if not inp.exists():
        print(f"Error: not found: {inp}", file=sys.stderr)
        return 2

    if inp.is_file():
        if inp.suffix.lower() not in (".laz", ".las"):
            print(f"Error: expected .laz or .las file: {inp}", file=sys.stderr)
            return 2
        files = [inp.resolve()]
    elif inp.is_dir():
        files = collect_laz_files(inp, recursive=args.recursive)
        if not files:
            print(f"Error: no .laz/.las files under {inp}", file=sys.stderr)
            return 2
    else:
        print(f"Error: not a file or directory: {inp}", file=sys.stderr)
        return 2

    worst = 0
    for i, fpath in enumerate(files):
        if len(files) > 1:
            print("-" * 60)
            print(f"[{i + 1}/{len(files)}]")
        rc = check_one(fpath, args.ignore_instance)
        worst = max(worst, rc)

    return worst


if __name__ == "__main__":
    raise SystemExit(main())
