#!/usr/bin/env python3
"""
Remap LAZ/LAS instance IDs so they are globally unique across classifications and contiguous.

Semantic classes listed in ``--stuff-classes`` (default: 0,1,4 = ground, vegetation, fence) are
treated as "stuff": every point in those classes gets output instance **0**.

Every point whose **classification is not** in that set is a non-stuff object: each distinct
(classification, source_instance) pair gets a new ID in **1 .. N** (sorted by class, old id),
so non-stuff instances are never 0.

Does not change classification or coordinates.

Run from the repository root:
  python scripts/dev/remap_laz_instance_ids.py in.laz
  python scripts/dev/remap_laz_instance_ids.py in.laz -o out.laz
  python scripts/dev/remap_laz_instance_ids.py data/tiles/
  python scripts/dev/remap_laz_instance_ids.py data/tiles/ -o data/tiles_out/

If ``-o`` / ``--output`` is omitted, each input file is overwritten in place (via a temp file).

Optional: ``--dry-run`` to print stats without writing.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import laspy
import numpy as np


def _parse_stuff_classes(s: str) -> frozenset[int]:
    s = (s or "").strip()
    if not s:
        return frozenset()
    return frozenset(int(x.strip()) for x in s.split(",") if x.strip())


def remap_instances(
    classification: np.ndarray,
    instance: np.ndarray,
    stuff_classes: frozenset[int],
) -> tuple[np.ndarray, int, int]:
    """
    Returns (new_instance_array, n_stuff_points, n_distinct_non_stuff_objects).

    Points whose semantic class is in ``stuff_classes`` → instance 0.
    All other points → contiguous ids 1..N per (classification, source_instance).
    """
    cls_arr = np.asarray(classification).astype(np.int64, copy=False)
    ins_old = np.asarray(instance).astype(np.int64, copy=False)
    n = len(ins_old)
    if len(cls_arr) != n:
        raise ValueError("classification and instance length mismatch")

    if not stuff_classes:
        stuff_mask = np.zeros(n, dtype=bool)
    else:
        stuff_list = sorted(stuff_classes)
        stuff_mask = np.isin(cls_arr, stuff_list)
    n_stuff = int(np.sum(stuff_mask))

    out_dtype = np.asarray(instance).dtype
    new_ins = np.zeros(n, dtype=np.uint64)  # stuff → 0

    non_stuff = ~stuff_mask
    n_objects = 0
    if np.any(non_stuff):
        pairs = np.column_stack([cls_arr[non_stuff], ins_old[non_stuff]])
        _, inv = np.unique(pairs, axis=0, return_inverse=True)
        n_objects = int(np.max(inv)) + 1 if len(inv) else 0
        new_ins[non_stuff] = inv.astype(np.uint64) + 1

    max_id = int(new_ins.max()) if n else 0
    if max_id > np.iinfo(np.uint32).max:
        raise ValueError(f"new max instance id {max_id} exceeds uint32; cannot represent in LAS")

    # Fit output dtype: prefer original if it fits, else uint32, else uint16 if fits
    if np.issubdtype(out_dtype, np.integer):
        info = np.iinfo(out_dtype)
        if max_id <= info.max:
            target_dtype = out_dtype
        elif max_id <= np.iinfo(np.uint16).max:
            target_dtype = np.uint16
        else:
            target_dtype = np.uint32
    else:
        target_dtype = np.uint32

    if max_id > np.iinfo(target_dtype).max:
        raise ValueError(
            f"new max instance id {max_id} exceeds target dtype {target_dtype} "
            f"(original file used {out_dtype}); use LAS 1.4+ with wider instance storage or fewer objects."
        )

    return new_ins.astype(target_dtype), n_stuff, n_objects


def collect_laz_files(root: Path, recursive: bool) -> list[Path]:
    root = root.resolve()
    if not root.is_dir():
        return []
    pattern = "**/*" if recursive else "*"
    out: list[Path] = []
    for p in root.glob(pattern):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf in (".laz", ".las"):
            out.append(p)
    return sorted(out)


def write_laz_atomic(las: laspy.LasData, final_path: Path) -> None:
    """Write LAZ/LAS to ``final_path``, replacing atomically when overwriting."""
    final_path = final_path.resolve()
    parent = final_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    suffix = final_path.suffix or ".laz"
    fd, tmp_name = tempfile.mkstemp(suffix=suffix, dir=str(parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        las.write(str(tmp_path))
        os.replace(tmp_path, final_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def process_file(
    path: Path,
    out_path: Path,
    *,
    stuff_classes: frozenset[int],
    dry_run: bool,
) -> int:
    """Process one file; write to ``out_path``. Returns 0 on success, 2 on error."""
    try:
        las = laspy.read(str(path))
    except Exception as e:
        print(f"Error reading {path}: {e}", file=sys.stderr)
        return 2

    if "classification" not in las.point_format.dimension_names:
        print(f"Error: {path}: missing 'classification' dimension.", file=sys.stderr)
        return 2
    if "instance" not in las.point_format.dimension_names:
        print(f"Error: {path}: missing 'instance' dimension.", file=sys.stderr)
        return 2

    cls = np.array(las.classification)
    ins = np.array(las.instance)

    try:
        new_ins, n_stuff, n_objects = remap_instances(cls, ins, stuff_classes=stuff_classes)
    except ValueError as e:
        print(f"Error: {path}: {e}", file=sys.stderr)
        return 2

    sc_str = ",".join(str(x) for x in sorted(stuff_classes)) if stuff_classes else "(none)"
    print(f"Input:  {path}")
    print(f"Points: {len(ins)}")
    print(f"Stuff classes {sc_str}: {n_stuff} points -> instance 0")
    if n_objects:
        print(
            f"Non-stuff: distinct (classification, source_instance) → {n_objects} objects "
            f"with ids 1..{n_objects}"
        )
    else:
        print("Non-stuff: no points (all classifications are stuff classes)")
    print(f"New instance range: 0 .. {int(new_ins.max())}")

    if dry_run:
        return 0

    las.instance = new_ins
    try:
        write_laz_atomic(las, out_path)
    except Exception as e:
        print(f"Error writing {out_path}: {e}", file=sys.stderr)
        return 2

    print(f"Wrote: {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "path",
        type=Path,
        help="Input .laz/.las file or directory of such files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file (single input) or output directory (directory input). "
        "Default: overwrite each input file in place.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When input is a directory, include subdirectories",
    )
    parser.add_argument(
        "--stuff-classes",
        type=str,
        default="0,1,4",
        help="Comma-separated semantic classification IDs treated as stuff (output instance 0). "
        "All other classes get contiguous instance IDs starting at 1. Empty string = no stuff class. "
        "Default: 0,1,4 (ground, vegetation, fence).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics only; do not write",
    )
    args = parser.parse_args()
    try:
        stuff_classes_fz = _parse_stuff_classes(args.stuff_classes)
    except ValueError as e:
        print(f"Error: invalid --stuff-classes: {e}", file=sys.stderr)
        return 2

    inp = args.path
    if not inp.exists():
        print(f"Error: not found: {inp}", file=sys.stderr)
        return 2

    files: list[Path]
    if inp.is_file():
        suf = inp.suffix.lower()
        if suf not in (".laz", ".las"):
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

    out_arg = args.output
    if out_arg is not None:
        out_arg = out_arg.resolve()
        if len(files) > 1 or inp.is_dir():
            if out_arg.exists() and not out_arg.is_dir():
                print(
                    "Error: when processing a directory (or multiple files), "
                    "-o/--output must be a directory path.",
                    file=sys.stderr,
                )
                return 2
            out_is_dir = True
        else:
            out_is_dir = out_arg.is_dir()
    else:
        out_is_dir = False

    worst = 0
    for fpath in files:
        if out_arg is None:
            target = fpath
        elif inp.is_file() and not out_is_dir:
            target = out_arg
        else:
            target = out_arg / fpath.name

        rc = process_file(
            fpath, target, stuff_classes=stuff_classes_fz, dry_run=args.dry_run
        )
        worst = max(worst, rc)

    return worst


if __name__ == "__main__":
    raise SystemExit(main())
