#!/usr/bin/env python3
"""Rewrite Colab Drive paths in train/eval.jsonl to a local data root (RunPod/Spark).

Usage:
  python rewrite_jsonl_paths.py /workspace/stereo_roman_mvp
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def rewrite(jsonl_path: Path, data_root: Path) -> int:
    rows = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            old = Path(row["path"])
            row["path"] = str(data_root / old.name)
            rows.append(row)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return len(rows)


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__.strip())
        sys.exit(1)
    data_root = Path(sys.argv[1]).resolve()
    if not data_root.is_dir():
        raise SystemExit(f"not a directory: {data_root}")
    for name in ("train.jsonl", "eval.jsonl"):
        p = data_root / name
        if not p.is_file():
            raise SystemExit(f"missing {p}")
        n = rewrite(p, data_root)
        print(f"rewrote {n} rows in {p}")


if __name__ == "__main__":
    main()
