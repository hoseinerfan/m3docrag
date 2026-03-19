"""Merge sharded page-summary JSONL files into one deduplicated JSONL.

Deduplication key: (doc_id, page_idx). First seen row wins.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-glob", required=True, help="Glob path to shard jsonl files.")
    p.add_argument("--output-jsonl", type=Path, required=True)
    p.add_argument("--sort-output", action="store_true", help="Sort by (doc_id, page_idx) before write.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.input_glob}")

    merged: list[dict] = []
    seen: set[tuple[str, int]] = set()
    bad_lines = 0
    dup_rows = 0

    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    bad_lines += 1
                    continue
                doc_id = row.get("doc_id")
                page_idx = row.get("page_idx")
                if doc_id is None or page_idx is None:
                    bad_lines += 1
                    continue
                try:
                    key = (str(doc_id), int(page_idx))
                except Exception:
                    bad_lines += 1
                    continue
                if key in seen:
                    dup_rows += 1
                    continue
                seen.add(key)
                merged.append(row)

    if args.sort_output:
        merged.sort(key=lambda r: (str(r.get("doc_id", "")), int(r.get("page_idx", -1))))

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w") as out:
        for row in merged:
            out.write(json.dumps(row, ensure_ascii=True) + "\n")

    print("input_files:", len(files))
    print("merged_rows:", len(merged))
    print("duplicate_rows_skipped:", dup_rows)
    print("bad_lines_skipped:", bad_lines)
    print("output_jsonl:", args.output_jsonl)


if __name__ == "__main__":
    main()
