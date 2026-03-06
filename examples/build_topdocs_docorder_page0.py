#!/usr/bin/env python3
# Copyright 2024 Bloomberg Finance L.P.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build topdocs/top-pages JSON from doc-level retrieval parquet by keeping top-K docs "
            "and forcing a fixed page_idx (default page 0) for each doc."
        )
    )
    p.add_argument("--retrieval-doc-parquet", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--run-id", type=str, default=None, help="Optional run_id filter.")
    p.add_argument("--topk-docs", type=int, default=1000)
    p.add_argument("--max-qids", type=int, default=None)
    p.add_argument("--qids-file", type=Path, default=None, help="Optional newline-separated qids.")
    p.add_argument("--force-page-idx", type=int, default=0)
    return p.parse_args()


def _pick_column(cols: list[str], candidates: list[str], name: str) -> str:
    for c in candidates:
        if c in cols:
            return c
    raise ValueError(f"Could not infer {name}. Available cols: {cols}")


def _to_int(x: Any, default: int | None = None) -> int | None:
    if x is None:
        return default
    try:
        return int(x)
    except Exception:
        return default


def _to_float(x: Any, default: float | None = None) -> float | None:
    if x is None:
        return default
    try:
        return float(x)
    except Exception:
        return default


def _load_qid_filter(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    qids: set[str] = set()
    with path.open() as f:
        for line in f:
            q = line.strip()
            if q:
                qids.add(q)
    return qids


def main() -> int:
    args = _parse_args()
    try:
        import pyarrow.dataset as ds
    except Exception as exc:
        raise RuntimeError("pyarrow is required for parquet input") from exc

    qid_filter = _load_qid_filter(args.qids_file)

    doc_ds = ds.dataset(str(args.retrieval_doc_parquet), format="parquet")
    doc_cols = list(doc_ds.schema.names)
    doc_qid_col = _pick_column(doc_cols, ["qid", "query_id", "question_id"], "doc parquet qid column")
    doc_run_col = "run_id" if "run_id" in doc_cols else None
    doc_id_col = _pick_column(doc_cols, ["doc_id", "document_id", "pid"], "doc parquet doc_id column")
    doc_max_rank_col = "maxsim_rank" if "maxsim_rank" in doc_cols else None
    doc_faiss_rank_col = "faiss_rank" if "faiss_rank" in doc_cols else None
    doc_max_score_col = "maxsim_score" if "maxsim_score" in doc_cols else None
    doc_faiss_score_col = "faiss_score" if "faiss_score" in doc_cols else None

    read_doc_cols = [doc_qid_col, doc_id_col]
    if doc_run_col:
        read_doc_cols.append(doc_run_col)
    if doc_max_rank_col:
        read_doc_cols.append(doc_max_rank_col)
    if doc_faiss_rank_col:
        read_doc_cols.append(doc_faiss_rank_col)
    if doc_max_score_col:
        read_doc_cols.append(doc_max_score_col)
    if doc_faiss_score_col:
        read_doc_cols.append(doc_faiss_score_col)

    filters = []
    if qid_filter:
        filters.append(ds.field(doc_qid_col).isin(list(qid_filter)))
    if args.run_id and doc_run_col:
        filters.append(ds.field(doc_run_col) == args.run_id)
    doc_filter = None
    if filters:
        doc_filter = filters[0]
        for f in filters[1:]:
            doc_filter = doc_filter & f

    doc_table = doc_ds.to_table(columns=read_doc_cols, filter=doc_filter)

    rows_by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in doc_table.to_pylist():
        qid = row.get(doc_qid_col)
        doc_id = row.get(doc_id_col)
        if qid is None or doc_id is None:
            continue
        run_val = row.get(doc_run_col) if doc_run_col else None
        if args.run_id and doc_run_col and str(run_val) != str(args.run_id):
            continue

        max_rank = _to_int(row.get(doc_max_rank_col)) if doc_max_rank_col else None
        faiss_rank = _to_int(row.get(doc_faiss_rank_col)) if doc_faiss_rank_col else None
        rank = max_rank if max_rank is not None else faiss_rank
        if rank is None:
            rank = 1_000_000_000

        max_score = _to_float(row.get(doc_max_score_col)) if doc_max_score_col else None
        faiss_score = _to_float(row.get(doc_faiss_score_col)) if doc_faiss_score_col else None
        score = max_score if max_score is not None else faiss_score
        if score is None:
            score = -float(rank)

        rows_by_qid[str(qid)].append(
            {
                "doc_id": str(doc_id),
                "rank": int(rank),
                "faiss_rank": int(faiss_rank) if faiss_rank is not None else 1_000_000_000,
                "score": float(score),
            }
        )

    qids = sorted(rows_by_qid.keys())
    if args.max_qids is not None:
        qids = qids[: max(0, args.max_qids)]

    top_pages: dict[str, list[dict[str, Any]]] = {}
    for qid in qids:
        rows = sorted(rows_by_qid[qid], key=lambda r: (int(r["rank"]), int(r["faiss_rank"]), str(r["doc_id"])))
        seen_docs = set()
        out_rows: list[dict[str, Any]] = []
        for r in rows:
            doc_id = str(r["doc_id"])
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            out_rows.append(
                {
                    "doc_id": doc_id,
                    "page_idx": int(args.force_page_idx),
                    "score": float(r["score"]),
                }
            )
            if len(out_rows) >= args.topk_docs:
                break
        top_pages[qid] = out_rows

    out = {
        "meta": {
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "method": "doc_topk_forced_page_idx",
            "retrieval_doc_parquet": str(args.retrieval_doc_parquet),
            "run_id": args.run_id,
            "topk_docs": args.topk_docs,
            "max_qids": args.max_qids,
            "force_page_idx": int(args.force_page_idx),
        },
        "top_pages": top_pages,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as f:
        json.dump(out, f, indent=2)

    total_candidates = sum(len(v) for v in top_pages.values())
    print("saved:", args.output_json)
    print("n_qids:", len(top_pages))
    print("n_candidates_total:", total_candidates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
