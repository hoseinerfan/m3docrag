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
            "Build topdocs/top-pages JSON by taking top-K docs from doc-level retrieval parquet "
            "and attaching top-N pages per doc from page-level retrieval parquet."
        )
    )
    p.add_argument("--retrieval-doc-parquet", type=Path, required=True)
    p.add_argument("--retrieval-page-parquet", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--run-id", type=str, default=None, help="Optional run_id filter.")
    p.add_argument("--topk-docs", type=int, default=1000)
    p.add_argument("--pages-per-doc", type=int, default=6)
    p.add_argument("--max-qids", type=int, default=None)
    p.add_argument("--qids-file", type=Path, default=None, help="Optional newline-separated qids.")
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
    doc_seed_page_col = None
    for c in ["maxsim_best_page", "best_page", "page_idx", "page_num"]:
        if c in doc_cols:
            doc_seed_page_col = c
            break

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
    if doc_seed_page_col:
        read_doc_cols.append(doc_seed_page_col)

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
        max_score = _to_float(row.get(doc_max_score_col)) if doc_max_score_col else None
        faiss_score = _to_float(row.get(doc_faiss_score_col)) if doc_faiss_score_col else None
        score = max_score if max_score is not None else faiss_score
        if score is None:
            score = -float(rank if rank is not None else 1_000_000_000)
        seed_page = _to_int(row.get(doc_seed_page_col), default=0) if doc_seed_page_col else 0
        rows_by_qid[str(qid)].append(
            {
                "doc_id": str(doc_id),
                "rank": rank if rank is not None else 1_000_000_000,
                "score": float(score),
                "seed_page": int(seed_page if seed_page is not None else 0),
            }
        )

    qids = sorted(rows_by_qid.keys())
    if args.max_qids is not None:
        qids = qids[: max(0, args.max_qids)]

    page_ds = ds.dataset(str(args.retrieval_page_parquet), format="parquet")
    page_cols = list(page_ds.schema.names)
    page_qid_col = _pick_column(page_cols, ["qid", "query_id", "question_id"], "page parquet qid column")
    page_run_col = "run_id" if "run_id" in page_cols else None
    page_doc_col = _pick_column(page_cols, ["pid", "doc_id", "document_id", "page_uid"], "page parquet doc column")
    page_idx_col = _pick_column(page_cols, ["page_num", "page_idx", "page"], "page parquet page column")
    page_rank_col = None
    for c in ["page_rank", "rank", "faiss_rank", "maxsim_rank"]:
        if c in page_cols:
            page_rank_col = c
            break
    page_score_col = None
    for c in ["faiss_score", "maxsim_score", "score"]:
        if c in page_cols:
            page_score_col = c
            break

    read_page_cols = [page_qid_col, page_doc_col, page_idx_col]
    if page_run_col:
        read_page_cols.append(page_run_col)
    if page_rank_col:
        read_page_cols.append(page_rank_col)
    if page_score_col:
        read_page_cols.append(page_score_col)

    top_pages: dict[str, list[dict[str, Any]]] = {}
    for qid in qids:
        doc_rows = rows_by_qid[qid]
        doc_rows = sorted(doc_rows, key=lambda r: (int(r["rank"]), -float(r["score"])))
        seen = set()
        docs = []
        for r in doc_rows:
            d = str(r["doc_id"])
            if d in seen:
                continue
            seen.add(d)
            docs.append(r)
            if len(docs) >= args.topk_docs:
                break

        doc_set = {str(r["doc_id"]) for r in docs}
        if not doc_set:
            top_pages[qid] = []
            continue

        page_filter = ds.field(page_qid_col) == qid
        if args.run_id and page_run_col:
            page_filter = page_filter & (ds.field(page_run_col) == args.run_id)

        page_table = page_ds.to_table(columns=read_page_cols, filter=page_filter)
        per_doc: dict[str, dict[int, tuple[int, float]]] = defaultdict(dict)
        for row in page_table.to_pylist():
            run_val = row.get(page_run_col) if page_run_col else None
            if args.run_id and page_run_col and str(run_val) != str(args.run_id):
                continue
            d = str(row.get(page_doc_col))
            if d not in doc_set:
                continue
            p = _to_int(row.get(page_idx_col))
            if p is None:
                continue
            rk = _to_int(row.get(page_rank_col), default=1_000_000_000) if page_rank_col else 1_000_000_000
            sc = _to_float(row.get(page_score_col)) if page_score_col else None
            if sc is None:
                sc = -float(rk)
            prev = per_doc[d].get(int(p))
            if prev is None or rk < prev[0]:
                per_doc[d][int(p)] = (int(rk), float(sc))

        q_rows: list[dict[str, Any]] = []
        for doc in docs:
            d = str(doc["doc_id"])
            seed = int(doc["seed_page"])
            entries = [(rk, p, sc) for p, (rk, sc) in per_doc.get(d, {}).items()]
            entries.sort(key=lambda x: x[0])

            chosen_pages = set()
            for rk, p, sc in entries[: args.pages_per_doc]:
                q_rows.append({"doc_id": d, "page_idx": int(p), "score": float(sc)})
                chosen_pages.add(int(p))
            if seed not in chosen_pages:
                q_rows.append({"doc_id": d, "page_idx": int(seed), "score": float(doc["score"])})
            if not entries:
                q_rows.append({"doc_id": d, "page_idx": int(seed), "score": float(doc["score"])})

        top_pages[qid] = q_rows

    out = {
        "meta": {
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "method": "doc_topk_plus_page_pool",
            "retrieval_doc_parquet": str(args.retrieval_doc_parquet),
            "retrieval_page_parquet": str(args.retrieval_page_parquet),
            "run_id": args.run_id,
            "topk_docs": args.topk_docs,
            "pages_per_doc": args.pages_per_doc,
            "max_qids": args.max_qids,
            "n_qids": len(top_pages),
        },
        "top_pages": top_pages,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {args.output_json}")
    print(f"QIDs: {len(top_pages)}")
    if top_pages:
        first_qid = next(iter(top_pages.keys()))
        print(f"Example qid={first_qid} n_candidates={len(top_pages[first_qid])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
