#!/usr/bin/env python3
"""Rerank retrieval parquet using baseline summaries + visual-lexicon metadata."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional
from urllib import error as urlerror
from urllib import request as urlrequest


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Rerank page-level retrieval candidates from parquet using both baseline "
            "page summaries and visual-lexicon metadata."
        )
    )
    p.add_argument("--retrieval-parquet", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--mmqa-jsonl", type=Path, default=None, help="MMQA split file for qid->question.")
    p.add_argument("--qrels-parquet", type=Path, default=None, help="Optional qrels parquet for before/after eval.")
    p.add_argument("--baseline-meta-jsonl", type=Path, required=True, help="Baseline metadata JSONL.")
    p.add_argument("--visual-meta-jsonl", type=Path, required=True, help="Visual lexicon metadata JSONL.")
    p.add_argument("--retrieval-run-id", type=str, default=None)
    p.add_argument("--retrieval-qid-col", type=str, default=None)
    p.add_argument("--retrieval-doc-col", type=str, default=None)
    p.add_argument("--retrieval-page-col", type=str, default=None)
    p.add_argument("--retrieval-score-col", type=str, default=None)
    p.add_argument("--retrieval-rank-col", type=str, default=None)
    p.add_argument("--qrels-qid-col", type=str, default=None)
    p.add_argument("--qrels-doc-col", type=str, default=None)
    p.add_argument("--qrels-page-col", type=str, default=None)
    p.add_argument("--qid", type=str, default=None, help="Optional single qid.")
    p.add_argument("--query", type=str, default=None, help="Optional query text for --qid.")
    p.add_argument("--qids-file", type=Path, default=None, help="Optional newline-separated qid allowlist.")
    p.add_argument("--max-qids", type=int, default=None, help="Optional cap after filtering/sorting qids.")
    p.add_argument("--topk-candidates", type=int, default=1000)
    p.add_argument("--save-top-k", type=int, default=1000)
    p.add_argument("--base-weight", type=float, default=1.0)
    p.add_argument("--summary-weight", type=float, default=0.9)
    p.add_argument("--visual-weight", type=float, default=1.1)
    p.add_argument("--visual-min-confidence", type=float, default=0.4)
    p.add_argument("--visual-uncertain-multiplier", type=float, default=0.35)
    p.add_argument(
        "--freeze-top-n",
        type=int,
        default=0,
        help="Keep top-N baseline ranks fixed before any reranking is applied.",
    )
    p.add_argument(
        "--window-start-rank",
        type=int,
        default=1,
        help="1-based rank where reranking window starts (inclusive).",
    )
    p.add_argument(
        "--window-end-rank",
        type=int,
        default=0,
        help="1-based rank where reranking window ends (inclusive). 0 means up to topk-candidates.",
    )
    p.add_argument(
        "--boost-only",
        action="store_true",
        help="Apply metadata as positive boost only (never subtract).",
    )
    p.add_argument(
        "--no-demotion",
        action="store_true",
        help="Never allow metadata to lower a candidate below its base-score-only value.",
    )
    p.add_argument(
        "--llm-rerank",
        action="store_true",
        help="Use an LLM to rerank candidates inside the configured rerank window.",
    )
    p.add_argument(
        "--llm-base-url",
        type=str,
        default=None,
        help="OpenAI-compatible base URL (for example http://127.0.0.1:8010/v1).",
    )
    p.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Model name used for LLM reranking.",
    )
    p.add_argument(
        "--llm-api-key-file",
        type=Path,
        default=None,
        help="Optional API key file for LLM reranking.",
    )
    p.add_argument(
        "--llm-window-max-candidates",
        type=int,
        default=30,
        help="Maximum candidates from rerank window shown to LLM.",
    )
    p.add_argument(
        "--llm-summary-max-chars",
        type=int,
        default=280,
        help="Max summary chars per candidate sent to LLM.",
    )
    p.add_argument(
        "--llm-max-retries",
        type=int,
        default=2,
        help="Retry attempts for each LLM rerank call.",
    )
    p.add_argument(
        "--llm-timeout-s",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds for LLM rerank call.",
    )
    p.add_argument(
        "--llm-max-tokens",
        type=int,
        default=512,
        help="Max completion tokens for LLM rerank call.",
    )
    p.add_argument(
        "--llm-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for LLM rerank call.",
    )
    p.add_argument(
        "--rewrite-query",
        action="store_true",
        help="Enable query-rewrite stage before second-pass reranking.",
    )
    p.add_argument(
        "--rewrite-base-url",
        type=str,
        default=None,
        help="OpenAI-compatible base URL for query rewrite. Defaults to --llm-base-url.",
    )
    p.add_argument(
        "--rewrite-model",
        type=str,
        default=None,
        help="Model for query rewrite. Defaults to --llm-model.",
    )
    p.add_argument(
        "--rewrite-api-key-file",
        type=Path,
        default=None,
        help="Optional API key file for query rewrite. Defaults to --llm-api-key-file.",
    )
    p.add_argument(
        "--rewrite-context-candidates",
        type=int,
        default=20,
        help="Number of top baseline candidates used as context for query rewrite.",
    )
    p.add_argument(
        "--rewrite-summary-max-chars",
        type=int,
        default=180,
        help="Max summary chars per candidate sent to query rewriter.",
    )
    p.add_argument(
        "--rewrite-max-retries",
        type=int,
        default=2,
        help="Retry attempts for each query-rewrite call.",
    )
    p.add_argument(
        "--rewrite-timeout-s",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds for query rewrite call.",
    )
    p.add_argument(
        "--rewrite-max-tokens",
        type=int,
        default=128,
        help="Max completion tokens for query rewrite call.",
    )
    p.add_argument(
        "--rewrite-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for query rewrite call.",
    )
    p.add_argument("--summary-ks", type=str, default="1,2,4,10,20,50,100,500")
    return p.parse_args()


def _pick_column(cols: list[str], explicit: str | None, candidates: list[str], label: str) -> str:
    if explicit is not None:
        if explicit not in cols:
            raise ValueError(f"{label} column {explicit!r} not found in columns: {cols}")
        return explicit
    for c in candidates:
        if c in cols:
            return c
    raise ValueError(f"Could not infer {label}. Available columns: {cols}")


def _parse_page_idx(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return int(value)
    s = str(value).strip()
    if not s:
        return None
    if re.fullmatch(r"-?\d+", s):
        try:
            return int(s)
        except Exception:
            return None
    m = re.search(r"(?:_page|#p|/|:)(\d+)$", s, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _parse_doc_id(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if "_page" in s:
        return s.split("_page", 1)[0]
    m = re.match(r"(.+?)[\s:/#-]*p(?:age)?[_-]?(\d+)$", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return s


def _load_retrieval_parquet(
    parquet_path: Path,
    qid_filter: set[str] | None,
    run_id_filter: str | None,
    qid_col_override: str | None,
    doc_col_override: str | None,
    page_col_override: str | None,
    score_col_override: str | None,
    rank_col_override: str | None,
) -> dict[str, list[dict[str, Any]]]:
    try:
        import pyarrow.dataset as ds
    except Exception as exc:
        raise ImportError("pyarrow is required to read --retrieval-parquet") from exc

    dataset = ds.dataset(str(parquet_path), format="parquet")
    cols = list(dataset.schema.names)
    run_id_col = "run_id" if "run_id" in cols else None
    qid_col = _pick_column(cols, qid_col_override, ["qid", "query_id", "question_id"], "retrieval qid")
    doc_col = _pick_column(cols, doc_col_override, ["doc_id", "document_id", "page_uid", "pid"], "retrieval doc")
    page_col = _pick_column(
        cols,
        page_col_override,
        ["page_idx", "page", "page_num", "page_no", "maxsim_best_page", "best_page", "page_id"],
        "retrieval page",
    )

    score_col = None
    if score_col_override:
        score_col = _pick_column(cols, score_col_override, [score_col_override], "retrieval score")
    else:
        for c in ["maxsim_score", "faiss_score", "score", "sim", "similarity"]:
            if c in cols:
                score_col = c
                break

    rank_col = None
    if rank_col_override:
        rank_col = _pick_column(cols, rank_col_override, [rank_col_override], "retrieval rank")
    else:
        for c in ["maxsim_rank", "faiss_rank", "rank", "page_rank", "retrieval_rank"]:
            if c in cols:
                rank_col = c
                break

    read_cols = [qid_col, doc_col, page_col]
    if run_id_col:
        read_cols.append(run_id_col)
    if score_col:
        read_cols.append(score_col)
    if rank_col:
        read_cols.append(rank_col)
    if "page_uid" in cols and "page_uid" not in read_cols:
        read_cols.append("page_uid")

    filters = []
    if qid_filter:
        filters.append(ds.field(qid_col).isin(list(qid_filter)))
    if run_id_filter and run_id_col and not qid_filter:
        filters.append(ds.field(run_id_col) == run_id_filter)
    flt = None
    if filters:
        flt = filters[0]
        for f in filters[1:]:
            flt = flt & f

    try:
        table = dataset.to_table(columns=read_cols, filter=flt)
    except Exception:
        table = dataset.to_table(columns=read_cols)

    dedup: dict[str, dict[tuple[str, int], dict[str, Any]]] = defaultdict(dict)
    for row in table.to_pylist():
        qid = row.get(qid_col)
        if qid is None:
            continue
        if qid_filter and str(qid) not in qid_filter:
            continue
        if run_id_filter and run_id_col and str(row.get(run_id_col)) != str(run_id_filter):
            continue

        doc_id = _parse_doc_id(row.get(doc_col))
        page_idx = _parse_page_idx(row.get(page_col))
        if doc_id is None and row.get("page_uid") is not None:
            doc_id = _parse_doc_id(row.get("page_uid"))
        if page_idx is None and row.get("page_uid") is not None:
            page_idx = _parse_page_idx(row.get("page_uid"))
        if page_idx is None and row.get(doc_col) is not None:
            page_idx = _parse_page_idx(row.get(doc_col))
        if doc_id is None or page_idx is None:
            continue

        score = row.get(score_col) if score_col else None
        rank = row.get(rank_col) if rank_col else None
        if score is None:
            try:
                score = -float(rank)
            except Exception:
                score = 0.0

        rec = {
            "doc_id": str(doc_id),
            "page_idx": int(page_idx),
            "score": float(score),
            "_rank": rank,
        }
        key = (str(doc_id), int(page_idx))
        q = str(qid)
        prev = dedup[q].get(key)
        if prev is None:
            dedup[q][key] = rec
            continue
        prev_rank = prev.get("_rank")
        cur_rank = rec.get("_rank")
        choose_cur = False
        if prev_rank is None and cur_rank is not None:
            choose_cur = True
        elif prev_rank is not None and cur_rank is not None:
            try:
                choose_cur = float(cur_rank) < float(prev_rank)
            except Exception:
                choose_cur = float(rec["score"]) > float(prev["score"])
        else:
            choose_cur = float(rec["score"]) > float(prev["score"])
        if choose_cur:
            dedup[q][key] = rec

    out: dict[str, list[dict[str, Any]]] = {}
    for qid, kv in dedup.items():
        rows = list(kv.values())
        if any(r.get("_rank") is not None for r in rows):
            def rank_key(r: dict[str, Any]) -> tuple[float, float]:
                rv = r.get("_rank")
                try:
                    rr = float(rv)
                except Exception:
                    rr = float("inf")
                return (rr, -float(r["score"]))
            rows.sort(key=rank_key)
        else:
            rows.sort(key=lambda r: float(r["score"]), reverse=True)
        for r in rows:
            r.pop("_rank", None)
        out[qid] = rows
    return out


def _load_qid2query(mmqa_jsonl: Path | None) -> dict[str, str]:
    if mmqa_jsonl is None:
        return {}
    out: dict[str, str] = {}
    with mmqa_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            qid = obj.get("qid") or obj.get("query_id") or obj.get("question_id") or obj.get("id")
            q = obj.get("question") or obj.get("query")
            if qid is None or q is None:
                continue
            out[str(qid)] = str(q)
    return out


def _load_qids_file(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    out: set[str] = set()
    with path.open() as f:
        for line in f:
            q = line.strip()
            if q:
                out.add(q)
    return out


def _load_qrels_targets_from_parquet(
    parquet_path: Path | None,
    qid_col_override: str | None,
    doc_col_override: str | None,
    page_col_override: str | None,
) -> dict[str, list[tuple[str, int | None]]]:
    if parquet_path is None:
        return {}
    try:
        import pyarrow.dataset as ds
    except Exception as exc:
        raise ImportError("pyarrow is required to read --qrels-parquet") from exc

    dataset = ds.dataset(str(parquet_path), format="parquet")
    cols = list(dataset.schema.names)
    qid_col = _pick_column(cols, qid_col_override, ["qid", "query_id", "question_id"], "qrels qid")
    doc_col = _pick_column(cols, doc_col_override, ["doc_id", "document_id", "pid", "page_uid"], "qrels doc")

    if page_col_override is not None:
        page_col = _pick_column(cols, page_col_override, [page_col_override], "qrels page")
    else:
        page_col = None
        for c in ["page_idx", "page_num", "page", "gold_page", "page_id", "maxsim_best_page"]:
            if c in cols:
                page_col = c
                break

    read_cols = [qid_col, doc_col]
    if page_col:
        read_cols.append(page_col)
    table = dataset.to_table(columns=read_cols)

    grouped: dict[str, dict[tuple[str, int | None], None]] = defaultdict(dict)
    for row in table.to_pylist():
        qid = row.get(qid_col)
        doc = _parse_doc_id(row.get(doc_col))
        if qid is None or doc is None:
            continue
        page = _parse_page_idx(row.get(page_col)) if page_col else None
        grouped[str(qid)][(str(doc), None if page is None else int(page))] = None

    return {qid: list(kv.keys()) for qid, kv in grouped.items()}


def _candidate_key_variants(doc_id: str, page_idx: int) -> list[str]:
    return [
        f"{doc_id}_page{page_idx}",
        f"{doc_id}#p{page_idx}",
        f"{doc_id}:{page_idx}",
        f"{doc_id}/{page_idx}",
    ]


def _extract_context_text(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for key in ("summary", "text", "snippet", "page_summary", "content"):
            v = obj.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def _load_context_map(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    suffixes = {s.lower() for s in path.suffixes}
    rows: list[Any] = []
    if ".jsonl" in suffixes:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            out: dict[str, str] = {}
            for key, value in payload.items():
                text = _extract_context_text(value)
                if text:
                    out[str(key)] = _norm_text(text)
            return out
        if isinstance(payload, list):
            rows = payload
        else:
            return {}

    out: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        doc_id = row.get("doc_id") or row.get("document_id")
        page_idx = row.get("page_idx", row.get("page", row.get("page_id")))
        text = _extract_context_text(row)
        if doc_id is None or page_idx is None or not text:
            continue
        try:
            page_idx = int(page_idx)
        except Exception:
            continue
        out[f"{doc_id}_page{page_idx}"] = _norm_text(text)
    return out


def _extract_first_json_object(text: str) -> Optional[dict]:
    if not text:
        return None
    candidate = text.strip()
    if candidate.startswith("{") and candidate.endswith("}"):
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    m = re.search(r"\{.*\}", candidate, flags=re.S)
    if not m:
        return None
    try:
        payload = json.loads(m.group(0))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _load_visual_lexicon_map(path: Path) -> dict[str, dict[str, tuple[str, float]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    out: dict[str, dict[str, tuple[str, float]]] = {}
    suffixes = {s.lower() for s in path.suffixes}
    rows: list[Any] = []
    if ".jsonl" in suffixes:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            rows = list(payload.values())

    for row in rows:
        if not isinstance(row, dict):
            continue
        doc_id = row.get("doc_id") or row.get("document_id")
        page_idx = row.get("page_idx", row.get("page", row.get("page_id")))
        if doc_id is None or page_idx is None:
            continue
        try:
            page_idx = int(page_idx)
        except Exception:
            continue
        summary_obj = row if isinstance(row.get("visual_lexicons"), list) else _extract_first_json_object(str(row.get("summary", "")))
        if not isinstance(summary_obj, dict):
            continue
        items = summary_obj.get("visual_lexicons")
        if not isinstance(items, list):
            continue
        page_terms: dict[str, tuple[str, float]] = {}
        for it in items:
            if not isinstance(it, dict):
                continue
            term = str(it.get("term", "")).strip()
            if not term:
                continue
            present = str(it.get("present", "")).strip().lower()
            conf_raw = it.get("confidence", 0.0)
            try:
                conf = float(conf_raw)
            except Exception:
                conf = 0.0
            if conf > 1.0:
                conf = conf / 100.0
            conf = max(0.0, min(conf, 1.0))
            page_terms[term] = (present, conf)
        if page_terms:
            out[f"{doc_id}_page{page_idx}"] = page_terms
    return out


def _norm_text(text: str) -> str:
    return " ".join(str(text).split())


def _lookup_context(context_map: dict[str, str], doc_id: str, page_idx: int) -> Optional[str]:
    for key in _candidate_key_variants(doc_id, page_idx):
        value = context_map.get(key)
        if value:
            return value
    return None


def _lookup_visual(page_map: dict[str, dict[str, tuple[str, float]]], doc_id: str, page_idx: int) -> Optional[dict[str, tuple[str, float]]]:
    for key in _candidate_key_variants(doc_id, page_idx):
        value = page_map.get(key)
        if value:
            return value
    return None


def _zscore(xs: list[float]) -> list[float]:
    if not xs:
        return []
    mean = sum(xs) / len(xs)
    var = sum((x - mean) * (x - mean) for x in xs) / len(xs)
    std = math.sqrt(var)
    if std < 1e-12:
        return [0.0 for _ in xs]
    return [(x - mean) / std for x in xs]


def _summary_match_score(query: str, summary_text: Optional[str]) -> float:
    if not summary_text:
        return 0.0
    stopwords = {
        "a", "an", "and", "as", "at", "by", "for", "from", "has", "have", "in", "is", "it", "its", "of", "on", "or",
        "that", "the", "their", "this", "to", "what", "when", "where", "which", "who", "with",
    }
    query_tokens = [tok for tok in re.findall(r"[A-Za-z0-9']+", query.casefold()) if tok not in stopwords]
    if not query_tokens:
        return 0.0
    summary_norm = summary_text.casefold()
    summary_tokens = set(re.findall(r"[A-Za-z0-9']+", summary_norm))
    overlap = [tok for tok in query_tokens if tok in summary_tokens]
    if not overlap:
        return 0.0
    overlap_count = len(overlap)
    coverage = overlap_count / max(len(set(query_tokens)), 1)
    bigram_hits = 0
    for i in range(len(query_tokens) - 1):
        bg = f"{query_tokens[i]} {query_tokens[i + 1]}"
        if bg in summary_norm:
            bigram_hits += 1
    exact_hit = 1.0 if _norm_text(query).casefold() in _norm_text(summary_text).casefold() else 0.0
    return (coverage * 4.0) + (overlap_count * 0.35) + (bigram_hits * 1.5) + (exact_hit * 6.0)


_VISUAL_QUERY_PATTERNS: dict[str, list[str]] = {
    "bald_man": [r"\bbald\b", r"\bshaved head\b", r"\bbald man\b"],
    "woman": [r"\bwoman\b", r"\bfemale\b", r"\blady\b", r"\bgirl\b"],
    "child": [r"\bchild\b", r"\bkid\b", r"\bboy\b", r"\bgirl\b", r"\bbaby\b", r"\btoddler\b"],
    "portrait_photo": [r"\bportrait\b", r"\bheadshot\b", r"\bphoto\b", r"\bpicture\b", r"\bpictured\b"],
    "logo": [r"\blogo\b", r"\bemblem\b", r"\bsymbol\b", r"\bcrest\b", r"\bseal\b", r"\bwordmark\b", r"\bbadge\b"],
    "map": [r"\bmap\b", r"\blocated\b", r"\bwhere\b"],
    "chart": [r"\bchart\b", r"\bgraph\b", r"\bplot\b", r"\btrend\b", r"\bstatistics\b"],
    "table": [r"\btable\b", r"\btabular\b", r"\brows?\b", r"\bcolumns?\b", r"\bstandings\b", r"\branking\b"],
    "building_or_campus": [r"\bbuilding\b", r"\bcampus\b", r"\bstadium\b", r"\bchurch\b", r"\bschool\b", r"\bhouse\b", r"\btower\b"],
    "sports_uniform": [r"\buniform\b", r"\bjersey\b", r"\bkit\b", r"\bhelmet\b"],
    "flag": [r"\bflag\b", r"\bbanner\b", r"\bnational flag\b", r"\bcountry flag\b"],
    "vehicle": [r"\bvehicle\b", r"\bcar\b", r"\btruck\b", r"\bbus\b", r"\btrain\b", r"\bplane\b", r"\baircraft\b", r"\bboat\b", r"\bship\b", r"\bmotorcycle\b", r"\bbike\b"],
}


def _query_visual_term_weights(query: str) -> dict[str, float]:
    q = query.casefold()
    weights: dict[str, float] = {}
    for term, pats in _VISUAL_QUERY_PATTERNS.items():
        hits = 0
        for pat in pats:
            if re.search(pat, q):
                hits += 1
        if hits > 0:
            weights[term] = 1.0 + (0.15 * (hits - 1))
    return weights


def _visual_match_score(
    query: str,
    visual_terms: Optional[dict[str, tuple[str, float]]],
    *,
    min_confidence: float,
    uncertain_multiplier: float,
) -> float:
    if not visual_terms:
        return 0.0
    term_weights = _query_visual_term_weights(query)
    if not term_weights:
        return 0.0
    total = 0.0
    for term, weight in term_weights.items():
        state_conf = visual_terms.get(term)
        if state_conf is None:
            continue
        state, conf = state_conf
        if conf < min_confidence:
            continue
        if state == "yes":
            total += float(weight) * float(conf)
        elif state == "uncertain":
            total += float(weight) * float(conf) * float(uncertain_multiplier)
    return total


def _gold_rank_multi(rows: list[dict[str, Any]], gold_targets: list[tuple[str, int | None]]) -> int | None:
    if not gold_targets:
        return None
    for i, r in enumerate(rows, start=1):
        d = str(r["doc_id"])
        p = int(r["page_idx"])
        for gd, gp in gold_targets:
            if d != str(gd):
                continue
            if gp is None or p == int(gp):
                return i
    return None


def _doc_rank_multi_from_page_rows(rows: list[dict[str, Any]], gold_targets: list[tuple[str, int | None]]) -> int | None:
    if not gold_targets:
        return None
    gold_docs = {str(gd) for gd, _ in gold_targets}
    if not gold_docs:
        return None
    seen: set[str] = set()
    rank = 0
    for r in rows:
        d = str(r["doc_id"])
        if d in seen:
            continue
        seen.add(d)
        rank += 1
        if d in gold_docs:
            return rank
    return None


def _doc_rank_multi_from_doc_rows(doc_ids: list[str], gold_targets: list[tuple[str, int | None]]) -> int | None:
    if not gold_targets:
        return None
    gold_docs = {str(gd) for gd, _ in gold_targets}
    if not gold_docs:
        return None
    for i, d in enumerate(doc_ids, start=1):
        if str(d) in gold_docs:
            return i
    return None


def _has_any_page_labeled_target(gold_targets: list[tuple[str, int | None]]) -> bool:
    return any(gp is not None for _, gp in gold_targets)


def _parse_summary_ks(ks_text: str) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for tok in str(ks_text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            k = int(tok)
        except Exception:
            continue
        if k <= 0 or k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out or [1, 2, 4, 10, 20, 50, 100, 500]


def _summary_from_diagnostics(diagnostics: dict[str, dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    rows = [d for d in diagnostics.values() if int(d.get("n_gold_targets", 0)) > 0]
    n = len(rows)
    before = [d.get("gold_before_rank") for d in rows]
    after = [d.get("gold_after_rank") for d in rows]
    before_num = [int(r) for r in before if r is not None]
    after_num = [int(r) for r in after if r is not None]
    both = [(int(b), int(a)) for b, a in zip(before, after) if b is not None and a is not None]
    improved = sum(1 for b, a in both if a < b)
    worsened = sum(1 for b, a in both if a > b)
    unchanged = sum(1 for b, a in both if a == b)
    out: dict[str, Any] = {
        "qids_with_gold_targets": n,
        "qids_with_before_rank": len(before_num),
        "qids_with_after_rank": len(after_num),
        "qids_with_both_ranks": len(both),
        "improved_count": improved,
        "worsened_count": worsened,
        "unchanged_count": unchanged,
        "avg_before_rank": (sum(before_num) / len(before_num)) if before_num else None,
        "avg_after_rank": (sum(after_num) / len(after_num)) if after_num else None,
        "median_before_rank": (float(statistics.median(before_num)) if before_num else None),
        "median_after_rank": (float(statistics.median(after_num)) if after_num else None),
        "mrr_before": (sum(1.0 / r for r in before_num) / n) if n else None,
        "mrr_after": (sum(1.0 / r for r in after_num) / n) if n else None,
    }
    for k in ks:
        out[f"recall@{k}_before"] = (sum(1 for r in before_num if r <= k) / n) if n else None
        out[f"recall@{k}_after"] = (sum(1 for r in after_num if r <= k) / n) if n else None
    return out


def _window_bounds(
    n_rows: int,
    *,
    freeze_top_n: int,
    window_start_rank: int,
    window_end_rank: int,
) -> tuple[int, int]:
    if n_rows <= 0:
        return (1, 0)
    start = max(1, int(window_start_rank), int(freeze_top_n) + 1)
    end = n_rows if int(window_end_rank) <= 0 else min(n_rows, int(window_end_rank))
    return (start, end)


def _read_api_key(path: Path | None) -> str:
    if path is None:
        return ""
    if not path.exists():
        return ""
    key = path.read_text().strip()
    return key


def _chat_completions_url(base_url: str) -> str:
    u = str(base_url).strip()
    if u.endswith("/chat/completions"):
        return u
    return u.rstrip("/") + "/chat/completions"


def _build_llm_rerank_prompt(
    *,
    query: str,
    rows: list[dict[str, Any]],
    summary_max_chars: int,
) -> str:
    def _visual_brief(terms: Optional[dict[str, tuple[str, float]]]) -> str:
        if not terms:
            return "none"
        yes = []
        uncertain = []
        for term, state_conf in terms.items():
            if not isinstance(state_conf, tuple) or len(state_conf) != 2:
                continue
            state, conf = state_conf
            s = str(state).strip().lower()
            try:
                c = float(conf)
            except Exception:
                c = 0.0
            if s == "yes":
                yes.append((str(term), c))
            elif s == "uncertain":
                uncertain.append((str(term), c))
        yes.sort(key=lambda z: z[1], reverse=True)
        uncertain.sort(key=lambda z: z[1], reverse=True)
        parts = []
        if yes:
            parts.append(
                "yes=" + ",".join(f"{t}({c:.2f})" for t, c in yes[:6])
            )
        if uncertain:
            parts.append(
                "uncertain=" + ",".join(f"{t}({c:.2f})" for t, c in uncertain[:4])
            )
        return "; ".join(parts) if parts else "none"

    lines = []
    for i, row in enumerate(rows, start=1):
        summary = _norm_text(str(row.get("summary_text", "")))[: max(1, int(summary_max_chars))]
        visual_brief = _visual_brief(row.get("visual_terms"))
        lines.append(
            f"[{i}] doc_id={row['doc_id']} page_idx={int(row['page_idx'])} "
            f"base_score={float(row['base_score']):.6f} summary={summary} visual={visual_brief}"
        )
    joined = "\n".join(lines)
    return (
        "You are a retrieval reranker.\n"
        "Given the query, candidate page summaries, and visual lexicon cues, reorder candidates from most relevant to least relevant.\n"
        "Favor candidates with direct evidence for answering the query.\n\n"
        f"QUERY:\n{_norm_text(query)}\n\n"
        "CANDIDATES:\n"
        f"{joined}\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "ordered_candidate_indices": [1, 2, 3]\n'
        "}\n"
        "Rules:\n"
        "- Use candidate indices shown above.\n"
        "- Do not invent new indices.\n"
        "- Include each chosen index at most once.\n"
    )


def _extract_ordered_indices(raw_reply: str, max_index: int) -> list[int]:
    if not raw_reply:
        return []
    payload = _extract_first_json_object(raw_reply)
    values = None
    if isinstance(payload, dict):
        for key in (
            "ordered_candidate_indices",
            "ordered_indices",
            "order",
            "ranking",
            "selected",
            "selected_indices",
        ):
            val = payload.get(key)
            if isinstance(val, list):
                values = val
                break
    out: list[int] = []
    seen: set[int] = set()

    def _push(idx_val: Any) -> None:
        try:
            idx = int(idx_val)
        except Exception:
            return
        if idx < 1 or idx > max_index:
            return
        if idx in seen:
            return
        seen.add(idx)
        out.append(idx)

    if isinstance(values, list):
        for item in values:
            if isinstance(item, dict):
                _push(item.get("candidate_index"))
            else:
                _push(item)
    else:
        for m in re.findall(r"\d+", raw_reply):
            _push(m)
    return out


def _reorder_rows_by_indices(rows: list[dict[str, Any]], ordered_indices: list[int]) -> list[dict[str, Any]]:
    if not rows:
        return []
    if not ordered_indices:
        return list(rows)
    by_idx = {i: row for i, row in enumerate(rows, start=1)}
    out: list[dict[str, Any]] = []
    used: set[int] = set()
    for idx in ordered_indices:
        row = by_idx.get(idx)
        if row is None or idx in used:
            continue
        out.append(row)
        used.add(idx)
    for i, row in enumerate(rows, start=1):
        if i in used:
            continue
        out.append(row)
    return out


def _call_llm_rerank_once(
    *,
    base_url: str,
    model: str,
    api_key: str,
    prompt: str,
    timeout_s: float,
    max_tokens: int,
    temperature: float,
) -> str:
    url = _chat_completions_url(base_url)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = urlrequest.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=float(timeout_s)) as resp:
        body = resp.read().decode("utf-8")
    obj = json.loads(body)
    return str(((obj.get("choices") or [{}])[0].get("message") or {}).get("content") or "")


def _build_query_rewrite_prompt(
    *,
    query: str,
    rows: list[dict[str, Any]],
    summary_max_chars: int,
) -> str:
    lines = []
    for i, row in enumerate(rows, start=1):
        summary = _norm_text(str(row.get("summary_text", "")))[: max(1, int(summary_max_chars))]
        lines.append(
            f"[{i}] doc_id={row['doc_id']} page_idx={int(row['page_idx'])} "
            f"base_score={float(row['base_score']):.6f} summary={summary}"
        )
    joined = "\n".join(lines)
    return (
        "You rewrite retrieval queries for document QA.\n"
        "Goal: preserve original intent, but add discriminative keywords so retrieval finds evidence pages.\n"
        "Do not broaden the question; keep answer target unchanged.\n\n"
        f"ORIGINAL_QUERY:\n{_norm_text(query)}\n\n"
        "TOP_CANDIDATE_SUMMARIES:\n"
        f"{joined}\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "rewritten_query": "<text>"\n'
        "}\n"
        "Rules:\n"
        "- Keep semantics equivalent to ORIGINAL_QUERY.\n"
        "- Add specific entities/aliases/terms only if supported by candidate summaries.\n"
        "- One sentence only.\n"
    )


def _extract_rewritten_query(raw_reply: str, original_query: str) -> str:
    if not raw_reply:
        return _norm_text(original_query)
    payload = _extract_first_json_object(raw_reply)
    if isinstance(payload, dict):
        for k in ("rewritten_query", "query", "rewrite"):
            v = payload.get(k)
            if isinstance(v, str) and v.strip():
                return _norm_text(v)
    first = _norm_text(raw_reply)
    if first:
        return first
    return _norm_text(original_query)


def _rewrite_query_with_llm(
    *,
    original_query: str,
    rows: list[dict[str, Any]],
    base_url: str,
    model: str,
    api_key: str,
    summary_max_chars: int,
    max_retries: int,
    timeout_s: float,
    max_tokens: int,
    temperature: float,
) -> tuple[str, dict[str, Any]]:
    trace = {
        "enabled": True,
        "attempted": False,
        "success": False,
        "error": None,
        "raw_reply_preview": None,
    }
    if not rows:
        return _norm_text(original_query), trace
    prompt = _build_query_rewrite_prompt(
        query=original_query,
        rows=rows,
        summary_max_chars=summary_max_chars,
    )
    attempts = max(1, int(max_retries))
    for attempt in range(1, attempts + 1):
        trace["attempted"] = True
        try:
            raw = _call_llm_rerank_once(
                base_url=base_url,
                model=model,
                api_key=api_key,
                prompt=prompt,
                timeout_s=timeout_s,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            trace["raw_reply_preview"] = _norm_text(raw)[:400]
            rewritten = _extract_rewritten_query(raw, original_query)
            if rewritten and rewritten.casefold() != _norm_text(original_query).casefold():
                trace["success"] = True
            return rewritten, trace
        except (urlerror.URLError, TimeoutError, OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            trace["error"] = str(exc)
        if attempt < attempts:
            time.sleep(0.6 * attempt)
    return _norm_text(original_query), trace


def _llm_rerank_rows(
    *,
    query: str,
    rows: list[dict[str, Any]],
    base_url: str,
    model: str,
    api_key: str,
    summary_max_chars: int,
    max_retries: int,
    timeout_s: float,
    max_tokens: int,
    temperature: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace = {
        "enabled": True,
        "attempted": False,
        "success": False,
        "selected_count": 0,
        "error": None,
        "raw_reply_preview": None,
    }
    if not rows:
        return rows, trace
    prompt = _build_llm_rerank_prompt(query=query, rows=rows, summary_max_chars=summary_max_chars)
    attempts = max(1, int(max_retries))
    for attempt in range(1, attempts + 1):
        trace["attempted"] = True
        try:
            raw = _call_llm_rerank_once(
                base_url=base_url,
                model=model,
                api_key=api_key,
                prompt=prompt,
                timeout_s=timeout_s,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            trace["raw_reply_preview"] = _norm_text(raw)[:400]
            indices = _extract_ordered_indices(raw, max_index=len(rows))
            if indices:
                trace["success"] = True
                trace["selected_count"] = len(indices)
                return _reorder_rows_by_indices(rows, indices), trace
            trace["error"] = "empty_or_unparseable_order"
        except (urlerror.URLError, TimeoutError, OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            trace["error"] = str(exc)
        if attempt < attempts:
            time.sleep(0.6 * attempt)
    return rows, trace


def main() -> int:
    args = _parse_args()
    summary_ks = _parse_summary_ks(args.summary_ks)
    llm_api_key = _read_api_key(args.llm_api_key_file)
    rewrite_api_key = _read_api_key(args.rewrite_api_key_file) if args.rewrite_api_key_file else llm_api_key
    if args.llm_rerank:
        if not args.llm_base_url:
            raise ValueError("--llm-rerank requires --llm-base-url")
        if not args.llm_model:
            raise ValueError("--llm-rerank requires --llm-model")
    if args.rewrite_query:
        rewrite_base_url = args.rewrite_base_url or args.llm_base_url
        rewrite_model = args.rewrite_model or args.llm_model
        if not rewrite_base_url:
            raise ValueError("--rewrite-query requires --rewrite-base-url or --llm-base-url")
        if not rewrite_model:
            raise ValueError("--rewrite-query requires --rewrite-model or --llm-model")

    qid_filter = {args.qid} if args.qid else None
    qids_allow = _load_qids_file(args.qids_file)
    qid2pages = _load_retrieval_parquet(
        parquet_path=args.retrieval_parquet,
        qid_filter=qid_filter,
        run_id_filter=args.retrieval_run_id,
        qid_col_override=args.retrieval_qid_col,
        doc_col_override=args.retrieval_doc_col,
        page_col_override=args.retrieval_page_col,
        score_col_override=args.retrieval_score_col,
        rank_col_override=args.retrieval_rank_col,
    )
    qid2query = _load_qid2query(args.mmqa_jsonl)
    qid2gold = _load_qrels_targets_from_parquet(
        parquet_path=args.qrels_parquet,
        qid_col_override=args.qrels_qid_col,
        doc_col_override=args.qrels_doc_col,
        page_col_override=args.qrels_page_col,
    )
    base_context_map = _load_context_map(args.baseline_meta_jsonl)
    visual_map = _load_visual_lexicon_map(args.visual_meta_jsonl)

    if args.qid is not None:
        if args.qid not in qid2pages:
            raise KeyError(f"qid={args.qid} not found in retrieval candidates")
        qids = [args.qid]
    else:
        qids = sorted(qid2pages.keys())
        if qids_allow is not None:
            qids = [q for q in qids if q in qids_allow]
        if args.max_qids is not None:
            qids = qids[: max(0, int(args.max_qids))]
    if not qids:
        raise ValueError("No qids to rerank")

    reranked_top_pages: dict[str, list[dict[str, Any]]] = {}
    reranked_top_docs: dict[str, list[str]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}

    for qid in qids:
        original_query = args.query if (args.qid == qid and args.query is not None) else qid2query.get(qid)
        if original_query is None:
            raise ValueError(f"No query text for qid={qid}. Provide --mmqa-jsonl or --query with --qid.")
        query = _norm_text(original_query)

        cands = qid2pages[qid][: args.topk_candidates]
        if not cands:
            reranked_top_pages[qid] = []
            reranked_top_docs[qid] = []
            diagnostics[qid] = {"n_candidates": 0}
            continue

        rewrite_trace = {
            "enabled": bool(args.rewrite_query),
            "attempted": False,
            "success": False,
            "error": None,
            "raw_reply_preview": None,
        }
        if args.rewrite_query:
            rewrite_rows = []
            for row in cands[: max(1, int(args.rewrite_context_candidates))]:
                d = str(row["doc_id"])
                p = int(row["page_idx"])
                rewrite_rows.append(
                    {
                        "doc_id": d,
                        "page_idx": p,
                        "base_score": float(row["score"]),
                        "summary_text": _lookup_context(base_context_map, d, p) or "",
                    }
                )
            rewrite_base_url = str(args.rewrite_base_url or args.llm_base_url)
            rewrite_model = str(args.rewrite_model or args.llm_model)
            query, rewrite_trace = _rewrite_query_with_llm(
                original_query=_norm_text(original_query),
                rows=rewrite_rows,
                base_url=rewrite_base_url,
                model=rewrite_model,
                api_key=rewrite_api_key,
                summary_max_chars=int(args.rewrite_summary_max_chars),
                max_retries=int(args.rewrite_max_retries),
                timeout_s=float(args.rewrite_timeout_s),
                max_tokens=int(args.rewrite_max_tokens),
                temperature=float(args.rewrite_temperature),
            )

        scored: list[dict[str, Any]] = []
        base_vals: list[float] = []
        summ_vals: list[float] = []
        vis_vals: list[float] = []
        for row in cands:
            d = str(row["doc_id"])
            p = int(row["page_idx"])
            base_score = float(row["score"])
            summary_text = _lookup_context(base_context_map, d, p)
            visual_terms = _lookup_visual(visual_map, d, p)
            summary_score = _summary_match_score(query, summary_text)
            visual_score = _visual_match_score(
                query=query,
                visual_terms=visual_terms,
                min_confidence=float(args.visual_min_confidence),
                uncertain_multiplier=float(args.visual_uncertain_multiplier),
            )
            rec = {
                "doc_id": d,
                "page_idx": p,
                "base_score": base_score,
                "summary_score": float(summary_score),
                "visual_score": float(visual_score),
                "summary_text": summary_text or "",
                "visual_terms": visual_terms or {},
            }
            scored.append(rec)
            base_vals.append(base_score)
            summ_vals.append(float(summary_score))
            vis_vals.append(float(visual_score))

        bz = _zscore(base_vals)
        sz = _zscore(summ_vals)
        vz = _zscore(vis_vals)
        for i, rec in enumerate(scored):
            rec["base_z"] = float(bz[i])
            rec["summary_z"] = float(sz[i])
            rec["visual_z"] = float(vz[i])
            base_component = float(args.base_weight) * rec["base_z"]
            meta_component = (
                (float(args.summary_weight) * rec["summary_z"])
                + (float(args.visual_weight) * rec["visual_z"])
            )
            if args.boost_only:
                meta_component = max(meta_component, 0.0)
            final_score = base_component + meta_component
            if args.no_demotion:
                final_score = max(final_score, base_component)
            rec["base_component"] = float(base_component)
            rec["meta_component"] = float(meta_component)
            rec["final_score"] = float(final_score)

        start_rank, end_rank = _window_bounds(
            len(scored),
            freeze_top_n=int(args.freeze_top_n),
            window_start_rank=int(args.window_start_rank),
            window_end_rank=int(args.window_end_rank),
        )
        if start_rank <= end_rank:
            prefix = scored[: start_rank - 1]
            window = scored[start_rank - 1 : end_rank]
            suffix = scored[end_rank:]
            window = sorted(window, key=lambda r: r["final_score"], reverse=True)
            llm_trace = {
                "enabled": bool(args.llm_rerank),
                "attempted": False,
                "success": False,
                "selected_count": 0,
                "error": None,
                "raw_reply_preview": None,
            }
            if args.llm_rerank and window:
                max_cands = max(1, int(args.llm_window_max_candidates))
                head = window[:max_cands]
                tail = window[max_cands:]
                head, llm_trace = _llm_rerank_rows(
                    query=query,
                    rows=head,
                    base_url=str(args.llm_base_url),
                    model=str(args.llm_model),
                    api_key=llm_api_key,
                    summary_max_chars=int(args.llm_summary_max_chars),
                    max_retries=int(args.llm_max_retries),
                    timeout_s=float(args.llm_timeout_s),
                    max_tokens=int(args.llm_max_tokens),
                    temperature=float(args.llm_temperature),
                )
                window = head + tail
            reranked = prefix + window + suffix
        else:
            reranked = list(scored)
            llm_trace = {
                "enabled": bool(args.llm_rerank),
                "attempted": False,
                "success": False,
                "selected_count": 0,
                "error": None,
                "raw_reply_preview": None,
            }

        save_rows = reranked[: args.save_top_k]
        reranked_top_pages[qid] = [
            {
                "doc_id": r["doc_id"],
                "page_idx": int(r["page_idx"]),
                "score": float(r["final_score"]),
                "base_score": float(r["base_score"]),
                "summary_score": float(r["summary_score"]),
                "visual_score": float(r["visual_score"]),
                "base_z": float(r["base_z"]),
                "summary_z": float(r["summary_z"]),
                "visual_z": float(r["visual_z"]),
                "base_component": float(r["base_component"]),
                "meta_component": float(r["meta_component"]),
            }
            for r in save_rows
        ]

        seen = set()
        docs: list[str] = []
        for r in save_rows:
            d = str(r["doc_id"])
            if d in seen:
                continue
            seen.add(d)
            docs.append(d)
        reranked_top_docs[qid] = docs

        gold_targets = qid2gold.get(qid, [])
        eval_granularity = "page" if _has_any_page_labeled_target(gold_targets) else "doc"
        if eval_granularity == "page":
            before_rank = _gold_rank_multi(cands, gold_targets)
            after_rank = _gold_rank_multi(reranked, gold_targets)
        else:
            before_rank = _doc_rank_multi_from_page_rows(cands, gold_targets)
            after_rank = _doc_rank_multi_from_doc_rows(reranked_top_docs[qid], gold_targets)

        diagnostics[qid] = {
            "n_candidates": len(cands),
            "n_gold_targets": len(gold_targets),
            "gold_before_rank": before_rank,
            "gold_after_rank": after_rank,
            "eval_granularity": eval_granularity,
            "weights": {
                "base_weight": float(args.base_weight),
                "summary_weight": float(args.summary_weight),
                "visual_weight": float(args.visual_weight),
            },
            "original_query": _norm_text(original_query),
            "effective_query": _norm_text(query),
            "query_rewrite_trace": rewrite_trace,
            "freeze_top_n": int(args.freeze_top_n),
            "window_start_rank": int(args.window_start_rank),
            "window_end_rank": int(args.window_end_rank),
            "boost_only": bool(args.boost_only),
            "no_demotion": bool(args.no_demotion),
            "llm_rerank_trace": llm_trace,
            "visual_min_confidence": float(args.visual_min_confidence),
        }

    out: dict[str, Any] = {
        "meta": {
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "method": "metadata_rerank_baseline_plus_visual_lexicon",
            "source_retrieval_parquet": str(args.retrieval_parquet),
            "source_baseline_meta_jsonl": str(args.baseline_meta_jsonl),
            "source_visual_meta_jsonl": str(args.visual_meta_jsonl),
            "source_mmqa_jsonl": None if args.mmqa_jsonl is None else str(args.mmqa_jsonl),
            "source_qrels_parquet": None if args.qrels_parquet is None else str(args.qrels_parquet),
            "retrieval_run_id": args.retrieval_run_id,
            "topk_candidates": int(args.topk_candidates),
            "save_top_k": int(args.save_top_k),
            "base_weight": float(args.base_weight),
            "summary_weight": float(args.summary_weight),
            "visual_weight": float(args.visual_weight),
            "visual_min_confidence": float(args.visual_min_confidence),
            "visual_uncertain_multiplier": float(args.visual_uncertain_multiplier),
            "freeze_top_n": int(args.freeze_top_n),
            "window_start_rank": int(args.window_start_rank),
            "window_end_rank": int(args.window_end_rank),
            "boost_only": bool(args.boost_only),
            "no_demotion": bool(args.no_demotion),
            "llm_rerank": bool(args.llm_rerank),
            "llm_base_url": args.llm_base_url,
            "llm_model": args.llm_model,
            "llm_api_key_file": None if args.llm_api_key_file is None else str(args.llm_api_key_file),
            "llm_window_max_candidates": int(args.llm_window_max_candidates),
            "llm_summary_max_chars": int(args.llm_summary_max_chars),
            "llm_max_retries": int(args.llm_max_retries),
            "llm_timeout_s": float(args.llm_timeout_s),
            "llm_max_tokens": int(args.llm_max_tokens),
            "llm_temperature": float(args.llm_temperature),
            "rewrite_query": bool(args.rewrite_query),
            "rewrite_base_url": args.rewrite_base_url,
            "rewrite_model": args.rewrite_model,
            "rewrite_api_key_file": None if args.rewrite_api_key_file is None else str(args.rewrite_api_key_file),
            "rewrite_context_candidates": int(args.rewrite_context_candidates),
            "rewrite_summary_max_chars": int(args.rewrite_summary_max_chars),
            "rewrite_max_retries": int(args.rewrite_max_retries),
            "rewrite_timeout_s": float(args.rewrite_timeout_s),
            "rewrite_max_tokens": int(args.rewrite_max_tokens),
            "rewrite_temperature": float(args.rewrite_temperature),
            "qids_file": None if args.qids_file is None else str(args.qids_file),
            "max_qids": args.max_qids,
            "summary_ks": summary_ks,
        },
        "top_docs": reranked_top_docs,
        "top_pages": reranked_top_pages,
        "diagnostics": diagnostics,
    }
    if args.qrels_parquet is not None:
        out["summary_metrics"] = _summary_from_diagnostics(diagnostics, ks=summary_ks)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as f:
        json.dump(out, f, indent=2)

    if "summary_metrics" in out:
        sm = out["summary_metrics"]
        print(
            "[summary] "
            f"qids_with_gold={sm.get('qids_with_gold_targets')} "
            f"recall@10 before={sm.get('recall@10_before')} after={sm.get('recall@10_after')} "
            f"mrr before={sm.get('mrr_before')} after={sm.get('mrr_after')}"
        )
    print(f"Saved reranked output: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
