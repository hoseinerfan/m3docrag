"""Run the agent scaffold on a subset of M3DocVQA examples.

This runner is intended for prototype validation, not final benchmark scoring.
It uses each example's supporting documents as the retrieval pool (oracle doc pool)
and runs the iterative agent loop with optional page-text context snippets.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from safetensors import safe_open

from m3docrag.agent import run_agent_session
from m3docrag.utils.paths import LOCAL_DATA_DIR, LOCAL_EMBEDDINGS_DIR

# Reuse helpers from existing example scripts to keep behavior consistent.
from run_agent_rag import (  # type: ignore
    build_rag_model,
    configure_warning_filters,
    make_candidate_context_fn,
    make_llm_call_local_hf,
    make_llm_call_stub,
    move_doc_embs,
    to_jsonable,
)
from build_agent_context_file import (  # type: ignore
    extract_doc_contexts,
    normalize_text,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev", choices=["dev", "train"])
    p.add_argument(
        "--mmqa-jsonl",
        type=Path,
        default=None,
        help="Path to MMQA_<split>.jsonl (defaults to LOCAL_DATA_DIR/m3-docvqa/multimodalqa/MMQA_<split>.jsonl).",
    )
    p.add_argument(
        "--pdf-dir",
        type=Path,
        default=None,
        help="Path to pdfs_<split> directory (defaults under LOCAL_DATA_DIR/m3-docvqa/splits).",
    )
    p.add_argument(
        "--embeddings-dir",
        type=Path,
        default=None,
        help="Directory with <doc_id>.safetensors (defaults to LOCAL_EMBEDDINGS_DIR/<embedding-name>).",
    )
    p.add_argument(
        "--embedding-name",
        default=None,
        help="Embedding folder under LOCAL_EMBEDDINGS_DIR if --embeddings-dir is not provided. "
        "Defaults to colpali-v1.2_m3-docvqa_<split>.",
    )
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-examples", type=int, default=20)
    p.add_argument(
        "--qid-file",
        type=Path,
        default=None,
        help="Optional txt/json/jsonl file to filter examples by qid.",
    )
    p.add_argument(
        "--max-pages-per-doc",
        type=int,
        default=None,
        help="Optional page cap per supporting doc (applies to embeddings and context extraction).",
    )
    p.add_argument("--context-backend", choices=["pypdf", "pymupdf", "easyocr", "auto"], default="pypdf")
    p.add_argument("--context-max-chars", type=int, default=400)
    p.add_argument(
        "--doc-pool-source",
        default="supporting-docs",
        choices=["supporting-docs", "retrieval-parquet"],
        help="Document pool per example: oracle supporting docs or top retrieved docs from parquet.",
    )
    p.add_argument(
        "--retrieval-edges-parquet",
        type=Path,
        default=None,
        help="Parquet file with per-qid retrieved docs (required when --doc-pool-source retrieval-parquet).",
    )
    p.add_argument(
        "--retrieval-topk-docs",
        type=int,
        default=1000,
        help="Max docs to load from retrieval parquet per qid.",
    )
    p.add_argument("--retrieval-qid-col", default=None, help="Optional override for qid column in retrieval parquet.")
    p.add_argument("--retrieval-doc-col", default=None, help="Optional override for doc_id column in retrieval parquet.")
    p.add_argument(
        "--retrieval-rank-col",
        default=None,
        help="Optional override for rank column in retrieval parquet (ascending rank).",
    )
    p.add_argument(
        "--retrieval-score-col",
        default=None,
        help="Optional override for score column in retrieval parquet (used if rank column absent).",
    )
    p.add_argument(
        "--rank-gold-source",
        default="mmqa-supporting-docs",
        choices=["mmqa-supporting-docs", "qrels-parquet", "union"],
        help="Gold doc source for rank diagnostics / 'selected gold doc' analysis.",
    )
    p.add_argument(
        "--qrels-parquet",
        type=Path,
        default=None,
        help="Optional qrels parquet for retrieval-rank gold docs (qid, doc_id).",
    )
    p.add_argument("--qrels-qid-col", default=None, help="Optional override for qid column in qrels parquet.")
    p.add_argument("--qrels-doc-col", default=None, help="Optional override for doc_id column in qrels parquet.")
    p.add_argument("--max-turns", type=int, default=2)
    p.add_argument("--pages-per-turn", type=int, default=3)
    p.add_argument("--n-return-pages", type=int, default=5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--policy-backend", default="stub", choices=["stub", "local-hf"])
    p.add_argument("--policy-model", default=None)
    p.add_argument("--policy-device", default=None)
    p.add_argument("--output-jsonl", type=Path, required=True, help="Output JSONL with per-example traces.")
    p.add_argument("--summary-json", type=Path, default=None, help="Optional summary JSON path.")
    p.add_argument("--save-context-dir", type=Path, default=None, help="Optional directory to save per-doc context maps.")
    p.add_argument(
        "--clear-doc-cache-each-example",
        action="store_true",
        help="Clear cached embeddings/page contexts after each example to reduce memory growth on long runs.",
    )
    p.add_argument("--stop-on-error", action="store_true")
    return p.parse_args()


def _load_json_or_jsonl(path: Path):
    suffixes = {s.lower() for s in path.suffixes}
    if ".jsonl" in suffixes:
        rows = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
    return json.loads(path.read_text())


def _load_qid_filter(path: Optional[Path]) -> Optional[set[str]]:
    if path is None:
        return None
    try:
        payload = _load_json_or_jsonl(path)
    except Exception:
        return {line.strip() for line in path.read_text().splitlines() if line.strip()}
    out = set()
    if isinstance(payload, dict):
        out.update(str(k) for k in payload.keys())
        return out
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, str):
                out.add(item)
            elif isinstance(item, dict):
                qid = item.get("qid") or item.get("id")
                if qid is not None:
                    out.add(str(qid))
        return out
    if isinstance(payload, str):
        out.add(payload)
        return out
    # Plain-text fallback if JSON parsing produced something unexpected.
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def load_mmqa_examples(path: Path) -> list[dict]:
    examples = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def extract_gold_answers(example: dict) -> list[str]:
    vals = example.get("answers", [])
    out: list[str] = []
    if isinstance(vals, list):
        for v in vals:
            if isinstance(v, str):
                out.append(v)
            elif isinstance(v, dict):
                for key in ("answer", "text", "value"):
                    if key in v and v[key] is not None:
                        out.append(str(v[key]))
                        break
    elif isinstance(vals, str):
        out.append(vals)
    return out


def extract_supporting_doc_ids(example: dict) -> list[str]:
    out: list[str] = []
    for row in example.get("supporting_context", []) or []:
        if isinstance(row, dict):
            doc_id = row.get("doc_id")
            if doc_id is not None:
                out.append(str(doc_id))
    # preserve order, dedupe
    seen = set()
    ordered = []
    for d in out:
        if d in seen:
            continue
        seen.add(d)
        ordered.append(d)
    return ordered


def _norm_answer(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = text.casefold()
    text = " ".join(text.split())
    return text or None


def _pick_column(
    names: list[str],
    explicit: Optional[str],
    candidates: list[str],
    label: str,
    required: bool = True,
) -> Optional[str]:
    if explicit is not None:
        if explicit not in names:
            raise ValueError(f"{label} column {explicit!r} not found in parquet columns: {names}")
        return explicit
    lower_map = {n.lower(): n for n in names}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(f"Could not auto-detect {label} column. Available columns: {names}")
    return None


def load_retrieval_doc_pools_from_parquet(
    *,
    parquet_path: Path,
    qids: list[str],
    topk_docs: int,
    qid_col: Optional[str],
    doc_col: Optional[str],
    rank_col: Optional[str],
    score_col: Optional[str],
) -> dict[str, list[dict]]:
    """Load top retrieved docs per qid from a parquet file.

    Returns:
        {qid: [{"doc_id": str, "rank": int, "score": float|None, "source_rank": Any}, ...]}
    """
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    qid_set = {str(q) for q in qids}

    try:
        import pyarrow.dataset as ds
    except Exception as exc:
        raise RuntimeError(
            "pyarrow is required to load retrieval parquet for --doc-pool-source retrieval-parquet"
        ) from exc

    dataset = ds.dataset(str(parquet_path), format="parquet")
    col_names = list(dataset.schema.names)
    qid_col = _pick_column(col_names, qid_col, ["qid", "query_id", "question_id"], "qid")
    doc_col = _pick_column(
        col_names,
        doc_col,
        [
            "doc_id",
            "document_id",
            "candidate_doc_id",
            "dst_doc_id",
            "target_doc_id",
            "node_id_dst",
        ],
        "doc_id",
    )
    rank_col = _pick_column(
        col_names,
        rank_col,
        ["rank", "retrieval_rank", "position", "pos", "idx"],
        "rank",
        required=False,
    )
    score_col = _pick_column(
        col_names,
        score_col,
        ["score", "retrieval_score", "sim", "similarity", "weight"],
        "score",
        required=False,
    )

    cols = [qid_col, doc_col]
    if rank_col:
        cols.append(rank_col)
    if score_col and score_col not in cols:
        cols.append(score_col)

    try:
        table = dataset.to_table(columns=cols, filter=ds.field(qid_col).isin(list(qid_set)))
    except Exception:
        # Fallback if parquet/qid dtype mismatch prevents pushdown filtering.
        table = dataset.to_table(columns=cols)

    rows = []
    for row in table.to_pylist():
        qid = row.get(qid_col)
        doc_id = row.get(doc_col)
        if qid is None or doc_id is None:
            continue
        qid = str(qid)
        if qid not in qid_set:
            continue
        rows.append(
            {
                "qid": qid,
                "doc_id": str(doc_id),
                "source_rank": row.get(rank_col) if rank_col else None,
                "score": row.get(score_col) if score_col else None,
            }
        )

    grouped: dict[str, list[dict]] = {q: [] for q in qids}
    for row in rows:
        grouped.setdefault(row["qid"], []).append(row)

    out: dict[str, list[dict]] = {}
    for qid in qids:
        items = grouped.get(qid, [])
        if rank_col:
            def _rank_key(x):
                r = x.get("source_rank")
                try:
                    return (0, int(r))
                except Exception:
                    return (1, float("inf"))
            items = sorted(items, key=_rank_key)
        elif score_col:
            def _score_key(x):
                s = x.get("score")
                try:
                    return float(s)
                except Exception:
                    return float("-inf")
            items = sorted(items, key=_score_key, reverse=True)

        dedup = []
        seen_docs = set()
        for row in items:
            doc_id = row["doc_id"]
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            dedup.append(row)
            if topk_docs is not None and len(dedup) >= topk_docs:
                break

        normalized = []
        for i, row in enumerate(dedup, start=1):
            normalized.append(
                {
                    "doc_id": row["doc_id"],
                    "rank": i,
                    "score": (float(row["score"]) if row.get("score") is not None else None),
                    "source_rank": row.get("source_rank"),
                }
            )
        out[qid] = normalized

    logger.info(
        f"Loaded retrieval doc pools from {parquet_path} for {len(out)} qids "
        f"(avg docs/qid={sum(len(v) for v in out.values()) / max(len(out), 1):.1f})"
    )
    return out


def load_qrels_doc_map_from_parquet(
    *,
    parquet_path: Path,
    qids: list[str],
    qid_col: Optional[str],
    doc_col: Optional[str],
) -> dict[str, list[str]]:
    """Load gold doc ids per qid from qrels parquet."""
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    qid_set = {str(q) for q in qids}

    try:
        import pyarrow.dataset as ds
    except Exception as exc:
        raise RuntimeError("pyarrow is required to load qrels parquet for rank diagnostics") from exc

    dataset = ds.dataset(str(parquet_path), format="parquet")
    col_names = list(dataset.schema.names)
    qid_col = _pick_column(col_names, qid_col, ["qid", "query_id", "question_id"], "qrels qid")
    doc_col = _pick_column(
        col_names,
        doc_col,
        ["doc_id", "document_id", "candidate_doc_id", "target_doc_id"],
        "qrels doc_id",
    )

    try:
        table = dataset.to_table(columns=[qid_col, doc_col], filter=ds.field(qid_col).isin(list(qid_set)))
    except Exception:
        table = dataset.to_table(columns=[qid_col, doc_col])

    grouped: dict[str, list[str]] = {q: [] for q in qids}
    seen_per_qid: dict[str, set[str]] = {q: set() for q in qids}
    for row in table.to_pylist():
        qid = row.get(qid_col)
        doc_id = row.get(doc_col)
        if qid is None or doc_id is None:
            continue
        qid = str(qid)
        if qid not in qid_set:
            continue
        doc_id = str(doc_id)
        if doc_id in seen_per_qid.setdefault(qid, set()):
            continue
        seen_per_qid[qid].add(doc_id)
        grouped.setdefault(qid, []).append(doc_id)

    logger.info(
        f"Loaded qrels gold docs from {parquet_path} for {len(grouped)} qids "
        f"(avg gold docs/qid={sum(len(v) for v in grouped.values()) / max(len(grouped), 1):.2f})"
    )
    return grouped


def compute_gold_doc_rank_info(gold_doc_ids: list[str], doc_pool_ids: list[str]) -> dict:
    pos = {doc_id: i + 1 for i, doc_id in enumerate(doc_pool_ids)}
    gold_ranks = {doc_id: pos[doc_id] for doc_id in gold_doc_ids if doc_id in pos}
    best = min(gold_ranks.values()) if gold_ranks else None
    return {
        "gold_doc_ranks_in_doc_pool": gold_ranks,
        "num_gold_docs_in_doc_pool": len(gold_ranks),
        "best_gold_doc_rank_in_doc_pool": best,
    }


def compute_agent_selection_rank_info(agent_payload: dict, gold_doc_ids: list[str], doc_pool_ids: list[str]) -> dict:
    pos = {doc_id: i + 1 for i, doc_id in enumerate(doc_pool_ids)}
    gold_set = set(gold_doc_ids)
    turns = []
    first_turn_with_gold = None
    all_selected_gold = []

    for step in agent_payload.get("steps", []) or []:
        seen = set()
        selected_doc_ids = []
        for item in step.get("selected_pages", []) or []:
            if not isinstance(item, list) or len(item) < 1:
                continue
            doc_id = str(item[0])
            if doc_id in seen:
                continue
            seen.add(doc_id)
            selected_doc_ids.append(doc_id)

        selected_gold = [d for d in selected_doc_ids if d in gold_set]
        if selected_gold and first_turn_with_gold is None:
            first_turn_with_gold = step.get("turn")
        all_selected_gold.extend([d for d in selected_gold if d not in all_selected_gold])

        turns.append(
            {
                "turn": step.get("turn"),
                "selected_doc_ids": selected_doc_ids,
                "selected_doc_ranks_in_doc_pool": {d: pos.get(d) for d in selected_doc_ids},
                "selected_gold_docs": selected_gold,
                "selected_gold_doc_ranks_in_doc_pool": {d: pos.get(d) for d in selected_gold},
            }
        )

    return {
        "first_turn_with_gold_doc_selected": first_turn_with_gold,
        "selected_any_gold_doc": first_turn_with_gold is not None,
        "selected_gold_docs_any_turn": all_selected_gold,
        "per_turn_doc_selection": turns,
    }


class DocCache:
    def __init__(
        self,
        *,
        embeddings_dir: Path,
        pdf_dir: Path,
        max_pages_per_doc: Optional[int],
        context_backend: str,
        context_max_chars: int,
        save_context_dir: Optional[Path],
    ):
        self.embeddings_dir = embeddings_dir
        self.pdf_dir = pdf_dir
        self.max_pages_per_doc = max_pages_per_doc
        self.context_backend = context_backend
        self.context_max_chars = context_max_chars
        self.save_context_dir = save_context_dir
        self._emb_cache: dict[str, torch.Tensor] = {}
        self._ctx_cache: dict[str, dict[str, str]] = {}

    def clear(self) -> None:
        self._emb_cache.clear()
        self._ctx_cache.clear()

    def get_doc_embeddings(self, doc_id: str) -> torch.Tensor:
        if doc_id in self._emb_cache:
            return self._emb_cache[doc_id]
        path = self.embeddings_dir / f"{doc_id}.safetensors"
        if not path.exists():
            raise FileNotFoundError(f"Missing embeddings for doc {doc_id}: {path}")
        with safe_open(str(path), framework="pt", device="cpu") as f:
            embs = f.get_tensor("embeddings")
        if self.max_pages_per_doc is not None:
            embs = embs[: self.max_pages_per_doc]
        embs = embs.to(torch.bfloat16)
        self._emb_cache[doc_id] = embs
        return embs

    def get_doc_context_map(self, doc_id: str) -> dict[str, str]:
        if doc_id in self._ctx_cache:
            return self._ctx_cache[doc_id]

        embs = self.get_doc_embeddings(doc_id)
        n_pages = int(embs.shape[0])
        pdf_path = self.pdf_dir / f"{doc_id}.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing PDF for doc {doc_id}: {pdf_path}")
        page_idxs = list(range(n_pages))
        page_texts = extract_doc_contexts(pdf_path, page_idxs, backend=self.context_backend, ocr_reader=None)

        context_map: dict[str, str] = {}
        for page_idx in page_idxs:
            text = normalize_text(page_texts.get(page_idx, ""), max_chars=self.context_max_chars)
            if text:
                context_map[f"{doc_id}_page{page_idx}"] = text

        self._ctx_cache[doc_id] = context_map
        if self.save_context_dir is not None:
            self.save_context_dir.mkdir(parents=True, exist_ok=True)
            (self.save_context_dir / f"{doc_id}.json").write_text(json.dumps(context_map, indent=2) + "\n")
        return context_map


def main():
    args = parse_args()
    configure_warning_filters()

    mmqa_jsonl = args.mmqa_jsonl
    if mmqa_jsonl is None:
        mmqa_jsonl = Path(LOCAL_DATA_DIR) / "m3-docvqa" / "multimodalqa" / f"MMQA_{args.split}.jsonl"
    if not mmqa_jsonl.exists():
        raise FileNotFoundError(mmqa_jsonl)

    pdf_dir = args.pdf_dir
    if pdf_dir is None:
        pdf_dir = Path(LOCAL_DATA_DIR) / "m3-docvqa" / "splits" / f"pdfs_{args.split}"
    if not pdf_dir.exists():
        raise FileNotFoundError(pdf_dir)

    embeddings_dir = args.embeddings_dir
    if embeddings_dir is None:
        embedding_name = args.embedding_name or f"colpali-v1.2_m3-docvqa_{args.split}"
        embeddings_dir = Path(LOCAL_EMBEDDINGS_DIR) / embedding_name
    if not embeddings_dir.exists():
        raise FileNotFoundError(embeddings_dir)

    examples = load_mmqa_examples(mmqa_jsonl)
    qid_filter = _load_qid_filter(args.qid_file)
    if qid_filter is not None:
        examples = [ex for ex in examples if str(ex.get("qid")) in qid_filter]

    if args.start_index:
        examples = examples[args.start_index :]
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    if not examples:
        raise ValueError("No examples selected.")

    logger.info(f"Selected {len(examples)} examples from {mmqa_jsonl}")

    selected_qids = [str(ex.get("qid", f"idx_{i}")) for i, ex in enumerate(examples)]

    retrieval_doc_pools = None
    if args.doc_pool_source == "retrieval-parquet":
        if args.retrieval_edges_parquet is None:
            raise ValueError("--retrieval-edges-parquet is required when --doc-pool-source retrieval-parquet")
        retrieval_doc_pools = load_retrieval_doc_pools_from_parquet(
            parquet_path=args.retrieval_edges_parquet,
            qids=selected_qids,
            topk_docs=args.retrieval_topk_docs,
            qid_col=args.retrieval_qid_col,
            doc_col=args.retrieval_doc_col,
            rank_col=args.retrieval_rank_col,
            score_col=args.retrieval_score_col,
        )
    qrels_doc_map = None
    if args.rank_gold_source in {"qrels-parquet", "union"}:
        if args.qrels_parquet is None:
            raise ValueError("--qrels-parquet is required when --rank-gold-source is qrels-parquet or union")
        qrels_doc_map = load_qrels_doc_map_from_parquet(
            parquet_path=args.qrels_parquet,
            qids=selected_qids,
            qid_col=args.qrels_qid_col,
            doc_col=args.qrels_doc_col,
        )

    rag_model = build_rag_model(device=args.device)
    policy_device = args.policy_device or args.device
    if args.policy_backend == "stub":
        llm_call = make_llm_call_stub()
    else:
        if not args.policy_model:
            raise ValueError("--policy-model is required when --policy-backend local-hf")
        llm_call = make_llm_call_local_hf(args.policy_model, device=policy_device)

    cache = DocCache(
        embeddings_dir=embeddings_dir,
        pdf_dir=pdf_dir,
        max_pages_per_doc=args.max_pages_per_doc,
        context_backend=args.context_backend,
        context_max_chars=args.context_max_chars,
        save_context_dir=args.save_context_dir,
    )
    if args.clear_doc_cache_each_example:
        logger.info("Enabled --clear-doc-cache-each-example (lower memory, slower runtime).")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary_json or (args.output_jsonl.parent / f"{args.output_jsonl.stem}_summary.json")

    summary_rows = []
    counts = {}
    retrieval_diag_counts = {
        "examples_with_any_gold_in_doc_pool": 0,
        "examples_with_gold_rank_le_10": 0,
        "examples_with_gold_rank_le_100": 0,
        "examples_with_agent_selected_gold_doc": 0,
        "examples_with_agent_selected_gold_doc_turn1": 0,
    }
    best_gold_ranks = []

    with args.output_jsonl.open("w") as fout:
        for i, example in enumerate(examples, start=1):
            qid = str(example.get("qid", f"idx_{i-1}"))
            question = str(example.get("question", "")).strip()
            gold_answers = extract_gold_answers(example)
            supporting_doc_ids = extract_supporting_doc_ids(example)
            qrels_gold_doc_ids = list((qrels_doc_map or {}).get(qid, []))

            if args.rank_gold_source == "mmqa-supporting-docs":
                rank_gold_doc_ids = list(supporting_doc_ids)
            elif args.rank_gold_source == "qrels-parquet":
                rank_gold_doc_ids = qrels_gold_doc_ids
            else:
                rank_gold_doc_ids = []
                seen_rank_gold = set()
                for doc_id in supporting_doc_ids + qrels_gold_doc_ids:
                    if doc_id in seen_rank_gold:
                        continue
                    seen_rank_gold.add(doc_id)
                    rank_gold_doc_ids.append(doc_id)

            if args.doc_pool_source == "supporting-docs":
                doc_pool_entries = [{"doc_id": d, "rank": j + 1, "score": None, "source_rank": None} for j, d in enumerate(supporting_doc_ids)]
            else:
                assert retrieval_doc_pools is not None
                doc_pool_entries = retrieval_doc_pools.get(qid, [])
            doc_pool_ids = [str(x["doc_id"]) for x in doc_pool_entries]
            gold_rank_info = compute_gold_doc_rank_info(rank_gold_doc_ids, doc_pool_ids)
            best_gold_rank = gold_rank_info["best_gold_doc_rank_in_doc_pool"]
            if best_gold_rank is not None:
                retrieval_diag_counts["examples_with_any_gold_in_doc_pool"] += 1
                best_gold_ranks.append(best_gold_rank)
                if best_gold_rank <= 10:
                    retrieval_diag_counts["examples_with_gold_rank_le_10"] += 1
                if best_gold_rank <= 100:
                    retrieval_diag_counts["examples_with_gold_rank_le_100"] += 1

            started = time.time()
            row = {
                "example_index": i - 1 + args.start_index,
                "qid": qid,
                "question": question,
                "gold_answers": gold_answers,
                "supporting_doc_ids": supporting_doc_ids,
                "rank_gold_source": args.rank_gold_source,
                "rank_gold_doc_ids": rank_gold_doc_ids,
                "qrels_gold_doc_ids": (qrels_gold_doc_ids if qrels_doc_map is not None else None),
                "doc_pool_source": args.doc_pool_source,
                "doc_pool_doc_ids": doc_pool_ids,
                "doc_pool_size": len(doc_pool_ids),
                **gold_rank_info,
            }

            try:
                if not question:
                    raise ValueError("Missing question")
                if not supporting_doc_ids:
                    raise ValueError("No supporting_doc_ids in example")
                if not doc_pool_ids:
                    raise ValueError(f"No doc pool docs for qid={qid} (source={args.doc_pool_source})")

                docid2embs_cpu = {}
                merged_context_map: dict[str, str] = {}
                for doc_id in doc_pool_ids:
                    embs = cache.get_doc_embeddings(doc_id)
                    docid2embs_cpu[doc_id] = embs
                    merged_context_map.update(cache.get_doc_context_map(doc_id))

                candidate_context_fn = make_candidate_context_fn(
                    merged_context_map,
                    max_chars=args.context_max_chars,
                )
                docid2embs = move_doc_embs(docid2embs_cpu, args.device)

                result = run_agent_session(
                    query=question,
                    rag_model=rag_model,
                    docid2embs=docid2embs,
                    token2pageuid=None,
                    all_token_embeddings=None,
                    max_turns=args.max_turns,
                    pages_per_turn=args.pages_per_turn,
                    n_return_pages=args.n_return_pages,
                    llm_call=llm_call,
                    candidate_context_fn=candidate_context_fn,
                )

                payload = to_jsonable(result)
                reason = payload.get("reason")
                counts[reason] = counts.get(reason, 0) + 1
                pred_answer = payload.get("answer")
                agent_rank_info = compute_agent_selection_rank_info(payload, rank_gold_doc_ids, doc_pool_ids)
                if agent_rank_info["selected_any_gold_doc"]:
                    retrieval_diag_counts["examples_with_agent_selected_gold_doc"] += 1
                if agent_rank_info["first_turn_with_gold_doc_selected"] == 1:
                    retrieval_diag_counts["examples_with_agent_selected_gold_doc_turn1"] += 1
                row.update(
                    {
                        "reason": reason,
                        "pred_answer": pred_answer,
                        "pred_answer_exact_in_gold": (
                            _norm_answer(pred_answer) in {_norm_answer(a) for a in gold_answers if _norm_answer(a)}
                            if pred_answer is not None and gold_answers
                            else None
                        ),
                        **agent_rank_info,
                        "agent": payload,
                    }
                )
                logger.info(
                    f"[{i}/{len(examples)}] qid={qid} | reason={reason} | "
                    f"pool_docs={len(doc_pool_ids)} | best_gold_rank={best_gold_rank} | answer={pred_answer}"
                )
            except Exception as exc:
                row.update(
                    {
                        "reason": "error",
                        "pred_answer": None,
                        "pred_answer_exact_in_gold": None,
                        "error": str(exc),
                    }
                )
                counts["error"] = counts.get("error", 0) + 1
                logger.exception(f"[{i}/{len(examples)}] qid={qid} failed: {exc}")
                if args.stop_on_error:
                    raise
            finally:
                row["elapsed_sec"] = round(time.time() - started, 3)
                fout.write(json.dumps(row) + "\n")
                fout.flush()
                if args.clear_doc_cache_each_example:
                    cache.clear()
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()

    summary_rows = {
        "num_examples": len(examples),
        "counts": counts,
        "output_jsonl": str(args.output_jsonl),
        "mmqa_jsonl": str(mmqa_jsonl),
        "pdf_dir": str(pdf_dir),
        "embeddings_dir": str(embeddings_dir),
        "agent_config": {
            "max_turns": args.max_turns,
            "pages_per_turn": args.pages_per_turn,
            "n_return_pages": args.n_return_pages,
            "policy_backend": args.policy_backend,
            "policy_model": args.policy_model,
            "policy_device": policy_device,
            "device": args.device,
            "context_backend": args.context_backend,
            "context_max_chars": args.context_max_chars,
            "max_pages_per_doc": args.max_pages_per_doc,
            "doc_pool": args.doc_pool_source,
            "rank_gold_source": args.rank_gold_source,
            "retrieval_edges_parquet": (str(args.retrieval_edges_parquet) if args.retrieval_edges_parquet else None),
            "retrieval_topk_docs": args.retrieval_topk_docs if args.doc_pool_source == "retrieval-parquet" else None,
            "qrels_parquet": (str(args.qrels_parquet) if args.qrels_parquet else None),
        },
        "retrieval_diagnostics": {
            **retrieval_diag_counts,
            "avg_best_gold_doc_rank_in_doc_pool": (
                sum(best_gold_ranks) / len(best_gold_ranks) if best_gold_ranks else None
            ),
            "median_best_gold_doc_rank_in_doc_pool": (
                sorted(best_gold_ranks)[len(best_gold_ranks) // 2] if best_gold_ranks else None
            ),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_rows, indent=2) + "\n")
    logger.info(f"Saved subset summary: {summary_path}")


if __name__ == "__main__":
    main()
