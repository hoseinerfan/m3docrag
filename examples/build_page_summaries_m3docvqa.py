"""Build offline page summaries for M3DocVQA PDFs.

This is a subset-friendly preprocessor intended to generate page-level summaries
that can be reused by the lightweight selector via
`examples/run_agent_m3docvqa_subset.py --page-summaries-file`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

from m3docrag.utils.paths import LOCAL_DATA_DIR, LOCAL_MODEL_DIR
from m3docrag.utils.pdfs import get_images_from_pdf
from m3docrag.vqa import VQAModel

from run_agent_m3docvqa_subset import (  # type: ignore
    _load_json_or_jsonl,
    _load_qid_filter,
    extract_supporting_doc_ids,
    load_mmqa_examples,
    load_retrieval_doc_pools_from_parquet,
)


SUMMARY_PROMPT = (
    "Summarize this single document page for retrieval. "
    "Mention the main topic, key names/titles/dates, and important visible figures, logos, symbols, or charts. "
    "Keep the summary concise and factual, under 80 words."
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev", choices=["dev", "train"])
    p.add_argument(
        "--mmqa-jsonl",
        type=Path,
        default=None,
        help="Path to MMQA_<split>.jsonl (defaults under LOCAL_DATA_DIR).",
    )
    p.add_argument(
        "--pdf-dir",
        type=Path,
        default=None,
        help="Path to pdfs_<split> directory (defaults under LOCAL_DATA_DIR).",
    )
    p.add_argument(
        "--qid-file",
        type=Path,
        default=None,
        help="Optional txt/json/jsonl file to filter examples by qid.",
    )
    p.add_argument(
        "--doc-id-file",
        type=Path,
        default=None,
        help="Optional txt/json/jsonl file with doc_ids. If provided, qid-based discovery is skipped.",
    )
    p.add_argument(
        "--doc-pool-source",
        default="retrieval-parquet",
        choices=["supporting-docs", "retrieval-parquet"],
        help="How to discover doc ids when --doc-id-file is not provided.",
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
        default=100,
        help="Max docs to load from retrieval parquet per qid when discovering doc ids.",
    )
    p.add_argument("--retrieval-qid-col", default=None)
    p.add_argument("--retrieval-doc-col", default=None)
    p.add_argument("--retrieval-rank-col", default=None)
    p.add_argument("--retrieval-score-col", default=None)
    p.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional cap on the number of discovered docs to summarize.",
    )
    p.add_argument(
        "--max-pages-per-doc",
        type=int,
        default=None,
        help="Optional cap on pages summarized per document.",
    )
    p.add_argument(
        "--model-name-or-path",
        default="Qwen2-VL-7B-Instruct",
        help="Local model path or folder name under LOCAL_MODEL_DIR.",
    )
    p.add_argument(
        "--model-type",
        default="qwen2",
        help="VQA model type passed to m3docrag.vqa.VQAModel (default: qwen2).",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--bits", type=int, default=16, choices=[4, 16])
    p.add_argument(
        "--qwen-use-fast-processor",
        default="auto",
        choices=["auto", "true", "false"],
        help="Override Qwen processor speed path. 'auto' keeps model defaults.",
    )
    p.add_argument(
        "--page-max-retries",
        type=int,
        default=2,
        help="Retries per page for transient CUDA/runtime failures.",
    )
    p.add_argument(
        "--page-retry-wait-seconds",
        type=float,
        default=1.0,
        help="Sleep between page retries.",
    )
    p.add_argument(
        "--fail-on-page-error",
        action="store_true",
        help="Abort run if a page still fails after retries.",
    )
    p.add_argument(
        "--prompt",
        default=SUMMARY_PROMPT,
        help="Prompt used to summarize each page.",
    )
    p.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        help="Output JSONL path. Rows are compatible with --page-summaries-file.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output-jsonl instead of appending and resuming.",
    )
    return p.parse_args()


def _resolve_local_model_path(model_name_or_path: str) -> Path:
    p = Path(model_name_or_path)
    if p.exists():
        return p
    candidate = Path(LOCAL_MODEL_DIR) / model_name_or_path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Model path does not exist: {model_name_or_path}. "
        f"Tried {p} and {candidate}."
    )


def _load_doc_id_file(path: Path) -> list[str]:
    try:
        payload = _load_json_or_jsonl(path)
    except Exception:
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if isinstance(payload, dict):
        if "doc_ids" in payload and isinstance(payload["doc_ids"], list):
            return [str(x) for x in payload["doc_ids"]]
        return [str(k) for k in payload.keys()]
    if isinstance(payload, list):
        out = []
        for item in payload:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                doc_id = item.get("doc_id") or item.get("document_id")
                if doc_id is not None:
                    out.append(str(doc_id))
        return out
    if isinstance(payload, str):
        return [payload]
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _dedupe_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = str(item).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def discover_doc_ids(args) -> list[str]:
    if args.doc_id_file is not None:
        return _dedupe_keep_order(_load_doc_id_file(args.doc_id_file))

    mmqa_jsonl = args.mmqa_jsonl
    if mmqa_jsonl is None:
        mmqa_jsonl = Path(LOCAL_DATA_DIR) / "m3-docvqa" / "multimodalqa" / f"MMQA_{args.split}.jsonl"
    if not mmqa_jsonl.exists():
        raise FileNotFoundError(mmqa_jsonl)

    examples = load_mmqa_examples(mmqa_jsonl)
    qid_filter = _load_qid_filter(args.qid_file)
    if qid_filter is not None:
        examples = [ex for ex in examples if str(ex.get("qid")) in qid_filter]
    if not examples:
        raise ValueError("No examples selected for summary generation.")

    selected_qids = [str(ex.get("qid")) for ex in examples]

    if args.doc_pool_source == "supporting-docs":
        doc_ids: list[str] = []
        for ex in examples:
            doc_ids.extend(extract_supporting_doc_ids(ex))
        return _dedupe_keep_order(doc_ids)

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
    doc_ids = []
    for qid in selected_qids:
        doc_ids.extend(str(row["doc_id"]) for row in retrieval_doc_pools.get(qid, []))
    return _dedupe_keep_order(doc_ids)


def _load_existing_keys(path: Path) -> set[tuple[str, int]]:
    existing: set[tuple[str, int]] = set()
    if not path.exists():
        return existing
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            doc_id = row.get("doc_id")
            page_idx = row.get("page_idx")
            if doc_id is None or page_idx is None:
                continue
            try:
                existing.add((str(doc_id), int(page_idx)))
            except Exception:
                continue
    return existing


def _get_pdf_dir(args) -> Path:
    pdf_dir = args.pdf_dir
    if pdf_dir is None:
        pdf_dir = Path(LOCAL_DATA_DIR) / "m3-docvqa" / "splits" / f"pdfs_{args.split}"
    if not pdf_dir.exists():
        raise FileNotFoundError(pdf_dir)
    return pdf_dir


def build_vqa_model(args) -> VQAModel:
    model_path = _resolve_local_model_path(args.model_name_or_path)
    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        # Prefer bf16 on A100/H100 when available; this matched the known-good run.
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        attn_impl = "flash_attention_2"
    else:
        dtype = torch.float32
        attn_impl = "eager"

    use_fast_processor = None
    if args.qwen_use_fast_processor == "true":
        use_fast_processor = True
    elif args.qwen_use_fast_processor == "false":
        use_fast_processor = False

    model = VQAModel(
        model_name_or_path=str(model_path),
        model_type=args.model_type,
        dtype=dtype,
        bits=args.bits,
        attn_implementation=attn_impl,
        use_fast_processor=use_fast_processor,
    )
    # 4-bit models are device-dispatched by accelerate/bitsandbytes and must not be moved via .to().
    if use_cuda and args.bits != 4 and isinstance(model.model, torch.nn.Module):
        model.model = model.model.to(args.device)
    logger.info(
        "Loaded summary model | path={} | type={} | device={} | bits={}",
        model_path,
        args.model_type,
        args.device if use_cuda else "cpu",
        args.bits,
    )
    return model


def _is_retryable_page_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    retry_signals = (
        "cuda driver error: invalid argument",
        "cuda out of memory",
        "cuda error",
        "cublas",
        "device-side assert",
    )
    return any(token in msg for token in retry_signals)


def _generate_with_retries(
    *,
    model: VQAModel,
    image,
    prompt: str,
    page_max_retries: int,
    page_retry_wait_seconds: float,
    doc_id: str,
    page_idx: int,
) -> str:
    attempts = max(1, int(page_max_retries) + 1)
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return model.generate(images=[image], question=prompt).strip()
        except Exception as exc:
            last_exc = exc
            retryable = _is_retryable_page_error(exc)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if attempt >= attempts or not retryable:
                raise
            logger.warning(
                "Retrying page after error | doc_id={} page_idx={} attempt={}/{} | {}",
                doc_id,
                page_idx,
                attempt,
                attempts,
                exc,
            )
            if page_retry_wait_seconds > 0:
                time.sleep(page_retry_wait_seconds)
    assert last_exc is not None
    raise last_exc


def summarize_doc_pages(
    *,
    doc_id: str,
    pdf_dir: Path,
    model: VQAModel,
    prompt: str,
    max_pages_per_doc: Optional[int],
) -> list[dict]:
    pdf_path = pdf_dir / f"{doc_id}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    images = get_images_from_pdf(pdf_path)
    if max_pages_per_doc is not None:
        images = images[: max(max_pages_per_doc, 0)]

    rows: list[dict] = []
    for page_idx, image in enumerate(images):
        summary = model.generate(images=[image], question=prompt).strip()
        rows.append(
            {
                "doc_id": doc_id,
                "page_idx": page_idx,
                "summary": summary,
            }
        )
    return rows


def main():
    args = parse_args()
    pdf_dir = _get_pdf_dir(args)

    doc_ids = discover_doc_ids(args)
    if args.max_docs is not None:
        doc_ids = doc_ids[: args.max_docs]
    if not doc_ids:
        raise ValueError("No documents selected for summary generation.")

    existing_keys = set() if args.overwrite else _load_existing_keys(args.output_jsonl)
    if existing_keys:
        logger.info("Resuming output file with {} existing page summaries", len(existing_keys))

    model = build_vqa_model(args)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "a"

    total_written = 0
    total_skipped = 0
    with args.output_jsonl.open(mode) as fout:
        for idx, doc_id in enumerate(doc_ids, start=1):
            try:
                pdf_path = pdf_dir / f"{doc_id}.pdf"
                if not pdf_path.exists():
                    logger.warning("[{}/{}] missing pdf for doc_id={}", idx, len(doc_ids), doc_id)
                    continue

                images = get_images_from_pdf(pdf_path)
                page_limit = len(images)
                if args.max_pages_per_doc is not None:
                    page_limit = min(page_limit, max(args.max_pages_per_doc, 0))

                unsummarized_pages = [
                    page_idx
                    for page_idx in range(page_limit)
                    if (doc_id, page_idx) not in existing_keys
                ]
                if not unsummarized_pages:
                    total_skipped += page_limit
                    logger.info("[{}/{}] doc_id={} | all {} pages already summarized", idx, len(doc_ids), doc_id, page_limit)
                    continue

                logger.info(
                    "[{}/{}] doc_id={} | pdf_pages={} | to_summarize={}",
                    idx,
                    len(doc_ids),
                    doc_id,
                    len(images),
                    len(unsummarized_pages),
                )

                for page_idx in unsummarized_pages:
                    try:
                        summary = _generate_with_retries(
                            model=model,
                            image=images[page_idx],
                            prompt=args.prompt,
                            page_max_retries=args.page_max_retries,
                            page_retry_wait_seconds=args.page_retry_wait_seconds,
                            doc_id=doc_id,
                            page_idx=page_idx,
                        )
                        row = {
                            "doc_id": doc_id,
                            "page_idx": page_idx,
                            "summary": summary,
                        }
                        fout.write(json.dumps(row) + "\n")
                        fout.flush()
                        existing_keys.add((doc_id, page_idx))
                        total_written += 1
                    except Exception as exc:
                        logger.exception(
                            "Failed summarizing page | doc_id={} page_idx={} | {}",
                            doc_id,
                            page_idx,
                            exc,
                        )
                        if args.fail_on_page_error:
                            raise
                        continue
            except Exception as exc:
                logger.exception("Failed summarizing doc_id={} | {}", doc_id, exc)

    logger.info(
        "Done | docs={} | page_summaries_written={} | page_summaries_skipped={}",
        len(doc_ids),
        total_written,
        total_skipped,
    )


if __name__ == "__main__":
    main()
