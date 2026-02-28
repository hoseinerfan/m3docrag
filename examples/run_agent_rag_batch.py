"""Batch one-document smoke runner for the agent scaffold.

This script runs `run_agent_session(...)` across multiple documents, using one
document embedding tensor at a time plus a page-level context map extracted
from the corresponding PDF. It is intended for quick smoke tests (e.g., 10 docs)
with a single question template and saves one JSON result per document.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import torch
from loguru import logger
from safetensors import safe_open

from m3docrag.agent import run_agent_session

# Reuse the single-run script helpers to keep behavior aligned.
from run_agent_rag import (  # type: ignore
    build_rag_model,
    configure_warning_filters,
    make_candidate_context_fn,
    make_llm_call_local_hf,
    make_llm_call_stub,
    move_doc_embs,
    to_jsonable,
)

# Reuse PDF text extraction helpers used by the context builder.
from build_agent_context_file import (  # type: ignore
    extract_doc_contexts,
    normalize_text,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--question", required=True, help="Question asked for every document in the batch.")
    p.add_argument(
        "--source-embeddings-dir",
        type=Path,
        required=True,
        help="Directory containing per-document <doc_id>.safetensors files.",
    )
    p.add_argument(
        "--pdf-dir",
        type=Path,
        required=True,
        help="Directory containing <doc_id>.pdf files.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where per-doc outputs and summary files are written.",
    )
    p.add_argument(
        "--doc-ids-file",
        type=Path,
        default=None,
        help="Optional txt/json/jsonl file listing doc_ids to run. If omitted, docs are selected from source-embeddings-dir.",
    )
    p.add_argument("--max-docs", type=int, default=10)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-pages-per-doc", type=int, default=None, help="Optional cap on pages loaded per document.")
    p.add_argument("--context-backend", choices=["pypdf", "pymupdf", "easyocr", "auto"], default="pypdf")
    p.add_argument("--context-max-chars", type=int, default=400)
    p.add_argument("--save-context", action="store_true", help="Save per-doc context JSON files for inspection.")
    p.add_argument("--max-turns", type=int, default=2)
    p.add_argument("--pages-per-turn", type=int, default=3)
    p.add_argument("--n-return-pages", type=int, default=5)
    p.add_argument(
        "--explore-return-pages-multiplier",
        type=int,
        default=10,
        help="Expansion factor used when hop-mode exploration increases retrieval depth on later turns.",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--policy-backend", default="stub", choices=["stub", "local-hf"])
    p.add_argument("--policy-model", default=None)
    p.add_argument("--policy-device", default=None)
    return p.parse_args()


def _iter_doc_ids_from_file(path: Path) -> Iterable[str]:
    suffixes = {s.lower() for s in path.suffixes}
    if ".jsonl" in suffixes:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, str):
                yield row
                continue
            if isinstance(row, dict):
                doc_id = row.get("doc_id") or row.get("document_id") or row.get("id")
                if doc_id:
                    yield str(doc_id)
        return

    if ".json" in suffixes:
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            for row in payload:
                if isinstance(row, str):
                    yield row
                elif isinstance(row, dict):
                    doc_id = row.get("doc_id") or row.get("document_id") or row.get("id")
                    if doc_id:
                        yield str(doc_id)
            return
        if isinstance(payload, dict):
            for key in payload.keys():
                yield str(key)
            return
        raise ValueError(f"Unsupported JSON payload in doc ids file: {type(payload)}")

    # Default: plain text, one doc id per line.
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            yield line


def select_doc_ids(args) -> list[str]:
    if args.doc_ids_file is not None:
        doc_ids = list(dict.fromkeys(_iter_doc_ids_from_file(args.doc_ids_file)))
    else:
        doc_ids = sorted(p.stem for p in args.source_embeddings_dir.glob("*.safetensors"))

    if args.start_index:
        doc_ids = doc_ids[args.start_index :]
    if args.max_docs is not None:
        doc_ids = doc_ids[: args.max_docs]
    return doc_ids


def load_one_doc_embeddings(src_path: Path, *, max_pages: int | None) -> tuple[str, torch.Tensor]:
    doc_id = src_path.stem
    with safe_open(str(src_path), framework="pt", device="cpu") as f:
        embs = f.get_tensor("embeddings")
    if max_pages is not None:
        embs = embs[:max_pages]
    # Match the smoke setup dtype to reduce memory usage.
    return doc_id, embs.to(torch.bfloat16)


def build_context_map_for_doc(
    *,
    doc_id: str,
    pdf_dir: Path,
    n_pages: int,
    backend: str,
    max_chars: int,
) -> dict[str, str]:
    pdf_path = pdf_dir / f"{doc_id}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing PDF for {doc_id}: {pdf_path}")

    page_indices = list(range(n_pages))
    page_texts = extract_doc_contexts(pdf_path, page_indices, backend=backend, ocr_reader=None)

    out: dict[str, str] = {}
    for page_idx in page_indices:
        text = normalize_text(page_texts.get(page_idx, ""), max_chars=max_chars)
        if text:
            out[f"{doc_id}_page{page_idx}"] = text
    return out


def main():
    args = parse_args()
    configure_warning_filters()

    if not args.source_embeddings_dir.exists():
        raise FileNotFoundError(args.source_embeddings_dir)
    if not args.pdf_dir.exists():
        raise FileNotFoundError(args.pdf_dir)

    doc_ids = select_doc_ids(args)
    if not doc_ids:
        raise ValueError("No documents selected for batch run.")

    logger.info(f"Selected {len(doc_ids)} docs for batch smoke run.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = args.output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    context_dir = args.output_dir / "contexts"
    if args.save_context:
        context_dir.mkdir(parents=True, exist_ok=True)

    rag_model = build_rag_model(device=args.device)

    policy_device = args.policy_device or args.device
    if args.policy_backend == "stub":
        llm_call = make_llm_call_stub()
    else:
        if not args.policy_model:
            raise ValueError("--policy-model is required when --policy-backend local-hf")
        llm_call = make_llm_call_local_hf(args.policy_model, device=policy_device)

    summary_rows = []

    for idx, doc_id in enumerate(doc_ids, start=1):
        src_path = args.source_embeddings_dir / f"{doc_id}.safetensors"
        out_json = results_dir / f"{idx:02d}_{doc_id}.json"
        logger.info(f"[{idx}/{len(doc_ids)}] Running doc {doc_id}")

        started = time.time()
        try:
            loaded_doc_id, embs = load_one_doc_embeddings(src_path, max_pages=args.max_pages_per_doc)
            docid2embs = move_doc_embs({loaded_doc_id: embs}, args.device)

            context_map = build_context_map_for_doc(
                doc_id=loaded_doc_id,
                pdf_dir=args.pdf_dir,
                n_pages=int(embs.shape[0]),
                backend=args.context_backend,
                max_chars=args.context_max_chars,
            )
            candidate_context_fn = make_candidate_context_fn(context_map, max_chars=args.context_max_chars)

            if args.save_context:
                (context_dir / f"{loaded_doc_id}.json").write_text(json.dumps(context_map, indent=2) + "\n")

            result = run_agent_session(
                query=args.question,
                rag_model=rag_model,
                docid2embs=docid2embs,
                token2pageuid=None,
                all_token_embeddings=None,
                max_turns=args.max_turns,
                pages_per_turn=args.pages_per_turn,
                n_return_pages=args.n_return_pages,
                explore_return_pages_multiplier=args.explore_return_pages_multiplier,
                llm_call=llm_call,
                candidate_context_fn=candidate_context_fn,
            )

            payload = to_jsonable(result)
            payload["_meta"] = {
                "doc_id": loaded_doc_id,
                "question": args.question,
                "elapsed_sec": round(time.time() - started, 3),
            }
            out_json.write_text(json.dumps(payload, indent=2) + "\n")

            summary_rows.append(
                {
                    "doc_id": loaded_doc_id,
                    "reason": payload.get("reason"),
                    "answer": payload.get("answer"),
                    "steps": len(payload.get("steps", [])),
                    "elapsed_sec": payload["_meta"]["elapsed_sec"],
                    "output_json": str(out_json),
                }
            )
            logger.info(
                f"[{idx}/{len(doc_ids)}] done | reason={payload.get('reason')} | answer={payload.get('answer')}"
            )
        except Exception as exc:
            elapsed = round(time.time() - started, 3)
            logger.exception(f"[{idx}/{len(doc_ids)}] failed for {doc_id}: {exc}")
            summary_rows.append(
                {
                    "doc_id": doc_id,
                    "reason": "error",
                    "answer": None,
                    "steps": 0,
                    "elapsed_sec": elapsed,
                    "error": str(exc),
                    "output_json": str(out_json),
                }
            )
        finally:
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    summary_json = args.output_dir / "summary.json"
    summary_jsonl = args.output_dir / "summary.jsonl"
    summary_json.write_text(json.dumps(summary_rows, indent=2) + "\n")
    with summary_jsonl.open("w") as f:
        for row in summary_rows:
            f.write(json.dumps(row) + "\n")

    counts = {}
    for row in summary_rows:
        counts[row["reason"]] = counts.get(row["reason"], 0) + 1
    logger.info(f"Batch complete | counts={counts} | summary={summary_json}")


if __name__ == "__main__":
    main()
