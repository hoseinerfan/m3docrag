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
from typing import Iterable, Optional

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

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary_json or (args.output_jsonl.parent / f"{args.output_jsonl.stem}_summary.json")

    summary_rows = []
    counts = {}

    with args.output_jsonl.open("w") as fout:
        for i, example in enumerate(examples, start=1):
            qid = str(example.get("qid", f"idx_{i-1}"))
            question = str(example.get("question", "")).strip()
            gold_answers = extract_gold_answers(example)
            supporting_doc_ids = extract_supporting_doc_ids(example)

            started = time.time()
            row = {
                "example_index": i - 1 + args.start_index,
                "qid": qid,
                "question": question,
                "gold_answers": gold_answers,
                "supporting_doc_ids": supporting_doc_ids,
            }

            try:
                if not question:
                    raise ValueError("Missing question")
                if not supporting_doc_ids:
                    raise ValueError("No supporting_doc_ids in example")

                docid2embs_cpu = {}
                merged_context_map: dict[str, str] = {}
                for doc_id in supporting_doc_ids:
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
                row.update(
                    {
                        "reason": reason,
                        "pred_answer": pred_answer,
                        "pred_answer_exact_in_gold": (
                            _norm_answer(pred_answer) in {_norm_answer(a) for a in gold_answers if _norm_answer(a)}
                            if pred_answer is not None and gold_answers
                            else None
                        ),
                        "agent": payload,
                    }
                )
                logger.info(
                    f"[{i}/{len(examples)}] qid={qid} | reason={reason} | "
                    f"docs={len(supporting_doc_ids)} | answer={pred_answer}"
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
            "doc_pool": "supporting_docs_only",
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_rows, indent=2) + "\n")
    logger.info(f"Saved subset summary: {summary_path}")


if __name__ == "__main__":
    main()
