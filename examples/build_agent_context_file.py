"""Build a page-context file for the agent policy prompt.

This script extracts page-level snippets from local PDFs and saves them as JSON/JSONL
with keys compatible with `examples/run_agent_rag.py --context-file`.

Primary use case: generate real context for the smoke subset created from
`agent_smoke_docid2embs.pt`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from loguru import logger

from m3docrag.utils.paths import LOCAL_DATA_DIR
from m3docrag.utils.pdfs import get_images_from_pdf


PageKey = Tuple[str, int]


def normalize_text(text: str, max_chars: int) -> str:
    text = " ".join((text or "").split())
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text


def load_targets_from_embeddings(path: Path, max_docs: Optional[int], max_pages_per_doc: Optional[int]) -> list[PageKey]:
    docid2embs = torch.load(path, map_location="cpu")
    if not isinstance(docid2embs, dict):
        raise ValueError(f"Expected dict in embeddings bundle: {path}")

    targets: list[PageKey] = []
    items = list(docid2embs.items())
    if max_docs is not None:
        items = items[:max_docs]

    for doc_id, embs in items:
        if not hasattr(embs, "shape") or len(embs.shape) < 1:
            continue
        n_pages = int(embs.shape[0])
        if max_pages_per_doc is not None:
            n_pages = min(n_pages, max_pages_per_doc)
        for page_idx in range(n_pages):
            targets.append((str(doc_id), page_idx))
    return targets


def load_targets_from_trace(path: Path) -> list[PageKey]:
    payload = json.load(path.open())
    steps = payload.get("steps", [])
    targets: list[PageKey] = []
    for step in steps:
        for item in step.get("selected_pages", []):
            if not isinstance(item, list) or len(item) < 2:
                continue
            doc_id = str(item[0])
            page_idx = int(item[1])
            targets.append((doc_id, page_idx))
    # Preserve order but deduplicate.
    seen = set()
    ordered: list[PageKey] = []
    for t in targets:
        if t in seen:
            continue
        seen.add(t)
        ordered.append(t)
    return ordered


def group_targets_by_doc(targets: Sequence[PageKey]) -> Dict[str, list[int]]:
    out: Dict[str, list[int]] = {}
    for doc_id, page_idx in targets:
        out.setdefault(doc_id, []).append(page_idx)
    for doc_id in out:
        out[doc_id] = sorted(set(out[doc_id]))
    return out


def extract_doc_texts_pymupdf(pdf_path: Path, page_indices: Sequence[int]) -> Optional[dict[int, str]]:
    try:
        import fitz  # PyMuPDF
    except Exception:
        return None

    doc = fitz.open(pdf_path)
    try:
        out = {}
        for page_idx in page_indices:
            if page_idx < 0 or page_idx >= len(doc):
                continue
            out[page_idx] = doc[page_idx].get_text("text") or ""
        return out
    finally:
        doc.close()


def extract_doc_texts_pypdf(pdf_path: Path, page_indices: Sequence[int]) -> Optional[dict[int, str]]:
    try:
        from pypdf import PdfReader
    except Exception:
        return None

    reader = PdfReader(str(pdf_path))
    out = {}
    for page_idx in page_indices:
        if page_idx < 0 or page_idx >= len(reader.pages):
            continue
        text = reader.pages[page_idx].extract_text() or ""
        out[page_idx] = text
    return out


def extract_doc_texts_easyocr(
    pdf_path: Path,
    page_indices: Sequence[int],
    *,
    ocr_reader=None,
) -> dict[int, str]:
    import numpy as np

    if ocr_reader is None:
        import easyocr

        ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

    images = get_images_from_pdf(pdf_path)
    out = {}
    for page_idx in page_indices:
        if page_idx < 0 or page_idx >= len(images):
            continue
        img = images[page_idx]
        # easyocr accepts numpy arrays.
        results = ocr_reader.readtext(
            np.array(img),
            detail=0,
            paragraph=True,
        )
        out[page_idx] = "\n".join([str(x) for x in results])
    return out


def extract_doc_contexts(
    pdf_path: Path,
    page_indices: Sequence[int],
    *,
    backend: str,
    ocr_reader=None,
) -> dict[int, str]:
    if backend in ("auto", "pymupdf"):
        out = extract_doc_texts_pymupdf(pdf_path, page_indices)
        if out is not None:
            return out
        if backend == "pymupdf":
            raise RuntimeError("PyMuPDF (fitz) not available")

    if backend in ("auto", "pypdf"):
        out = extract_doc_texts_pypdf(pdf_path, page_indices)
        if out is not None:
            return out
        if backend == "pypdf":
            raise RuntimeError("pypdf not available")

    if backend in ("auto", "easyocr"):
        return extract_doc_texts_easyocr(pdf_path, page_indices, ocr_reader=ocr_reader)

    raise ValueError(f"Unknown backend: {backend}")


def save_context_map(context_map: dict[str, str], out_path: Path, fmt: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        with out_path.open("w") as f:
            json.dump(context_map, f, indent=2)
        return
    if fmt == "jsonl":
        with out_path.open("w") as f:
            for key, text in context_map.items():
                if "_page" in key:
                    doc_id, page = key.split("_page", 1)
                    row = {"doc_id": doc_id, "page_idx": int(page), "summary": text}
                else:
                    row = {"key": key, "summary": text}
                f.write(json.dumps(row) + "\n")
        return
    raise ValueError(f"Unknown output format: {fmt}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output context file (.json or .jsonl).",
    )
    p.add_argument(
        "--pdf-dir",
        type=Path,
        default=None,
        help="Directory containing <doc_id>.pdf. Defaults to LOCAL_DATA_DIR/m3-docvqa/splits/pdfs_dev.",
    )
    p.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Optional docid2embs.pt bundle to define target docs/pages.",
    )
    p.add_argument(
        "--trace-json",
        type=Path,
        default=None,
        help="Optional agent output JSON to define target pages from selected_pages traces.",
    )
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--max-pages-per-doc", type=int, default=None)
    p.add_argument(
        "--backend",
        choices=["auto", "pymupdf", "pypdf", "easyocr"],
        default="auto",
        help="Text extraction backend. auto tries PyMuPDF, then pypdf, then EasyOCR.",
    )
    p.add_argument(
        "--max-chars",
        type=int,
        default=400,
        help="Max chars per page snippet in the output context map.",
    )
    p.add_argument(
        "--output-format",
        choices=["json", "jsonl"],
        default="json",
        help="Output serialization format.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.embeddings is None and args.trace_json is None:
        raise ValueError("Provide at least one of --embeddings or --trace-json")

    pdf_dir = args.pdf_dir
    if pdf_dir is None:
        pdf_dir = Path(LOCAL_DATA_DIR) / "m3-docvqa" / "splits" / "pdfs_dev"
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF dir not found: {pdf_dir}")

    targets: list[PageKey] = []
    if args.embeddings is not None:
        targets.extend(
            load_targets_from_embeddings(
                args.embeddings,
                max_docs=args.max_docs,
                max_pages_per_doc=args.max_pages_per_doc,
            )
        )
    if args.trace_json is not None:
        targets.extend(load_targets_from_trace(args.trace_json))

    # Ordered dedupe.
    seen = set()
    deduped_targets: list[PageKey] = []
    for t in targets:
        if t in seen:
            continue
        seen.add(t)
        deduped_targets.append(t)
    targets = deduped_targets

    if not targets:
        raise ValueError("No target pages found to extract context for.")

    grouped = group_targets_by_doc(targets)
    logger.info(f"Target docs: {len(grouped)} | target pages: {len(targets)}")

    ocr_reader = None
    if args.backend == "easyocr":
        import easyocr

        ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

    context_map: dict[str, str] = {}

    for doc_i, (doc_id, page_indices) in enumerate(grouped.items(), start=1):
        pdf_path = pdf_dir / f"{doc_id}.pdf"
        if not pdf_path.exists():
            logger.warning(f"Missing PDF for {doc_id}: {pdf_path}")
            continue
        logger.info(f"[{doc_i}/{len(grouped)}] extracting {len(page_indices)} pages from {doc_id}")

        try:
            page_texts = extract_doc_contexts(
                pdf_path,
                page_indices,
                backend=args.backend,
                ocr_reader=ocr_reader,
            )
        except Exception as exc:
            logger.warning(f"Failed to extract {doc_id}: {exc}")
            continue

        for page_idx in page_indices:
            text = page_texts.get(page_idx, "")
            text = normalize_text(text, max_chars=args.max_chars)
            if not text:
                continue
            context_map[f"{doc_id}_page{page_idx}"] = text

    logger.info(f"Context entries written: {len(context_map)}")
    save_context_map(context_map, args.output, args.output_format)
    logger.info(f"Saved context file: {args.output}")


if __name__ == "__main__":
    main()
