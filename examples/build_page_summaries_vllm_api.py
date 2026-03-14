"""Build page summaries through an OpenAI-compatible VLM API (SimpleDoc-style).

Outputs JSONL rows with keys: doc_id, page_idx, summary
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import requests
from loguru import logger

from m3docrag.utils.paths import LOCAL_DATA_DIR
from m3docrag.utils.pdfs import get_images_from_pdf


DEFAULT_PROMPT = """You are tasked with creating a comprehensive summary of a given page from a document.
Focus on the main textual content and any visible tables, figures, charts, logos, or images.
Return your answer inside <summary>...</summary> tags.
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev", choices=["dev", "train"])
    p.add_argument("--pdf-dir", type=Path, default=None)
    p.add_argument("--doc-id-file", type=Path, default=None, help="txt/json/jsonl file of doc ids")
    p.add_argument("--all-docs-in-pdf-dir", action="store_true", help="Use all PDFs under --pdf-dir")
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--max-pages-per-doc", type=int, default=None)
    p.add_argument("--page-start-per-doc", type=int, default=0)

    p.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, e.g. http://host:8000/v1")
    p.add_argument("--api-key-file", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-32B-Instruct")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--request-timeout-seconds", type=int, default=300)
    p.add_argument("--api-retries", type=int, default=3)
    p.add_argument("--api-retry-wait-seconds", type=float, default=2.0)

    p.add_argument("--image-dpi", type=int, default=150)
    p.add_argument("--image-max-side", type=int, default=1344)
    p.add_argument("--image-detail", default="high", choices=["low", "high", "auto"])

    p.add_argument("--prompt-file", type=Path, default=None)
    p.add_argument("--prompt-text", default=None)
    p.add_argument("--preserve-tags", action="store_true", help="Keep XML-like tags in output summary")

    p.add_argument("--output-jsonl", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fail-on-page-error", action="store_true")
    return p.parse_args()


def _get_pdf_dir(args) -> Path:
    pdf_dir = args.pdf_dir
    if pdf_dir is None:
        pdf_dir = Path(LOCAL_DATA_DIR) / "m3-docvqa" / "splits" / f"pdfs_{args.split}"
    if not pdf_dir.exists():
        raise FileNotFoundError(pdf_dir)
    return pdf_dir


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


def discover_doc_ids(args, pdf_dir: Path) -> list[str]:
    if args.doc_id_file is not None:
        return _dedupe_keep_order(_load_doc_id_file(args.doc_id_file))
    if args.all_docs_in_pdf_dir:
        return _dedupe_keep_order([p.stem for p in sorted(pdf_dir.glob("*.pdf"))])
    raise ValueError("Provide --doc-id-file or --all-docs-in-pdf-dir")


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


def _read_api_key(path: Path) -> str:
    key = path.read_text().strip()
    if not key:
        raise ValueError(f"Empty API key file: {path}")
    return key


def _chat_url(base_url: str) -> str:
    u = base_url.rstrip("/")
    if u.endswith("/chat/completions"):
        return u
    if u.endswith("/v1"):
        return u + "/chat/completions"
    return u + "/v1/chat/completions"


def _load_prompt(args) -> str:
    if args.prompt_text:
        return args.prompt_text
    if args.prompt_file is not None:
        return args.prompt_file.read_text()
    return DEFAULT_PROMPT


def _resize_image(image, max_side: int):
    if max_side is None or max_side <= 0:
        return image
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    scale = max_side / float(max(w, h))
    return image.resize((max(1, int(w * scale)), max(1, int(h * scale))))


def _image_to_data_url(image) -> str:
    buff = BytesIO()
    image.save(buff, format="PNG")
    b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _extract_summary(text: str, preserve_tags: bool) -> str:
    txt = text or ""
    m = re.search(r"<summary>(.*?)</summary>", txt, flags=re.I | re.S)
    if m:
        txt = m.group(1)
    if not preserve_tags:
        txt = re.sub(r"</?[^>]+>", " ", txt)
    return " ".join(txt.split())


def _call_vlm_with_retries(*, chat_url: str, api_key: str, model: str, prompt: str, image_data_url: str, detail: str, max_tokens: int, temperature: float, top_p: float, timeout_s: int, retries: int, retry_wait_s: float, preserve_tags: bool) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url,
                            "detail": detail,
                        },
                    },
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    last_exc = None
    attempts = max(1, retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.post(chat_url, headers=headers, json=payload, timeout=timeout_s)
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:1000]}")
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, list):
                # Some APIs may return mixed chunks.
                content = " ".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in content)
            return _extract_summary(str(content), preserve_tags=preserve_tags)
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            time.sleep(max(0.0, retry_wait_s))
    raise RuntimeError(f"VLM request failed after {attempts} attempts") from last_exc


def main():
    args = parse_args()
    pdf_dir = _get_pdf_dir(args)

    doc_ids = discover_doc_ids(args, pdf_dir)
    if args.max_docs is not None:
        doc_ids = doc_ids[: args.max_docs]
    if not doc_ids:
        raise ValueError("No documents selected for summary generation")

    existing_keys = set() if args.overwrite else _load_existing_keys(args.output_jsonl)
    if existing_keys:
        logger.info("Resuming output file with {} existing page summaries", len(existing_keys))

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "a"

    api_key = _read_api_key(args.api_key_file)
    chat_url = _chat_url(args.base_url)
    prompt = _load_prompt(args)

    total_written = 0
    total_skipped = 0

    with args.output_jsonl.open(mode) as fout:
        for idx, doc_id in enumerate(doc_ids, start=1):
            try:
                pdf_path = pdf_dir / f"{doc_id}.pdf"
                if not pdf_path.exists():
                    logger.warning("[{}/{}] missing pdf for doc_id={}", idx, len(doc_ids), doc_id)
                    continue

                images = get_images_from_pdf(pdf_path, dpi_resolution=args.image_dpi)
                page_start = max(0, int(args.page_start_per_doc))
                page_end = len(images)
                if args.max_pages_per_doc is not None:
                    page_end = min(page_end, page_start + max(args.max_pages_per_doc, 0))
                if page_start >= len(images):
                    logger.warning(
                        "[{}/{}] doc_id={} | page_start {} out of range for pdf_pages={}",
                        idx,
                        len(doc_ids),
                        doc_id,
                        page_start,
                        len(images),
                    )
                    continue

                unsummarized_pages = [
                    page_idx
                    for page_idx in range(page_start, page_end)
                    if (doc_id, page_idx) not in existing_keys
                ]
                if not unsummarized_pages:
                    total_skipped += max(0, page_end - page_start)
                    logger.info(
                        "[{}/{}] doc_id={} | all {} pages already summarized",
                        idx,
                        len(doc_ids),
                        doc_id,
                        max(0, page_end - page_start),
                    )
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
                        image = _resize_image(images[page_idx], args.image_max_side)
                        image_data_url = _image_to_data_url(image)
                        summary = _call_vlm_with_retries(
                            chat_url=chat_url,
                            api_key=api_key,
                            model=args.model,
                            prompt=prompt,
                            image_data_url=image_data_url,
                            detail=args.image_detail,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            timeout_s=args.request_timeout_seconds,
                            retries=args.api_retries,
                            retry_wait_s=args.api_retry_wait_seconds,
                            preserve_tags=args.preserve_tags,
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
                            "Failed summarizing page via API | doc_id={} page_idx={} | {}",
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
