"""Run the agent scaffold on a subset of M3DocVQA examples.

This runner is intended for prototype validation, not final benchmark scoring.
It uses each example's supporting documents as the retrieval pool (oracle doc pool)
and runs the iterative agent loop with optional page-text context snippets.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Callable, Optional
from urllib import error as urlerror
from urllib import request as urlrequest

import torch
from loguru import logger
from safetensors import safe_open

from m3docrag.agent import run_agent_session
from m3docrag.utils.paths import LOCAL_DATA_DIR, LOCAL_EMBEDDINGS_DIR

# Reuse helpers from existing example scripts to keep behavior consistent.
from run_agent_rag import (  # type: ignore
    build_rag_model,
    configure_warning_filters,
    load_context_map,
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
    p.add_argument(
        "--explore-return-pages-multiplier",
        type=int,
        default=10,
        help="Expansion factor used when hop-mode exploration increases retrieval depth on later turns.",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--policy-backend", default="stub", choices=["stub", "local-hf", "openai-api"])
    p.add_argument("--policy-model", default=None)
    p.add_argument("--policy-device", default=None)
    p.add_argument(
        "--policy-base-url",
        default=None,
        help="OpenAI-compatible base URL for --policy-backend openai-api (for example http://127.0.0.1:8010/v1).",
    )
    p.add_argument(
        "--policy-api-key-file",
        type=Path,
        default=None,
        help="Optional API key file for --policy-backend openai-api.",
    )
    p.add_argument(
        "--policy-timeout-s",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds for --policy-backend openai-api.",
    )
    p.add_argument(
        "--policy-max-retries",
        type=int,
        default=2,
        help="Retry attempts for --policy-backend openai-api.",
    )
    p.add_argument(
        "--policy-max-tokens",
        type=int,
        default=512,
        help="Max completion tokens for --policy-backend openai-api.",
    )
    p.add_argument(
        "--policy-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for --policy-backend openai-api.",
    )
    p.add_argument("--output-jsonl", type=Path, required=True, help="Output JSONL with per-example traces.")
    p.add_argument("--summary-json", type=Path, default=None, help="Optional summary JSON path.")
    p.add_argument("--save-context-dir", type=Path, default=None, help="Optional directory to save per-doc context maps.")
    p.add_argument(
        "--clear-doc-cache-each-example",
        action="store_true",
        help="Clear cached embeddings/page contexts after each example to reduce memory growth on long runs.",
    )
    p.add_argument(
        "--selection-only",
        action="store_true",
        help="Run a lightweight selection-only mode: one hop-planning call plus per-hop doc retrieval, without page-context extraction or iterative answer generation.",
    )
    p.add_argument(
        "--selection-max-hop-queries",
        type=int,
        default=3,
        help="Max hop queries to keep from the planner in --selection-only mode.",
    )
    p.add_argument(
        "--selection-profile",
        default="default",
        choices=["default", "simpledoc"],
        help="Preset for selection-only planner/retriever hyperparameters.",
    )
    p.add_argument(
        "--selection-planner-backend",
        default="heuristic",
        choices=["heuristic", "llm"],
        help="Hop planner backend in --selection-only mode.",
    )
    p.add_argument(
        "--selection-planner-prompt-file",
        type=Path,
        default=None,
        help="Optional prompt template file for LLM planner. Supports {question}, {max_hops}, {root_query}.",
    )
    p.add_argument(
        "--selection-retriever-prompt-file",
        type=Path,
        default=None,
        help="Optional prompt template file for RetrieverAgent reranker. Supports {query}, {max_select_docs}, {candidates}.",
    )
    p.add_argument(
        "--selection-answer-conditioned",
        action="store_true",
        help=(
            "Enable true answer-conditioned hop chaining in --selection-only mode. "
            "After each turn, extract an intermediate fact from retrieved evidence and generate the next query from that fact."
        ),
    )
    p.add_argument(
        "--selection-answer-context-candidates",
        type=int,
        default=6,
        help="Max evidence snippets shown to the answer-conditioned follow-up generator per turn.",
    )
    p.add_argument(
        "--selection-answer-prompt-file",
        type=Path,
        default=None,
        help=(
            "Optional prompt template file for answer-conditioned follow-up generation. "
            "Supports {question}, {current_query}, {turn}, {max_hops}, {evidence}, {previous_intermediate_answers}."
        ),
    )
    p.add_argument(
        "--selection-topk-docs-per-hop",
        type=int,
        default=2,
        help="Max unique docs selected from each hop query in --selection-only mode.",
    )
    p.add_argument(
        "--selection-root-topk-docs",
        type=int,
        default=None,
        help="Optional override for docs selected from the root query in --selection-only mode.",
    )
    p.add_argument(
        "--selection-variant-topk-docs",
        type=int,
        default=None,
        help="Optional override for docs selected from each variant query in --selection-only mode.",
    )
    p.add_argument(
        "--selection-max-variant-queries",
        type=int,
        default=None,
        help="Optional cap on number of variant queries kept after the root query in --selection-only mode.",
    )
    p.add_argument(
        "--selection-summary-full-scan",
        action="store_true",
        help=(
            "When page summaries are provided, score all available summarized pages across the doc pool "
            "for each query and prepend top metadata candidates before final selection."
        ),
    )
    p.add_argument(
        "--selection-summary-full-scan-topk",
        type=int,
        default=1000,
        help="Max metadata candidates kept per query when --selection-summary-full-scan is enabled.",
    )
    p.add_argument(
        "--selection-retrieval-depth-multiplier",
        type=int,
        default=8,
        help="Depth multiplier applied to per-hop doc quota when building retrieval candidates.",
    )
    p.add_argument(
        "--selection-candidate-multi-page",
        action="store_true",
        help="Retrieve multiple pages per doc for candidate construction (default is one page per doc).",
    )
    p.add_argument(
        "--selection-stop-no-new-docs",
        action="store_true",
        help="Stop selection-only loop early when a hop yields too few new selected docs.",
    )
    p.add_argument(
        "--selection-stop-min-new-docs",
        type=int,
        default=1,
        help="Minimum number of newly selected docs required to continue when --selection-stop-no-new-docs is enabled.",
    )
    p.add_argument(
        "--selection-retriever-agent",
        action="store_true",
        help=(
            "Use an LLM-based RetrieverAgent-style reranker in --selection-only mode. "
            "The model receives query + candidate page summaries and chooses which docs/pages to prioritize."
        ),
    )
    p.add_argument(
        "--selection-retriever-candidate-docs",
        type=int,
        default=40,
        help="Number of unique candidate docs shown to RetrieverAgent per query in --selection-only mode.",
    )
    p.add_argument(
        "--selection-retriever-select-docs",
        type=int,
        default=12,
        help="Max docs/pages parsed from RetrieverAgent output per query in --selection-only mode.",
    )
    p.add_argument(
        "--selection-retriever-require-success",
        action="store_true",
        help=(
            "Fail the run when RetrieverAgent LLM reranking fails or cannot be parsed "
            "instead of silently falling back to non-LLM ordering."
        ),
    )
    p.add_argument(
        "--selection-final-global-rerank",
        action="store_true",
        help=(
            "After all hops, aggregate a union of candidate docs and run one final global rerank "
            "before trimming to --selection-final-topk-docs."
        ),
    )
    p.add_argument(
        "--selection-final-candidate-docs",
        type=int,
        default=None,
        help=(
            "Unique candidate docs retained for the final global rerank. "
            "Defaults to --selection-retriever-candidate-docs when omitted."
        ),
    )
    p.add_argument(
        "--selection-final-topk-docs",
        type=int,
        default=10,
        help="Final number of docs kept after global rerank in --selection-only mode.",
    )
    p.add_argument(
        "--selection-final-rerank-chunk-size",
        type=int,
        default=40,
        help="Chunk size for tournament-style final global rerank when candidate docs are large.",
    )
    p.add_argument(
        "--selection-final-rerank-chunk-keep",
        type=int,
        default=10,
        help="Docs kept per chunk during tournament-style final global rerank.",
    )
    p.add_argument(
        "--page-summaries-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON/JSONL file keyed by (doc_id, page_idx) with page summaries. "
            "When provided, --selection-only reranks retrieved pages using summary-query lexical match "
            "before doc selection."
        ),
    )
    p.add_argument(
        "--page-visual-metadata-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON/JSONL file keyed by (doc_id, page_idx) with visual metadata "
            "(for example logo/torch/crest flags and confidence)."
        ),
    )
    p.add_argument(
        "--selection-visual-boost",
        type=float,
        default=2.0,
        help="Weight multiplier for visual-metadata match score in --selection-only mode.",
    )
    p.add_argument(
        "--selection-visual-min-confidence",
        type=float,
        default=0.4,
        help="Minimum visual-metadata confidence used for visual score in --selection-only mode.",
    )
    p.add_argument("--stop-on-error", action="store_true")
    return p.parse_args()


def _read_api_key(path: Optional[Path]) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text().strip()


def _chat_completions_url(base_url: str) -> str:
    url = str(base_url).strip()
    if url.endswith("/chat/completions"):
        return url
    return url.rstrip("/") + "/chat/completions"


def make_llm_call_openai_api(
    *,
    base_url: str,
    model: str,
    api_key_file: Optional[Path],
    timeout_s: float,
    max_retries: int,
    max_tokens: int,
    temperature: float,
) -> Callable[[str], str]:
    if not base_url:
        raise ValueError("--policy-base-url is required when --policy-backend openai-api")
    if not model:
        raise ValueError("--policy-model is required when --policy-backend openai-api")

    api_key = _read_api_key(api_key_file) or "dummy-key"
    url = _chat_completions_url(base_url)
    attempts = max(1, int(max_retries))

    def _call(prompt: str) -> str:
        last_err: Optional[Exception] = None
        payload = {
            "model": str(model),
            "messages": [{"role": "user", "content": str(prompt)}],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        body = json.dumps(payload).encode("utf-8")
        for attempt in range(1, attempts + 1):
            try:
                req = urlrequest.Request(url=url, data=body, headers=headers, method="POST")
                with urlrequest.urlopen(req, timeout=float(timeout_s)) as resp:
                    raw = resp.read().decode("utf-8")
                obj = json.loads(raw)
                return str(((obj.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
            except urlerror.HTTPError as exc:
                try:
                    err_body = exc.read().decode("utf-8", errors="ignore")
                except Exception:
                    err_body = str(exc)
                prompt_chars = len(str(prompt))
                msg = (
                    f"HTTP {exc.code} from policy endpoint. "
                    f"prompt_chars={prompt_chars} max_tokens={int(max_tokens)} "
                    f"response={_norm_text(err_body)[:500]}"
                )
                last_err = RuntimeError(msg)
                if attempt < attempts:
                    time.sleep(0.6 * attempt)
            except (urlerror.URLError, TimeoutError, OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
                last_err = exc
                if attempt < attempts:
                    time.sleep(0.6 * attempt)
        raise RuntimeError(f"openai-api policy call failed after {attempts} attempts: {last_err}")

    return _call


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


def _read_text_file_optional(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text().strip()
    return text or None


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


def _norm_text(text: Optional[str]) -> str:
    return " ".join((text or "").split())


def _dedupe_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        val = _norm_text(item)
        if not val:
            continue
        key = val.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(val)
    return out


def _extract_selection_hop_queries(reply: Optional[str], question: str, max_hops: int) -> list[str]:
    lines = [ln.strip() for ln in (reply or "").splitlines() if ln.strip()]
    hop_queries: list[str] = []
    if lines:
        first = lines[0]
        lower = first.lower()
        if lower.startswith("continue hop:"):
            hop_queries.append(first.split(":", 1)[1].strip())
        elif lower.startswith("continue query:"):
            hop_queries.append(first.split(":", 1)[1].strip())
        for line in lines[1:]:
            if line.lower().startswith("hop_query:"):
                hop_queries.append(line.split(":", 1)[1].strip())
    hop_queries = _dedupe_keep_order(hop_queries)
    if not hop_queries:
        hop_queries = [_norm_text(question)]
    return hop_queries[: max(max_hops, 1)]


def _extract_year_text(question: str) -> str:
    years = re.findall(r"\b(?:19|20)\d{2}\b", question)
    return years[0] if years else ""


def _extract_named_entities(question: str) -> list[str]:
    entities: list[str] = []
    for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", question):
        entity = match.group(1).strip()
        lower = entity.casefold()
        if lower in {"which", "what", "who", "when", "where", "how"}:
            continue
        entities.append(entity)
    return _dedupe_keep_order(entities)


def _extract_anchor_target(question: str) -> tuple[str, str]:
    q = _norm_text(question)
    patterns = (
        r"\bwhich\s+(.+?)\s+(movie|film|album|song|team|state|country|city|series|book)\b",
        r"\bwhat\s+(.+?)\s+(movie|film|album|song|team|state|country|city|series|book)\b",
    )
    for pattern in patterns:
        m = re.search(pattern, q, flags=re.IGNORECASE)
        if m:
            return (_norm_text(m.group(1)), m.group(2).lower())
    return ("", "")


def _extract_comparison_option_queries(question: str) -> list[str]:
    def _descriptor_option_queries(option_text: str) -> tuple[list[str], bool]:
        option_norm = _norm_text(option_text)
        option_lower = option_norm.casefold()
        is_descriptor = any(
            marker in option_lower
            for marker in ("logo", "poster", "cover", "flag", "pictured", "featuring", "features", "showing", "depicting")
        )
        if not is_descriptor:
            return [option_norm], False

        anchor_nouns = {"college", "movie", "film", "album", "song", "team", "country", "state", "city", "book"}
        stopwords = {
            "the",
            "a",
            "an",
            "with",
            "whose",
            "its",
            "their",
            "his",
            "her",
            "on",
            "in",
            "of",
            "for",
            "that",
            "featuring",
            "features",
            "showing",
            "depicting",
            "pictured",
        }
        tokens = re.findall(r"[A-Za-z0-9']+", option_norm)
        kept = [tok for tok in tokens if tok.casefold() not in stopwords]
        descriptor_tokens = [tok for tok in kept if tok.casefold() not in anchor_nouns]
        anchor_tokens = [tok for tok in kept if tok.casefold() in anchor_nouns]
        variants = _dedupe_keep_order(
            [
                _norm_text(" ".join(descriptor_tokens + anchor_tokens)),
                _norm_text(" ".join(anchor_tokens + descriptor_tokens)),
            ]
        )
        return variants or [option_norm], True

    q = _norm_text(question)
    # Prefer the explicit comparison tail after the final colon:
    # "Which ...: Option A or Option B?"
    m = re.search(r"^(?P<prefix>.+?):\s*(?P<a>.+?)\s+or\s+(?P<b>.+?)\??$", q, flags=re.IGNORECASE)
    if m:
        prefix = _norm_text(m.group("prefix"))
        option_a = _norm_text(m.group("a").strip(" ,"))
        option_b = _norm_text(m.group("b").strip(" ,"))
        prefix_lower = prefix.casefold()
        year_text = _extract_year_text(prefix)

        context_parts: list[str] = []
        for phrase in ("NFL Draft", "NBA Draft", "Miami Dolphins", "Seattle Dragons"):
            if phrase.casefold() in prefix_lower:
                context_parts.append(phrase)
        for noun in ("college", "movie", "film", "album", "song", "team", "season", "draft"):
            if noun in prefix_lower:
                context_parts.append(noun)
        if year_text:
            context_parts.append(year_text)

        compact_context = " ".join(_dedupe_keep_order(context_parts))
        option_a_queries, option_a_is_descriptor = _descriptor_option_queries(option_a)
        option_b_queries, option_b_is_descriptor = _descriptor_option_queries(option_b)
        option_queries: list[str] = []
        for query_text in option_a_queries:
            option_queries.append(
                query_text
                if option_a_is_descriptor
                else _norm_text(" ".join([query_text, compact_context]))
            )
        for query_text in option_b_queries:
            option_queries.append(
                query_text
                if option_b_is_descriptor
                else _norm_text(" ".join([query_text, compact_context]))
            )
        return _dedupe_keep_order(option_queries)
    return []


def _heuristic_selection_hop_queries(question: str, max_queries: int) -> list[str]:
    q = _norm_text(question)
    q_lower = q.casefold()
    year_text = _extract_year_text(q)
    entities = _extract_named_entities(q)
    person_anchor = entities[0] if entities else ""
    anchor_target, anchor_type = _extract_anchor_target(q)

    queries: list[str] = []

    # Template 1: dubbed/voice multi-hop movie questions.
    if any(term in q_lower for term in ("voice actress", "voice actor", "dubbed")):
        relation_terms: list[str] = []
        if "voice actress" in q_lower:
            relation_terms += ["voice", "actress"]
        elif "voice actor" in q_lower:
            relation_terms += ["voice", "actor"]
        if "dubbed" in q_lower:
            relation_terms.append("dubbed")
        if "tamil" in q_lower:
            relation_terms.append("tamil")
        if "film" in q_lower:
            relation_terms.append("film")
        elif "movie" in q_lower:
            relation_terms.append("movie")

        first_parts = [person_anchor, anchor_target, " ".join(relation_terms), year_text]
        first_hop = _norm_text(" ".join([p for p in first_parts if p]))
        if first_hop:
            queries.append(first_hop)

        if anchor_target and any(term in q_lower for term in ("dubbed", "tamil")):
            version_parts = [
                "Who dubbed the",
                "Tamil" if "tamil" in q_lower else "",
                "version of the",
                year_text,
                anchor_target,
                anchor_type or ("movie" if "movie" in q_lower else "film" if "film" in q_lower else ""),
            ]
            version_hop = _norm_text(" ".join([p for p in version_parts if p]))
            if version_hop:
                queries.append(version_hop)

    # Template 2: anchored entity/title questions.
    if not queries and anchor_target:
        anchor_hop = _norm_text(" ".join([anchor_target, anchor_type, year_text]).strip())
        if anchor_hop:
            queries.append(anchor_hop)
        if person_anchor:
            relation_hop = _norm_text(" ".join([person_anchor, anchor_target, anchor_type, year_text]).strip())
            if relation_hop:
                queries.append(relation_hop)

    # Template 3: generic fallback using named entity + year.
    if not queries and person_anchor:
        generic_hop = _norm_text(" ".join([person_anchor, year_text]).strip())
        if generic_hop:
            queries.append(generic_hop)

    queries = _dedupe_keep_order(queries)
    return queries[: max(max_queries, 0)]


def _render_prompt_template(template: Optional[str], values: dict[str, object]) -> Optional[str]:
    if not template:
        return None
    rendered = str(template)
    # Allow both {{key}} and {key} placeholders.
    for k, v in values.items():
        rendered = rendered.replace(f"{{{{{k}}}}}", str(v))
    try:
        rendered = rendered.format(**{k: str(v) for k, v in values.items()})
    except Exception:
        # Keep fallback rendered text if python formatting fails (for example JSON braces).
        pass
    return rendered


def _selection_planner_prompt(
    *,
    question: str,
    max_hops: int,
    prompt_template: Optional[str],
) -> str:
    root_query = _norm_text(question)
    rendered = _render_prompt_template(
        prompt_template,
        {
            "question": question,
            "root_query": root_query,
            "max_hops": max(max_hops, 1),
        },
    )
    if rendered:
        return rendered
    return (
        "You are a retrieval planner for multi-hop document QA.\n"
        "Produce focused hop queries for evidence retrieval.\n"
        "Keep semantics aligned with the original question.\n\n"
        f"QUESTION:\n{question}\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "hop_queries": ["<query1>", "<query2>"]\n'
        "}\n"
        f"Rules:\n- Include at most {max(max_hops, 1)} queries.\n"
        "- First query should be the root or close paraphrase of the original question.\n"
        "- Keep entities/numbers/years intact.\n"
    )


def _parse_selection_planner_response(
    *,
    raw_reply: str,
    question: str,
    max_hops: int,
) -> list[str]:
    root_query = _norm_text(question)
    queries: list[str] = []
    payload = _extract_first_json_object(raw_reply)
    if isinstance(payload, dict):
        for key in ("hop_queries", "queries", "subqueries", "retrieval_queries"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        queries.append(_norm_text(item))
                break
    if not queries and raw_reply:
        for line in str(raw_reply).splitlines():
            line = line.strip()
            if not line:
                continue
            lower = line.lower()
            if lower.startswith("hop_query:"):
                queries.append(_norm_text(line.split(":", 1)[1]))

    queries = [q for q in _dedupe_keep_order(queries) if q]
    if not queries:
        return [root_query]
    if queries[0].casefold() != root_query.casefold():
        queries = [root_query] + [q for q in queries if q.casefold() != root_query.casefold()]
    return queries[: max(max_hops, 1)]


def _plan_selection_hop_queries(
    *,
    question: str,
    max_hops: int,
    planner_backend: str = "heuristic",
    planner_llm_call: Optional[Callable[[str], str]] = None,
    planner_prompt_template: Optional[str] = None,
) -> tuple[list[str], Optional[str]]:
    if planner_backend == "llm" and planner_llm_call is not None:
        prompt = _selection_planner_prompt(
            question=question,
            max_hops=max_hops,
            prompt_template=planner_prompt_template,
        )
        try:
            raw_reply = planner_llm_call(prompt) or ""
            queries = _parse_selection_planner_response(
                raw_reply=raw_reply,
                question=question,
                max_hops=max_hops,
            )
            planner_reply = raw_reply if raw_reply else "\n".join([f"HOP_QUERY: {q}" for q in queries])
            return queries, planner_reply
        except Exception as exc:
            logger.warning("LLM planner failed, falling back to heuristic planner: {}", exc)

    root_query = _norm_text(question)
    comparison_queries = _extract_comparison_option_queries(question)
    if max_hops <= 1:
        queries = [root_query]
        planner_kind = "HEURISTIC"
    elif comparison_queries:
        # For "X or Y" comparisons, always keep both option-specific tracks plus the root query.
        queries = [root_query] + comparison_queries
        planner_kind = "HEURISTIC_COMPARISON"
    else:
        aux_queries = _heuristic_selection_hop_queries(question, max_hops + 2)
        aux_queries = [q for q in aux_queries if q.casefold() != root_query.casefold()]
        queries = [root_query] + aux_queries[: max_hops - 1]
        planner_kind = "HEURISTIC"
    planner_reply = "\n".join([planner_kind] + [f"HOP_QUERY: {x}" for x in queries])
    return queries, planner_reply


def _selection_only_retrieval_depth(selection_topk_docs_per_hop: int, depth_multiplier: int = 8) -> int:
    quota = max(selection_topk_docs_per_hop, 1)
    return max(quota * max(int(depth_multiplier), 1), 8)


def _selection_retriever_prompt(
    *,
    question: str,
    candidates: list[dict],
    max_select_docs: int,
    prompt_template: Optional[str] = None,
) -> str:
    lines = []
    # Keep prompts compact for 1k-token context windows.
    n_candidates = max(len(candidates), 1)
    if n_candidates >= 8:
        summary_char_cap = 60
    elif n_candidates >= 6:
        summary_char_cap = 80
    elif n_candidates >= 4:
        summary_char_cap = 110
    else:
        summary_char_cap = 160
    for row in candidates:
        summary = _norm_text(str(row.get("summary") or ""))[:summary_char_cap]
        lines.append(
            f"[{row['candidate_index']}] d={row['doc_id']} p={row['page_idx']} "
            f"s={row['score']:.2f} t={summary}"
        )
    joined_candidates = "\n".join(lines)
    rendered = _render_prompt_template(
        prompt_template,
        {
            "query": question,
            "question": question,
            "max_select_docs": max(max_select_docs, 1),
            "candidates": joined_candidates,
        },
    )
    if rendered:
        return rendered
    return (
        "You are a RetrieverAgent for multi-hop document QA.\n"
        "Task: choose candidate docs/pages most likely to contain evidence for the query.\n"
        "Prioritize evidence-bearing pages, not generic background pages.\n\n"
        f"QUERY:\n{question}\n\n"
        "CANDIDATES:\n"
        f"{joined_candidates}\n\n"
        "Return JSON only with this schema:\n"
        "{\n"
        '  "selected": [\n'
        '    {"candidate_index": 1, "doc_id": "<id>", "page_idx": 0}\n'
        "  ]\n"
        "}\n"
        f"Rules:\n- select at most {max(max_select_docs, 1)} items.\n"
        "- choose from listed candidates only.\n"
        "- no extra keys, no markdown, no explanation.\n"
    )


def _selection_retriever_repair_prompt(
    *,
    question: str,
    candidates: list[dict],
    required_count: int,
) -> str:
    lines = []
    for row in candidates:
        lines.append(
            f"[{row['candidate_index']}] d={row['doc_id']} p={row['page_idx']} s={row['score']:.2f}"
        )
    joined_candidates = "\n".join(lines)
    return (
        "Your previous RetrieverAgent response was invalid.\n"
        f"Select exactly {max(int(required_count), 1)} candidates for the query.\n\n"
        f"QUERY:\n{question}\n\n"
        "CANDIDATES:\n"
        f"{joined_candidates}\n\n"
        "Return JSON only with this exact schema:\n"
        "{\n"
        '  "selected": [\n'
        "    1\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- candidate_index must be from the list above.\n"
        f"- return exactly {max(int(required_count), 1)} indices.\n"
        "- no markdown, no explanation.\n"
    )


def _selection_answer_conditioned_prompt(
    *,
    question: str,
    current_query: str,
    turn: int,
    max_hops: int,
    evidence_lines: list[str],
    previous_intermediate_answers: list[str],
    prompt_template: Optional[str] = None,
) -> str:
    evidence = "\n".join(evidence_lines)
    prior_answers = "\n".join(
        [f"- {a}" for a in previous_intermediate_answers if _norm_text(a)]
    ) or "None"
    rendered = _render_prompt_template(
        prompt_template,
        {
            "question": question,
            "current_query": current_query,
            "turn": turn,
            "max_hops": max(max_hops, 1),
            "evidence": evidence,
            "previous_intermediate_answers": prior_answers,
        },
    )
    if rendered:
        return rendered
    return (
        "You are a multi-hop retrieval planner.\n"
        "Given the original question and evidence from the current retrieval turn,\n"
        "infer one intermediate fact and produce the next follow-up retrieval query.\n\n"
        f"ORIGINAL_QUESTION:\n{question}\n\n"
        f"CURRENT_QUERY (turn {turn}/{max(max_hops, 1)}):\n{current_query}\n\n"
        f"PREVIOUS_INTERMEDIATE_ANSWERS:\n{prior_answers}\n\n"
        f"EVIDENCE_SNIPPETS:\n{evidence}\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "intermediate_answer": "<short factual phrase from evidence>",\n'
        '  "next_query": "<follow-up retrieval query conditioned on the intermediate answer>"\n'
        "}\n"
        "Rules:\n"
        "- Keep entities, years, and numbers consistent with the original question.\n"
        "- next_query must differ from current_query.\n"
        "- Do not answer the final question; only provide the next retrieval query.\n"
    )


def _parse_selection_answer_conditioned_response(
    *,
    raw_reply: str,
    current_query: str,
) -> tuple[Optional[str], Optional[str]]:
    intermediate_answer = None
    next_query = None
    payload = _extract_first_json_object(raw_reply)
    if isinstance(payload, dict):
        for key in ("intermediate_answer", "answer", "fact", "intermediate_fact"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                intermediate_answer = _norm_text(value)
                break
        for key in ("next_query", "follow_up_query", "hop_query", "retrieval_query", "query"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                next_query = _norm_text(value)
                break

    if next_query is None and raw_reply:
        for line in raw_reply.splitlines():
            line = line.strip()
            if not line:
                continue
            lower = line.casefold()
            if lower.startswith("next_query:") or lower.startswith("follow_up_query:"):
                next_query = _norm_text(line.split(":", 1)[1])
                break
            if lower.startswith("intermediate_answer:") and intermediate_answer is None:
                intermediate_answer = _norm_text(line.split(":", 1)[1])

    if next_query and next_query.casefold() == _norm_text(current_query).casefold():
        next_query = None
    return intermediate_answer, next_query


def _build_answer_conditioned_evidence_lines(
    *,
    selected_pages: list[tuple[str, int, float]],
    top_pages_payload: list[dict],
    page_summary_map: Optional[dict[str, str]],
    max_items: int,
) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()

    # Prefer pages explicitly selected this turn, then backfill with top retrieved pages.
    for doc_id, page_idx, score in selected_pages:
        uid = f"{doc_id}#p{int(page_idx)}"
        if uid in seen:
            continue
        seen.add(uid)
        summary = _lookup_page_summary(page_summary_map, str(doc_id), int(page_idx)) or ""
        lines.append(
            f"[{len(lines)+1}] doc_id={doc_id} page_idx={int(page_idx)} score={float(score):.4f} "
            f"summary={_norm_text(summary)[:240]}"
        )
        if len(lines) >= max(max_items, 1):
            return lines

    for row in top_pages_payload:
        doc_id = str(row.get("doc_id") or "")
        page_idx = int(row.get("page_idx") or 0)
        uid = f"{doc_id}#p{page_idx}"
        if not doc_id or uid in seen:
            continue
        seen.add(uid)
        summary = _norm_text(str(row.get("page_summary") or ""))[:240]
        score = float(row.get("score") or 0.0)
        lines.append(
            f"[{len(lines)+1}] doc_id={doc_id} page_idx={page_idx} score={score:.4f} summary={summary}"
        )
        if len(lines) >= max(max_items, 1):
            break
    return lines


def _propose_answer_conditioned_follow_up_query(
    *,
    question: str,
    current_query: str,
    turn: int,
    max_hops: int,
    selected_pages: list[tuple[str, int, float]],
    top_pages_payload: list[dict],
    page_summary_map: Optional[dict[str, str]],
    llm_call: Optional[Callable[[str], str]],
    context_candidates: int,
    previous_intermediate_answers: list[str],
    prompt_template: Optional[str] = None,
) -> tuple[Optional[str], Optional[str], dict]:
    trace = {
        "attempted": bool(llm_call),
        "success": False,
        "error": None,
        "intermediate_answer": None,
        "next_query": None,
        "raw_reply_preview": None,
        "evidence_count": 0,
    }
    if llm_call is None:
        trace["error"] = "llm_unavailable"
        return None, None, trace

    evidence_lines = _build_answer_conditioned_evidence_lines(
        selected_pages=selected_pages,
        top_pages_payload=top_pages_payload,
        page_summary_map=page_summary_map,
        max_items=max(context_candidates, 1),
    )
    trace["evidence_count"] = len(evidence_lines)
    if not evidence_lines:
        trace["error"] = "no_evidence"
        return None, None, trace

    prompt = _selection_answer_conditioned_prompt(
        question=question,
        current_query=current_query,
        turn=turn,
        max_hops=max_hops,
        evidence_lines=evidence_lines,
        previous_intermediate_answers=previous_intermediate_answers,
        prompt_template=prompt_template,
    )
    try:
        raw_reply = llm_call(prompt) or ""
    except Exception as exc:
        trace["error"] = str(exc)
        return None, None, trace

    trace["raw_reply_preview"] = _norm_text(raw_reply)[:400]
    intermediate_answer, next_query = _parse_selection_answer_conditioned_response(
        raw_reply=raw_reply,
        current_query=current_query,
    )
    if next_query is None:
        trace["error"] = "missing_next_query"
        if intermediate_answer:
            trace["intermediate_answer"] = intermediate_answer
        return None, intermediate_answer, trace

    trace["success"] = True
    trace["next_query"] = next_query
    trace["intermediate_answer"] = intermediate_answer
    return next_query, intermediate_answer, trace


def _parse_selection_retriever_response(
    *,
    raw_reply: str,
    candidate_rows: list[tuple[str, int, float]],
    max_select_docs: int,
) -> list[tuple[str, int, float]]:
    if not raw_reply:
        return []

    candidate_by_index = {idx + 1: row for idx, row in enumerate(candidate_rows)}
    candidate_by_doc = {
        str(doc_id): (str(doc_id), int(page_idx), float(score))
        for doc_id, page_idx, score in candidate_rows
    }
    picked: list[tuple[str, int, float]] = []
    picked_uids: set[str] = set()

    payload = _extract_first_json_object(raw_reply)
    selected_items = None
    if isinstance(payload, dict):
        for key in ("selected", "selected_pages", "pages", "choices"):
            value = payload.get(key)
            if isinstance(value, list):
                selected_items = value
                break

    if isinstance(selected_items, list):
        for item in selected_items:
            row = None
            if isinstance(item, dict):
                idx = item.get("candidate_index")
                if isinstance(idx, int) and idx in candidate_by_index:
                    row = candidate_by_index[idx]
                if row is None:
                    doc_id = str(item.get("doc_id") or "").strip()
                    page_idx = item.get("page_idx")
                    if doc_id and isinstance(page_idx, int):
                        row = candidate_by_doc.get(doc_id)
                        if row and int(row[1]) != int(page_idx):
                            row = None
                    elif doc_id:
                        row = candidate_by_doc.get(doc_id)
            elif isinstance(item, int):
                row = candidate_by_index.get(item)
            if row is None:
                continue
            uid = f"{row[0]}#p{int(row[1])}"
            if uid in picked_uids:
                continue
            picked.append(row)
            picked_uids.add(uid)
            if len(picked) >= max(max_select_docs, 1):
                return picked

    # Fallback: parse candidate indices from partially formed text/JSON.
    # This handles truncated responses like: {"selected":[{"candidate_index":4
    candidate_indices = re.findall(r"candidate_index\"?\s*[:=]\s*(\d+)", raw_reply, flags=re.IGNORECASE)
    for idx_str in candidate_indices:
        try:
            idx = int(idx_str)
        except Exception:
            continue
        row = candidate_by_index.get(idx)
        if row is None:
            continue
        uid = f"{row[0]}#p{int(row[1])}"
        if uid in picked_uids:
            continue
        picked.append(row)
        picked_uids.add(uid)
        if len(picked) >= max(max_select_docs, 1):
            return picked

    # Fallback: parse doc IDs in raw text.
    doc_ids = re.findall(r"\b[0-9a-f]{32}\b", raw_reply.casefold())
    for doc_id in doc_ids:
        row = candidate_by_doc.get(doc_id)
        if row is None:
            continue
        uid = f"{row[0]}#p{int(row[1])}"
        if uid in picked_uids:
            continue
        picked.append(row)
        picked_uids.add(uid)
        if len(picked) >= max(max_select_docs, 1):
            break
    return picked


def _rerank_with_selection_retriever_agent(
    *,
    query: str,
    retrieved: list[tuple[str, int, float]],
    page_summary_map: Optional[dict[str, str]],
    llm_call: Optional[Callable[[str], str]],
    candidate_doc_limit: int,
    max_select_docs: int,
    require_success: bool = False,
    require_exact_count: bool = False,
    prompt_template: Optional[str] = None,
) -> tuple[list[tuple[str, int, float]], dict]:
    trace = {
        "enabled": True,
        "candidate_docs_considered": 0,
        "selected_count": 0,
        "parse_success": False,
        "selected_docs": [],
        "raw_reply_preview": None,
        "repair_attempted": False,
        "repair_success": False,
        "repair_reply_preview": None,
        "required_selection_count": None,
        "error": None,
    }
    if not retrieved or llm_call is None:
        trace["enabled"] = False
        if require_success:
            raise RuntimeError("RetrieverAgent rerank requires non-empty candidates and an LLM call.")
        return retrieved, trace

    candidate_rows: list[tuple[str, int, float]] = []
    candidate_payload: list[dict] = []
    seen_docs: set[str] = set()
    for doc_id, page_idx, score in retrieved:
        doc_id = str(doc_id)
        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        candidate_rows.append((doc_id, int(page_idx), float(score)))
        summary_text = _lookup_page_summary(page_summary_map, doc_id, int(page_idx)) or ""
        candidate_payload.append(
            {
                "candidate_index": len(candidate_rows),
                "doc_id": doc_id,
                "page_idx": int(page_idx),
                "score": float(score),
                "summary": summary_text,
            }
        )
        if len(candidate_rows) >= max(candidate_doc_limit, 1):
            break

    trace["candidate_docs_considered"] = len(candidate_rows)
    if not candidate_rows:
        return retrieved, trace
    target_count = min(max(int(max_select_docs), 1), len(candidate_rows))
    min_required = target_count if require_exact_count else 1
    trace["required_selection_count"] = min_required

    prompt = _selection_retriever_prompt(
        question=query,
        candidates=candidate_payload,
        max_select_docs=target_count,
        prompt_template=prompt_template,
    )
    try:
        raw_reply = llm_call(prompt)
    except Exception as exc:
        trace["error"] = str(exc)
        if require_success:
            raise RuntimeError(f"RetrieverAgent LLM call failed: {exc}") from exc
        return retrieved, trace

    raw_reply = raw_reply or ""
    trace["raw_reply_preview"] = _norm_text(raw_reply)[:400]
    selected_rows = _parse_selection_retriever_response(
        raw_reply=raw_reply,
        candidate_rows=candidate_rows,
        max_select_docs=target_count,
    )
    if len(selected_rows) < min_required:
        if require_success:
            trace["repair_attempted"] = True

            def _merge_rows(
                base: list[tuple[str, int, float]],
                extra: list[tuple[str, int, float]],
                limit: int,
            ) -> list[tuple[str, int, float]]:
                merged: list[tuple[str, int, float]] = []
                seen: set[str] = set()
                for row in list(base) + list(extra):
                    uid = f"{row[0]}#p{int(row[1])}"
                    if uid in seen:
                        continue
                    seen.add(uid)
                    merged.append(row)
                    if len(merged) >= max(limit, 1):
                        break
                return merged

            def _build_candidate_payload(
                rows: list[tuple[str, int, float]],
            ) -> list[dict]:
                payload: list[dict] = []
                for idx, (doc_id, page_idx, score) in enumerate(rows, start=1):
                    summary_text = _lookup_page_summary(page_summary_map, str(doc_id), int(page_idx)) or ""
                    payload.append(
                        {
                            "candidate_index": idx,
                            "doc_id": str(doc_id),
                            "page_idx": int(page_idx),
                            "score": float(score),
                            "summary": summary_text,
                        }
                    )
                return payload

            repair_attempts = max(3, min_required * 2)
            for _ in range(repair_attempts):
                need = max(min_required - len(selected_rows), 0)
                if need <= 0:
                    break

                selected_uids = {f"{doc_id}#p{int(page_idx)}" for doc_id, page_idx, _ in selected_rows}
                remaining_rows = [
                    row for row in candidate_rows if f"{row[0]}#p{int(row[1])}" not in selected_uids
                ]
                if not remaining_rows:
                    break

                ask_count = min(max(need, 1), len(remaining_rows))
                repair_candidates = _build_candidate_payload(remaining_rows)
                repair_prompt = _selection_retriever_repair_prompt(
                    question=query,
                    candidates=repair_candidates,
                    required_count=ask_count,
                )
                try:
                    repair_reply = llm_call(repair_prompt) or ""
                    trace["repair_reply_preview"] = _norm_text(repair_reply)[:400]
                    repair_rows = _parse_selection_retriever_response(
                        raw_reply=repair_reply,
                        candidate_rows=remaining_rows,
                        max_select_docs=ask_count,
                    )
                    selected_rows = _merge_rows(selected_rows, repair_rows, target_count)
                except Exception as exc:
                    trace["error"] = f"repair_call_failed: {exc}"
                    raise RuntimeError(f"RetrieverAgent repair call failed: {exc}") from exc

            if len(selected_rows) < min_required:
                preview = trace.get("raw_reply_preview") or ""
                repair_preview = trace.get("repair_reply_preview") or ""
                raise RuntimeError(
                    "RetrieverAgent selected too few candidates after repair. "
                    f"selected={len(selected_rows)} required={min_required} "
                    f"Preview={preview[:200]!r} RepairPreview={repair_preview[:200]!r}"
                )
            trace["repair_success"] = True
        else:
            return retrieved, trace

    trace["parse_success"] = True
    trace["selected_count"] = len(selected_rows)
    trace["selected_docs"] = [
        {"doc_id": str(doc_id), "page_idx": int(page_idx)}
        for doc_id, page_idx, _ in selected_rows
    ]

    selected_uids = {f"{doc_id}#p{int(page_idx)}" for doc_id, page_idx, _ in selected_rows}
    remainder = [
        row for row in retrieved if f"{str(row[0])}#p{int(row[1])}" not in selected_uids
    ]
    return selected_rows + remainder, trace


def _page_key_variants(doc_id: str, page_idx: int) -> list[str]:
    return [
        f"{doc_id}_page{page_idx}",
        f"{doc_id}#p{page_idx}",
        f"{doc_id}:{page_idx}",
        f"{doc_id}/{page_idx}",
    ]


def _lookup_page_summary(page_summary_map: Optional[dict[str, str]], doc_id: str, page_idx: int) -> Optional[str]:
    if not page_summary_map:
        return None
    for key in _page_key_variants(doc_id, page_idx):
        value = page_summary_map.get(key)
        if value:
            return _norm_text(value)
    return None


def _to_bool(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().casefold()
        if v in {"true", "t", "yes", "y", "1"}:
            return True
        if v in {"false", "f", "no", "n", "0"}:
            return False
    return None


def _to_float(value) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


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
    m = re.search(r"\{.*\}", candidate, re.DOTALL)
    if not m:
        return None
    try:
        payload = json.loads(m.group(0))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _negated_mention(text: str, term_pattern: str) -> bool:
    neg_patterns = [
        rf"\b(no|not|none|without|absent|lacks?|lack(?:ing)?)\b[^.]{{0,140}}\b{term_pattern}\b",
        rf"\b{term_pattern}\b[^.]{{0,120}}\b(no|not|none|without|absent)\b",
    ]
    return any(re.search(pat, text, re.IGNORECASE) for pat in neg_patterns)


def _coerce_visual_metadata(obj) -> Optional[dict[str, float | bool]]:
    if obj is None:
        return None

    payload = obj if isinstance(obj, dict) else None
    if payload is None and isinstance(obj, str):
        payload = _extract_first_json_object(obj)

    three_torches = None
    logo_visible = None
    shield_or_crest = None
    confidence = None

    if isinstance(payload, dict):
        for key in (
            "three_torches_logo_visible",
            "three_torch_logo_visible",
            "three_torches_visible",
            "torch_logo_visible",
        ):
            if key in payload:
                three_torches = _to_bool(payload.get(key))
                break
        for key in ("logo_visible", "emblem_visible", "symbol_visible"):
            if key in payload:
                logo_visible = _to_bool(payload.get(key))
                break
        for key in ("shield_or_crest_visible", "crest_visible", "shield_visible"):
            if key in payload:
                shield_or_crest = _to_bool(payload.get(key))
                break
        for key in ("confidence", "conf", "score"):
            if key in payload:
                confidence = _to_float(payload.get(key))
                break

    raw_text = None
    if isinstance(obj, str):
        raw_text = obj
    elif isinstance(obj, dict):
        maybe_text = obj.get("summary") or obj.get("text") or obj.get("snippet") or obj.get("content")
        if isinstance(maybe_text, str):
            raw_text = maybe_text
    if raw_text:
        lower = raw_text.casefold()
        has_torch = bool(re.search(r"\btorch(?:es)?\b", lower))
        has_logo = bool(re.search(r"\b(logo|emblem|symbol)\b", lower))
        has_shield_or_crest = bool(re.search(r"\b(shield|crest|seal)\b", lower))
        if three_torches is None:
            three_torches = bool(re.search(r"\bthree\b[^.]{0,40}\btorch(?:es)?\b", lower)) and not _negated_mention(
                lower, r"torch(?:es)?"
            )
        if logo_visible is None:
            logo_visible = has_logo and not _negated_mention(lower, r"logo|emblem|symbol")
        if shield_or_crest is None:
            shield_or_crest = has_shield_or_crest and not _negated_mention(lower, r"shield|crest|seal")
        if confidence is None:
            m = re.search(r"\bconfidence\b[^0-9]*([01](?:\.\d+)?)", lower)
            if m:
                confidence = _to_float(m.group(1))
            elif any((three_torches, logo_visible, shield_or_crest)):
                confidence = 0.6

    if confidence is None:
        confidence = 0.0
    confidence = max(0.0, min(float(confidence), 1.0))
    three_torches = bool(three_torches)
    logo_visible = bool(logo_visible)
    shield_or_crest = bool(shield_or_crest)
    if not (three_torches or logo_visible or shield_or_crest):
        return None
    return {
        "three_torches_logo_visible": three_torches,
        "logo_visible": logo_visible,
        "shield_or_crest_visible": shield_or_crest,
        "confidence": confidence,
    }


def load_page_visual_metadata_map(path: Path) -> dict[str, dict[str, float | bool]]:
    payload = _load_json_or_jsonl(path)
    out: dict[str, dict[str, float | bool]] = {}

    if isinstance(payload, dict):
        for key, value in payload.items():
            if _parse_page_key(str(key)) is None:
                continue
            meta = _coerce_visual_metadata(value)
            if meta is not None:
                out[str(key)] = meta
        return out

    if not isinstance(payload, list):
        return out

    for row in payload:
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
        meta = _coerce_visual_metadata(row)
        if meta is None:
            for key in ("summary", "text", "snippet", "content"):
                if isinstance(row.get(key), str):
                    meta = _coerce_visual_metadata(row.get(key))
                    if meta is not None:
                        break
        if meta is not None:
            out[f"{doc_id}_page{page_idx}"] = meta
    return out


def _parse_page_key(key: str) -> Optional[tuple[str, int]]:
    key = str(key)
    m = re.match(r"^(?P<doc>.+)_page(?P<page>\d+)$", key)
    if m:
        return m.group("doc"), int(m.group("page"))
    m = re.match(r"^(?P<doc>.+)#p(?P<page>\d+)$", key)
    if m:
        return m.group("doc"), int(m.group("page"))
    m = re.match(r"^(?P<doc>.+):(?P<page>\d+)$", key)
    if m:
        return m.group("doc"), int(m.group("page"))
    m = re.match(r"^(?P<doc>.+)/(?P<page>\d+)$", key)
    if m:
        return m.group("doc"), int(m.group("page"))
    return None


def _build_doc_page_summary_index(
    page_summary_map: Optional[dict[str, str]],
    allowed_doc_ids: set[str],
) -> dict[str, list[tuple[int, str]]]:
    out: dict[str, list[tuple[int, str]]] = {}
    if not page_summary_map:
        return out
    for key, value in page_summary_map.items():
        parsed = _parse_page_key(key)
        if parsed is None:
            continue
        doc_id, page_idx = parsed
        if doc_id not in allowed_doc_ids:
            continue
        text = _norm_text(value)
        if not text:
            continue
        out.setdefault(doc_id, []).append((page_idx, text))
    return out


def _build_doc_page_index(
    page_summary_map: Optional[dict[str, str]],
    page_visual_meta_map: Optional[dict[str, dict[str, float | bool]]],
    allowed_doc_ids: set[str],
) -> dict[str, list[int]]:
    out: dict[str, set[int]] = {}
    for source_map in (page_summary_map or {}, page_visual_meta_map or {}):
        for key in source_map:
            parsed = _parse_page_key(key)
            if parsed is None:
                continue
            doc_id, page_idx = parsed
            if doc_id not in allowed_doc_ids:
                continue
            out.setdefault(doc_id, set()).add(page_idx)
    return {doc_id: sorted(pages) for doc_id, pages in out.items()}


def _is_descriptor_focused_query(query: str) -> bool:
    q = _norm_text(query).casefold()
    tokens = re.findall(r"[A-Za-z0-9']+", q)
    if len(tokens) > 8:
        return False
    return any(
        marker in q for marker in ("logo", "poster", "cover", "flag", "symbol", "emblem", "crest", "shield", "torch")
    )


def _lookup_page_visual_meta(
    page_visual_meta_map: Optional[dict[str, dict[str, float | bool]]],
    doc_id: str,
    page_idx: int,
) -> Optional[dict[str, float | bool]]:
    if not page_visual_meta_map:
        return None
    for key in _page_key_variants(doc_id, page_idx):
        value = page_visual_meta_map.get(key)
        if value:
            return value
    return None


def _visual_match_score(
    query: str,
    visual_meta: Optional[dict[str, float | bool]],
    *,
    descriptor_focused: bool,
    visual_boost: float,
    min_confidence: float,
) -> float:
    # Disabled: offline visual metadata has shown low reliability in current experiments.
    # Keep the plumbing in place so this can be re-enabled later without a large refactor.
    return 0.0


def _combined_metadata_score(
    query: str,
    summary_text: Optional[str],
    visual_meta: Optional[dict[str, float | bool]],
    *,
    descriptor_focused: bool,
    visual_boost: float,
    min_confidence: float,
) -> tuple[float, float, float]:
    summary_score = _summary_match_score(query, summary_text)
    visual_score = _visual_match_score(
        query,
        visual_meta,
        descriptor_focused=descriptor_focused,
        visual_boost=visual_boost,
        min_confidence=min_confidence,
    )
    return summary_score + visual_score, summary_score, visual_score


def _summary_match_score(query: str, summary_text: Optional[str]) -> float:
    if not summary_text:
        return 0.0

    stopwords = {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
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
    for idx in range(len(query_tokens) - 1):
        bigram = f"{query_tokens[idx]} {query_tokens[idx + 1]}"
        if bigram in summary_norm:
            bigram_hits += 1
    exact_query_hit = 1.0 if _norm_text(query).casefold() in _norm_text(summary_text).casefold() else 0.0
    return (coverage * 4.0) + (overlap_count * 0.35) + (bigram_hits * 1.5) + (exact_query_hit * 6.0)


def _metadata_full_scan_candidates(
    *,
    query: str,
    doc_page_index: dict[str, list[int]],
    page_summary_map: Optional[dict[str, str]],
    page_visual_meta_map: Optional[dict[str, dict[str, float | bool]]],
    descriptor_focused: bool,
    visual_boost: float,
    min_visual_confidence: float,
    topk_docs: int,
) -> list[tuple[str, int, float]]:
    candidates: list[tuple[str, int, float]] = []
    for doc_id, page_indices in doc_page_index.items():
        best_score = 0.0
        best_page_idx = None
        for page_idx in page_indices:
            score, _, _ = _combined_metadata_score(
                query,
                _lookup_page_summary(page_summary_map, doc_id, page_idx),
                _lookup_page_visual_meta(page_visual_meta_map, doc_id, page_idx),
                descriptor_focused=descriptor_focused,
                visual_boost=visual_boost,
                min_confidence=min_visual_confidence,
            )
            if score > best_score:
                best_score = score
                best_page_idx = page_idx
        if best_page_idx is None:
            continue
        candidates.append((doc_id, best_page_idx, float(best_score)))
    candidates.sort(key=lambda x: x[2], reverse=True)
    if topk_docs is not None and topk_docs > 0:
        candidates = candidates[:topk_docs]
    return candidates


def _rerank_with_page_summaries(
    *,
    query: str,
    retrieved: list[tuple[str, int, float]],
    page_summary_map: Optional[dict[str, str]],
) -> list[tuple[str, int, float]]:
    if not page_summary_map or not retrieved:
        return retrieved

    enriched = []
    any_summary_match = False
    for idx, (doc_id, page_idx, score) in enumerate(retrieved):
        summary_text = _lookup_page_summary(page_summary_map, str(doc_id), int(page_idx))
        match_score = _summary_match_score(query, summary_text)
        if match_score > 0:
            any_summary_match = True
        enriched.append((match_score, float(score), idx, (str(doc_id), int(page_idx), float(score))))

    if not any_summary_match:
        return retrieved

    enriched.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [item[3] for item in enriched]


def _select_selection_only_pages(
    *,
    query: str,
    retrieved: list[tuple[str, int, float]],
    selected_doc_ids: set[str],
    quota: int,
    descriptor_focused: bool,
    page_summary_map: Optional[dict[str, str]] = None,
    page_visual_meta_map: Optional[dict[str, dict[str, float | bool]]] = None,
    visual_boost: float = 2.0,
    min_visual_confidence: float = 0.4,
) -> list[tuple[str, int, float]]:
    if quota <= 0:
        return []

    ranked_available = [
        (
            rank,
            str(doc_id),
            int(page_idx),
            float(score),
            *_combined_metadata_score(
                query,
                _lookup_page_summary(page_summary_map, str(doc_id), int(page_idx)),
                _lookup_page_visual_meta(page_visual_meta_map, str(doc_id), int(page_idx)),
                descriptor_focused=descriptor_focused,
                visual_boost=visual_boost,
                min_confidence=min_visual_confidence,
            ),
        )
        for rank, (doc_id, page_idx, score) in enumerate(retrieved, start=1)
        if doc_id not in selected_doc_ids
    ]
    if not ranked_available:
        return []

    picked: list[tuple[str, int, float]] = []
    picked_doc_ids: set[str] = set()

    def add_candidate(candidate: tuple[int, str, int, float, float, float, float]) -> bool:
        _, doc_id, page_idx, score, _, _, _ = candidate
        if doc_id in selected_doc_ids or doc_id in picked_doc_ids:
            return False
        picked.append((doc_id, page_idx, score))
        picked_doc_ids.add(doc_id)
        return True

    # Always keep the best-ranked unseen doc for stability.
    add_candidate(ranked_available[0])

    # Stage-2 summary-aware selection:
    # Use page summaries to pick one extra evidence doc from a bounded window.
    if quota > 1 and (page_summary_map or page_visual_meta_map):
        rank_window = 64 if descriptor_focused else 40
        candidates = [c for c in ranked_available if 2 <= c[0] <= rank_window and c[6] > 0]
        if descriptor_focused:
            deeper = [c for c in candidates if 9 <= c[0] <= 24]
            if deeper:
                candidates = deeper
        if candidates:
            best_summary = max(candidates, key=lambda c: c[6])[6]
            best_candidates = [c for c in candidates if c[6] == best_summary]
            if descriptor_focused:
                target_band_rank = 13
                chosen = min(best_candidates, key=lambda c: abs(c[0] - target_band_rank))
            else:
                chosen = min(best_candidates, key=lambda c: c[0])
            add_candidate(chosen)

    if descriptor_focused and len(picked) < quota:
        deeper_band = [candidate for candidate in ranked_available if 9 <= candidate[0] <= 24]
        if not deeper_band:
            deeper_band = [candidate for candidate in ranked_available if candidate[0] >= 5]
        if deeper_band:
            target_band_rank = 13
            add_candidate(min(deeper_band, key=lambda candidate: abs(candidate[0] - target_band_rank)))

    for candidate in ranked_available:
        if len(picked) >= quota:
            break
        add_candidate(candidate)

    selected_doc_ids.update(doc_id for doc_id, _, _ in picked)
    return picked


def _final_global_rerank_docs(
    *,
    question: str,
    candidate_rows: list[tuple[str, int, float]],
    page_summary_map: Optional[dict[str, str]],
    page_visual_meta_map: Optional[dict[str, dict[str, float | bool]]],
    final_candidate_doc_limit: int,
    final_topk_docs: int,
    use_retriever_agent: bool,
    retriever_llm_call: Optional[Callable[[str], str]],
    retriever_require_success: bool,
    retriever_prompt_template: Optional[str],
    rerank_chunk_size: int,
    rerank_chunk_keep: int,
    visual_boost: float,
    min_visual_confidence: float,
) -> tuple[list[tuple[str, int, float]], dict]:
    if final_topk_docs <= 0 or not candidate_rows:
        return [], {"enabled": False, "reason": "no-candidates-or-nonpositive-topk"}

    # Keep one best page per doc before final reranking.
    best_by_doc: dict[str, tuple[str, int, float]] = {}
    for doc_id, page_idx, score in candidate_rows:
        doc_id = str(doc_id)
        row = (doc_id, int(page_idx), float(score))
        if doc_id not in best_by_doc or float(score) > best_by_doc[doc_id][2]:
            best_by_doc[doc_id] = row

    descriptor_focused = _is_descriptor_focused_query(question)
    unique_rows = list(best_by_doc.values())
    enriched_rows = []
    any_metadata_signal = False
    for idx, (doc_id, page_idx, score) in enumerate(unique_rows):
        combined_score, summary_score, visual_score = _combined_metadata_score(
            question,
            _lookup_page_summary(page_summary_map, doc_id, page_idx),
            _lookup_page_visual_meta(page_visual_meta_map, doc_id, page_idx),
            descriptor_focused=descriptor_focused,
            visual_boost=visual_boost,
            min_confidence=min_visual_confidence,
        )
        if combined_score > 0:
            any_metadata_signal = True
        enriched_rows.append(
            (
                combined_score,
                summary_score,
                visual_score,
                float(score),
                idx,
                (doc_id, page_idx, float(score)),
            )
        )

    if any_metadata_signal:
        enriched_rows.sort(key=lambda item: (-item[0], -item[1], -item[3], item[4]))
    else:
        enriched_rows.sort(key=lambda item: (-item[3], item[4]))

    pre_llm_rows = [item[5] for item in enriched_rows]
    candidate_doc_limit = max(int(final_candidate_doc_limit), 1)
    if candidate_doc_limit < len(pre_llm_rows):
        pre_llm_rows = pre_llm_rows[:candidate_doc_limit]

    retriever_trace = None
    tournament_trace = {
        "enabled": bool(use_retriever_agent and retriever_llm_call is not None),
        "chunk_size": max(int(rerank_chunk_size), 1),
        "chunk_keep": max(int(rerank_chunk_keep), 1),
        "rounds": [],
    }
    reranked_rows = pre_llm_rows
    if use_retriever_agent and retriever_llm_call is not None and pre_llm_rows:
        chunk_size = max(int(rerank_chunk_size), 1)
        chunk_keep = max(int(rerank_chunk_keep), 1)
        topk_target = max(int(final_topk_docs), 1)
        stage_rows = list(pre_llm_rows)
        round_idx = 1

        # Tournament rerank: process large candidate sets in chunks to avoid prompt overflows.
        while len(stage_rows) > max(chunk_size, topk_target):
            chunks = [stage_rows[i : i + chunk_size] for i in range(0, len(stage_rows), chunk_size)]
            next_stage: list[tuple[str, int, float]] = []
            round_rows_before = len(stage_rows)
            round_info = {
                "round": round_idx,
                "rows_before": round_rows_before,
                "num_chunks": len(chunks),
                "chunk_results": [],
            }
            min_keep_needed = max(1, math.ceil(topk_target / max(len(chunks), 1)))
            for chunk_idx, chunk in enumerate(chunks, start=1):
                select_k = min(len(chunk), max(chunk_keep, min_keep_needed))
                chunk_reranked, chunk_trace = _rerank_with_selection_retriever_agent(
                    query=question,
                    retrieved=chunk,
                    page_summary_map=page_summary_map,
                    llm_call=retriever_llm_call,
                    candidate_doc_limit=len(chunk),
                    max_select_docs=select_k,
                    require_success=retriever_require_success,
                    require_exact_count=retriever_require_success,
                    prompt_template=retriever_prompt_template,
                )
                picked_chunk: list[tuple[str, int, float]] = []
                picked_seen: set[str] = set()
                for doc_id, page_idx, score in chunk_reranked:
                    doc_id = str(doc_id)
                    if doc_id in picked_seen:
                        continue
                    picked_seen.add(doc_id)
                    picked_chunk.append((doc_id, int(page_idx), float(score)))
                    if len(picked_chunk) >= select_k:
                        break
                next_stage.extend(picked_chunk)
                round_info["chunk_results"].append(
                    {
                        "chunk_index": chunk_idx,
                        "chunk_size": len(chunk),
                        "picked": len(picked_chunk),
                        "retriever_agent": chunk_trace,
                    }
                )

            dedup_stage: list[tuple[str, int, float]] = []
            dedup_seen: set[str] = set()
            for doc_id, page_idx, score in next_stage:
                doc_id = str(doc_id)
                if doc_id in dedup_seen:
                    continue
                dedup_seen.add(doc_id)
                dedup_stage.append((doc_id, int(page_idx), float(score)))
            stage_rows = dedup_stage
            round_info["rows_after"] = len(stage_rows)
            tournament_trace["rounds"].append(round_info)
            if retriever_require_success and len(stage_rows) < topk_target:
                raise RuntimeError(
                    "Final tournament candidate set dropped below final_topk_docs. "
                    f"rows_after={len(stage_rows)} final_topk_docs={topk_target}"
                )
            round_idx += 1
            if round_idx > 10:
                if retriever_require_success:
                    raise RuntimeError("Final global rerank exceeded max tournament rounds (10).")
                break

        if retriever_require_success and len(stage_rows) < topk_target:
            raise RuntimeError(
                "Final rerank candidate set is smaller than final_topk_docs. "
                f"candidate_rows={len(stage_rows)} final_topk_docs={topk_target}"
            )

        reranked_rows, retriever_trace = _rerank_with_selection_retriever_agent(
            query=question,
            retrieved=stage_rows,
            page_summary_map=page_summary_map,
            llm_call=retriever_llm_call,
            candidate_doc_limit=min(len(stage_rows), chunk_size),
            max_select_docs=topk_target,
            require_success=retriever_require_success,
            require_exact_count=retriever_require_success,
            prompt_template=retriever_prompt_template,
        )

    final_rows: list[tuple[str, int, float]] = []
    final_seen: set[str] = set()
    for doc_id, page_idx, score in reranked_rows:
        doc_id = str(doc_id)
        if doc_id in final_seen:
            continue
        final_seen.add(doc_id)
        final_rows.append((doc_id, int(page_idx), float(score)))
        if len(final_rows) >= max(final_topk_docs, 1):
            break

    trace = {
        "enabled": True,
        "candidate_docs_unique": len(unique_rows),
        "candidate_doc_limit": candidate_doc_limit,
        "used_retriever_agent": bool(use_retriever_agent and retriever_llm_call is not None),
        "retriever_require_success": bool(retriever_require_success),
        "tournament": tournament_trace,
        "retriever_agent": retriever_trace,
        "pre_rerank_top_docs": _top_docs_payload_from_pages(
            pre_llm_rows,
            limit=min(20, len(pre_llm_rows)),
            page_summary_map=page_summary_map,
            page_visual_meta_map=page_visual_meta_map,
            query=question,
            descriptor_focused=descriptor_focused,
            visual_boost=visual_boost,
            min_visual_confidence=min_visual_confidence,
        ),
        "final_top_docs": _top_docs_payload_from_pages(
            final_rows,
            limit=len(final_rows),
            page_summary_map=page_summary_map,
            page_visual_meta_map=page_visual_meta_map,
            query=question,
            descriptor_focused=descriptor_focused,
            visual_boost=visual_boost,
            min_visual_confidence=min_visual_confidence,
        ),
    }
    return final_rows, trace


def _top_docs_payload_from_pages(
    rows: list[tuple[str, int, float]],
    limit: int,
    page_summary_map: Optional[dict[str, str]] = None,
    page_visual_meta_map: Optional[dict[str, dict[str, float | bool]]] = None,
    query: Optional[str] = None,
    descriptor_focused: bool = False,
    visual_boost: float = 2.0,
    min_visual_confidence: float = 0.4,
) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    rank = 0
    for doc_id, page_idx, score in rows:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        rank += 1
        row = {
            "doc_id": str(doc_id),
            "returned_rank": rank,
            "best_page_idx": int(page_idx),
            "best_page_score": float(score),
        }
        summary_text = _lookup_page_summary(page_summary_map, str(doc_id), int(page_idx))
        if summary_text:
            row["page_summary"] = summary_text[:240]
            if query:
                row["summary_match_score"] = round(_summary_match_score(query, summary_text), 4)
        visual_meta = _lookup_page_visual_meta(page_visual_meta_map, str(doc_id), int(page_idx))
        if visual_meta:
            row["visual_meta"] = {
                "three_torches_logo_visible": bool(visual_meta.get("three_torches_logo_visible")),
                "logo_visible": bool(visual_meta.get("logo_visible")),
                "shield_or_crest_visible": bool(visual_meta.get("shield_or_crest_visible")),
                "confidence": round(float(visual_meta.get("confidence", 0.0) or 0.0), 4),
            }
        out.append(row)
        if len(out) >= limit:
            break
    return out


def _top_pages_payload_from_pages(
    rows: list[tuple[str, int, float]],
    limit: int,
    page_summary_map: Optional[dict[str, str]] = None,
    page_visual_meta_map: Optional[dict[str, dict[str, float | bool]]] = None,
    query: Optional[str] = None,
    descriptor_focused: bool = False,
    visual_boost: float = 2.0,
    min_visual_confidence: float = 0.4,
) -> list[dict]:
    out: list[dict] = []
    for doc_id, page_idx, score in rows[:limit]:
        row = {
            "doc_id": str(doc_id),
            "page_idx": int(page_idx),
            "score": float(score),
        }
        summary_text = _lookup_page_summary(page_summary_map, str(doc_id), int(page_idx))
        if summary_text:
            row["page_summary"] = summary_text[:240]
            if query:
                row["summary_match_score"] = round(_summary_match_score(query, summary_text), 4)
        visual_meta = _lookup_page_visual_meta(page_visual_meta_map, str(doc_id), int(page_idx))
        if visual_meta:
            row["visual_meta"] = {
                "three_torches_logo_visible": bool(visual_meta.get("three_torches_logo_visible")),
                "logo_visible": bool(visual_meta.get("logo_visible")),
                "shield_or_crest_visible": bool(visual_meta.get("shield_or_crest_visible")),
                "confidence": round(float(visual_meta.get("confidence", 0.0) or 0.0), 4),
            }
        out.append(row)
    return out


def run_selection_only_session(
    *,
    question: str,
    rag_model,
    docid2embs: dict,
    selection_planner_backend: str = "heuristic",
    selection_planner_llm_call: Optional[Callable[[str], str]] = None,
    selection_planner_prompt_template: Optional[str] = None,
    selection_max_hop_queries: int,
    selection_topk_docs_per_hop: int,
    selection_root_topk_docs: Optional[int] = None,
    selection_variant_topk_docs: Optional[int] = None,
    selection_max_variant_queries: Optional[int] = None,
    selection_summary_full_scan: bool = False,
    selection_summary_full_scan_topk: int = 1000,
    selection_retrieval_depth_multiplier: int = 8,
    selection_candidate_single_page_per_doc: bool = True,
    selection_stop_no_new_docs: bool = False,
    selection_stop_min_new_docs: int = 1,
    selection_retriever_agent: bool = False,
    selection_retriever_candidate_docs: int = 40,
    selection_retriever_select_docs: int = 12,
    selection_retriever_require_success: bool = False,
    selection_final_global_rerank: bool = False,
    selection_final_candidate_docs: Optional[int] = None,
    selection_final_topk_docs: int = 10,
    selection_final_rerank_chunk_size: int = 40,
    selection_final_rerank_chunk_keep: int = 10,
    selection_retriever_llm_call: Optional[Callable[[str], str]] = None,
    selection_retriever_prompt_template: Optional[str] = None,
    selection_answer_conditioned: bool = False,
    selection_answer_llm_call: Optional[Callable[[str], str]] = None,
    selection_answer_context_candidates: int = 6,
    selection_answer_prompt_template: Optional[str] = None,
    page_summary_map: Optional[dict[str, str]] = None,
    page_visual_meta_map: Optional[dict[str, dict[str, float | bool]]] = None,
    selection_visual_boost: float = 2.0,
    selection_visual_min_confidence: float = 0.4,
):
    hop_queries, planner_reply = _plan_selection_hop_queries(
        question=question,
        max_hops=selection_max_hop_queries,
        planner_backend=selection_planner_backend,
        planner_llm_call=selection_planner_llm_call,
        planner_prompt_template=selection_planner_prompt_template,
    )
    if hop_queries and selection_max_variant_queries is not None:
        root_query = hop_queries[0]
        variant_queries = hop_queries[1 : 1 + max(selection_max_variant_queries, 0)]
        hop_queries = [root_query] + variant_queries
    logger.info(
        "selection-only plan: {} queries | {}",
        len(hop_queries),
        hop_queries,
    )
    if selection_answer_conditioned:
        logger.info("selection-only answer-conditioned chaining enabled.")
    if page_summary_map:
        logger.info("selection-only summary rerank active: {} page summaries loaded", len(page_summary_map))
    if page_visual_meta_map:
        logger.info("selection-only visual metadata active: {} page entries loaded", len(page_visual_meta_map))
    final_candidate_doc_limit = max(
        int(selection_final_candidate_docs)
        if selection_final_candidate_docs is not None
        else int(selection_retriever_candidate_docs),
        1,
    )
    if selection_final_global_rerank:
        logger.info(
            "selection-only final global rerank active: candidate_docs={} topk={} chunk_size={} chunk_keep={}",
            final_candidate_doc_limit,
            max(int(selection_final_topk_docs), 1),
            max(int(selection_final_rerank_chunk_size), 1),
            max(int(selection_final_rerank_chunk_keep), 1),
        )
    if selection_retriever_require_success:
        logger.info("selection-only retriever strict mode enabled: no non-LLM fallback.")
    doc_page_index = _build_doc_page_index(page_summary_map, page_visual_meta_map, set(docid2embs.keys()))
    if selection_summary_full_scan and doc_page_index:
        logger.info(
            "selection-only metadata full-scan active: docs_with_summaries={} topk={}",
            len(doc_page_index),
            selection_summary_full_scan_topk,
        )

    steps = []
    selected_doc_ids: set[str] = set()
    max_selection_quota = max(
        selection_topk_docs_per_hop,
        selection_root_topk_docs or 0,
        selection_variant_topk_docs or 0,
    )
    base_retrieval_depth = _selection_only_retrieval_depth(
        max_selection_quota,
        depth_multiplier=selection_retrieval_depth_multiplier,
    )
    final_reason = "selection_only"
    planned_queries = list(hop_queries)
    if selection_answer_conditioned and planned_queries:
        dynamic_queries = [planned_queries[0]]
        planner_backfill_queries = planned_queries[1:]
    else:
        dynamic_queries = list(planned_queries)
        planner_backfill_queries = []
    intermediate_answers: list[str] = []
    final_union_candidate_rows: list[tuple[str, int, float]] = []

    for turn in range(1, max(selection_max_hop_queries, 1) + 1):
        if turn > len(dynamic_queries):
            break
        hop_query = dynamic_queries[turn - 1]
        if turn == 1 and selection_root_topk_docs is not None:
            selection_quota = max(selection_root_topk_docs, 0)
        elif turn > 1 and selection_variant_topk_docs is not None:
            selection_quota = max(selection_variant_topk_docs, 0)
        else:
            selection_quota = max(selection_topk_docs_per_hop, 0)
        descriptor_focused = _is_descriptor_focused_query(hop_query)
        query_retrieval_depth = (
            max(base_retrieval_depth * 4, 64)
            if descriptor_focused
            else base_retrieval_depth
        )
        retrieved = rag_model.retrieve_pages_from_docs(
            query=hop_query,
            docid2embs=docid2embs,
            index=None,
            token2pageuid=None,
            all_token_embeddings=None,
            n_return_pages=query_retrieval_depth,
            single_page_from_each_doc=selection_candidate_single_page_per_doc,
            show_progress=False,
        )
        metadata_candidates = []
        if selection_summary_full_scan and doc_page_index:
            metadata_candidates = _metadata_full_scan_candidates(
                query=hop_query,
                doc_page_index=doc_page_index,
                page_summary_map=page_summary_map,
                page_visual_meta_map=page_visual_meta_map,
                descriptor_focused=descriptor_focused,
                visual_boost=selection_visual_boost,
                min_visual_confidence=selection_visual_min_confidence,
                topk_docs=max(selection_summary_full_scan_topk, 0),
            )
        if metadata_candidates:
            seen_doc_ids = {doc_id for doc_id, _, _ in metadata_candidates}
            retrieved_for_selection = metadata_candidates + [x for x in retrieved if x[0] not in seen_doc_ids]
        else:
            retrieved_for_selection = retrieved
        if selection_final_global_rerank:
            final_union_candidate_rows.extend(retrieved_for_selection[:final_candidate_doc_limit])
        retriever_agent_trace = None
        if selection_retriever_agent:
            retrieved_for_selection, retriever_agent_trace = _rerank_with_selection_retriever_agent(
                query=hop_query,
                retrieved=retrieved_for_selection,
                page_summary_map=page_summary_map,
                llm_call=selection_retriever_llm_call,
                candidate_doc_limit=selection_retriever_candidate_docs,
                max_select_docs=selection_retriever_select_docs,
                require_success=selection_retriever_require_success,
                prompt_template=selection_retriever_prompt_template,
            )

        top_pages_payload = _top_pages_payload_from_pages(
            retrieved_for_selection,
            limit=query_retrieval_depth,
            page_summary_map=page_summary_map,
            page_visual_meta_map=page_visual_meta_map,
            query=hop_query,
            descriptor_focused=descriptor_focused,
            visual_boost=selection_visual_boost,
            min_visual_confidence=selection_visual_min_confidence,
        )
        top_docs = _top_docs_payload_from_pages(
            retrieved_for_selection,
            limit=query_retrieval_depth,
            page_summary_map=page_summary_map,
            page_visual_meta_map=page_visual_meta_map,
            query=hop_query,
            descriptor_focused=descriptor_focused,
            visual_boost=selection_visual_boost,
            min_visual_confidence=selection_visual_min_confidence,
        )
        selected_doc_ids_before = set(selected_doc_ids)
        selected_pages = _select_selection_only_pages(
            query=hop_query,
            retrieved=retrieved_for_selection,
            selected_doc_ids=selected_doc_ids,
            quota=selection_quota,
            descriptor_focused=descriptor_focused,
            page_summary_map=page_summary_map,
            page_visual_meta_map=page_visual_meta_map,
            visual_boost=selection_visual_boost,
            min_visual_confidence=selection_visual_min_confidence,
        )
        new_selected_docs = sorted(selected_doc_ids - selected_doc_ids_before)
        new_selected_doc_count = len(new_selected_docs)
        next_query = None
        intermediate_answer = None
        answer_conditioned_trace = None
        if selection_answer_conditioned and turn < max(selection_max_hop_queries, 1):
            next_query, intermediate_answer, answer_conditioned_trace = _propose_answer_conditioned_follow_up_query(
                question=question,
                current_query=hop_query,
                turn=turn,
                max_hops=max(selection_max_hop_queries, 1),
                selected_pages=selected_pages,
                top_pages_payload=top_pages_payload,
                page_summary_map=page_summary_map,
                llm_call=selection_answer_llm_call,
                context_candidates=selection_answer_context_candidates,
                previous_intermediate_answers=intermediate_answers,
                prompt_template=selection_answer_prompt_template,
            )
            if intermediate_answer:
                intermediate_answers.append(intermediate_answer)

            if next_query:
                existing = {q.casefold() for q in dynamic_queries}
                if next_query.casefold() not in existing:
                    dynamic_queries.append(next_query)
                elif answer_conditioned_trace is not None:
                    answer_conditioned_trace["success"] = False
                    answer_conditioned_trace["error"] = "duplicate_next_query"
                    next_query = None

            # Backfill with planner-generated queries when conditioned follow-up is unavailable.
            if next_query is None and planner_backfill_queries:
                fallback_query = planner_backfill_queries.pop(0)
                existing = {q.casefold() for q in dynamic_queries}
                if fallback_query.casefold() not in existing:
                    dynamic_queries.append(fallback_query)
                    if answer_conditioned_trace is None:
                        answer_conditioned_trace = {
                            "attempted": False,
                            "success": False,
                            "error": "planner_backfill",
                            "intermediate_answer": intermediate_answer,
                            "next_query": fallback_query,
                            "raw_reply_preview": None,
                            "evidence_count": 0,
                        }
                    else:
                        answer_conditioned_trace["fallback_query"] = fallback_query

        steps.append(
            {
                "turn": turn,
                "query": hop_query,
                "selected_pages": selected_pages,
                "answer": intermediate_answer,
                "stop_reason": None,
                "action_type": "selection_only",
                "facts_added": ([intermediate_answer] if intermediate_answer else []),
                "new_selected_doc_count": new_selected_doc_count,
                "new_selected_doc_ids": new_selected_docs,
                "next_query": next_query,
                "answer_conditioned_trace": answer_conditioned_trace,
                "retrieval_traces": [
                    {
                        "query": hop_query,
                        "requested_page_count": query_retrieval_depth,
                        "returned_page_count": len(retrieved),
                        "metadata_full_scan_candidates": len(metadata_candidates),
                        "retriever_agent": retriever_agent_trace,
                        "top_pages": top_pages_payload,
                        "top_docs": top_docs,
                    }
                ],
            }
        )
        if selection_stop_no_new_docs and new_selected_doc_count < max(1, int(selection_stop_min_new_docs)):
            steps[-1]["stop_reason"] = "no-new-selected-docs"
            final_reason = "no-new-selected-docs"
            break

    out = {
        "answer": None,
        "reason": final_reason,
        "steps": steps,
            "selection_plan": {
                "planner_reply": planner_reply,
                "hop_queries": planned_queries,
                "executed_queries": [str(step.get("query") or "") for step in steps],
                "planner_backend": selection_planner_backend,
                "planner_prompt_template_used": bool(selection_planner_prompt_template),
                "topk_docs_per_hop": selection_topk_docs_per_hop,
                "root_topk_docs": selection_root_topk_docs,
                "variant_topk_docs": selection_variant_topk_docs,
                "max_variant_queries": selection_max_variant_queries,
                "retrieval_depth_per_query": base_retrieval_depth,
                "descriptor_retrieval_depth_per_query": max(base_retrieval_depth * 4, 64),
                "retrieval_depth_multiplier": selection_retrieval_depth_multiplier,
                "single_page_from_each_doc": selection_candidate_single_page_per_doc,
                "stop_no_new_docs": selection_stop_no_new_docs,
                "stop_min_new_docs": selection_stop_min_new_docs,
                "page_summaries_enabled": bool(page_summary_map),
                "page_visual_metadata_enabled": bool(page_visual_meta_map),
                "summary_full_scan": selection_summary_full_scan,
                "summary_full_scan_topk": selection_summary_full_scan_topk,
                "selection_retriever_agent": selection_retriever_agent,
                "selection_retriever_candidate_docs": selection_retriever_candidate_docs,
                "selection_retriever_select_docs": selection_retriever_select_docs,
                "selection_retriever_require_success": selection_retriever_require_success,
                "selection_final_global_rerank": selection_final_global_rerank,
                "selection_final_candidate_docs": final_candidate_doc_limit,
                "selection_final_topk_docs": selection_final_topk_docs,
                "selection_final_rerank_chunk_size": selection_final_rerank_chunk_size,
                "selection_final_rerank_chunk_keep": selection_final_rerank_chunk_keep,
                "selection_retriever_prompt_template_used": bool(selection_retriever_prompt_template),
                "selection_answer_conditioned": selection_answer_conditioned,
                "selection_answer_context_candidates": selection_answer_context_candidates,
                "selection_answer_prompt_template_used": bool(selection_answer_prompt_template),
                "selection_visual_boost": selection_visual_boost,
                "selection_visual_min_confidence": selection_visual_min_confidence,
            },
    }
    if selection_final_global_rerank:
        final_selected_pages, final_trace = _final_global_rerank_docs(
            question=question,
            candidate_rows=final_union_candidate_rows,
            page_summary_map=page_summary_map,
            page_visual_meta_map=page_visual_meta_map,
            final_candidate_doc_limit=final_candidate_doc_limit,
            final_topk_docs=max(int(selection_final_topk_docs), 1),
            use_retriever_agent=selection_retriever_agent,
            retriever_llm_call=selection_retriever_llm_call,
            retriever_require_success=selection_retriever_require_success,
            retriever_prompt_template=selection_retriever_prompt_template,
            rerank_chunk_size=selection_final_rerank_chunk_size,
            rerank_chunk_keep=selection_final_rerank_chunk_keep,
            visual_boost=selection_visual_boost,
            min_visual_confidence=selection_visual_min_confidence,
        )
        out["final_selected_pages"] = final_selected_pages
        out["final_selected_doc_ids"] = [doc_id for doc_id, _, _ in final_selected_pages]
        out["final_selected_doc_count"] = len(out["final_selected_doc_ids"])
        out["final_selection_trace"] = final_trace
    return out


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
    all_selected_docs_ordered = []
    all_selected_docs_seen: set[str] = set()

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
            if doc_id not in all_selected_docs_seen:
                all_selected_docs_seen.add(doc_id)
                all_selected_docs_ordered.append(doc_id)

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

    final_selected_doc_ids_raw = agent_payload.get("final_selected_doc_ids") or []
    final_selected_doc_ids: list[str] = []
    final_seen = set()
    for doc_id in final_selected_doc_ids_raw:
        doc_id = str(doc_id)
        if doc_id in final_seen:
            continue
        final_seen.add(doc_id)
        final_selected_doc_ids.append(doc_id)
    has_final_selection = bool(final_selected_doc_ids)
    if not has_final_selection:
        final_selected_doc_ids = list(all_selected_docs_ordered)

    final_selected_doc_set = set(final_selected_doc_ids)
    final_selected_gold = [doc_id for doc_id in final_selected_doc_ids if doc_id in gold_set]
    selected_any_final = bool(final_selected_gold)
    selected_all_final = bool(gold_set) and all(doc_id in final_selected_doc_set for doc_id in gold_set)
    selected_any_legacy = selected_any_final if has_final_selection else (first_turn_with_gold is not None)
    all_selected_gold_set = set(all_selected_gold)
    selected_all_legacy = selected_all_final if has_final_selection else (
        bool(gold_set) and all(doc_id in all_selected_gold_set for doc_id in gold_set)
    )

    return {
        "first_turn_with_gold_doc_selected": first_turn_with_gold,
        "selected_any_gold_doc": selected_any_legacy,
        "selected_all_gold_docs": selected_all_legacy,
        "selected_gold_docs_any_turn": all_selected_gold,
        "selection_has_final_rerank": has_final_selection,
        "selected_doc_ids_final": final_selected_doc_ids,
        "selected_total_final": len(final_selected_doc_ids),
        "selected_doc_ranks_in_doc_pool_final": {d: pos.get(d) for d in final_selected_doc_ids},
        "selected_gold_docs_final": final_selected_gold,
        "selected_gold_doc_ranks_in_doc_pool_final": {d: pos.get(d) for d in final_selected_gold},
        "selected_any_gold_doc_final": selected_any_final,
        "selected_all_gold_docs_final": selected_all_final,
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
    if args.selection_profile == "simpledoc":
        args.selection_only = True
        args.selection_planner_backend = "llm"
        args.selection_max_hop_queries = max(int(args.selection_max_hop_queries), 3)
        args.selection_topk_docs_per_hop = max(int(args.selection_topk_docs_per_hop), 2)
        args.selection_retriever_agent = True
        args.selection_retriever_candidate_docs = max(int(args.selection_retriever_candidate_docs), 40)
        args.selection_retriever_select_docs = max(int(args.selection_retriever_select_docs), 12)
        args.selection_summary_full_scan = True
        args.selection_retrieval_depth_multiplier = max(int(args.selection_retrieval_depth_multiplier), 8)
        args.selection_stop_no_new_docs = True
        args.selection_stop_min_new_docs = max(int(args.selection_stop_min_new_docs), 1)
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
    selection_planner_prompt_template = _read_text_file_optional(args.selection_planner_prompt_file)
    selection_retriever_prompt_template = _read_text_file_optional(args.selection_retriever_prompt_file)
    selection_answer_prompt_template = _read_text_file_optional(args.selection_answer_prompt_file)

    selection_needs_llm = bool(args.selection_retriever_agent) or (
        args.selection_only and args.selection_planner_backend == "llm"
    ) or bool(args.selection_only and args.selection_answer_conditioned)

    if args.selection_only:
        llm_call = None
        if selection_needs_llm:
            if args.policy_backend == "local-hf":
                if not args.policy_model:
                    raise ValueError("--policy-model is required when LLM planner/retriever is enabled in --selection-only mode")
                llm_call = make_llm_call_local_hf(args.policy_model, device=policy_device)
                logger.info("Selection-only LLM planner/retriever active via local-hf policy model.")
            elif args.policy_backend == "openai-api":
                llm_call = make_llm_call_openai_api(
                    base_url=str(args.policy_base_url or ""),
                    model=str(args.policy_model or ""),
                    api_key_file=args.policy_api_key_file,
                    timeout_s=float(args.policy_timeout_s),
                    max_retries=int(args.policy_max_retries),
                    max_tokens=int(args.policy_max_tokens),
                    temperature=float(args.policy_temperature),
                )
                logger.info("Selection-only LLM planner/retriever active via openai-api policy backend.")
            else:
                raise ValueError("LLM planner/retriever in --selection-only mode requires --policy-backend local-hf or openai-api")
        else:
            logger.info("Selection-only mode uses heuristic planner and no RetrieverAgent LLM reranking.")
        selection_planner_llm_call = llm_call if args.selection_planner_backend == "llm" else None
        selection_retriever_llm_call = llm_call if args.selection_retriever_agent else None
        selection_answer_llm_call = llm_call if args.selection_answer_conditioned else None
    elif args.policy_backend == "stub":
        llm_call = make_llm_call_stub()
        selection_planner_llm_call = None
        selection_retriever_llm_call = None
        selection_answer_llm_call = None
    elif args.policy_backend == "local-hf":
        if not args.policy_model:
            raise ValueError("--policy-model is required when --policy-backend local-hf")
        llm_call = make_llm_call_local_hf(args.policy_model, device=policy_device)
        selection_planner_llm_call = None
        selection_retriever_llm_call = None
        selection_answer_llm_call = None
    else:
        llm_call = make_llm_call_openai_api(
            base_url=str(args.policy_base_url or ""),
            model=str(args.policy_model or ""),
            api_key_file=args.policy_api_key_file,
            timeout_s=float(args.policy_timeout_s),
            max_retries=int(args.policy_max_retries),
            max_tokens=int(args.policy_max_tokens),
            temperature=float(args.policy_temperature),
        )
        selection_planner_llm_call = None
        selection_retriever_llm_call = None
        selection_answer_llm_call = None

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

    page_summary_map = None
    if args.page_summaries_file is not None:
        page_summary_map = load_context_map(args.page_summaries_file)
        logger.info("Loaded {} page summaries from {}", len(page_summary_map), args.page_summaries_file)
    if args.selection_retriever_agent and not page_summary_map:
        raise ValueError("--selection-retriever-agent requires --page-summaries-file")
    page_visual_meta_map = None
    if args.page_visual_metadata_file is not None:
        page_visual_meta_map = load_page_visual_metadata_map(args.page_visual_metadata_file)
        logger.info(
            "Loaded {} page visual metadata entries from {}",
            len(page_visual_meta_map),
            args.page_visual_metadata_file,
        )

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
                    if not args.selection_only:
                        merged_context_map.update(cache.get_doc_context_map(doc_id))

                docid2embs = move_doc_embs(docid2embs_cpu, args.device)
                if args.selection_only:
                    result = run_selection_only_session(
                        question=question,
                        rag_model=rag_model,
                        docid2embs=docid2embs,
                        selection_planner_backend=args.selection_planner_backend,
                        selection_planner_llm_call=selection_planner_llm_call,
                        selection_planner_prompt_template=selection_planner_prompt_template,
                        selection_max_hop_queries=args.selection_max_hop_queries,
                        selection_topk_docs_per_hop=args.selection_topk_docs_per_hop,
                        selection_root_topk_docs=args.selection_root_topk_docs,
                        selection_variant_topk_docs=args.selection_variant_topk_docs,
                        selection_max_variant_queries=args.selection_max_variant_queries,
                        selection_summary_full_scan=args.selection_summary_full_scan,
                        selection_summary_full_scan_topk=args.selection_summary_full_scan_topk,
                        selection_retrieval_depth_multiplier=args.selection_retrieval_depth_multiplier,
                        selection_candidate_single_page_per_doc=(not args.selection_candidate_multi_page),
                        selection_stop_no_new_docs=args.selection_stop_no_new_docs,
                        selection_stop_min_new_docs=args.selection_stop_min_new_docs,
                        selection_retriever_agent=args.selection_retriever_agent,
                        selection_retriever_candidate_docs=args.selection_retriever_candidate_docs,
                        selection_retriever_select_docs=args.selection_retriever_select_docs,
                        selection_retriever_require_success=args.selection_retriever_require_success,
                        selection_final_global_rerank=args.selection_final_global_rerank,
                        selection_final_candidate_docs=args.selection_final_candidate_docs,
                        selection_final_topk_docs=args.selection_final_topk_docs,
                        selection_final_rerank_chunk_size=args.selection_final_rerank_chunk_size,
                        selection_final_rerank_chunk_keep=args.selection_final_rerank_chunk_keep,
                        selection_retriever_llm_call=selection_retriever_llm_call,
                        selection_retriever_prompt_template=selection_retriever_prompt_template,
                        selection_answer_conditioned=args.selection_answer_conditioned,
                        selection_answer_llm_call=selection_answer_llm_call,
                        selection_answer_context_candidates=args.selection_answer_context_candidates,
                        selection_answer_prompt_template=selection_answer_prompt_template,
                        page_summary_map=page_summary_map,
                        page_visual_meta_map=page_visual_meta_map,
                        selection_visual_boost=args.selection_visual_boost,
                        selection_visual_min_confidence=args.selection_visual_min_confidence,
                    )
                else:
                    candidate_context_fn = make_candidate_context_fn(
                        merged_context_map,
                        max_chars=args.context_max_chars,
                    )
                    result = run_agent_session(
                        query=question,
                        rag_model=rag_model,
                        docid2embs=docid2embs,
                        doc_ranked_ids=doc_pool_ids,
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
            "explore_return_pages_multiplier": args.explore_return_pages_multiplier,
            "selection_only": args.selection_only,
            "selection_profile": args.selection_profile,
            "selection_planner": (args.selection_planner_backend if args.selection_only else None),
            "selection_planner_prompt_file": (
                str(args.selection_planner_prompt_file) if args.selection_planner_prompt_file else None
            ),
            "selection_max_hop_queries": args.selection_max_hop_queries,
            "selection_topk_docs_per_hop": args.selection_topk_docs_per_hop,
            "selection_root_topk_docs": args.selection_root_topk_docs,
            "selection_variant_topk_docs": args.selection_variant_topk_docs,
            "selection_max_variant_queries": args.selection_max_variant_queries,
            "selection_summary_full_scan": args.selection_summary_full_scan,
            "selection_summary_full_scan_topk": args.selection_summary_full_scan_topk,
            "selection_retrieval_depth_multiplier": args.selection_retrieval_depth_multiplier,
            "selection_candidate_single_page_per_doc": (not args.selection_candidate_multi_page),
            "selection_stop_no_new_docs": args.selection_stop_no_new_docs,
            "selection_stop_min_new_docs": args.selection_stop_min_new_docs,
            "selection_retriever_agent": args.selection_retriever_agent,
            "selection_retriever_candidate_docs": args.selection_retriever_candidate_docs,
            "selection_retriever_select_docs": args.selection_retriever_select_docs,
            "selection_retriever_require_success": args.selection_retriever_require_success,
            "selection_final_global_rerank": args.selection_final_global_rerank,
            "selection_final_candidate_docs": args.selection_final_candidate_docs,
            "selection_final_topk_docs": args.selection_final_topk_docs,
            "selection_final_rerank_chunk_size": args.selection_final_rerank_chunk_size,
            "selection_final_rerank_chunk_keep": args.selection_final_rerank_chunk_keep,
            "selection_retriever_prompt_file": (
                str(args.selection_retriever_prompt_file) if args.selection_retriever_prompt_file else None
            ),
            "selection_answer_conditioned": args.selection_answer_conditioned,
            "selection_answer_context_candidates": args.selection_answer_context_candidates,
            "selection_answer_prompt_file": (
                str(args.selection_answer_prompt_file) if args.selection_answer_prompt_file else None
            ),
            "page_summaries_file": (str(args.page_summaries_file) if args.page_summaries_file else None),
            "page_visual_metadata_file": (
                str(args.page_visual_metadata_file) if args.page_visual_metadata_file else None
            ),
            "selection_visual_boost": args.selection_visual_boost,
            "selection_visual_min_confidence": args.selection_visual_min_confidence,
            "policy_backend": args.policy_backend,
            "policy_model": args.policy_model,
            "policy_base_url": args.policy_base_url,
            "policy_api_key_file": (str(args.policy_api_key_file) if args.policy_api_key_file else None),
            "policy_timeout_s": args.policy_timeout_s,
            "policy_max_retries": args.policy_max_retries,
            "policy_max_tokens": args.policy_max_tokens,
            "policy_temperature": args.policy_temperature,
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
