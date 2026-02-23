from __future__ import annotations

from dataclasses import dataclass
import re
from textwrap import dedent
from typing import List, Optional, Sequence

from loguru import logger

from .memory import AgentMemory, PageRef


AGENT_PROMPT = dedent(
    """
    You are a concise research agent answering document questions with a tight page budget.
    You have: a question, previously seen pages, and newly retrieved candidate pages (with brief summaries).

    Tools you can implicitly call:
    - READ: read provided candidate pages (you already have their summaries; use them to decide).
    - STOP: when confident in a short answer or no further progress is likely.

    Policy:
    1) If a candidate page directly contains the answer, respond with ANSWER: <short answer> and STOP.
       If the question asks for a title/name and a candidate summary explicitly contains that title/name, answer immediately.
    2) Otherwise, pick the smallest subset of candidate pages that likely advance the answer and say CONTINUE with a refined query if needed.
       Do NOT repeat the same question as CONTINUE QUERY unless you substantially refine it.
    3) If stuck or evidence is insufficient, respond with UNANSWERABLE and STOP.

    Format your reply as one of:
    - ANSWER: <text>
    - CONTINUE QUERY: <refined query>
    - UNANSWERABLE: <reason>
    """
)


_TITLE_QUERY_HINTS = (
    "title",
    "name appears",
    "name shown",
    "series name",
    "movie name",
    "article title",
)

_TITLE_CUTOFF_MARKERS = (
    " 31 languages",
    " ArticleTalk",
    " Read Edit",
    " View history",
    " Genre ",
    " Created by ",
    " Directed by ",
)


def _normalize_ws(text: str) -> str:
    return " ".join(text.split())


def _extract_title_from_summary(summary: str) -> Optional[str]:
    text = _normalize_ws(summary)
    if not text:
        return None

    cut = len(text)
    for marker in _TITLE_CUTOFF_MARKERS:
        idx = text.find(marker)
        if idx > 0:
            cut = min(cut, idx)

    candidate = text[:cut].strip(" -:|.,;")
    if not candidate:
        return None

    # If no known marker is present, keep only a short prefix to avoid over-answering.
    if cut == len(text):
        candidate = " ".join(candidate.split()[:6]).strip(" -:|.,;")

    if len(candidate) < 2 or len(candidate) > 80:
        return None
    if not re.search(r"[A-Za-z]", candidate):
        return None

    return candidate


def _maybe_direct_answer_from_context(
    query: str,
    candidates: Sequence[PageRef],
    candidate_contexts: Optional[Sequence[Optional[str]]],
) -> Optional[dict]:
    if not candidates or not candidate_contexts:
        return None

    q = query.lower()
    if not any(hint in q for hint in _TITLE_QUERY_HINTS):
        return None

    for i, context in enumerate(candidate_contexts):
        if not context:
            continue
        title = _extract_title_from_summary(context)
        if not title:
            continue
        return {"type": "answer", "text": title, "chosen": [candidates[i]]}

    return None


@dataclass
class AgentPolicy:
    """Lightweight heuristic policy wrapping an LLM (prompt provided to caller)."""

    def select_action(
        self,
        *,
        query: str,
        candidates: Sequence[PageRef],
        memory: AgentMemory,
        llm_call,
        candidate_contexts: Optional[Sequence[Optional[str]]] = None,
    ) -> dict:
        """Decide next action using the provided llm_call(query: str) -> str.

        Returns a dict with keys: {type: 'answer'|'continue'|'unanswerable', 'text': str, 'chosen': list[PageRef]}
        """
        direct = _maybe_direct_answer_from_context(query, candidates, candidate_contexts)
        if direct is not None:
            logger.debug(f"Policy direct-answer heuristic hit: {direct['text']}")
            return direct

        prompt_parts: List[str] = [AGENT_PROMPT]

        prompt_parts.append(f"QUESTION: {query}")
        if memory.steps:
            prompt_parts.append(
                "SEEN PAGES:\n"
                + "\n".join(
                    [f"- turn {s.turn}: {[ (d, p) for d, p, _ in s.selected_pages ]}" for s in memory.steps]
                )
            )
        prompt_parts.append("CANDIDATE PAGES:")
        for i, (doc_id, page_idx, score) in enumerate(candidates):
            prompt_parts.append(f"- [{i}] doc={doc_id} page={page_idx} score={score:.3f}")
            if candidate_contexts and i < len(candidate_contexts):
                context = candidate_contexts[i]
                if context:
                    prompt_parts.append(f"  summary: {context}")

        full_prompt = "\n".join(prompt_parts)

        logger.debug(full_prompt)

        raw = llm_call(full_prompt)
        logger.debug(f"LLM raw reply: {raw}")

        action = {"type": "continue", "text": query, "chosen": list(candidates)}

        if raw is None:
            return action

        text = raw.strip()
        lower = text.lower()
        if lower.startswith("answer:"):
            action = {"type": "answer", "text": text[len("answer:") :].strip(), "chosen": list(candidates)}
        elif lower.startswith("continue query:"):
            refined = text[len("continue query:") :].strip()
            action = {"type": "continue", "text": refined or query, "chosen": list(candidates)}
        elif lower.startswith("unanswerable"):
            reason = text.split(":", 1)[-1].strip() if ":" in text else ""
            action = {"type": "unanswerable", "text": reason, "chosen": list(candidates)}

        return action
