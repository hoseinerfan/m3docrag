from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from typing import List, Sequence

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
    2) Otherwise, pick the smallest subset of candidate pages that likely advance the answer and say CONTINUE with a refined query if needed.
    3) If stuck or evidence is insufficient, respond with UNANSWERABLE and STOP.

    Format your reply as one of:
    - ANSWER: <text>
    - CONTINUE QUERY: <refined query>
    - UNANSWERABLE: <reason>
    """
)


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
    ) -> dict:
        """Decide next action using the provided llm_call(query: str) -> str.

        Returns a dict with keys: {type: 'answer'|'continue'|'unanswerable', 'text': str, 'chosen': list[PageRef]}
        """

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

