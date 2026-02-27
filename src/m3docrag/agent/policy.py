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
    You have: a question, known intermediate facts, previously seen pages, and newly retrieved candidate pages (with brief summaries).

    Tools you can implicitly call:
    - READ: read provided candidate pages (you already have their summaries; use them to decide).
    - STOP: when confident in a short answer or no further progress is likely.

    Policy:
    1) If a candidate page directly contains the answer, respond with ANSWER: <short answer> and STOP.
       If the question asks for a title/name and a candidate summary explicitly contains that title/name, answer immediately.
    2) For multi-hop questions, prefer decomposing into a single-hop subquestion and respond with CONTINUE HOP.
       You may extract and store intermediate facts from evidence using FACT lines.
       If hop-2 depends on hop-1, emit one or more HOP_QUERY lines in dependency order.
       When evidence for the current hop exists, emit HOP_ANSWER with a short value (or UNKNOWN).
       Do not stop after finding only one likely entity if the question still requires a relation/attribute/second clue.
    3) Otherwise, pick the smallest subset of candidate pages that likely advance the answer and say CONTINUE with a refined query if needed.
       Do NOT repeat the same question as CONTINUE QUERY unless you substantially refine it.
    4) If stuck or evidence is insufficient, respond with UNANSWERABLE and STOP.

    Reply format:
    - ANSWER: <text>
    - CONTINUE QUERY: <refined query>
    - CONTINUE HOP: <single-hop subquestion>
    - UNANSWERABLE: <reason>
    Optional additional lines after the first line:
    - FACT: <intermediate fact>
    - HOP_QUERY: <independent subquery for retrieval>
    - HOP_ANSWER: <short answer for current hop, or UNKNOWN>
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
    " ArticleTalk",
    " Read Edit",
    " View history",
    " Genre ",
    " Created by ",
    " Directed by ",
)

_REJECT_TITLE_PREFIXES = (
    "this page was last edited",
    "privacy policy",
    "notes [edit]",
    "references [edit]",
    "internet portal",
    "music portal",
    "production executive producers",
    "outstanding writing for",
    "brown served as",
    "in november",
)

_REJECT_TITLE_SUBSTRINGS = (
    "cookie statement",
    "about wikipedia",
    "disclaimerscontact wikipedia",
)

_TV_SERIES_QUERY_HINTS = (
    "tv series",
    "television series",
)

_TV_SERIES_CONTEXT_HINTS = (
    " sitcom",
    " showrunner",
    " created by ",
    " starring ",
    " running time ",
    " production companies ",
    " narrated by ",
    " comedy series",
    " television series",
)

_NON_TV_CONTEXT_HINTS = (
    " song ",
    " single ",
    " album ",
    " music video",
    " discography",
    " track listing",
    " released ",
    " recorded ",
    " b-side",
    " a-side",
)

_FACT_REJECT_SUBSTRINGS = (
    "privacy policy",
    "about wikipedia",
    "disclaimerscontact wikipedia",
    "cookie statement",
    "last edited on",
    "mobile view",
)

_FACT_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "for",
    "to",
    "and",
    "or",
    "by",
    "with",
    "from",
    "at",
    "as",
    "is",
    "are",
    "was",
    "were",
    "did",
    "does",
    "who",
    "which",
    "what",
    "when",
    "where",
    "how",
}


_HOP_QUERY_STOP_PHRASES = {
    "who",
    "what",
    "which",
    "when",
    "where",
    "why",
    "how",
}


def _normalize_ws(text: str) -> str:
    return " ".join(text.split())


def _strip_wikipedia_suffixes(text: str) -> str:
    # Remove "<N> languages" and any trailing UI tokens if they remain.
    text = re.sub(r"\s+\d+\s+languages?\b.*$", "", text, flags=re.IGNORECASE)
    return text.strip()


def _looks_like_boilerplate(title: str) -> bool:
    lower = title.lower()
    if any(lower.startswith(prefix) for prefix in _REJECT_TITLE_PREFIXES):
        return True
    if any(token in lower for token in _REJECT_TITLE_SUBSTRINGS):
        return True
    return False


def _query_type_context_guard(query: str, context: str) -> bool:
    q = query.lower()
    c = f" {context.lower()} "
    if any(hint in q for hint in _TV_SERIES_QUERY_HINTS):
        if any(hint in c for hint in _NON_TV_CONTEXT_HINTS):
            return False
        return any(hint in c for hint in _TV_SERIES_CONTEXT_HINTS)
    return True


def _extract_title_from_summary(summary: str) -> Optional[str]:
    text = _normalize_ws(summary)
    if not text:
        return None

    cut = len(text)
    lang_match = re.search(r"\b\d+\s+languages?\b", text, flags=re.IGNORECASE)
    if lang_match and lang_match.start() > 0:
        cut = min(cut, lang_match.start())
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

    candidate = _strip_wikipedia_suffixes(candidate)
    if len(candidate) < 2 or len(candidate) > 80:
        return None
    if not re.search(r"[A-Za-z]", candidate):
        return None
    if _looks_like_boilerplate(candidate):
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
        if i >= len(candidates):
            break
        _, page_idx, _ = candidates[i]
        # Only auto-answer from the document's first page; other pages are too noisy.
        if page_idx != 0:
            continue
        if not context:
            continue
        if not _query_type_context_guard(query, context):
            continue
        title = _extract_title_from_summary(context)
        if not title:
            continue
        return {"type": "answer", "text": title, "chosen": [candidates[i]]}

    return None


def _extract_fact_lines(lines: Sequence[str]) -> list[str]:
    facts: list[str] = []
    for line in lines:
        if not isinstance(line, str):
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("fact:"):
            fact = stripped.split(":", 1)[1].strip()
            if fact:
                facts.append(fact)
    return facts


def _extract_hop_query_lines(lines: Sequence[str]) -> list[str]:
    queries: list[str] = []
    for line in lines:
        if not isinstance(line, str):
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("hop_query:"):
            q = stripped.split(":", 1)[1].strip()
            if q:
                queries.append(q)
    return queries


def _extract_hop_answer(lines: Sequence[str]) -> Optional[str]:
    for line in lines:
        if not isinstance(line, str):
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("hop_answer:"):
            answer = stripped.split(":", 1)[1].strip()
            if answer:
                lower = answer.lower()
                if lower in {"unknown", "n/a", "none", "not found"}:
                    return None
                return answer
    return None


def _dedupe_queries(queries: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for query in queries:
        q = _normalize_ws(query)
        if not q:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out


def _fallback_fact_from_candidate_contexts(
    query: str,
    candidate_contexts: Optional[Sequence[Optional[str]]],
) -> Optional[str]:
    if not candidate_contexts:
        return None
    query_tokens = {
        t
        for t in re.findall(r"[a-z0-9]+", (query or "").lower())
        if len(t) >= 3 and t not in _FACT_STOPWORDS
    }

    best_text: Optional[str] = None
    best_score = -1

    for ctx in candidate_contexts:
        if not ctx:
            continue
        text = " ".join(ctx.split()).strip()
        if len(text) < 24:
            continue
        lower = text.lower()
        if any(bad in lower for bad in _FACT_REJECT_SUBSTRINGS):
            continue

        if query_tokens:
            overlap = len(query_tokens & set(re.findall(r"[a-z0-9]+", lower)))
        else:
            overlap = 0

        # Avoid injecting noisy facts that do not anchor to the query at all.
        if query_tokens and overlap == 0:
            continue

        if overlap < best_score:
            continue

        # Keep a short snippet; enough to seed next-hop query without bloating the prompt.
        text = text[:220]
        if "." in text:
            text = text.split(".", 1)[0].strip()
        if text:
            best_text = text
            best_score = overlap

    if not best_text:
        return None
    return f"Candidate evidence snippet: {best_text}"


def _looks_multi_hop_like_query(query: str) -> bool:
    q = " ".join((query or "").lower().split())
    if not q:
        return False
    patterns = (
        r"\bwho\b.*\bthat\b",
        r"\bwhich\b.*\bthat\b",
        r"\bwhose\b",
        r"\bwho\b.*\bwith\b",
        r"\bwhich\b.*\bwith\b",
        r"\bin\b.+\bwho\b",
        r"\bin\b.+\bwhich\b",
        r"\baccording to\b",
        r"\bbased on\b",
        r"\bfrom\b.+\bwhich\b",
        r"\bwhich\b.+\bdid\b",
        r"\bwho\b.+\bdid\b",
        r"\bwork(?:ed)? as\b",
        r"\bvoice actor\b",
        r"\bvoice actress\b",
        r"\bdubbed\b",
        r"\bversion of\b",
    )
    return any(re.search(p, q) for p in patterns)


def _has_nontrivial_candidate_evidence(candidate_contexts: Optional[Sequence[Optional[str]]]) -> bool:
    if not candidate_contexts:
        return False
    for c in candidate_contexts:
        if not c:
            continue
        if len(c.strip()) >= 24:
            return True
    return False


def _suggest_hop_query(query: str, facts: Sequence[str]) -> str:
    q = " ".join((query or "").split())
    q_lower = q.lower()
    years = re.findall(r"\b(?:19|20)\d{2}\b", q)
    year_text = years[0] if years else ""
    anchor_movie_match = re.search(r"\bwhich\s+(.+?)\s+movie\b", q_lower)
    anchor_movie = anchor_movie_match.group(1).strip() if anchor_movie_match else ""
    if anchor_movie:
        anchor_movie = re.sub(r"[^a-z0-9 ]+", " ", anchor_movie).strip()

    # Prefer a 2+ token title-cased person/entity anchor from the original question.
    entities = []
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", q):
        ent = m.group(1).strip()
        if ent.lower() in _HOP_QUERY_STOP_PHRASES:
            continue
        entities.append(ent)
    person_anchor = entities[0] if entities else ""

    relation_parts = []
    if "voice actress" in q_lower:
        relation_parts.append("voice actress")
    elif "voice actor" in q_lower:
        relation_parts.append("voice actor")
    if "dubbed" in q_lower:
        relation_parts.append("dubbed")
    if "tamil" in q_lower:
        relation_parts.append("tamil")
    if "film" in q_lower:
        relation_parts.append("film")
    elif "movie" in q_lower:
        relation_parts.append("movie")
    relation_text = " ".join(relation_parts).strip()

    if not facts:
        parts = [person_anchor, relation_text, year_text]
        hop = " ".join([p for p in parts if p]).strip()
        if hop:
            return hop

    if facts and person_anchor:
        parts = [anchor_movie, "movie", person_anchor, relation_text, year_text]
        hop = " ".join([p for p in parts if p]).strip()
        if hop:
            return hop

    if facts:
        return (
            "Find a different supporting document/page for the missing clue needed to answer: "
            f"{q}"
        )
    return f"Identify one intermediate clue (entity/title/attribute) needed before answering: {q}"


def _suggest_hop_queries(query: str, facts: Sequence[str]) -> list[str]:
    q = " ".join((query or "").split())
    q_lower = q.lower()
    years = re.findall(r"\b(?:19|20)\d{2}\b", q)
    year_text = years[0] if years else ""
    entities = []
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", q):
        ent = m.group(1).strip()
        if ent.lower() in _HOP_QUERY_STOP_PHRASES:
            continue
        entities.append(ent)
    person_anchor = entities[0] if entities else ""
    relation_parts = []
    if "voice actress" in q_lower:
        relation_parts.append("voice actress")
    elif "voice actor" in q_lower:
        relation_parts.append("voice actor")
    if "dubbed" in q_lower:
        relation_parts.append("dubbed")
    if "tamil" in q_lower:
        relation_parts.append("tamil")
    if "film" in q_lower:
        relation_parts.append("film")
    elif "movie" in q_lower:
        relation_parts.append("movie")
    relation_text = " ".join(relation_parts).strip()

    queries: list[str] = []
    if person_anchor:
        parts = [person_anchor, relation_text, year_text]
        first_hop = " ".join([p for p in parts if p]).strip()
        if first_hop:
            queries.append(first_hop)
    if "which " in q_lower and (" movie" in q_lower or " film" in q_lower):
        focus = re.sub(r"^which\s+", "", q, flags=re.IGNORECASE)
        if focus:
            queries.append(focus)
    if person_anchor and relation_text:
        queries.append(f"{person_anchor} {relation_text}")
    queries.append(_suggest_hop_query(query, facts))
    return _dedupe_queries(queries)[:3]


def _is_near_duplicate_query(a: str, b: str) -> bool:
    ta = [t for t in re.findall(r"[a-z0-9]+", (a or "").lower()) if t not in _FACT_STOPWORDS]
    tb = [t for t in re.findall(r"[a-z0-9]+", (b or "").lower()) if t not in _FACT_STOPWORDS]
    if not ta or not tb:
        return _normalize_ws(a).lower() == _normalize_ws(b).lower()
    sa = set(ta)
    sb = set(tb)
    overlap = len(sa & sb) / max(len(sa | sb), 1)
    return overlap >= 0.8


def _is_generic_hop_query(text: str) -> bool:
    t = _normalize_ws(text).lower()
    if not t:
        return False
    generic_starts = (
        "who is ",
        "what is ",
        "what was ",
        "which is ",
        "which was ",
    )
    if t.startswith(generic_starts):
        return True
    generic_phrases = (
        "and what is",
        "and who is",
        "what is her",
        "what is his",
    )
    return any(p in t for p in generic_phrases)


def _should_defer_unanswerable(
    *,
    query: str,
    memory: AgentMemory,
    candidate_contexts: Optional[Sequence[Optional[str]]],
) -> bool:
    # Be conservative: only override early UNANSWERABLE for likely multi-hop questions when we still have evidence to inspect.
    prior_continue_hop = bool(memory.steps and memory.steps[-1].action_type == "continue_hop")
    has_fact_memory = bool(getattr(memory, "facts", None))
    if not (_looks_multi_hop_like_query(query) or prior_continue_hop or has_fact_memory):
        return False
    # Allow a few extra turns once hop behavior has started.
    max_steps_before_stop = 4 if (prior_continue_hop or has_fact_memory) else 2
    if len(memory.steps) >= max_steps_before_stop:
        return False
    if not _has_nontrivial_candidate_evidence(candidate_contexts):
        return False
    return True


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
        if getattr(memory, "facts", None):
            prompt_parts.append("KNOWN FACTS:\n" + "\n".join([f"- {fact}" for fact in memory.facts]))
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

        action = {
            "type": "continue",
            "text": query,
            "chosen": list(candidates),
            "facts": [],
            "hop_queries": [],
            "hop_answer": None,
        }

        if raw is None:
            return action

        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if not lines:
            return action

        facts = _extract_fact_lines(lines[1:])
        hop_queries = _extract_hop_query_lines(lines[1:])
        hop_answer = _extract_hop_answer(lines[1:])
        text = lines[0]
        lower = text.lower()
        if lower.startswith("answer:"):
            action = {
                "type": "answer",
                "text": text[len("answer:") :].strip(),
                "chosen": list(candidates),
                "facts": facts,
                "hop_queries": [],
                "hop_answer": None,
            }
        elif lower.startswith("continue query:"):
            refined = text[len("continue query:") :].strip()
            action = {
                "type": "continue",
                "text": refined or query,
                "chosen": list(candidates),
                "facts": facts,
                "hop_queries": _dedupe_queries(hop_queries),
                "hop_answer": hop_answer,
            }
        elif lower.startswith("continue hop:"):
            refined = text[len("continue hop:") :].strip()
            action = {
                "type": "continue_hop",
                "text": refined or query,
                "chosen": list(candidates),
                "facts": facts,
                "hop_queries": _dedupe_queries(hop_queries),
                "hop_answer": hop_answer,
            }
        elif lower.startswith("unanswerable"):
            reason = text.split(":", 1)[-1].strip() if ":" in text else ""
            action = {
                "type": "unanswerable",
                "text": reason,
                "chosen": list(candidates),
                "facts": facts,
                "hop_queries": [],
                "hop_answer": None,
            }

        prior_continue_hop = bool(memory.steps and memory.steps[-1].action_type == "continue_hop")
        likely_multihop = _looks_multi_hop_like_query(query) or prior_continue_hop or bool(getattr(memory, "facts", None))

        # If the model doesn't emit FACT lines, synthesize one lightweight snippet for hop-style continuation.
        if action["type"] in ("continue", "continue_hop") and not action.get("facts") and likely_multihop:
            fallback_fact = _fallback_fact_from_candidate_contexts(query, candidate_contexts)
            if fallback_fact:
                logger.debug("Policy synthesized fallback FACT from candidate context")
                action = {**action, "facts": [fallback_fact]}

        # Multi-hop guard: avoid early stopping before collecting enough intermediate evidence.
        if action["type"] == "unanswerable" and _should_defer_unanswerable(
            query=query,
            memory=memory,
            candidate_contexts=candidate_contexts,
        ):
            hop_query = _suggest_hop_query(query, list(getattr(memory, "facts", [])) + list(action.get("facts", [])))
            logger.debug(f"Policy UNANSWERABLE overridden to CONTINUE HOP for likely multi-hop query: {hop_query}")
            action = {
                "type": "continue_hop",
                "text": hop_query,
                "chosen": list(candidates),
                "facts": action.get("facts", []),
                "hop_queries": _suggest_hop_queries(
                    query, list(getattr(memory, "facts", [])) + list(action.get("facts", []))
                ),
                "hop_answer": action.get("hop_answer"),
            }

        # If the LLM asks to continue but simply repeats a multi-hop question, coerce to an explicit hop query.
        if action["type"] in ("continue", "continue_hop"):
            action_text = action.get("text", "")
            same_query = _normalize_ws(action_text) == _normalize_ws(query)
            near_duplicate_query = _is_near_duplicate_query(action_text, query)
            generic_hop_query = _is_generic_hop_query(action_text)
            if likely_multihop and (same_query or near_duplicate_query or generic_hop_query):
                hop_query = _suggest_hop_query(
                    query,
                    list(getattr(memory, "facts", [])) + list(action.get("facts", [])),
                )
                logger.debug(f"Policy repeated continue normalized to CONTINUE HOP for likely multi-hop query: {hop_query}")
                action = {
                    "type": "continue_hop",
                    "text": hop_query,
                    "chosen": list(candidates),
                    "facts": action.get("facts", []),
                    "hop_queries": _suggest_hop_queries(
                        query, list(getattr(memory, "facts", [])) + list(action.get("facts", []))
                    ),
                    "hop_answer": action.get("hop_answer"),
                }

        if action["type"] == "continue_hop":
            suggested_hops = _suggest_hop_queries(
                query, list(getattr(memory, "facts", [])) + list(action.get("facts", []))
            )
            hop_queries = _dedupe_queries(list(action.get("hop_queries", [])) + suggested_hops)
            if action.get("text"):
                hop_queries = _dedupe_queries([action["text"]] + hop_queries)
            action = {**action, "hop_queries": hop_queries[:3]}

        return action
