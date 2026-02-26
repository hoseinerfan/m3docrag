from __future__ import annotations

from typing import Callable, Optional, Sequence

from loguru import logger

from m3docrag.rag import MultimodalRAGModel

from .memory import AgentMemory, PageRef
from .policy import AgentPolicy


def _seen_doc_ids(memory: AgentMemory) -> set[str]:
    seen: set[str] = set()
    for step in memory.steps:
        for doc_id, _, _ in step.selected_pages:
            seen.add(doc_id)
    return seen


def _should_diversify_docs(memory: AgentMemory) -> bool:
    if memory.facts:
        return True
    if memory.steps and memory.steps[-1].action_type == "continue_hop":
        return True
    return False


def _select_turn_candidates(
    candidates: Sequence[PageRef],
    *,
    pages_per_turn: int,
    memory: AgentMemory,
) -> list[PageRef]:
    if pages_per_turn <= 0:
        return []
    if not candidates:
        return []

    if not _should_diversify_docs(memory):
        return list(candidates[:pages_per_turn])

    seen_docs_prev_turns = _seen_doc_ids(memory)
    selected: list[PageRef] = []
    selected_doc_ids: set[str] = set()
    selected_page_uids: set[str] = set()

    def _append_candidate(c: PageRef) -> bool:
        doc_id, page_idx, _ = c
        uid = f"{doc_id}#p{page_idx}"
        if uid in selected_page_uids:
            return False
        selected.append(c)
        selected_doc_ids.add(doc_id)
        selected_page_uids.add(uid)
        return len(selected) >= pages_per_turn

    # Pass 1: prioritize docs never selected in prior turns.
    for c in candidates:
        doc_id = c[0]
        if doc_id in seen_docs_prev_turns or doc_id in selected_doc_ids:
            continue
        if _append_candidate(c):
            return selected

    # Pass 2: preserve doc diversity within the turn, even if docs were seen before.
    for c in candidates:
        doc_id = c[0]
        if doc_id in selected_doc_ids:
            continue
        if _append_candidate(c):
            return selected

    # Pass 3: fill remaining slots by score order.
    for c in candidates:
        if _append_candidate(c):
            return selected

    return selected


def run_agent_session(
    *,
    query: str,
    rag_model: MultimodalRAGModel,
    docid2embs: dict,
    token2pageuid: Optional[Sequence[str]] = None,
    all_token_embeddings=None,
    max_turns: int = 4,
    pages_per_turn: int = 3,
    llm_call: Callable[[str], str],
    n_return_pages: int = 6,
    stop_if_seen: bool = True,
    candidate_context_fn: Optional[Callable[[str, int], Optional[str]]] = None,
):
    """Iterative agent loop over the existing RAG model.

    The agent reuses the underlying retrieval & VQA models. It only decides which pages
    to consider and when to stop.
    """

    memory = AgentMemory()
    policy = AgentPolicy()

    current_query = query
    for turn in range(1, max_turns + 1):
        logger.info(f"[turn {turn}] query: {current_query}")

        # 1) retrieve candidates
        candidates: list[PageRef] = rag_model.retrieve_pages_from_docs(
            query=current_query,
            docid2embs=docid2embs,
            index=None,
            token2pageuid=token2pageuid,
            all_token_embeddings=all_token_embeddings,
            n_return_pages=n_return_pages,
            show_progress=False,
        )

        if stop_if_seen:
            candidates = memory.unseen(candidates)
            logger.info(f"filtered to unseen candidates: {len(candidates)}")

        if not candidates:
            memory.add_step(
                turn,
                current_query,
                [],
                answer=None,
                stop_reason="no-new-candidates",
                action_type="no-new-candidates",
            )
            return {"answer": None, "reason": "no-new-candidates", "steps": memory.steps}

        top_for_turn = _select_turn_candidates(
            candidates,
            pages_per_turn=pages_per_turn,
            memory=memory,
        )
        if _should_diversify_docs(memory):
            logger.info(
                "doc-diversity selection active: selected {} pages from {} docs",
                len(top_for_turn),
                len({doc_id for doc_id, _, _ in top_for_turn}),
            )
        candidate_contexts = None
        if candidate_context_fn is not None:
            candidate_contexts = []
            for doc_id, page_idx, _ in top_for_turn:
                try:
                    candidate_contexts.append(candidate_context_fn(doc_id, page_idx))
                except Exception as exc:
                    logger.warning(
                        f"candidate_context_fn failed for doc={doc_id} page={page_idx}: {exc}"
                    )
                    candidate_contexts.append(None)

        # 2) decide action
        action = policy.select_action(
            query=current_query,
            candidates=top_for_turn,
            memory=memory,
            llm_call=llm_call,
            candidate_contexts=candidate_contexts,
        )
        added_facts = memory.add_facts(action.get("facts", []))

        if action["type"] == "answer":
            memory.add_step(
                turn,
                current_query,
                top_for_turn,
                answer=action["text"],
                stop_reason="answered",
                action_type=action["type"],
                facts_added=added_facts,
            )
            return {"answer": action["text"], "reason": "answered", "steps": memory.steps}

        if action["type"] == "unanswerable":
            memory.add_step(
                turn,
                current_query,
                top_for_turn,
                answer=None,
                stop_reason="unanswerable",
                action_type=action["type"],
                facts_added=added_facts,
            )
            return {"answer": None, "reason": "unanswerable", "steps": memory.steps}

        # continue
        memory.add_step(
            turn,
            current_query,
            top_for_turn,
            answer=None,
            stop_reason=None,
            action_type=action["type"],
            facts_added=added_facts,
        )
        current_query = action["text"] or current_query

    return {"answer": None, "reason": "max_turns", "steps": memory.steps}
