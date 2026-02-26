from __future__ import annotations

from typing import Callable, Optional, Sequence

from loguru import logger

from m3docrag.rag import MultimodalRAGModel

from .memory import AgentMemory, PageRef
from .policy import AgentPolicy


def _page_uid(page: PageRef) -> str:
    doc_id, page_idx, _ = page
    return f"{doc_id}#p{page_idx}"


def _merge_candidate_lists(*candidate_lists: Sequence[PageRef]) -> list[PageRef]:
    merged: list[PageRef] = []
    seen_uids: set[str] = set()
    for candidate_list in candidate_lists:
        for page in candidate_list:
            uid = _page_uid(page)
            if uid in seen_uids:
                continue
            seen_uids.add(uid)
            merged.append(page)
    return merged


def _norm_query(text: str) -> str:
    return " ".join((text or "").lower().split())


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
    exploration_offset: int = 0,
) -> list[PageRef]:
    if pages_per_turn <= 0:
        return []
    if not candidates:
        return []

    if not _should_diversify_docs(memory):
        return list(candidates[:pages_per_turn])

    ordered_candidates = list(candidates)
    if ordered_candidates and exploration_offset > 0:
        # Rotate candidates so later turns can inspect deeper-ranked documents.
        offset = exploration_offset % len(ordered_candidates)
        if offset > 0:
            ordered_candidates = ordered_candidates[offset:] + ordered_candidates[:offset]

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
    for c in ordered_candidates:
        doc_id = c[0]
        if doc_id in seen_docs_prev_turns or doc_id in selected_doc_ids:
            continue
        if _append_candidate(c):
            return selected

    # Pass 2: preserve doc diversity within the turn, even if docs were seen before.
    for c in ordered_candidates:
        doc_id = c[0]
        if doc_id in selected_doc_ids:
            continue
        if _append_candidate(c):
            return selected

    # Pass 3: fill remaining slots by score order.
    for c in ordered_candidates:
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
    explore_return_pages_multiplier: int = 10,
    doc_ranked_ids: Optional[Sequence[str]] = None,
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

        explore_mode = _should_diversify_docs(memory)
        explore_span = pages_per_turn * max(explore_return_pages_multiplier, 2)
        n_return_pages_turn = n_return_pages
        if explore_mode and turn > 1:
            n_return_pages_turn = max(
                n_return_pages,
                explore_span * turn,
            )
            logger.info(
                "exploration retrieval depth active: n_return_pages {} -> {}",
                n_return_pages,
                n_return_pages_turn,
            )

        # 1) retrieve candidates
        candidates_main: list[PageRef] = rag_model.retrieve_pages_from_docs(
            query=current_query,
            docid2embs=docid2embs,
            index=None,
            token2pageuid=token2pageuid,
            all_token_embeddings=all_token_embeddings,
            n_return_pages=n_return_pages_turn,
            show_progress=False,
        )
        candidate_lists: list[list[PageRef]] = [candidates_main]

        if explore_mode and turn > 1 and _norm_query(current_query) != _norm_query(query):
            candidates_seed = rag_model.retrieve_pages_from_docs(
                query=query,
                docid2embs=docid2embs,
                index=None,
                token2pageuid=token2pageuid,
                all_token_embeddings=all_token_embeddings,
                n_return_pages=n_return_pages_turn,
                show_progress=False,
            )
            candidate_lists.insert(0, candidates_seed)
            logger.info(
                "exploration seed-query retrieval active: merged {} seed candidates with {} current-query candidates",
                len(candidates_seed),
                len(candidates_main),
            )

        if explore_mode and turn > 1 and doc_ranked_ids:
            doc_explore_window = max(explore_span, pages_per_turn * 8)
            window_start = min(doc_explore_window * (turn - 1), len(doc_ranked_ids))
            window_end = min(window_start + doc_explore_window, len(doc_ranked_ids))
            if window_end > window_start:
                explore_doc_ids = [
                    doc_id
                    for doc_id in doc_ranked_ids[window_start:window_end]
                    if doc_id in docid2embs
                ]
            else:
                explore_doc_ids = []
            if explore_doc_ids:
                explore_docid2embs = {doc_id: docid2embs[doc_id] for doc_id in explore_doc_ids}
                explore_candidates = rag_model.retrieve_pages_from_docs(
                    query=query,
                    docid2embs=explore_docid2embs,
                    index=None,
                    token2pageuid=token2pageuid,
                    all_token_embeddings=all_token_embeddings,
                    n_return_pages=max(n_return_pages, len(explore_doc_ids)),
                    show_progress=False,
                )
                candidate_lists.insert(0, explore_candidates)
                logger.info(
                    "rank-window exploration active: docs[{}:{}] -> {} docs, {} candidate pages",
                    window_start,
                    window_end,
                    len(explore_doc_ids),
                    len(explore_candidates),
                )

        candidates = _merge_candidate_lists(*candidate_lists)
        logger.info("merged candidate count before unseen filtering: {}", len(candidates))

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
            exploration_offset=(explore_span * (turn - 1)) if explore_mode and turn > 1 else 0,
        )
        if explore_mode:
            logger.info(
                "doc-diversity selection active: selected {} pages from {} docs (offset={})",
                len(top_for_turn),
                len({doc_id for doc_id, _, _ in top_for_turn}),
                (explore_span * (turn - 1)) if turn > 1 else 0,
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
