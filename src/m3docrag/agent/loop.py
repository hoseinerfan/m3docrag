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


def _norm_query_key(text: str) -> str:
    return _norm_query(text)


def _dedupe_queries(queries: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for query in queries:
        q = " ".join((query or "").split())
        if not q:
            continue
        key = _norm_query_key(q)
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out


def _select_seed_candidates_from_query_lists(
    query_candidate_lists: Sequence[Sequence[PageRef]],
    *,
    pages_per_turn: int,
    allowed_uids: set[str],
) -> list[PageRef]:
    if pages_per_turn <= 0:
        return []
    seeds: list[PageRef] = []
    selected_uids: set[str] = set()
    selected_doc_ids: set[str] = set()
    for cand_list in query_candidate_lists:
        for candidate in cand_list:
            uid = _page_uid(candidate)
            if uid not in allowed_uids or uid in selected_uids:
                continue
            doc_id = candidate[0]
            if doc_id in selected_doc_ids:
                continue
            selected_uids.add(uid)
            selected_doc_ids.add(doc_id)
            seeds.append(candidate)
            break
        if len(seeds) >= pages_per_turn:
            break
    return seeds


def _top_pages_payload(candidates: Sequence[PageRef], *, limit: int = 20) -> list[dict]:
    return [
        {"doc_id": doc_id, "page_idx": int(page_idx), "score": float(score)}
        for doc_id, page_idx, score in list(candidates)[:limit]
    ]


def _top_docs_payload(candidates: Sequence[PageRef], *, limit: int = 20) -> list[dict]:
    out: list[dict] = []
    seen_docs: set[str] = set()
    for doc_id, page_idx, score in candidates:
        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        out.append(
            {
                "doc_id": doc_id,
                "best_page_idx": int(page_idx),
                "best_page_score": float(score),
            }
        )
        if len(out) >= limit:
            break
    return out


def _slice_rank_window(
    ranked_ids: Sequence[str],
    *,
    start: int,
    size: int,
    docid2embs: dict,
) -> list[str]:
    if size <= 0:
        return []
    if start < 0:
        start = 0
    if start >= len(ranked_ids):
        return []
    end = min(start + size, len(ranked_ids))
    return [doc_id for doc_id in ranked_ids[start:end] if doc_id in docid2embs]


def _retrieve_rank_window_candidates(
    *,
    rag_model: MultimodalRAGModel,
    query: str,
    explore_doc_ids: Sequence[str],
    docid2embs: dict,
    token2pageuid: Optional[Sequence[str]],
    all_token_embeddings,
    n_return_pages: int,
) -> list[PageRef]:
    if not explore_doc_ids:
        return []
    explore_docid2embs = {doc_id: docid2embs[doc_id] for doc_id in explore_doc_ids}
    return rag_model.retrieve_pages_from_docs(
        query=query,
        docid2embs=explore_docid2embs,
        index=None,
        token2pageuid=token2pageuid,
        all_token_embeddings=all_token_embeddings,
        n_return_pages=max(n_return_pages, len(explore_doc_ids) * 2),
        show_progress=False,
    )


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
    pending_hop_queries: list[str] = []
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
        retrieval_queries = [current_query]
        if explore_mode and pending_hop_queries:
            retrieval_queries = _dedupe_queries([current_query] + pending_hop_queries)
            logger.info("multi-query retrieval active: {} queries", len(retrieval_queries))

        query_candidate_lists: list[list[PageRef]] = []
        retrieval_traces: list[dict] = []
        for i, retrieval_query in enumerate(retrieval_queries, start=1):
            retrieved = rag_model.retrieve_pages_from_docs(
                query=retrieval_query,
                docid2embs=docid2embs,
                index=None,
                token2pageuid=token2pageuid,
                all_token_embeddings=all_token_embeddings,
                n_return_pages=n_return_pages_turn,
                show_progress=False,
            )
            query_candidate_lists.append(retrieved)
            logger.info(
                "retrieval query [{} / {}]: {} candidates | {}",
                i,
                len(retrieval_queries),
                len(retrieved),
                retrieval_query,
            )
            top_docs = _top_docs_payload(retrieved, limit=min(20, max(pages_per_turn * 4, 8)))
            retrieval_traces.append(
                {
                    "query": retrieval_query,
                    "returned_page_count": len(retrieved),
                    "top_pages": _top_pages_payload(
                        retrieved, limit=min(20, max(pages_per_turn * 4, 8))
                    ),
                    "top_docs": top_docs,
                }
            )
            logger.info(
                "retrieval query [{}] top docs: {}",
                i,
                [x["doc_id"] for x in top_docs[: min(5, len(top_docs))]],
            )
        candidate_lists: list[list[PageRef]] = list(query_candidate_lists)
        candidates_main = query_candidate_lists[0] if query_candidate_lists else []

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
            head_explore_doc_ids = _slice_rank_window(
                doc_ranked_ids,
                start=window_start,
                size=doc_explore_window,
                docid2embs=docid2embs,
            )
            if head_explore_doc_ids:
                explore_candidates = _retrieve_rank_window_candidates(
                    rag_model=rag_model,
                    query=query,
                    explore_doc_ids=head_explore_doc_ids,
                    docid2embs=docid2embs,
                    token2pageuid=token2pageuid,
                    all_token_embeddings=all_token_embeddings,
                    n_return_pages=n_return_pages,
                )
                candidate_lists.insert(0, explore_candidates)
                logger.info(
                    "rank-window exploration active (head): docs[{}:{}] -> {} docs, {} candidate pages",
                    window_start,
                    min(window_start + doc_explore_window, len(doc_ranked_ids)),
                    len(head_explore_doc_ids),
                    len(explore_candidates),
                )

            tail_end = max(0, len(doc_ranked_ids) - doc_explore_window * (turn - 2))
            tail_start = max(0, tail_end - doc_explore_window)
            tail_explore_doc_ids = _slice_rank_window(
                doc_ranked_ids,
                start=tail_start,
                size=tail_end - tail_start,
                docid2embs=docid2embs,
            )
            if tail_explore_doc_ids:
                tail_candidates = _retrieve_rank_window_candidates(
                    rag_model=rag_model,
                    query=query,
                    explore_doc_ids=tail_explore_doc_ids,
                    docid2embs=docid2embs,
                    token2pageuid=token2pageuid,
                    all_token_embeddings=all_token_embeddings,
                    n_return_pages=n_return_pages,
                )
                candidate_lists.insert(0, tail_candidates)
                logger.info(
                    "rank-window exploration active (tail): docs[{}:{}] -> {} docs, {} candidate pages",
                    tail_start,
                    tail_end,
                    len(tail_explore_doc_ids),
                    len(tail_candidates),
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
                retrieval_traces=retrieval_traces,
            )
            return {"answer": None, "reason": "no-new-candidates", "steps": memory.steps}

        top_for_turn: list[PageRef] = []
        if len(retrieval_queries) > 1:
            allowed_uids = {_page_uid(c) for c in candidates}
            seed_candidates = _select_seed_candidates_from_query_lists(
                query_candidate_lists,
                pages_per_turn=pages_per_turn,
                allowed_uids=allowed_uids,
            )
            if seed_candidates:
                seed_uids = {_page_uid(c) for c in seed_candidates}
                remainder = [c for c in candidates if _page_uid(c) not in seed_uids]
                remainder_budget = max(0, pages_per_turn - len(seed_candidates))
                if remainder_budget > 0:
                    remainder_selected = _select_turn_candidates(
                        remainder,
                        pages_per_turn=remainder_budget,
                        memory=memory,
                        exploration_offset=(explore_span * (turn - 1)) if explore_mode and turn > 1 else 0,
                    )
                else:
                    remainder_selected = []
                top_for_turn = seed_candidates + remainder_selected
                logger.info(
                    "multi-query seed selection: {} seeded + {} remainder",
                    len(seed_candidates),
                    len(remainder_selected),
                )

        if not top_for_turn:
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
                retrieval_traces=retrieval_traces,
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
                retrieval_traces=retrieval_traces,
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
            retrieval_traces=retrieval_traces,
        )
        pending_hop_queries = _dedupe_queries(action.get("hop_queries", []))
        current_query = action["text"] or current_query

    return {"answer": None, "reason": "max_turns", "steps": memory.steps}
