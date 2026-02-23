from __future__ import annotations

from typing import Callable, Optional, Sequence

from loguru import logger

from m3docrag.rag import MultimodalRAGModel

from .memory import AgentMemory, PageRef
from .policy import AgentPolicy


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
            memory.add_step(turn, current_query, [], answer=None, stop_reason="no-new-candidates")
            return {"answer": None, "reason": "no-new-candidates", "steps": memory.steps}

        top_for_turn = candidates[:pages_per_turn]
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

        if action["type"] == "answer":
            memory.add_step(turn, current_query, top_for_turn, answer=action["text"], stop_reason="answered")
            return {"answer": action["text"], "reason": "answered", "steps": memory.steps}

        if action["type"] == "unanswerable":
            memory.add_step(turn, current_query, top_for_turn, answer=None, stop_reason="unanswerable")
            return {"answer": None, "reason": "unanswerable", "steps": memory.steps}

        # continue
        memory.add_step(turn, current_query, top_for_turn, answer=None, stop_reason=None)
        current_query = action["text"] or current_query

    return {"answer": None, "reason": "max_turns", "steps": memory.steps}
