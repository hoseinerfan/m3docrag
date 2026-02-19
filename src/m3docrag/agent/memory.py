from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


PageRef = Tuple[str, int, float]


@dataclass
class StepRecord:
    """A single agent turn.

    Attributes
    ----------
    turn : int
        1-based turn counter.
    query : str
        Query issued to retrieval/VQA for this turn.
    selected_pages : list[PageRef]
        Pages (doc_id, page_idx, score) considered this turn.
    answer : str | None
        Model output for this turn (if produced).
    stop_reason : str | None
        Why the loop stopped on this turn (if it did).
    """

    turn: int
    query: str
    selected_pages: List[PageRef]
    answer: Optional[str] = None
    stop_reason: Optional[str] = None


@dataclass
class AgentMemory:
    """Tracks visited pages and per-turn records."""

    steps: List[StepRecord] = field(default_factory=list)
    seen_pages: set[str] = field(default_factory=set)

    def page_uid(self, doc_id: str, page_idx: int) -> str:
        return f"{doc_id}#p{page_idx}"

    def add_step(
        self,
        turn: int,
        query: str,
        selected_pages: Sequence[PageRef],
        answer: Optional[str] = None,
        stop_reason: Optional[str] = None,
    ) -> None:
        for doc_id, page_idx, _ in selected_pages:
            self.seen_pages.add(self.page_uid(doc_id, page_idx))
        self.steps.append(
            StepRecord(
                turn=turn,
                query=query,
                selected_pages=list(selected_pages),
                answer=answer,
                stop_reason=stop_reason,
            )
        )

    def unseen(self, candidates: Sequence[PageRef]) -> list[PageRef]:
        """Filter out pages already visited."""

        return [c for c in candidates if self.page_uid(c[0], c[1]) not in self.seen_pages]

