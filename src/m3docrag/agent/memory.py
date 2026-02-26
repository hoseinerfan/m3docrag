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
    action_type : str | None
        Policy action selected for this turn (answer/continue/continue_hop/unanswerable).
    facts_added : list[str]
        New intermediate facts stored from this turn.
    """

    turn: int
    query: str
    selected_pages: List[PageRef]
    answer: Optional[str] = None
    stop_reason: Optional[str] = None
    action_type: Optional[str] = None
    facts_added: List[str] = field(default_factory=list)


@dataclass
class AgentMemory:
    """Tracks visited pages and per-turn records."""

    steps: List[StepRecord] = field(default_factory=list)
    seen_pages: set[str] = field(default_factory=set)
    facts: List[str] = field(default_factory=list)
    _fact_set: set[str] = field(default_factory=set, repr=False)

    def page_uid(self, doc_id: str, page_idx: int) -> str:
        return f"{doc_id}#p{page_idx}"

    def add_step(
        self,
        turn: int,
        query: str,
        selected_pages: Sequence[PageRef],
        answer: Optional[str] = None,
        stop_reason: Optional[str] = None,
        action_type: Optional[str] = None,
        facts_added: Sequence[str] = (),
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
                action_type=action_type,
                facts_added=list(facts_added),
            )
        )

    def add_facts(self, facts: Sequence[str]) -> list[str]:
        """Store new intermediate facts, preserving insertion order and de-duplicating."""

        added: list[str] = []
        for fact in facts:
            fact = " ".join(str(fact).split()).strip()
            if not fact or fact in self._fact_set:
                continue
            self._fact_set.add(fact)
            self.facts.append(fact)
            added.append(fact)
        return added

    def unseen(self, candidates: Sequence[PageRef]) -> list[PageRef]:
        """Filter out pages already visited."""

        return [c for c in candidates if self.page_uid(c[0], c[1]) not in self.seen_pages]
