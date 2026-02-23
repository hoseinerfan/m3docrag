"""Score outputs from `run_agent_m3docvqa_subset.py`.

This script summarizes:
- reason counts
- answered / abstain behavior
- exact-match style metrics using `pred_answer_exact_in_gold` when available
- light normalization fallback for OCR-spaced answers
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional


def _norm(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = text.casefold()
    text = " ".join(text.split())
    # Normalize common OCR spacing artifacts and punctuation.
    text = re.sub(r"[\"'`“”‘’]", "", text)
    text = re.sub(r"[\-–—/:;,.|(){}\[\]]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _read_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _exact_in_gold(pred: Optional[str], gold_answers: list[str]) -> Optional[bool]:
    if pred is None:
        return None
    pred_n = _norm(pred)
    if not pred_n:
        return None
    gold_norm = {_norm(g) for g in gold_answers if _norm(g)}
    if not gold_norm:
        return None
    return pred_n in gold_norm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-jsonl", type=Path, required=True, help="Output JSONL from run_agent_m3docvqa_subset.py")
    p.add_argument("--show-errors-only", action="store_true", help="Print only wrong answered / missed-answer examples.")
    p.add_argument("--show-details", action="store_true", help="Print per-example rows.")
    p.add_argument("--max-detail-rows", type=int, default=50)
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    rows = list(_read_jsonl(args.results_jsonl))
    if not rows:
        raise ValueError(f"No rows found in {args.results_jsonl}")

    reason_counts = {}
    total = 0
    error_count = 0
    answered_count = 0
    abstain_count = 0
    exact_match_overall = 0
    exact_match_answered = 0
    answered_with_gold = 0
    examples_with_gold = 0
    missed_with_gold = 0

    details = []

    for row in rows:
        total += 1
        reason = row.get("reason", "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

        if reason == "error":
            error_count += 1

        pred = row.get("pred_answer")
        gold_answers = row.get("gold_answers") or []
        if not isinstance(gold_answers, list):
            gold_answers = [str(gold_answers)]

        has_gold = len(gold_answers) > 0
        if has_gold:
            examples_with_gold += 1

        pred_answered = reason == "answered" and pred is not None
        if pred_answered:
            answered_count += 1
        else:
            abstain_count += 1
            if has_gold and reason != "error":
                missed_with_gold += 1

        exact = row.get("pred_answer_exact_in_gold")
        if exact is None and pred_answered:
            exact = _exact_in_gold(pred, gold_answers)

        status = None
        if pred_answered:
            if exact is True:
                exact_match_answered += 1
                exact_match_overall += 1
                status = "correct_answered"
            else:
                status = "wrong_answer"
            if has_gold:
                answered_with_gold += 1
        else:
            if has_gold and reason != "error":
                status = "missed_answer"
            elif reason != "error":
                status = "abstained_no_gold"
            else:
                status = "error"

        details.append(
            {
                "qid": row.get("qid"),
                "reason": reason,
                "status": status,
                "pred_answer": pred,
                "gold_answers": gold_answers,
                "pred_answer_exact_in_gold": exact,
                "supporting_doc_ids": row.get("supporting_doc_ids", []),
            }
        )

    metrics = {
        "total_examples": total,
        "error_count": error_count,
        "reason_counts": reason_counts,
        "answered_count": answered_count,
        "abstain_count": abstain_count,
        "examples_with_gold": examples_with_gold,
        "answered_with_gold": answered_with_gold,
        "missed_with_gold": missed_with_gold,
        "exact_match_overall_count": exact_match_overall,
        "exact_match_answered_count": exact_match_answered,
        "exact_match_overall_rate": (exact_match_overall / total) if total else None,
        "exact_match_on_answered_rate": (exact_match_answered / answered_count) if answered_count else None,
        "answer_coverage_rate": (answered_count / total) if total else None,
    }

    print("Metrics")
    print(f"- total_examples: {metrics['total_examples']}")
    print(f"- error_count: {metrics['error_count']}")
    print(f"- reason_counts: {metrics['reason_counts']}")
    print(f"- answered_count: {metrics['answered_count']}")
    print(f"- abstain_count: {metrics['abstain_count']}")
    print(f"- examples_with_gold: {metrics['examples_with_gold']}")
    print(f"- answered_with_gold: {metrics['answered_with_gold']}")
    print(f"- missed_with_gold: {metrics['missed_with_gold']}")
    print(f"- exact_match_overall_count: {metrics['exact_match_overall_count']}")
    print(f"- exact_match_answered_count: {metrics['exact_match_answered_count']}")
    print(f"- exact_match_overall_rate: {metrics['exact_match_overall_rate']}")
    print(f"- exact_match_on_answered_rate: {metrics['exact_match_on_answered_rate']}")
    print(f"- answer_coverage_rate: {metrics['answer_coverage_rate']}")

    if args.show_details or args.show_errors_only:
        print("\nDetails")
        shown = 0
        for d in details:
            if args.show_errors_only and d["status"] not in ("wrong_answer", "missed_answer", "error"):
                continue
            print(
                f"- {d['qid']} | {d['status']} | reason={d['reason']} | "
                f"pred={d['pred_answer']!r} | gold={d['gold_answers'][:2]}"
            )
            shown += 1
            if shown >= args.max_detail_rows:
                break

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({"metrics": metrics, "details": details}, indent=2) + "\n")


if __name__ == "__main__":
    main()
