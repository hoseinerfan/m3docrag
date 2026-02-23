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


def _norm_compact(text: Optional[str]) -> Optional[str]:
    text = _norm(text)
    if text is None:
        return None
    return text.replace(" ", "")


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


def _exact_in_gold_ocr_relaxed(pred: Optional[str], gold_answers: list[str]) -> Optional[bool]:
    if pred is None:
        return None
    pred_n = _norm_compact(pred)
    if not pred_n:
        return None
    gold_norm = {_norm_compact(g) for g in gold_answers if _norm_compact(g)}
    if not gold_norm:
        return None
    return pred_n in gold_norm


def _classify_question_type(question: Optional[str]) -> str:
    q = (question or "").strip().lower()
    if not q:
        return "unknown"

    if "what color" in q or "what colour" in q or "color of" in q or "colour of" in q:
        return "color"
    if q.startswith("how many") or q.startswith("how much") or "number of" in q:
        return "count"
    if "what year" in q or "which year" in q or "in what year" in q:
        return "year"
    if q.startswith("when ") or "what date" in q or "what time" in q:
        return "date_time"
    if q.startswith("who ") or "which person" in q or "whose " in q:
        return "person"
    if q.startswith("where ") or "which country" in q or "which city" in q or "which state" in q:
        return "location"
    if "title" in q or "name" in q:
        return "title_name"
    if q.startswith("is ") or q.startswith("does ") or q.startswith("did ") or q.startswith("was "):
        return "yes_no"
    return "other"


def _init_bucket() -> dict:
    return {
        "total": 0,
        "reason_counts": {},
        "answered_count": 0,
        "abstain_count": 0,
        "exact_match_count": 0,
        "exact_match_ocr_relaxed_count": 0,
    }


def _update_bucket(bucket: dict, *, reason: str, pred_answered: bool, exact: Optional[bool], exact_ocr: Optional[bool]):
    bucket["total"] += 1
    bucket["reason_counts"][reason] = bucket["reason_counts"].get(reason, 0) + 1
    if pred_answered:
        bucket["answered_count"] += 1
    else:
        bucket["abstain_count"] += 1
    if exact is True:
        bucket["exact_match_count"] += 1
    if exact_ocr is True:
        bucket["exact_match_ocr_relaxed_count"] += 1


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
    exact_match_ocr_relaxed_overall = 0
    exact_match_ocr_relaxed_answered = 0
    answered_with_gold = 0
    examples_with_gold = 0
    missed_with_gold = 0
    by_qtype: dict[str, dict] = {}

    details = []

    for row in rows:
        total += 1
        reason = row.get("reason", "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

        if reason == "error":
            error_count += 1

        pred = row.get("pred_answer")
        question = row.get("question")
        qtype = _classify_question_type(question)
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
        exact_ocr = None
        if pred_answered:
            exact_ocr = _exact_in_gold_ocr_relaxed(pred, gold_answers)

        status = None
        if pred_answered:
            if exact is True:
                exact_match_answered += 1
                exact_match_overall += 1
                status = "correct_answered"
            else:
                status = "wrong_answer"
            if exact_ocr is True:
                exact_match_ocr_relaxed_answered += 1
                exact_match_ocr_relaxed_overall += 1
            if has_gold:
                answered_with_gold += 1
        else:
            if has_gold and reason != "error":
                status = "missed_answer"
            elif reason != "error":
                status = "abstained_no_gold"
            else:
                status = "error"

        bucket = by_qtype.setdefault(qtype, _init_bucket())
        _update_bucket(bucket, reason=reason, pred_answered=pred_answered, exact=exact, exact_ocr=exact_ocr)

        details.append(
            {
                "qid": row.get("qid"),
                "question_type": qtype,
                "question": question,
                "reason": reason,
                "status": status,
                "pred_answer": pred,
                "gold_answers": gold_answers,
                "pred_answer_exact_in_gold": exact,
                "pred_answer_exact_in_gold_ocr_relaxed": exact_ocr,
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
        "exact_match_ocr_relaxed_overall_count": exact_match_ocr_relaxed_overall,
        "exact_match_ocr_relaxed_answered_count": exact_match_ocr_relaxed_answered,
        "exact_match_overall_rate": (exact_match_overall / total) if total else None,
        "exact_match_on_answered_rate": (exact_match_answered / answered_count) if answered_count else None,
        "exact_match_ocr_relaxed_overall_rate": (exact_match_ocr_relaxed_overall / total) if total else None,
        "exact_match_ocr_relaxed_on_answered_rate": (
            exact_match_ocr_relaxed_answered / answered_count
        ) if answered_count else None,
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
    print(f"- exact_match_ocr_relaxed_overall_count: {metrics['exact_match_ocr_relaxed_overall_count']}")
    print(f"- exact_match_ocr_relaxed_answered_count: {metrics['exact_match_ocr_relaxed_answered_count']}")
    print(f"- exact_match_overall_rate: {metrics['exact_match_overall_rate']}")
    print(f"- exact_match_on_answered_rate: {metrics['exact_match_on_answered_rate']}")
    print(f"- exact_match_ocr_relaxed_overall_rate: {metrics['exact_match_ocr_relaxed_overall_rate']}")
    print(f"- exact_match_ocr_relaxed_on_answered_rate: {metrics['exact_match_ocr_relaxed_on_answered_rate']}")
    print(f"- answer_coverage_rate: {metrics['answer_coverage_rate']}")

    print("\nQuestion-Type Breakdown")
    for qtype in sorted(by_qtype.keys()):
        b = by_qtype[qtype]
        cov = (b["answered_count"] / b["total"]) if b["total"] else None
        em = (b["exact_match_count"] / b["total"]) if b["total"] else None
        em_ocr = (b["exact_match_ocr_relaxed_count"] / b["total"]) if b["total"] else None
        print(
            f"- {qtype}: total={b['total']} answered={b['answered_count']} "
            f"reason_counts={b['reason_counts']} em={em} em_ocr={em_ocr} coverage={cov}"
        )

    if args.show_details or args.show_errors_only:
        print("\nDetails")
        shown = 0
        for d in details:
            if args.show_errors_only and d["status"] not in ("wrong_answer", "missed_answer", "error"):
                continue
            print(
                f"- {d['qid']} | {d['question_type']} | {d['status']} | reason={d['reason']} | "
                f"pred={d['pred_answer']!r} | gold={d['gold_answers'][:2]}"
            )
            shown += 1
            if shown >= args.max_detail_rows:
                break

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({"metrics": metrics, "question_type_breakdown": by_qtype, "details": details}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
