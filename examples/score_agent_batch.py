"""Score `run_agent_rag_batch.py` outputs against a small labeled set.

Supported labels formats (JSON or JSONL):
- {"<doc_id>": "Expected Title", "<doc_id2>": null}
- [{"doc_id": "...", "answer": "Expected Title", "aliases": ["..."]}, ...]

`null` / missing answer means the expected behavior is to abstain
(`unanswerable`, `max_turns`, etc.) rather than return an answer.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _norm(text: str | None) -> str | None:
    if text is None:
        return None
    text = text.casefold()
    text = " ".join(text.split())
    # Lightweight normalization for OCR artifacts and punctuation variance.
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\"'`“”‘’]", "", text)
    text = re.sub(r"\s*[-–—:;,./|]\s*", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _load_json_or_jsonl(path: Path) -> Any:
    suffixes = {s.lower() for s in path.suffixes}
    if ".jsonl" in suffixes:
        rows = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
    return json.loads(path.read_text())


def load_summary_rows(path: Path) -> list[dict]:
    payload = _load_json_or_jsonl(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in summary file: {path}")
    rows = []
    for row in payload:
        if isinstance(row, dict) and row.get("doc_id"):
            rows.append(row)
    return rows


def load_labels(path: Path) -> dict[str, dict]:
    payload = _load_json_or_jsonl(path)
    out: dict[str, dict] = {}

    def _row_to_record(doc_id: str, row: dict | None, value_fallback=None):
        expected = None
        aliases: list[str] = []
        if row is None:
            expected = value_fallback
        else:
            if "expect_answer" in row and row["expect_answer"] is False:
                expected = None
            else:
                for key in ("answer", "expected", "gold", "title"):
                    if key in row:
                        expected = row[key]
                        break
                if expected is None:
                    expected = value_fallback
            aliases_val = row.get("aliases", [])
            if isinstance(aliases_val, list):
                aliases = [str(x) for x in aliases_val if x is not None]

        if expected is not None:
            expected = str(expected)

        out[str(doc_id)] = {
            "expected": expected,
            "aliases": aliases,
        }

    if isinstance(payload, dict):
        for doc_id, value in payload.items():
            if isinstance(value, dict):
                _row_to_record(str(doc_id), value)
            else:
                _row_to_record(str(doc_id), None, value_fallback=value)
        return out

    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            doc_id = row.get("doc_id") or row.get("document_id") or row.get("id")
            if not doc_id:
                continue
            _row_to_record(str(doc_id), row)
        return out

    raise ValueError(f"Unsupported labels payload type: {type(payload)}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--summary-json", type=Path, required=True, help="Batch summary JSON from run_agent_rag_batch.py")
    p.add_argument("--labels", type=Path, required=True, help="JSON/JSONL labels file")
    p.add_argument(
        "--show-details",
        action="store_true",
        help="Print per-document comparison rows (all labeled docs).",
    )
    p.add_argument(
        "--show-errors-only",
        action="store_true",
        help="Print only mismatches / wrong answered rows.",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the computed metrics and mismatches as JSON.",
    )
    return p.parse_args()


def matches_expected(predicted: str | None, expected: str | None, aliases: list[str]) -> bool:
    if predicted is None or expected is None:
        return False
    p = _norm(predicted)
    candidates = [_norm(expected)] + [_norm(a) for a in aliases]
    candidates = [c for c in candidates if c]
    return p in candidates


def main():
    args = parse_args()
    rows = load_summary_rows(args.summary_json)
    labels = load_labels(args.labels)

    metrics = {
        "total_summary_rows": len(rows),
        "total_labeled_rows": 0,
        "unlabeled_rows_skipped": 0,
        "expected_answer_count": 0,
        "expected_abstain_count": 0,
        "pred_answered_count": 0,
        "pred_abstain_count": 0,
        "correct_answered_count": 0,
        "correct_abstain_count": 0,
        "false_answered_count": 0,
        "missed_answer_count": 0,
    }

    details = []

    for row in rows:
        doc_id = str(row.get("doc_id"))
        if doc_id not in labels:
            metrics["unlabeled_rows_skipped"] += 1
            continue

        metrics["total_labeled_rows"] += 1
        label = labels[doc_id]
        expected = label["expected"]
        aliases = label["aliases"]
        reason = row.get("reason")
        predicted = row.get("answer")
        pred_answered = reason == "answered" and predicted is not None

        if expected is None:
            metrics["expected_abstain_count"] += 1
        else:
            metrics["expected_answer_count"] += 1

        if pred_answered:
            metrics["pred_answered_count"] += 1
        else:
            metrics["pred_abstain_count"] += 1

        status = "unknown"
        if expected is None:
            if pred_answered:
                metrics["false_answered_count"] += 1
                status = "false_answered"
            else:
                metrics["correct_abstain_count"] += 1
                status = "correct_abstain"
        else:
            if pred_answered and matches_expected(str(predicted), expected, aliases):
                metrics["correct_answered_count"] += 1
                status = "correct_answered"
            elif pred_answered:
                metrics["false_answered_count"] += 1
                status = "wrong_answer"
            else:
                metrics["missed_answer_count"] += 1
                status = "missed_answer"

        details.append(
            {
                "doc_id": doc_id,
                "status": status,
                "reason": reason,
                "predicted_answer": predicted,
                "expected_answer": expected,
                "aliases": aliases,
            }
        )

    pa = metrics["pred_answered_count"]
    ea = metrics["expected_answer_count"]
    tl = metrics["total_labeled_rows"]
    metrics["answered_precision"] = (metrics["correct_answered_count"] / pa) if pa else None
    metrics["answer_recall_on_expected"] = (metrics["correct_answered_count"] / ea) if ea else None
    metrics["overall_accuracy"] = (
        (metrics["correct_answered_count"] + metrics["correct_abstain_count"]) / tl if tl else None
    )

    print("Metrics")
    for k in (
        "total_summary_rows",
        "total_labeled_rows",
        "unlabeled_rows_skipped",
        "expected_answer_count",
        "expected_abstain_count",
        "pred_answered_count",
        "pred_abstain_count",
        "correct_answered_count",
        "correct_abstain_count",
        "false_answered_count",
        "missed_answer_count",
    ):
        print(f"- {k}: {metrics[k]}")
    print(f"- answered_precision: {metrics['answered_precision']}")
    print(f"- answer_recall_on_expected: {metrics['answer_recall_on_expected']}")
    print(f"- overall_accuracy: {metrics['overall_accuracy']}")

    if args.show_details or args.show_errors_only:
        print("\nDetails")
        for d in details:
            if args.show_errors_only and d["status"] in ("correct_answered", "correct_abstain"):
                continue
            print(
                f"- {d['doc_id']} | {d['status']} | reason={d['reason']} | "
                f"pred={d['predicted_answer']!r} | expected={d['expected_answer']!r}"
            )

    if args.output_json is not None:
        payload = {"metrics": metrics, "details": details}
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
