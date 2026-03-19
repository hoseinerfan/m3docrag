"""Build a visual lexicon candidate list from train/dev question text.

The script can read MMQA JSONL or questions parquet, extract frequent visual
phrases, and map them to canonical visual terms.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

try:
    from m3docrag.utils.paths import LOCAL_DATA_DIR

    DEFAULT_MMQA_TRAIN = Path(LOCAL_DATA_DIR) / "m3-docvqa" / "multimodalqa" / "MMQA_train.jsonl"
except Exception:
    DEFAULT_MMQA_TRAIN = None


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "those",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whose",
    "why",
    "with",
    "how",
    "many",
    "much",
    "kind",
    "type",
    "name",
    "called",
    "shown",
    "seen",
    "visible",
}

VISUAL_TRIGGER_TOKENS = {
    "image",
    "photo",
    "picture",
    "portrait",
    "logo",
    "symbol",
    "emblem",
    "seal",
    "map",
    "chart",
    "graph",
    "plot",
    "table",
    "flag",
    "banner",
    "vehicle",
    "car",
    "truck",
    "bus",
    "train",
    "plane",
    "airplane",
    "aircraft",
    "boat",
    "ship",
    "bike",
    "bicycle",
    "motorcycle",
    "person",
    "man",
    "woman",
    "child",
    "boy",
    "girl",
    "baby",
    "bald",
    "building",
    "campus",
    "house",
    "church",
    "school",
    "stadium",
    "uniform",
    "jersey",
    "helmet",
    "color",
}

QUESTION_COL_CANDIDATES = ["question", "query", "question_text", "text", "prompt"]
SPLIT_COL_CANDIDATES = ["split", "set", "subset"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mmqa-jsonl", type=Path, default=DEFAULT_MMQA_TRAIN)
    p.add_argument("--questions-parquet", type=Path, default=None)
    p.add_argument("--split", type=str, default="train", help="Used only for parquet inputs.")
    p.add_argument("--question-column", type=str, default=None)
    p.add_argument("--split-column", type=str, default=None)
    p.add_argument("--max-ngram", type=int, default=3)
    p.add_argument("--min-count", type=int, default=5)
    p.add_argument("--top-k", type=int, default=250)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-terms-txt", type=Path, default=None)
    return p.parse_args()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def _extract_ngrams(tokens: list[str], max_ngram: int) -> set[str]:
    out: set[str] = set()
    n_tokens = len(tokens)
    for n in range(1, max_ngram + 1):
        for i in range(0, max(0, n_tokens - n + 1)):
            gram = tokens[i : i + n]
            if not gram:
                continue
            if gram[0] in STOPWORDS or gram[-1] in STOPWORDS:
                continue
            if all(tok.isdigit() for tok in gram):
                continue
            out.add(" ".join(gram))
    return out


def _contains_visual_trigger(phrase: str) -> bool:
    toks = phrase.split()
    return any(tok in VISUAL_TRIGGER_TOKENS for tok in toks)


def _canonical_term(phrase: str) -> str | None:
    p = f" {phrase} "

    if re.search(r"\bbald\b", p) and re.search(r"\b(man|male|guy)\b", p):
        return "bald_man"
    if re.search(r"\b(child|kid|boy|girl|baby|toddler)\b", p):
        return "child"
    if re.search(r"\b(woman|female|lady)\b", p):
        return "woman"
    if re.search(r"\b(portrait|headshot|profile photo|profile picture)\b", p):
        return "portrait_photo"
    if re.search(r"\b(logo|emblem|seal|symbol|wordmark|badge)\b", p):
        return "logo"
    if re.search(r"\bmap\b", p):
        return "map"
    if re.search(r"\b(chart|graph|plot|histogram|bar chart|line chart|pie chart)\b", p):
        return "chart"
    if re.search(r"\b(table|tabular|spreadsheet|grid)\b", p):
        return "table"
    if re.search(r"\b(building|campus|church|school|house|tower|stadium|office)\b", p):
        return "building_or_campus"
    if re.search(r"\b(uniform|jersey|kit|helmet)\b", p):
        return "sports_uniform"
    if re.search(r"\b(flag|banner|pennant)\b", p):
        return "flag"
    if re.search(r"\b(vehicle|car|truck|bus|train|plane|airplane|aircraft|boat|ship|bike|bicycle|motorcycle)\b", p):
        return "vehicle"
    return None


def _iter_questions_from_mmqa(path: Path) -> Iterable[str]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            q = None
            for k in QUESTION_COL_CANDIDATES:
                if k in row and row[k] is not None:
                    q = str(row[k])
                    break
            if q:
                yield q


def _iter_questions_from_parquet(path: Path, split: str, q_col_override: str | None, split_col_override: str | None) -> Iterable[str]:
    try:
        import pyarrow.dataset as ds
    except Exception as exc:
        raise ImportError("pyarrow is required for --questions-parquet") from exc

    data = ds.dataset(str(path), format="parquet")
    cols = list(data.schema.names)
    q_col = q_col_override or next((c for c in QUESTION_COL_CANDIDATES if c in cols), None)
    if q_col is None:
        raise RuntimeError(f"Could not infer question column. available={cols}")

    split_col = split_col_override or next((c for c in SPLIT_COL_CANDIDATES if c in cols), None)
    scan_cols = [q_col]
    filters = None
    if split_col is not None:
        filters = ds.field(split_col) == split
        scan_cols.append(split_col)

    table = data.to_table(columns=scan_cols, filter=filters)
    for row in table.to_pylist():
        q = row.get(q_col)
        if q is None:
            continue
        txt = str(q).strip()
        if txt:
            yield txt


def main() -> None:
    args = parse_args()

    if args.questions_parquet is not None:
        question_iter = _iter_questions_from_parquet(
            args.questions_parquet,
            split=args.split,
            q_col_override=args.question_column,
            split_col_override=args.split_column,
        )
        source = str(args.questions_parquet)
    else:
        if args.mmqa_jsonl is None:
            raise RuntimeError("Provide --mmqa-jsonl or --questions-parquet")
        if not args.mmqa_jsonl.exists():
            raise FileNotFoundError(args.mmqa_jsonl)
        question_iter = _iter_questions_from_mmqa(args.mmqa_jsonl)
        source = str(args.mmqa_jsonl)

    n_questions = 0
    raw_df: Counter[str] = Counter()
    canonical_df: Counter[str] = Counter()
    canonical_examples: dict[str, list[str]] = defaultdict(list)

    for question in question_iter:
        n_questions += 1
        toks = _tokenize(question)
        if not toks:
            continue

        phrases = _extract_ngrams(toks, args.max_ngram)
        visual_phrases = {ph for ph in phrases if _contains_visual_trigger(ph)}
        raw_df.update(visual_phrases)

        canonical_hits: set[str] = set()
        for ph in visual_phrases:
            term = _canonical_term(ph)
            if term is None:
                continue
            canonical_hits.add(term)
        canonical_df.update(canonical_hits)

        for term in sorted(canonical_hits):
            if len(canonical_examples[term]) < 3:
                canonical_examples[term].append(question)

    raw_terms = [
        {"phrase": phrase, "count_questions": int(cnt)}
        for phrase, cnt in raw_df.most_common()
        if cnt >= args.min_count
    ][: args.top_k]

    canonical_terms = []
    for term, cnt in canonical_df.most_common():
        canonical_terms.append(
            {
                "term": term,
                "count_questions": int(cnt),
                "question_rate": float(cnt) / max(1, n_questions),
                "example_questions": canonical_examples.get(term, []),
            }
        )

    out = {
        "source": source,
        "n_questions": n_questions,
        "max_ngram": args.max_ngram,
        "min_count": args.min_count,
        "top_k": args.top_k,
        "n_raw_terms_kept": len(raw_terms),
        "top_raw_visual_phrases": raw_terms,
        "canonical_terms": canonical_terms,
        "suggested_prompt_terms": [rec["term"] for rec in canonical_terms],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, ensure_ascii=True, indent=2) + "\n")

    if args.output_terms_txt is not None:
        args.output_terms_txt.parent.mkdir(parents=True, exist_ok=True)
        with args.output_terms_txt.open("w") as f:
            for term in out["suggested_prompt_terms"]:
                f.write(f"{term}\n")

    print("source:", source)
    print("n_questions:", n_questions)
    print("n_raw_terms_kept:", len(raw_terms))
    print("n_canonical_terms:", len(canonical_terms))
    print("output_json:", args.output_json)
    if args.output_terms_txt is not None:
        print("output_terms_txt:", args.output_terms_txt)


if __name__ == "__main__":
    main()
