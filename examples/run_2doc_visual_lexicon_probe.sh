#!/bin/bash
set -euo pipefail

ROOT="${ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/newproject}"
DOCS="${DOCS:-$ROOT/outputs/docids_two_gold.txt}"
PDF_DIR="${PDF_DIR:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/data/m3-docvqa/splits/pdfs_dev}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
API_KEY_FILE="${API_KEY_FILE:-$ROOT/deepinfrakey}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-32B-Instruct}"
MAX_PAGES="${MAX_PAGES:-3}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/lexicon_probe_two_docs}"

GENERAL_PROMPT="$ROOT/m3docrag/examples/prompts/page_summary_general.txt"
LEXICON_PROMPT="$ROOT/m3docrag/examples/prompts/page_summary_lexicon_probe.txt"
GENERAL_OUT="$OUT_DIR/page_summaries_general_p${MAX_PAGES}.jsonl"
LEXICON_OUT="$OUT_DIR/page_summaries_lexicon_p${MAX_PAGES}.jsonl"

mkdir -p "$OUT_DIR"
if [[ ! -s "$API_KEY_FILE" ]]; then
  printf 'dummy-key\n' > "$API_KEY_FILE"
  chmod 600 "$API_KEY_FILE"
fi

echo "[1/3] Generate baseline general summaries"
python "$ROOT/m3docrag/examples/build_page_summaries_vllm_api.py" \
  --split dev \
  --doc-id-file "$DOCS" \
  --pdf-dir "$PDF_DIR" \
  --base-url "$BASE_URL" \
  --api-key-file "$API_KEY_FILE" \
  --model "$MODEL" \
  --max-pages-per-doc "$MAX_PAGES" \
  --prompt-file "$GENERAL_PROMPT" \
  --output-jsonl "$GENERAL_OUT" \
  --overwrite

echo "[2/3] Generate targeted lexicon JSON summaries"
python "$ROOT/m3docrag/examples/build_page_summaries_vllm_api.py" \
  --split dev \
  --doc-id-file "$DOCS" \
  --pdf-dir "$PDF_DIR" \
  --base-url "$BASE_URL" \
  --api-key-file "$API_KEY_FILE" \
  --model "$MODEL" \
  --max-pages-per-doc "$MAX_PAGES" \
  --prompt-file "$LEXICON_PROMPT" \
  --output-jsonl "$LEXICON_OUT" \
  --overwrite

echo "[3/3] Compare outputs"
export GENERAL_OUT
export LEXICON_OUT
python - <<'PY'
import json
import os
import re


def load_rows(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[(str(row["doc_id"]), int(row["page_idx"]))] = str(row.get("summary", ""))
    return out


def parse_json_blob(text):
    text = text.strip()
    candidates = [text]
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            return json.loads(c)
        except Exception:
            pass
    return None


general = load_rows(os.environ["GENERAL_OUT"])
lexicon = load_rows(os.environ["LEXICON_OUT"])
keys = sorted(set(general).intersection(set(lexicon)))

parsed = 0
any_yes = 0
avg_yes = 0.0

for k in keys:
    obj = parse_json_blob(lexicon[k])
    if obj is None:
        continue
    parsed += 1
    items = obj.get("visual_lexicons", [])
    yes_count = sum(
        1 for item in items
        if isinstance(item, dict) and str(item.get("present", "")).strip().lower() == "yes"
    )
    avg_yes += yes_count
    if yes_count > 0:
        any_yes += 1

if parsed > 0:
    avg_yes = avg_yes / parsed

print("general_rows:", len(general))
print("lexicon_rows:", len(lexicon))
print("paired_rows:", len(keys))
print("lexicon_json_parse_rate:", f"{parsed}/{len(keys)}")
print("rows_with_any_yes:", f"{any_yes}/{parsed if parsed else 1}")
print("avg_yes_terms_per_parsed_row:", f"{avg_yes:.2f}")

print("\nSample side-by-side (first 6 rows):")
for k in keys[:6]:
    g = " ".join(general[k].split())
    l = " ".join(lexicon[k].split())
    print(f"\n=== {k[0]} p{k[1]} ===")
    print("GENERAL:", g[:260])
    print("LEXICON:", l[:260])
PY

echo
echo "Baseline output: $GENERAL_OUT"
echo "Lexicon output:  $LEXICON_OUT"
