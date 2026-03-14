#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/newproject}"
DOCS="${DOCS:-$ROOT/outputs/docids_two_gold.txt}"
MODEL_PATH="${MODEL_PATH:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/models/Qwen2.5-VL-32B-Instruct}"
OUT="${OUT:-$ROOT/outputs/smoke_qwen25vl32b_1page_venv.jsonl}"
USE_FAST="${USE_FAST:-true}" # true|false|auto
MAX_PAGES="${MAX_PAGES:-1}"

source "$ROOT/.venv_qwen32b/bin/activate"
cd "$ROOT/m3docrag"
unset TRANSFORMERS_CACHE

python "$ROOT/m3docrag/examples/build_page_summaries_m3docvqa.py" \
  --split dev \
  --doc-id-file "$DOCS" \
  --model-name-or-path "$MODEL_PATH" \
  --model-type qwen2_5_vl \
  --device cuda \
  --bits 16 \
  --max-pages-per-doc "$MAX_PAGES" \
  --qwen-use-fast-processor "$USE_FAST" \
  --page-max-retries 2 \
  --page-retry-wait-seconds 1.0 \
  --prompt "Write one short plain-text paragraph describing only visible page content for retrieval: key people, organizations, titles, logos/symbols, and concrete entities. No markdown, no bullet points, no hallucinations." \
  --output-jsonl "$OUT" \
  --overwrite

ls -lh "$OUT"
wc -l "$OUT"
