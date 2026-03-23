#!/bin/bash
#SBATCH -J hard10_mhop_llm
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -t 08:00:00
#SBATCH -o /mmfs1/scratch/jacks.local/aerfanshekooh/newproject/outputs/logs/hard10_multihop_llm_%j.out
#SBATCH -e /mmfs1/scratch/jacks.local/aerfanshekooh/newproject/outputs/logs/hard10_multihop_llm_%j.err

set -euo pipefail

# You can override any variable below via `sbatch --export=ALL,VAR=value ...`.
ROOT="${ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/newproject}"
MMQA_DEV="${MMQA_DEV:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/data/m3-docvqa/multimodalqa/MMQA_dev.jsonl}"
PDF_DIR="${PDF_DIR:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/data/m3-docvqa/splits/pdfs_dev}"
RET_PARQ="${RET_PARQ:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/datasets/m3docvqa_dev_ret1000_rerankexp_v1/parquet/retrieval_edges.parquet}"
QRELS="${QRELS:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/datasets/m3docvqa_dev_ret1000_rerankexp_v1/parquet/qrels.parquet}"
BASE_META="${BASE_META:-$ROOT/outputs/page_summaries_dev_full_allpages_qwen25vl32b_api.jsonl}"

QIDS_HARD50="${QIDS_HARD50:-$ROOT/outputs/qids_dev_hard50_rank11_500.txt}"
QIDS_HARD10="${QIDS_HARD10:-$ROOT/outputs/qids_dev_hard10_rank11_500.txt}"

MODEL="${MODEL:-Qwen/Qwen2.5-VL-32B-Instruct}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8010/v1}"
API_KEY_FILE="${API_KEY_FILE:-$ROOT/deepinfrakey}"

OUT_JSONL="${OUT_JSONL:-$ROOT/outputs/agent_simpledoc_hard10_multihop.jsonl}"
OUT_SUMMARY="${OUT_SUMMARY:-$ROOT/outputs/agent_simpledoc_hard10_multihop_summary.json}"

mkdir -p "$ROOT/outputs/logs"

eval "$(conda shell.bash hook)"
conda activate "$ROOT/.conda/m3docrag"
cd "$ROOT/m3docrag"

if [[ ! -f "$QIDS_HARD50" ]]; then
  echo "Missing QIDS_HARD50: $QIDS_HARD50"
  exit 2
fi

if [[ ! -s "$API_KEY_FILE" ]]; then
  printf 'dummy-key\n' > "$API_KEY_FILE"
  chmod 600 "$API_KEY_FILE"
fi

head -n 10 "$QIDS_HARD50" > "$QIDS_HARD10"
echo "Built QIDS_HARD10: $QIDS_HARD10 ($(wc -l < "$QIDS_HARD10") qids)"
echo "Using BASE_URL=$BASE_URL"

# Fail fast if the OpenAI-compatible server is not reachable.
curl -sS "$BASE_URL/models" >/dev/null || {
  echo "LLM server not reachable at $BASE_URL"
  echo "If running vLLM on another host, submit with --export=ALL,BASE_URL=http://<host>:8010/v1"
  exit 3
}

python "$ROOT/m3docrag/examples/run_agent_m3docvqa_subset.py" \
  --split dev \
  --mmqa-jsonl "$MMQA_DEV" \
  --pdf-dir "$PDF_DIR" \
  --qid-file "$QIDS_HARD10" \
  --max-examples 10 \
  --doc-pool-source retrieval-parquet \
  --retrieval-edges-parquet "$RET_PARQ" \
  --retrieval-topk-docs 1000 \
  --rank-gold-source qrels-parquet \
  --qrels-parquet "$QRELS" \
  --selection-only \
  --selection-profile default \
  --selection-planner-backend llm \
  --policy-backend openai-api \
  --policy-model "$MODEL" \
  --policy-base-url "$BASE_URL" \
  --policy-api-key-file "$API_KEY_FILE" \
  --policy-max-tokens 64 \
  --selection-max-hop-queries 3 \
  --selection-topk-docs-per-hop 50 \
  --selection-max-variant-queries 2 \
  --selection-summary-full-scan \
  --selection-summary-full-scan-topk 1000 \
  --selection-retrieval-depth-multiplier 10 \
  --selection-candidate-multi-page \
  --page-summaries-file "$BASE_META" \
  --device cuda \
  --output-jsonl "$OUT_JSONL" \
  --summary-json "$OUT_SUMMARY"

echo "Done."
echo "output_jsonl=$OUT_JSONL"
echo "summary_json=$OUT_SUMMARY"
