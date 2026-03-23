#!/bin/bash
#SBATCH -J hard10_mhop_all
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH -t 12:00:00
#SBATCH -o /mmfs1/scratch/jacks.local/aerfanshekooh/newproject/outputs/logs/hard10_mhop_all_%j.out
#SBATCH -e /mmfs1/scratch/jacks.local/aerfanshekooh/newproject/outputs/logs/hard10_mhop_all_%j.err

set -euo pipefail

# Full self-contained job: starts vLLM locally, then runs selection-only eval.
ROOT="${ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/newproject}"
MMQA_DEV="${MMQA_DEV:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/data/m3-docvqa/multimodalqa/MMQA_dev.jsonl}"
PDF_DIR="${PDF_DIR:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/data/m3-docvqa/splits/pdfs_dev}"
RET_PARQ="${RET_PARQ:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/datasets/m3docvqa_dev_ret1000_rerankexp_v1/parquet/retrieval_edges.parquet}"
QRELS="${QRELS:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/datasets/m3docvqa_dev_ret1000_rerankexp_v1/parquet/qrels.parquet}"
BASE_META="${BASE_META:-$ROOT/outputs/page_summaries_dev_full_allpages_qwen25vl32b_api.jsonl}"
QIDS_HARD50="${QIDS_HARD50:-$ROOT/outputs/qids_dev_hard50_rank11_500.txt}"
QIDS_HARD10="${QIDS_HARD10:-$ROOT/outputs/qids_dev_hard10_rank11_500.txt}"
API_KEY_FILE="${API_KEY_FILE:-$ROOT/deepinfrakey}"

MODEL_PATH="${MODEL_PATH:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/models/Qwen2.5-VL-32B-Instruct}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-32B-Instruct}"
VLLM_PORT="${VLLM_PORT:-8010}"
BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"

OUT_JSONL="${OUT_JSONL:-$ROOT/outputs/agent_simpledoc_hard10_multihop.jsonl}"
OUT_SUMMARY="${OUT_SUMMARY:-$ROOT/outputs/agent_simpledoc_hard10_multihop_summary.json}"
VLLM_STDOUT="${VLLM_STDOUT:-$ROOT/outputs/logs/vllm_inline_${SLURM_JOB_ID:-manual}.log}"

mkdir -p "$ROOT/outputs/logs"
if [[ ! -s "$API_KEY_FILE" ]]; then
  printf 'dummy-key\n' > "$API_KEY_FILE"
  chmod 600 "$API_KEY_FILE"
fi
if [[ ! -f "$QIDS_HARD50" ]]; then
  echo "Missing QIDS_HARD50: $QIDS_HARD50"
  exit 2
fi

head -n 10 "$QIDS_HARD50" > "$QIDS_HARD10"
echo "Built QIDS_HARD10: $QIDS_HARD10 ($(wc -l < "$QIDS_HARD10") qids)"

source "$(conda info --base)/etc/profile.d/conda.sh"

# Use very short cache/tmp paths to avoid quota and IPC path-length failures.
SHORT="/tmp/vllm_${SLURM_JOB_ID:-manual}"
mkdir -p "$SHORT"/triton "$SHORT"/xdg "$SHORT"/hf "$SHORT"/tmp
export TRITON_CACHE_DIR="$SHORT/triton"
export XDG_CACHE_HOME="$SHORT/xdg"
export HF_HOME="$SHORT/hf"
export TMPDIR="$SHORT/tmp"

conda activate "$ROOT/.conda/vllm_server"
python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port "$VLLM_PORT" \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  > "$VLLM_STDOUT" 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  if curl -fsS "${BASE_URL}/models" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
curl -fsS "${BASE_URL}/models" >/dev/null
echo "vLLM ready at ${BASE_URL}"

conda activate "$ROOT/.conda/m3docrag"
cd "$ROOT/m3docrag"
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
  --policy-model "$MODEL_NAME" \
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
echo "vllm_log=$VLLM_STDOUT"
