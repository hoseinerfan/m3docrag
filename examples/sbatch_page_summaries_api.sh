#!/bin/bash
#SBATCH -J pgsum_qwen32b
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=/mmfs1/scratch/jacks.local/aerfanshekooh/newproject/logs/%x_%j.out
#SBATCH --error=/mmfs1/scratch/jacks.local/aerfanshekooh/newproject/logs/%x_%j.err
#SBATCH --requeue

set -euo pipefail

ROOT="${ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/newproject}"
PDF_DIR="${PDF_DIR:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/data/m3-docvqa/splits/pdfs_dev}"
M2="${M2:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/models/Qwen2.5-VL-32B-Instruct}"
API_KEY_FILE="${API_KEY_FILE:-$ROOT/deepinfrakey}"
OUT="${OUT:-$ROOT/outputs/page_summaries_dev_full_allpages_qwen25vl32b_api.jsonl}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-32B-Instruct}"

mkdir -p "$ROOT/logs" "$(dirname "$OUT")"
if [[ ! -s "$API_KEY_FILE" ]]; then
  printf 'dummy-key\n' > "$API_KEY_FILE"
  chmod 600 "$API_KEY_FILE"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ROOT/.conda/vllm_server"

python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port 8000 \
  --model "$M2" \
  --served-model-name "$MODEL_NAME" \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --limit-mm-per-prompt '{"image":1}' \
  > "$ROOT/logs/vllm_${SLURM_JOB_ID:-manual}.log" 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

# Wait until local API is healthy before launching the client.
for _ in $(seq 1 120); do
  if curl -fsS "${BASE_URL%/v1}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
curl -fsS "${BASE_URL%/v1}/health" >/dev/null

conda activate "$ROOT/.conda/m3docrag"
python "$ROOT/m3docrag/examples/build_page_summaries_vllm_api.py" \
  --split dev \
  --pdf-dir "$PDF_DIR" \
  --all-docs-in-pdf-dir \
  --base-url "$BASE_URL" \
  --api-key-file "$API_KEY_FILE" \
  --model "$MODEL_NAME" \
  --request-timeout-seconds 120 \
  --api-retries 1 \
  --api-retry-wait-seconds 2 \
  --prompt-text "Write one short plain-text paragraph describing only visible page content for retrieval: key people, organizations, titles, logos/symbols, and concrete entities. No markdown, no bullet points, no hallucinations." \
  --output-jsonl "$OUT"

wc -l "$OUT"
