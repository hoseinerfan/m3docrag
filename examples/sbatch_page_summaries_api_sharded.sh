#!/bin/bash
#SBATCH -J pgsum_qw_shard
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/mmfs1/scratch/jacks.local/aerfanshekooh/newproject/logs/%x_%A_%a.out
#SBATCH --error=/mmfs1/scratch/jacks.local/aerfanshekooh/newproject/logs/%x_%A_%a.err
#SBATCH --requeue

set -euo pipefail

ROOT="${ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/newproject}"
PDF_DIR="${PDF_DIR:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/data/m3-docvqa/splits/pdfs_dev}"
M2="${M2:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/models/Qwen2.5-VL-32B-Instruct}"
API_KEY_FILE="${API_KEY_FILE:-$ROOT/deepinfrakey}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-32B-Instruct}"
PROMPT_FILE="${PROMPT_FILE:-$ROOT/m3docrag/examples/prompts/page_summary_lexicon_probe_v3.txt}"
PY_VLLM="${PY_VLLM:-$ROOT/.conda/vllm_server/bin/python}"
PY_CLIENT="${PY_CLIENT:-$ROOT/.conda/m3docrag/bin/python}"

OUT_DIR="${OUT_DIR:-$ROOT/outputs/page_summaries_dev_shards_qwen25vl32b_api_lexicon_train12_v3}"
DOC_LIST="${DOC_LIST:-$OUT_DIR/doc_ids_all.txt}"
SHARD_COUNT="${SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}"
SHARD_ID="${SHARD_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
PORT_BASE="${PORT_BASE:-8000}"
PORT=$((PORT_BASE + SHARD_ID))
BASE_URL="http://127.0.0.1:${PORT}/v1"
OUT_SHARD="${OUT_SHARD:-$OUT_DIR/page_summaries_shard_${SHARD_ID}_of_${SHARD_COUNT}.jsonl}"

if (( SHARD_ID < 0 || SHARD_ID >= SHARD_COUNT )); then
  echo "Invalid shard settings: SHARD_ID=$SHARD_ID SHARD_COUNT=$SHARD_COUNT" >&2
  exit 2
fi

mkdir -p "$ROOT/logs" "$OUT_DIR"
if [[ ! -s "$API_KEY_FILE" ]]; then
  printf 'dummy-key\n' > "$API_KEY_FILE"
  chmod 600 "$API_KEY_FILE"
fi

if [[ ! -x "$PY_VLLM" ]]; then
  echo "Missing vLLM python executable: $PY_VLLM" >&2
  exit 127
fi
if [[ ! -x "$PY_CLIENT" ]]; then
  echo "Missing client python executable: $PY_CLIENT" >&2
  exit 127
fi
if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "Prompt file not found: $PROMPT_FILE" >&2
  exit 1
fi

if [[ ! -s "$DOC_LIST" ]]; then
  "$PY_CLIENT" - <<'PY'
import os
from pathlib import Path

pdf_dir = Path(os.environ["PDF_DIR"])
doc_list = Path(os.environ["DOC_LIST"])
doc_list.parent.mkdir(parents=True, exist_ok=True)
docs = sorted(p.stem for p in pdf_dir.glob("*.pdf"))
with doc_list.open("w") as f:
    for d in docs:
        f.write(d + "\n")
print("wrote_doc_list:", doc_list, "n_docs:", len(docs))
PY
fi

SHARD_DOCS="$OUT_DIR/doc_ids_shard_${SHARD_ID}_of_${SHARD_COUNT}.txt"
export SHARD_DOCS SHARD_COUNT SHARD_ID DOC_LIST
"$PY_CLIENT" - <<'PY'
import os
from pathlib import Path

doc_list = Path(os.environ["DOC_LIST"])
shard_docs = Path(os.environ["SHARD_DOCS"])
shard_count = int(os.environ["SHARD_COUNT"])
shard_id = int(os.environ["SHARD_ID"])

docs = [x.strip() for x in doc_list.read_text().splitlines() if x.strip()]
picked = [d for i, d in enumerate(docs) if i % shard_count == shard_id]
with shard_docs.open("w") as f:
    for d in picked:
        f.write(d + "\n")
print("doc_list_total:", len(docs), "shard_docs:", len(picked), "shard:", shard_id, "of", shard_count)
PY

if [[ ! -s "$SHARD_DOCS" ]]; then
  echo "No docs for shard $SHARD_ID/$SHARD_COUNT; exiting."
  exit 0
fi

# Keep cache/tmp paths very short; vLLM uses Unix IPC sockets with a strict path length limit.
CACHE_BASE="${CACHE_BASE:-/tmp/vllm_${SLURM_JOB_ID:-manual}_${SHARD_ID}}"
mkdir -p "$CACHE_BASE/tmp" "$CACHE_BASE/xdg" "$CACHE_BASE/triton" "$CACHE_BASE/torchinductor"
export TMPDIR="$CACHE_BASE/tmp"
export XDG_CACHE_HOME="$CACHE_BASE/xdg"
export TRITON_CACHE_DIR="$CACHE_BASE/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_BASE/torchinductor"
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1
unset VLLM_NODE VLLM_JOB VLLM_PID || true

VLLM_LOG="$ROOT/logs/vllm_${SLURM_JOB_ID:-manual}_${SHARD_ID}.log"
"$PY_VLLM" -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port "$PORT" \
  --model "$M2" \
  --served-model-name "$MODEL_NAME" \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --limit-mm-per-prompt '{"image":1}' \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 150); do
  if curl -fsS "${BASE_URL%/v1}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
curl -fsS "${BASE_URL%/v1}/health" >/dev/null

"$PY_CLIENT" "$ROOT/m3docrag/examples/build_page_summaries_vllm_api.py" \
  --split dev \
  --doc-id-file "$SHARD_DOCS" \
  --pdf-dir "$PDF_DIR" \
  --base-url "$BASE_URL" \
  --api-key-file "$API_KEY_FILE" \
  --model "$MODEL_NAME" \
  --request-timeout-seconds 420 \
  --api-retries 6 \
  --api-retry-wait-seconds 4 \
  --prompt-file "$PROMPT_FILE" \
  --output-jsonl "$OUT_SHARD"

wc -l "$OUT_SHARD"
echo "DONE shard=$SHARD_ID/$SHARD_COUNT out=$OUT_SHARD"
