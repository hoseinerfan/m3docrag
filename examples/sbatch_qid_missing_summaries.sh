#!/bin/bash
#SBATCH -J pgsum_qids
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/mmfs1/scratch/jacks.local/aerfanshekooh/newproject/outputs/logs/pgsum_qids_%j.out
#SBATCH --error=/mmfs1/scratch/jacks.local/aerfanshekooh/newproject/outputs/logs/pgsum_qids_%j.err
#SBATCH --requeue

set -euo pipefail

# Build/append page summaries only for docs that appear in top-k retrieval pools
# of the requested qids and are missing from the summary metadata file.

ROOT="${ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/newproject}"
PDF_DIR="${PDF_DIR:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/data/m3-docvqa/splits/pdfs_dev}"
RET_PARQ="${RET_PARQ:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/datasets/m3docvqa_dev_ret1000_rerankexp_v1/parquet/retrieval_edges.parquet}"
QIDS_FILE="${QIDS_FILE:-$ROOT/outputs/qids_one_debug.txt}"
TOPK_DOCS="${TOPK_DOCS:-1000}"

API_KEY_FILE="${API_KEY_FILE:-$ROOT/deepinfrakey}"
MODEL_PATH="${MODEL_PATH:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/models/Qwen2.5-VL-32B-Instruct}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-32B-Instruct}"
PORT="${PORT:-8010}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}/v1}"

# Default behavior updates the main metadata file in place.
OUT="${OUT:-$ROOT/outputs/page_summaries_dev_full_allpages_qwen25vl32b_api.jsonl}"
DOC_IDS_TARGET="${DOC_IDS_TARGET:-$ROOT/outputs/doc_ids_target_from_qids_${SLURM_JOB_ID:-manual}.txt}"
DOC_IDS_MISSING="${DOC_IDS_MISSING:-$ROOT/outputs/doc_ids_missing_from_qids_${SLURM_JOB_ID:-manual}.txt}"

PY_VLLM="${PY_VLLM:-$ROOT/.conda/vllm_server/bin/python}"
PY_CLIENT="${PY_CLIENT:-$ROOT/.conda/m3docrag/bin/python}"

mkdir -p "$ROOT/outputs" "$ROOT/outputs/logs"

if [[ ! -x "$PY_VLLM" ]]; then
  echo "Missing vLLM python executable: $PY_VLLM" >&2
  exit 127
fi
if [[ ! -x "$PY_CLIENT" ]]; then
  echo "Missing client python executable: $PY_CLIENT" >&2
  exit 127
fi
if [[ ! -s "$API_KEY_FILE" ]]; then
  printf 'dummy-key\n' > "$API_KEY_FILE"
  chmod 600 "$API_KEY_FILE"
fi
if [[ ! -s "$QIDS_FILE" ]]; then
  echo "Missing qid file: $QIDS_FILE" >&2
  exit 2
fi
if [[ ! -s "$RET_PARQ" ]]; then
  echo "Missing retrieval parquet: $RET_PARQ" >&2
  exit 2
fi

export RET_PARQ QIDS_FILE TOPK_DOCS DOC_IDS_TARGET DOC_IDS_MISSING OUT
"$PY_CLIENT" - <<'PY'
import json
import os
from pathlib import Path

import pyarrow.dataset as ds


def pick(columns, candidates, required=True):
    cmap = {c.lower(): c for c in columns}
    for cand in candidates:
        hit = cmap.get(cand.lower())
        if hit is not None:
            return hit
    if required:
        raise RuntimeError(f"Could not find any of {candidates} in columns: {columns}")
    return None


def has_summary_text(row):
    for key in ("summary", "page_summary", "text", "snippet", "content"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return True
    return False


ret_parq = Path(os.environ["RET_PARQ"])
qids_file = Path(os.environ["QIDS_FILE"])
topk = int(os.environ["TOPK_DOCS"])
doc_ids_target = Path(os.environ["DOC_IDS_TARGET"])
doc_ids_missing = Path(os.environ["DOC_IDS_MISSING"])
out_meta = Path(os.environ["OUT"])

qids = [x.strip() for x in qids_file.read_text().splitlines() if x.strip()]
if not qids:
    raise RuntimeError(f"No qids found in {qids_file}")
qid_set = set(qids)

dataset = ds.dataset(str(ret_parq), format="parquet")
cols = list(dataset.schema.names)
qid_col = pick(cols, ["qid", "query_id", "question_id"])
doc_col = pick(cols, ["doc_id", "document_id", "candidate_doc_id", "dst_doc_id", "target_doc_id", "node_id_dst"])
rank_col = pick(cols, ["rank", "retrieval_rank", "position", "pos", "idx"], required=False)
score_col = pick(cols, ["score", "retrieval_score", "sim", "similarity", "weight"], required=False)

sel_cols = [qid_col, doc_col]
if rank_col:
    sel_cols.append(rank_col)
if score_col and score_col not in sel_cols:
    sel_cols.append(score_col)

try:
    table = dataset.to_table(columns=sel_cols, filter=ds.field(qid_col).isin(list(qid_set)))
except Exception:
    table = dataset.to_table(columns=sel_cols)

grouped = {q: [] for q in qids}
for row in table.to_pylist():
    qid = row.get(qid_col)
    doc_id = row.get(doc_col)
    if qid is None or doc_id is None:
        continue
    qid = str(qid)
    if qid not in qid_set:
        continue
    grouped[qid].append(
        {
            "doc_id": str(doc_id),
            "source_rank": row.get(rank_col) if rank_col else None,
            "score": row.get(score_col) if score_col else None,
        }
    )

target_docs = []
seen = set()
for qid in qids:
    rows = grouped.get(qid, [])
    if rank_col:
        def rank_key(x):
            try:
                return (0, int(x["source_rank"]))
            except Exception:
                return (1, 10**18)
        rows = sorted(rows, key=rank_key)
    elif score_col:
        def score_key(x):
            try:
                return float(x["score"])
            except Exception:
                return float("-inf")
        rows = sorted(rows, key=score_key, reverse=True)

    kept = []
    local_seen = set()
    for row in rows:
        d = row["doc_id"]
        if d in local_seen:
            continue
        local_seen.add(d)
        kept.append(d)
        if len(kept) >= topk:
            break

    for d in kept:
        if d not in seen:
            seen.add(d)
            target_docs.append(d)

existing_docs = set()
if out_meta.exists():
    with out_meta.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            doc_id = row.get("doc_id") or row.get("document_id")
            if doc_id is None:
                continue
            if has_summary_text(row):
                existing_docs.add(str(doc_id))

missing_docs = [d for d in target_docs if d not in existing_docs]

doc_ids_target.parent.mkdir(parents=True, exist_ok=True)
doc_ids_missing.parent.mkdir(parents=True, exist_ok=True)
with doc_ids_target.open("w") as f:
    for d in target_docs:
        f.write(d + "\n")
with doc_ids_missing.open("w") as f:
    for d in missing_docs:
        f.write(d + "\n")

print(f"qids={len(qids)}")
print(f"target_docs={len(target_docs)}")
print(f"existing_docs={len(existing_docs)}")
print(f"missing_docs={len(missing_docs)}")
print(f"doc_ids_target={doc_ids_target}")
print(f"doc_ids_missing={doc_ids_missing}")
print(f"output_meta={out_meta}")
PY

if [[ ! -s "$DOC_IDS_MISSING" ]]; then
  echo "No missing docs for requested qids. Nothing to generate."
  exit 0
fi

CACHE_BASE="${CACHE_BASE:-/tmp/vllm_${SLURM_JOB_ID:-manual}}"
mkdir -p "$CACHE_BASE/tmp" "$CACHE_BASE/xdg" "$CACHE_BASE/triton" "$CACHE_BASE/torchinductor"
export TMPDIR="$CACHE_BASE/tmp"
export XDG_CACHE_HOME="$CACHE_BASE/xdg"
export TRITON_CACHE_DIR="$CACHE_BASE/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_BASE/torchinductor"
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1

VLLM_LOG="${VLLM_LOG:-$ROOT/outputs/logs/vllm_pgsum_qids_${SLURM_JOB_ID:-manual}.log}"
"$PY_VLLM" -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port "$PORT" \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --limit-mm-per-prompt '{"image":1}' \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  if curl -fsS "${BASE_URL%/v1}/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited before readiness. Last logs:" >&2
    tail -n 120 "$VLLM_LOG" >&2 || true
    exit 3
  fi
  sleep 2
done
curl -fsS "${BASE_URL%/v1}/health" >/dev/null

"$PY_CLIENT" "$ROOT/m3docrag/examples/build_page_summaries_vllm_api.py" \
  --split dev \
  --doc-id-file "$DOC_IDS_MISSING" \
  --pdf-dir "$PDF_DIR" \
  --base-url "$BASE_URL" \
  --api-key-file "$API_KEY_FILE" \
  --model "$MODEL_NAME" \
  --request-timeout-seconds 240 \
  --api-retries 3 \
  --api-retry-wait-seconds 2 \
  --prompt-text "Write one short plain-text paragraph describing only visible page content for retrieval: key people, organizations, titles, logos/symbols, and concrete entities. No markdown, no bullet points, no hallucinations." \
  --output-jsonl "$OUT"

echo "Done."
echo "output_meta=$OUT"
echo "missing_doc_file=$DOC_IDS_MISSING"
echo "vllm_log=$VLLM_LOG"
