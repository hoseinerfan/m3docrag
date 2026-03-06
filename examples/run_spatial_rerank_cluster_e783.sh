#!/usr/bin/env bash
set -euo pipefail

# Run MaxSim spatial-coherence reranking for the target qid on cluster data.
# This script avoids `conda activate` and uses `conda run -n ...` for non-interactive shells.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_ENV="${CONDA_ENV:-my_cuda_env}"

ROOT="${ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/datasets/m3docvqa_dev_ret1000_rerankexp_v1}"
PARQ="${PARQ:-}"

EMB="${EMB:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali-v1.2_m3-docvqa_dev}"
MMQA="${MMQA:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/data/m3-docvqa/multimodalqa/MMQA_dev.jsonl}"
BACKBONE="${BACKBONE:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/models/colpali-v1.2-backbone}"
ADAPTER="${ADAPTER:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/models/colpali-v1.2}"

QID="${QID:-e783cba0b3df36372d11823e378e5437}"
QUERY="${QUERY:-Which completely bald person who wears thick glasses is among the members of LGBT billionaires?}"
GOLD_DOC_ID="${GOLD_DOC_ID:-d57e56eff064047af5a6ef074a570956}"
GOLD_PAGE_IDX="${GOLD_PAGE_IDX:-0}"

TOPK_CANDIDATES="${TOPK_CANDIDATES:-1000}"
SAVE_TOP_K="${SAVE_TOP_K:-1000}"

OUTDIR="${OUTDIR:-outputs}"
OUT_JSON="${OUT_JSON:-$OUTDIR/e783_spatial_reranked.json}"
DBG_JSON="${DBG_JSON:-$OUTDIR/e783_spatial_debug.json}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH." >&2
  exit 2
fi

mkdir -p "$OUTDIR"

echo "[1/4] Discovering parquet under ROOT=$ROOT"
find "$ROOT" -maxdepth 8 \( -type d -name "retrieval_edges*" -o -type f -name "*.parquet" \) | sort | head -n 200 || true

if [[ -z "$PARQ" ]]; then
  if [[ -d "$ROOT/parquet/retrieval_edges" ]]; then
    PARQ="$ROOT/parquet/retrieval_edges"
  elif [[ -f "$ROOT/parquet/retrieval_edges.parquet" ]]; then
    PARQ="$ROOT/parquet/retrieval_edges.parquet"
  elif [[ -d "$ROOT/parquet" ]]; then
    PARQ="$ROOT/parquet"
  else
    PARQ="$(find "$ROOT" -maxdepth 8 -type f -name '*.parquet' | head -n 1 || true)"
  fi
fi

if [[ -z "$PARQ" || ! -e "$PARQ" ]]; then
  echo "ERROR: PARQ not found. Set PARQ explicitly." >&2
  exit 3
fi
export PARQ
echo "[2/4] Using PARQ=$PARQ"

echo "[3/4] Parquet schema snapshot"
conda run -n "$CONDA_ENV" python - <<'PY'
import os
import pyarrow.dataset as ds
p = os.environ["PARQ"]
d = ds.dataset(p, format="parquet")
print("rows:", d.count_rows())
print("cols:", d.schema.names)
print("sample:", d.to_table(columns=d.schema.names[:min(8, len(d.schema.names))]).slice(0, 3).to_pylist())
PY

echo "[4/4] Running spatial reranker"
conda run -n "$CONDA_ENV" python examples/rerank_topdocs_spatial_coherence.py \
  --retrieval-parquet "$PARQ" \
  --output-json "$OUT_JSON" \
  --debug-qid-json "$DBG_JSON" \
  --qid "$QID" \
  --query "$QUERY" \
  --gold-doc-id "$GOLD_DOC_ID" \
  --gold-page-idx "$GOLD_PAGE_IDX" \
  --embedding-dir "$EMB" \
  --mmqa-jsonl "$MMQA" \
  --retrieval-model-name-or-path "$BACKBONE" \
  --retrieval-adapter-model-name-or-path "$ADAPTER" \
  --topk-candidates "$TOPK_CANDIDATES" \
  --save-top-k "$SAVE_TOP_K"

echo "[done] Summary"
export OUT_JSON QID
conda run -n "$CONDA_ENV" python - <<'PY'
import json, os
p = os.environ["OUT_JSON"]
qid = os.environ["QID"]
j = json.load(open(p))
print("diagnostics:", j["diagnostics"].get(qid))
print("top10:")
for i, r in enumerate(j["top_pages"][qid][:10], 1):
    print(i, r["doc_id"], r["page_idx"], "final=", round(r["score"], 4), "base=", round(r["base_score"], 4), "coh=", round(r["coherence"], 4))
PY
