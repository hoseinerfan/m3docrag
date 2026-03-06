#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_ENV="${CONDA_ENV:-my_cuda_env}"

DATASET_ROOT="${DATASET_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/datasets/m3docvqa_dev_ret1000_rerankexp_v1}"
DOC_PARQ="${DOC_PARQ:-$DATASET_ROOT/parquet/retrieval_edges.parquet}"
PAGE_PARQ="${PAGE_PARQ:-$DATASET_ROOT/parquet/retrieval_page_edges.parquet}"
QRELS_PARQ="${QRELS_PARQ:-$DATASET_ROOT/parquet/qrels.parquet}"

RUN_ID="${RUN_ID:-baseline_ret1000}"
TOPK_DOCS="${TOPK_DOCS:-1000}"
PAGES_PER_DOC="${PAGES_PER_DOC:-6}"
# Leave empty (or set to "all") to process every qid.
MAX_QIDS="${MAX_QIDS:-}"
CANDIDATE_MODE="${CANDIDATE_MODE:-doc_page_pool}" # doc_page_pool | docorder_page0
FORCE_PAGE_IDX="${FORCE_PAGE_IDX:-0}"             # used when CANDIDATE_MODE=docorder_page0

EMB="${EMB:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali-v1.2_m3-docvqa_dev}"
MMQA="${MMQA:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/data/m3-docvqa/multimodalqa/MMQA_dev.jsonl}"
BACKBONE="${BACKBONE:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/models/colpaligemma-3b-pt-448-base}"
ADAPTER="${ADAPTER:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/models/colpali-v1.2}"

COHERENCE_LAMBDA="${COHERENCE_LAMBDA:-0.15}"
RERANK_MODE="${RERANK_MODE:-per_doc_page}"

OUTDIR="${OUTDIR:-outputs/spatial_batch}"
mkdir -p "$OUTDIR"

if [[ "$CANDIDATE_MODE" == "docorder_page0" ]]; then
  TOPDOCS_JSON="${TOPDOCS_JSON:-$OUTDIR/topdocs_doc${TOPK_DOCS}_page${FORCE_PAGE_IDX}_${RUN_ID}.json}"
  RERANK_JSON="${RERANK_JSON:-$OUTDIR/reranked_doc${TOPK_DOCS}_page${FORCE_PAGE_IDX}_${RUN_ID}.json}"
  DEBUG_JSON="${DEBUG_JSON:-$OUTDIR/reranked_debug_page${FORCE_PAGE_IDX}_${RUN_ID}.json}"
else
  TOPDOCS_JSON="${TOPDOCS_JSON:-$OUTDIR/topdocs_doc${TOPK_DOCS}_p${PAGES_PER_DOC}_${RUN_ID}.json}"
  RERANK_JSON="${RERANK_JSON:-$OUTDIR/reranked_doc${TOPK_DOCS}_p${PAGES_PER_DOC}_${RUN_ID}.json}"
  DEBUG_JSON="${DEBUG_JSON:-$OUTDIR/reranked_debug_${RUN_ID}.json}"
fi

echo "[1/3] Build candidate topdocs pool"
if [[ "$CANDIDATE_MODE" == "doc_page_pool" ]]; then
  BUILD_ARGS=(
    --retrieval-doc-parquet "$DOC_PARQ"
    --retrieval-page-parquet "$PAGE_PARQ"
    --output-json "$TOPDOCS_JSON"
    --run-id "$RUN_ID"
    --topk-docs "$TOPK_DOCS"
    --pages-per-doc "$PAGES_PER_DOC"
  )
  if [[ -n "$MAX_QIDS" && "$MAX_QIDS" != "all" ]]; then
    BUILD_ARGS+=(--max-qids "$MAX_QIDS")
  fi
  conda run -n "$CONDA_ENV" python examples/build_topdocs_doc_page_pool.py "${BUILD_ARGS[@]}"
elif [[ "$CANDIDATE_MODE" == "docorder_page0" ]]; then
  BUILD_ARGS=(
    --retrieval-doc-parquet "$DOC_PARQ"
    --output-json "$TOPDOCS_JSON"
    --run-id "$RUN_ID"
    --topk-docs "$TOPK_DOCS"
    --force-page-idx "$FORCE_PAGE_IDX"
  )
  if [[ -n "$MAX_QIDS" && "$MAX_QIDS" != "all" ]]; then
    BUILD_ARGS+=(--max-qids "$MAX_QIDS")
  fi
  conda run -n "$CONDA_ENV" python examples/build_topdocs_docorder_page0.py "${BUILD_ARGS[@]}"
else
  echo "Unsupported CANDIDATE_MODE='$CANDIDATE_MODE'. Use 'doc_page_pool' or 'docorder_page0'." >&2
  exit 1
fi

echo "[2/3] Rerank and evaluate against qrels"
if [[ "$CANDIDATE_MODE" == "docorder_page0" ]]; then
  TOPK_CANDIDATES="$TOPK_DOCS"
else
  TOPK_CANDIDATES=$((TOPK_DOCS * PAGES_PER_DOC + TOPK_DOCS))
fi
RERANK_ARGS=(
  --topdocs-json "$TOPDOCS_JSON"
  --qrels-parquet "$QRELS_PARQ"
  --output-json "$RERANK_JSON"
  --debug-qid-json "$DEBUG_JSON"
  --embedding-dir "$EMB"
  --mmqa-jsonl "$MMQA"
  --retrieval-model-name-or-path "$BACKBONE"
  --retrieval-adapter-model-name-or-path "$ADAPTER"
  --topk-candidates "$TOPK_CANDIDATES"
  --save-top-k "$TOPK_DOCS"
  --coherence-lambda "$COHERENCE_LAMBDA"
  --rerank-mode "$RERANK_MODE"
)
if [[ -n "${QRELS_QID_COL:-}" ]]; then
  RERANK_ARGS+=(--qrels-qid-col "$QRELS_QID_COL")
fi
if [[ -n "${QRELS_DOC_COL:-}" ]]; then
  RERANK_ARGS+=(--qrels-doc-col "$QRELS_DOC_COL")
fi
if [[ -n "${QRELS_PAGE_COL:-}" ]]; then
  RERANK_ARGS+=(--qrels-page-col "$QRELS_PAGE_COL")
fi

conda run -n "$CONDA_ENV" python examples/rerank_topdocs_spatial_coherence.py "${RERANK_ARGS[@]}"

echo "[3/3] Summary"
export RERANK_JSON
conda run -n "$CONDA_ENV" python - <<'PY'
import json, os
p = os.environ["RERANK_JSON"]
j = json.load(open(p))
sm = j.get("summary_metrics", {})
print("output:", p)
print("summary_metrics:")
for k in [
    "qids_with_gold_targets",
    "qids_with_before_rank",
    "qids_with_after_rank",
    "improved_count",
    "worsened_count",
    "recall@1_before",
    "recall@1_after",
    "recall@5_before",
    "recall@5_after",
    "recall@10_before",
    "recall@10_after",
    "mrr_before",
    "mrr_after",
]:
    print(f"  {k}: {sm.get(k)}")
PY
