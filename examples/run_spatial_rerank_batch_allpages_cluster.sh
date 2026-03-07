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
# Leave empty (or "all") to process all qids.
MAX_QIDS="${MAX_QIDS:-all}"

# Keep retrieval_edges ordering from this rank column.
DOC_RANK_COL="${DOC_RANK_COL:-faiss_rank}"
DOC_SCORE_COL="${DOC_SCORE_COL:-faiss_score}"

EMB="${EMB:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali-v1.2_m3-docvqa_dev}"
MMQA="${MMQA:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/data/m3-docvqa/multimodalqa/MMQA_dev.jsonl}"
BACKBONE="${BACKBONE:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/models/colpaligemma-3b-pt-448-base}"
ADAPTER="${ADAPTER:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/models/colpali-v1.2}"

COHERENCE_LAMBDA="${COHERENCE_LAMBDA:-0.15}"
RERANK_MODE="${RERANK_MODE:-per_doc_reorder}"
SUMMARY_KS="${SUMMARY_KS:-1,2,4,10,50,100,500}"

OUTDIR="${OUTDIR:-outputs/spatial_batch_allpages}"
mkdir -p "$OUTDIR"

TOPDOCS_JSON="${TOPDOCS_JSON:-$OUTDIR/topdocs_doc${TOPK_DOCS}_allpages_${RUN_ID}.json}"
RERANK_JSON="${RERANK_JSON:-$OUTDIR/reranked_doc${TOPK_DOCS}_allpages_${RUN_ID}.json}"
DEBUG_JSON="${DEBUG_JSON:-$OUTDIR/reranked_debug_allpages_${RUN_ID}.json}"

echo "[1/4] Build all-pages candidate pool from retrieval_edges doc order"
BUILD_ARGS=(
  --retrieval-doc-parquet "$DOC_PARQ"
  --retrieval-page-parquet "$PAGE_PARQ"
  --output-json "$TOPDOCS_JSON"
  --run-id "$RUN_ID"
  --topk-docs "$TOPK_DOCS"
  --all-pages-per-doc
  --doc-rank-col "$DOC_RANK_COL"
  --doc-score-col "$DOC_SCORE_COL"
)
if [[ -n "$MAX_QIDS" && "$MAX_QIDS" != "all" ]]; then
  BUILD_ARGS+=(--max-qids "$MAX_QIDS")
fi
conda run -n "$CONDA_ENV" python examples/build_topdocs_doc_page_pool.py "${BUILD_ARGS[@]}"

echo "[2/4] Compute topk-candidates from built pool"
if [[ -z "${TOPK_CANDIDATES:-}" || "${TOPK_CANDIDATES}" == "auto" ]]; then
  TOPK_CANDIDATES="$(TOPDOCS_JSON="$TOPDOCS_JSON" conda run -n "$CONDA_ENV" python - <<'PY'
import json, os
p = os.environ["TOPDOCS_JSON"]
j = json.load(open(p))
mx = 0
for rows in j.get("top_pages", {}).values():
    if isinstance(rows, list):
        mx = max(mx, len(rows))
print(mx)
PY
  )"
fi
echo "TOPK_CANDIDATES=${TOPK_CANDIDATES}"

echo "[3/4] Rerank docs using all pages, then reorder docs"
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
  --summary-ks "$SUMMARY_KS"
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

echo "[4/4] Summary (retrieval_edges doc order vs fresh reranked docs)"
export RERANK_JSON SUMMARY_KS
conda run -n "$CONDA_ENV" python - <<'PY'
import json, os
p = os.environ["RERANK_JSON"]
ks = [int(x.strip()) for x in os.environ["SUMMARY_KS"].split(",") if x.strip()]
j = json.load(open(p))
sm = j.get("summary_metrics", {})

print("output:", p)
print("qids_with_gold_targets:", sm.get("qids_with_gold_targets"))
print("improved_count:", sm.get("improved_count"))
print("worsened_count:", sm.get("worsened_count"))
print("mrr_before:", sm.get("mrr_before"))
print("mrr_after :", sm.get("mrr_after"))
for k in ks:
    print(f"recall@{k}_before:", sm.get(f"recall@{k}_before"))
    print(f"recall@{k}_after :", sm.get(f"recall@{k}_after"))
PY
