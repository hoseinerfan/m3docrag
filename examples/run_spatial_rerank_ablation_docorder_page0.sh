#!/usr/bin/env bash
set -euo pipefail

# Ablation: keep top-K docs from retrieval_edges parquet and force page_idx=0 for each doc.
# This avoids page-pool expansion and gives a clean "doc-order + first-page" baseline.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_ENV="${CONDA_ENV:-my_cuda_env}"
DATASET_ROOT="${DATASET_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/datasets/m3docvqa_dev_ret1000_rerankexp_v1}"
DOC_PARQ="${DOC_PARQ:-$DATASET_ROOT/parquet/retrieval_edges.parquet}"

QID="${QID:-e783cba0b3df36372d11823e378e5437}"
QUERY="${QUERY:-Which completely bald person who wears thick glasses is among the members of LGBT billionaires?}"
RUN_ID="${RUN_ID:-baseline_ret1000}"
TOPK_DOCS="${TOPK_DOCS:-1000}"

GOLD_DOC_ID="${GOLD_DOC_ID:-d57e56eff064047af5a6ef074a570956}"
GOLD_PAGE_IDX="${GOLD_PAGE_IDX:-0}"

EMB="${EMB:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali-v1.2_m3-docvqa_dev}"
MMQA="${MMQA:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/data/m3-docvqa/multimodalqa/MMQA_dev.jsonl}"
BACKBONE="${BACKBONE:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/models/colpaligemma-3b-pt-448-base}"
ADAPTER="${ADAPTER:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/models/colpali-v1.2}"

COHERENCE_LAMBDA="${COHERENCE_LAMBDA:-0.15}"
RERANK_MODE="${RERANK_MODE:-per_doc_page}"

OUTDIR="${OUTDIR:-outputs}"
mkdir -p "$OUTDIR"
TOPDOCS_JSON="${TOPDOCS_JSON:-$OUTDIR/e783_docorder_page0_topdocs.json}"
OUT_JSON="${OUT_JSON:-$OUTDIR/e783_docorder_page0_reranked.json}"
DBG_JSON="${DBG_JSON:-$OUTDIR/e783_docorder_page0_debug.json}"

echo "[1/3] Build topdocs JSON (top-${TOPK_DOCS} docs, page_idx=0)"
export DOC_PARQ QID RUN_ID TOPK_DOCS TOPDOCS_JSON GOLD_DOC_ID
conda run -n "$CONDA_ENV" python - <<'PY'
import json
import os
import pyarrow.dataset as ds

parq = os.environ["DOC_PARQ"]
qid = os.environ["QID"]
run_id = os.environ["RUN_ID"]
topk = int(os.environ["TOPK_DOCS"])
out = os.environ["TOPDOCS_JSON"]
gold = os.environ["GOLD_DOC_ID"]

d = ds.dataset(parq, format="parquet")
cols = list(d.schema.names)
need = ["qid", "doc_id"]
for c in ["run_id", "maxsim_rank", "maxsim_score", "faiss_rank", "faiss_score"]:
    if c in cols:
        need.append(c)

rows = d.to_table(
    filter=(ds.field("qid") == qid) & (ds.field("run_id") == run_id),
    columns=need,
).to_pylist()

def _to_int(x, default=10**9):
    try:
        return int(x)
    except Exception:
        return default

def _to_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

rows = sorted(
    rows,
    key=lambda r: (
        _to_int(r.get("maxsim_rank")),
        _to_int(r.get("faiss_rank")),
        str(r.get("doc_id")),
    ),
)

out_rows = []
seen = set()
for r in rows:
    doc = str(r.get("doc_id"))
    if not doc or doc in seen:
        continue
    seen.add(doc)
    score = _to_float(r.get("maxsim_score"))
    if score is None:
        score = _to_float(r.get("faiss_score"))
    if score is None:
        score = -float(_to_int(r.get("maxsim_rank"), default=10**9))
    out_rows.append({"doc_id": doc, "page_idx": 0, "score": float(score)})
    if len(out_rows) >= topk:
        break

with open(out, "w") as f:
    json.dump({"top_pages": {qid: out_rows}}, f, indent=2)

gold_rank = next((i + 1 for i, r in enumerate(out_rows) if r["doc_id"] == gold), None)
print("saved:", out)
print("n_candidates:", len(out_rows))
print("gold_doc_rank_in_input:", gold_rank)
PY

echo "[2/3] Run reranker"
conda run -n "$CONDA_ENV" python examples/rerank_topdocs_spatial_coherence.py \
  --topdocs-json "$TOPDOCS_JSON" \
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
  --topk-candidates "$TOPK_DOCS" \
  --save-top-k "$TOPK_DOCS" \
  --coherence-lambda "$COHERENCE_LAMBDA" \
  --rerank-mode "$RERANK_MODE"

echo "[3/3] Done"
echo "TOPDOCS_JSON=$TOPDOCS_JSON"
echo "OUT_JSON=$OUT_JSON"
