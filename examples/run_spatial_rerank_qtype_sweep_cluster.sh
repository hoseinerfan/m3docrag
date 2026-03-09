#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_ENV="${CONDA_ENV:-my_cuda_env}"

DATASET_ROOT="${DATASET_ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/datasets/m3docvqa_dev_ret1000_rerankexp_v1}"
QUESTIONS_PARQ="${QUESTIONS_PARQ:-$DATASET_ROOT/parquet/questions.parquet}"

RUN_ID="${RUN_ID:-baseline_ret1000}"
TOPK_DOCS="${TOPK_DOCS:-1000}"
DOC_RANK_COL="${DOC_RANK_COL:-faiss_rank}"
DOC_SCORE_COL="${DOC_SCORE_COL:-faiss_score}"
RERANK_MODE="${RERANK_MODE:-per_doc_reorder}"
SCORE_MODE="${SCORE_MODE:-base_plus_coherence}"
COHERENCE_LAMBDA="${COHERENCE_LAMBDA:-0.15}"
EVIDENCE_MU="${EVIDENCE_MU:-0.05}"
MIN_SUPPORT_COUNT="${MIN_SUPPORT_COUNT:-3}"
MIN_SUPPORT_DIVERSITY="${MIN_SUPPORT_DIVERSITY:-2}"
SUMMARY_KS="${SUMMARY_KS:-1,2,4,10,50,100,500}"

N_PER_TYPE="${N_PER_TYPE:-20}"
SAMPLE_SEED="${SAMPLE_SEED:-13}"
TYPE_REGEX="${TYPE_REGEX:-}"
MAX_TYPES="${MAX_TYPES:-all}"

OUTROOT="${OUTROOT:-outputs/spatial_qtype_sweep}"
QIDS_DIR="$OUTROOT/qids"
RUNS_DIR="$OUTROOT/runs"
REPORTS_DIR="$OUTROOT/reports"
LOGS_DIR="$OUTROOT/logs"

mkdir -p "$QIDS_DIR" "$RUNS_DIR" "$REPORTS_DIR" "$LOGS_DIR"

MANIFEST_JSON="$REPORTS_DIR/qtype_manifest.json"
MANIFEST_TSV="$REPORTS_DIR/qtype_manifest.tsv"
SUMMARY_TSV="$REPORTS_DIR/qtype_sweep_summary.tsv"

echo "QUESTIONS_PARQ=$QUESTIONS_PARQ"
echo "RUN_ID=$RUN_ID TOPK_DOCS=$TOPK_DOCS"
echo "RERANK_MODE=$RERANK_MODE SCORE_MODE=$SCORE_MODE COHERENCE_LAMBDA=$COHERENCE_LAMBDA"
echo "N_PER_TYPE=$N_PER_TYPE SAMPLE_SEED=$SAMPLE_SEED TYPE_REGEX=${TYPE_REGEX:-<none>} MAX_TYPES=$MAX_TYPES"
echo "OUTROOT=$OUTROOT"

echo "[1/3] Build per-type qid files"
export QUESTIONS_PARQ QIDS_DIR MANIFEST_JSON MANIFEST_TSV N_PER_TYPE SAMPLE_SEED TYPE_REGEX MAX_TYPES
STEP1_PY="$OUTROOT/_qtype_step1_manifest.py"
cat > "$STEP1_PY" <<'PY'
import csv
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import pyarrow.dataset as ds

questions_parq = os.environ["QUESTIONS_PARQ"]
qids_dir = Path(os.environ["QIDS_DIR"])
manifest_json = Path(os.environ["MANIFEST_JSON"])
manifest_tsv = Path(os.environ["MANIFEST_TSV"])
n_per_type = int(os.environ["N_PER_TYPE"])
sample_seed = str(os.environ["SAMPLE_SEED"])
type_regex = os.environ.get("TYPE_REGEX", "").strip()
max_types_text = os.environ.get("MAX_TYPES", "all").strip().lower()
max_types = None if max_types_text in {"", "all"} else int(max_types_text)

qids_dir.mkdir(parents=True, exist_ok=True)

d = ds.dataset(questions_parq, format="parquet")
cols = list(d.schema.names)
qid_col = next((c for c in ["qid", "query_id", "question_id", "id"] if c in cols), None)
type_col = next(
    (c for c in ["question_type", "q_type", "questionType", "type", "question_category"] if c in cols),
    None,
)
if qid_col is None or type_col is None:
    raise RuntimeError(f"Could not infer qid/type columns. available={cols}")

rows = d.to_table(columns=[qid_col, type_col]).to_pylist()
grouped = defaultdict(set)
for r in rows:
    qid = r.get(qid_col)
    qtype = r.get(type_col)
    if qid is None:
        continue
    qtype_str = str(qtype).strip() if qtype is not None else "UNKNOWN"
    if not qtype_str:
        qtype_str = "UNKNOWN"
    grouped[qtype_str].add(str(qid))

type_pat = re.compile(type_regex) if type_regex else None

def slugify(x: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", x.strip()).strip("_").lower()
    return s or "unknown"

def key_for_sample(qtype: str, qid: str) -> str:
    return hashlib.md5(f"{sample_seed}|{qtype}|{qid}".encode("utf-8")).hexdigest()

manifest = []
for qtype in sorted(grouped.keys()):
    if type_pat and not type_pat.search(qtype):
        continue
    all_qids = sorted(grouped[qtype])
    selected = sorted(all_qids, key=lambda q: key_for_sample(qtype, q))[:n_per_type]
    slug = slugify(qtype)
    qf = qids_dir / f"{slug}.txt"
    with qf.open("w") as f:
        for q in selected:
            f.write(q + "\n")
    manifest.append(
        {
            "question_type": qtype,
            "slug": slug,
            "n_total_qids": len(all_qids),
            "n_selected_qids": len(selected),
            "qids_file": str(qf),
        }
    )

manifest.sort(key=lambda x: (-x["n_total_qids"], x["question_type"]))
if max_types is not None:
    manifest = manifest[: max_types]

with manifest_json.open("w") as f:
    json.dump(
        {
            "questions_parquet": questions_parq,
            "qid_col": qid_col,
            "type_col": type_col,
            "n_per_type": n_per_type,
            "sample_seed": sample_seed,
            "type_regex": type_regex,
            "max_types": max_types,
            "types": manifest,
        },
        f,
        indent=2,
    )

with manifest_tsv.open("w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=["question_type", "slug", "n_total_qids", "n_selected_qids", "qids_file"],
        delimiter="\t",
    )
    w.writeheader()
    for r in manifest:
        w.writerow(r)

print("saved manifest:", manifest_json)
print("saved manifest tsv:", manifest_tsv)
print("n_types:", len(manifest))
for r in manifest:
    print(f"{r['question_type']}: total={r['n_total_qids']} selected={r['n_selected_qids']}")
PY
conda run -n "$CONDA_ENV" python "$STEP1_PY"

echo "[2/3] Run rerank sweep by question type"
if [[ ! -s "$MANIFEST_TSV" ]]; then
  echo "Manifest TSV is empty: $MANIFEST_TSV" >&2
  exit 1
fi

tail -n +2 "$MANIFEST_TSV" | while IFS=$'\t' read -r QTYPE SLUG N_TOTAL N_SELECTED QFILE; do
  QTYPE="${QTYPE//$'\r'/}"
  SLUG="${SLUG//$'\r'/}"
  N_TOTAL="${N_TOTAL//$'\r'/}"
  N_SELECTED="${N_SELECTED//$'\r'/}"
  QFILE="${QFILE//$'\r'/}"
  if [[ -z "$SLUG" || -z "$QFILE" ]]; then
    continue
  fi
  if [[ "${N_SELECTED:-0}" -le 0 ]]; then
    echo "Skipping $QTYPE (no qids selected)"
    continue
  fi

  OUTDIR="$RUNS_DIR/$SLUG"
  mkdir -p "$OUTDIR"
  LOG="$LOGS_DIR/${SLUG}.log"

  echo "-----"
  echo "Type: $QTYPE"
  echo "slug: $SLUG"
  echo "n_total=$N_TOTAL n_selected=$N_SELECTED"
  echo "qids_file=$QFILE"
  echo "outdir=$OUTDIR"

  CONDA_ENV="$CONDA_ENV" \
  RUN_ID="$RUN_ID" \
  TOPK_DOCS="$TOPK_DOCS" \
  MAX_QIDS=all \
  QIDS_FILE="$QFILE" \
  DOC_RANK_COL="$DOC_RANK_COL" \
  DOC_SCORE_COL="$DOC_SCORE_COL" \
  RERANK_MODE="$RERANK_MODE" \
  SCORE_MODE="$SCORE_MODE" \
  COHERENCE_LAMBDA="$COHERENCE_LAMBDA" \
  EVIDENCE_MU="$EVIDENCE_MU" \
  MIN_SUPPORT_COUNT="$MIN_SUPPORT_COUNT" \
  MIN_SUPPORT_DIVERSITY="$MIN_SUPPORT_DIVERSITY" \
  SUMMARY_KS="$SUMMARY_KS" \
  OUTDIR="$OUTDIR" \
  bash examples/run_spatial_rerank_batch_allpages_cluster.sh 2>&1 | tee "$LOG"
done

echo "[3/3] Build summary table"
export MANIFEST_TSV RUNS_DIR SUMMARY_TSV
STEP3_PY="$OUTROOT/_qtype_step3_summary.py"
cat > "$STEP3_PY" <<'PY'
import csv
import json
import os
from pathlib import Path

manifest_tsv = Path(os.environ["MANIFEST_TSV"])
runs_dir = Path(os.environ["RUNS_DIR"])
summary_tsv = Path(os.environ["SUMMARY_TSV"])

rows = []
with manifest_tsv.open() as f:
    r = csv.DictReader(f, delimiter="\t")
    for m in r:
        slug = m["slug"]
        out_json = runs_dir / slug / "reranked_doc1000_allpages_baseline_ret1000.json"
        sm = {}
        if out_json.exists():
            try:
                sm = json.load(open(out_json)).get("summary_metrics", {}) or {}
            except Exception:
                sm = {}

        rec = {
            "question_type": m["question_type"],
            "slug": slug,
            "n_total_qids": m["n_total_qids"],
            "n_selected_qids": m["n_selected_qids"],
            "qids_with_gold_targets": sm.get("qids_with_gold_targets"),
            "improved_count": sm.get("improved_count"),
            "worsened_count": sm.get("worsened_count"),
            "recall@1_before": sm.get("recall@1_before"),
            "recall@1_after": sm.get("recall@1_after"),
            "recall@2_before": sm.get("recall@2_before"),
            "recall@2_after": sm.get("recall@2_after"),
            "recall@4_before": sm.get("recall@4_before"),
            "recall@4_after": sm.get("recall@4_after"),
            "recall@10_before": sm.get("recall@10_before"),
            "recall@10_after": sm.get("recall@10_after"),
            "mrr_before": sm.get("mrr_before"),
            "mrr_after": sm.get("mrr_after"),
            "delta_recall@10": None,
            "delta_mrr": None,
            "result_json": str(out_json),
        }
        if rec["recall@10_before"] is not None and rec["recall@10_after"] is not None:
            rec["delta_recall@10"] = rec["recall@10_after"] - rec["recall@10_before"]
        if rec["mrr_before"] is not None and rec["mrr_after"] is not None:
            rec["delta_mrr"] = rec["mrr_after"] - rec["mrr_before"]
        rows.append(rec)

rows.sort(
    key=lambda x: (
        -9999 if x["delta_mrr"] is None else -x["delta_mrr"],
        x["question_type"],
    )
)

headers = [
    "question_type",
    "n_total_qids",
    "n_selected_qids",
    "qids_with_gold_targets",
    "improved_count",
    "worsened_count",
    "recall@1_before",
    "recall@1_after",
    "recall@2_before",
    "recall@2_after",
    "recall@4_before",
    "recall@4_after",
    "recall@10_before",
    "recall@10_after",
    "delta_recall@10",
    "mrr_before",
    "mrr_after",
    "delta_mrr",
    "result_json",
]

summary_tsv.parent.mkdir(parents=True, exist_ok=True)
with summary_tsv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
    w.writeheader()
    for rec in rows:
        w.writerow(rec)

print("saved summary:", summary_tsv)
print("\t".join(["question_type", "n_selected_qids", "improved_count", "worsened_count", "delta_recall@10", "delta_mrr"]))
for rec in rows:
    print(
        "\t".join(
            [
                str(rec["question_type"]),
                str(rec["n_selected_qids"]),
                str(rec["improved_count"]),
                str(rec["worsened_count"]),
                str(rec["delta_recall@10"]),
                str(rec["delta_mrr"]),
            ]
        )
    )
PY
conda run -n "$CONDA_ENV" python "$STEP3_PY"

echo "Done. Summary table: $SUMMARY_TSV"
