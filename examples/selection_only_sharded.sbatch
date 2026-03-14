#!/bin/bash
#SBATCH -J m3doc_sel_shard
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --array=0-4

set -euo pipefail

# Usage:
#   sbatch --array=0-4 selection_only_sharded.sbatch <qid_file> <out_dir> [page_summaries_jsonl]
#
# Notes:
# - Override ROOT/PARQ/QRELS via environment variables if needed.
# - --array controls number of shards. Example: --array=0-7 => 8 shards.

ROOT="${ROOT:-/mmfs1/scratch/jacks.local/aerfanshekooh/newproject}"
PARQ="${PARQ:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/datasets/m3docvqa_dev_ret1000_rerankexp_v1/parquet/retrieval_edges.parquet}"
QRELS="${QRELS:-/mmfs1/scratch/jacks.local/aerfanshekooh/custom/datasets/m3docvqa_dev_ret1000_rerankexp_v1/parquet/qrels.parquet}"

if [ "$#" -lt 2 ]; then
  echo "Usage: sbatch --array=0-4 selection_only_sharded.sbatch <qid_file> <out_dir> [page_summaries_jsonl]"
  exit 2
fi

QID_FILE="$1"
OUT="$2"
SUMM_FILE="${3:-$ROOT/outputs/page_summaries_dev_full_allpages.jsonl}"

SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
N_SHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"

mkdir -p "$OUT" "$OUT/shards" "$OUT/logs"

eval "$(conda shell.bash hook)"
conda activate "$ROOT/.conda/m3docrag"
cd "$ROOT/m3docrag"
unset TRANSFORMERS_CACHE

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NA} SHARD=${SHARD_ID}/${N_SHARDS}"
echo "ROOT=$ROOT"
echo "QID_FILE=$QID_FILE"
echo "OUT=$OUT"
echo "SUMM_FILE=$SUMM_FILE"

SHARD_QID_FILE="$OUT/shards/qids_shard_${SHARD_ID}.txt"

python - <<PY
from pathlib import Path

qid_file = Path(r"""$QID_FILE""")
out_file = Path(r"""$SHARD_QID_FILE""")
shard_id = int("$SHARD_ID")
n_shards = int("$N_SHARDS")

qids = [x.strip() for x in qid_file.read_text().splitlines() if x.strip()]
picked = [q for i, q in enumerate(qids) if i % n_shards == shard_id]

out_file.parent.mkdir(parents=True, exist_ok=True)
out_file.write_text("\n".join(picked) + ("\n" if picked else ""))
print(f"total_qids={len(qids)} picked_for_shard={len(picked)} shard_id={shard_id} n_shards={n_shards}")
PY

if [ ! -s "$SHARD_QID_FILE" ]; then
  echo "No QIDs assigned to shard $SHARD_ID. Exiting."
  exit 0
fi

RESULTS_SHARD="$OUT/results_shard_${SHARD_ID}.jsonl"
SUMMARY_SHARD="$OUT/summary_shard_${SHARD_ID}.json"
RUNLOG_SHARD="$OUT/logs/run_shard_${SHARD_ID}.log"

python "$ROOT/m3docrag/examples/run_agent_m3docvqa_subset.py" \
  --split dev \
  --qid-file "$SHARD_QID_FILE" \
  --doc-pool-source retrieval-parquet \
  --retrieval-edges-parquet "$PARQ" \
  --retrieval-topk-docs 1000 \
  --retrieval-qid-col qid \
  --retrieval-doc-col doc_id \
  --retrieval-rank-col maxsim_rank \
  --retrieval-score-col maxsim_score \
  --rank-gold-source qrels-parquet \
  --qrels-parquet "$QRELS" \
  --selection-only \
  --selection-max-hop-queries 4 \
  --selection-root-topk-docs 10 \
  --selection-variant-topk-docs 10 \
  --selection-max-variant-queries 1 \
  --selection-summary-full-scan \
  --selection-summary-full-scan-topk 1000 \
  --page-summaries-file "$SUMM_FILE" \
  --output-jsonl "$RESULTS_SHARD" \
  --summary-json "$SUMMARY_SHARD" \
  2>&1 | tee "$RUNLOG_SHARD"

echo "Shard completed: $SHARD_ID"
echo "results=$RESULTS_SHARD"
echo "summary=$SUMMARY_SHARD"
echo "runlog=$RUNLOG_SHARD"
