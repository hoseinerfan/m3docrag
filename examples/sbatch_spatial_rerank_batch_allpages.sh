#!/usr/bin/env bash
#SBATCH --job-name=spatial_allpages
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# Usage:
#   sbatch examples/sbatch_spatial_rerank_batch_allpages.sh
#   RUN_SCOPE=full sbatch examples/sbatch_spatial_rerank_batch_allpages.sh
#   MAX_QIDS=200 sbatch examples/sbatch_spatial_rerank_batch_allpages.sh

PROJECT_DIR="${PROJECT_DIR:-/mmfs1/scratch/jacks.local/aerfanshekooh/newproject_2}"
cd "$PROJECT_DIR"

mkdir -p outputs/logs outputs/spatial_batch_allpages

RUN_SCOPE="${RUN_SCOPE:-sanity}" # sanity | full
if [[ "$RUN_SCOPE" == "full" ]]; then
  MAX_QIDS_DEFAULT="all"
else
  MAX_QIDS_DEFAULT="50"
fi

CONDA_ENV="${CONDA_ENV:-my_cuda_env}"
RUN_ID="${RUN_ID:-baseline_ret1000}"
TOPK_DOCS="${TOPK_DOCS:-1000}"
MAX_QIDS="${MAX_QIDS:-$MAX_QIDS_DEFAULT}"
DOC_RANK_COL="${DOC_RANK_COL:-faiss_rank}"
DOC_SCORE_COL="${DOC_SCORE_COL:-faiss_score}"
RERANK_MODE="${RERANK_MODE:-per_doc_reorder}"
COHERENCE_LAMBDA="${COHERENCE_LAMBDA:-0.15}"
SCORE_MODE="${SCORE_MODE:-base_plus_coherence}"
SUMMARY_KS="${SUMMARY_KS:-1,2,4,10,50,100,500}"
QIDS_FILE="${QIDS_FILE:-}"

LOG="outputs/logs/batch_allpages_${RUN_ID}_${RUN_SCOPE}_job${SLURM_JOB_ID:-manual}.log"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "RUN_SCOPE=$RUN_SCOPE MAX_QIDS=$MAX_QIDS"
echo "RUN_ID=$RUN_ID TOPK_DOCS=$TOPK_DOCS DOC_RANK_COL=$DOC_RANK_COL DOC_SCORE_COL=$DOC_SCORE_COL"
echo "RERANK_MODE=$RERANK_MODE SCORE_MODE=$SCORE_MODE COHERENCE_LAMBDA=$COHERENCE_LAMBDA SUMMARY_KS=$SUMMARY_KS"
echo "QIDS_FILE=${QIDS_FILE:-<none>}"
echo "LOG=$LOG"

CONDA_ENV="$CONDA_ENV" \
RUN_ID="$RUN_ID" \
TOPK_DOCS="$TOPK_DOCS" \
MAX_QIDS="$MAX_QIDS" \
DOC_RANK_COL="$DOC_RANK_COL" \
DOC_SCORE_COL="$DOC_SCORE_COL" \
RERANK_MODE="$RERANK_MODE" \
COHERENCE_LAMBDA="$COHERENCE_LAMBDA" \
SCORE_MODE="$SCORE_MODE" \
SUMMARY_KS="$SUMMARY_KS" \
QIDS_FILE="$QIDS_FILE" \
bash examples/run_spatial_rerank_batch_allpages_cluster.sh 2>&1 | tee "$LOG"
