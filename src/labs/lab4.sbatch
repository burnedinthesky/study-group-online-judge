#!/bin/bash -l
#SBATCH --job-name=gpt2-eval
#SBATCH --partition=dev
#SBATCH --account=ACD115198
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=H200:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00

set -euo pipefail

module purge
module load cuda/13.0

cd "$JUDGE_SUBMISSION_DIR"
uv sync --no-sources --no-dev
export PYTHONPATH="$JUDGE_TRUSTED_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
uv run --no-sync python -u -m judge.run_task \
  --task "$JUDGE_TASK_ID" \
  --submission "$JUDGE_SUBMISSION_DIR" \
  --output "$JUDGE_OUTPUT_DIR/result.json"
