#!/bin/bash
# Leave-one-out DMR discovery on SLURM, one fold per array task.
# Each task re-runs Metilene on the 41 training patients and predicts the
# held-out patient: run_loocv.py --start i --end i+1.
#
# Fold 0 also writes metilene_full_matrix_filtered.bed, which every later fold
# reads, so it runs on its own first (the default array below):
#   sbatch submit_loocv_array.sh
#   sbatch --array=1-41%5 submit_loocv_array.sh    # after fold 0 has finished
#
#SBATCH --job-name=brand_dmr_loocv
#SBATCH --array=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=brand_loocv_%A_%a.out
#SBATCH --error=brand_loocv_%A_%a.err

# Load Python 3.14 with the packages in requirements.txt, for example:
# module load python
# source ~/brand_env/bin/activate

cd "$SLURM_SUBMIT_DIR"
python run_loocv.py --start "$SLURM_ARRAY_TASK_ID" --end "$((SLURM_ARRAY_TASK_ID + 1))"
