#!/bin/bash
#SBATCH --job-name=ED_4x3_44_sigma0_consolidate
#SBATCH -p kim
#SBATCH --time=7-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=NONE
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/ED_4x3_44_sigma0_consolidate.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/ED_4x3_44_sigma0_consolidate.err
# Stages 2+3 of UCC_Hubbard's XDiag pipeline for the 4x3 (4,4) smallest-|E| run
# (stage 1: sbatch submit_HubbardED_XDiag.sh 4,3 4 4 8 --sigma 0, array job 698368):
# basis extraction, then orbit->Slater consolidation into the excited-state data folder.
# Submit with: sbatch --dependency=afterok:<array job id> jobs/consolidate_4x3_44_sigma0.sh
set -euo pipefail
cd "/home/jek354/research/UCC_Hubbard/.claude/worktrees/excited-states/XDiag ED pipeline"
export ED_DATA_DIR="/home/jek354/research/data/new_data/data_h5_fixed/N=(4, 4)_4x3_excited_sigma0"
julia --project=.. ExtractHubbardBasis.jl 4,3 4 4 "$ED_DATA_DIR"
python3 consolidateConvertHubbardED_XDiag.py 4,3 4 4 8 --sigma 0 --force-complex
