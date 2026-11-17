#!/bin/bash
#SBATCH --job-name=ED_4x3_54_level4_sector2
#SBATCH -p kim
#SBATCH --time=7-00:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=NONE
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/ED_4x3_54_level4_sector2.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/ED_4x3_54_level4_sector2.err
# 4x3 (5,4), sector 2 (q=(0,2), the ground-state sector of the existing data, momentum_sector.toml):
# the FOURTH EXCITED level (4th distinct energy above the ground level) at U=0.25, followed
# continuously to U=15 by overlap (substeps <= 0.05, overlap >= 0.95), each step using the 40
# lowest eigenpairs (KrylovKit Lanczos); the state must stay within the 35 lowest.
# Stored ONLY at U = 0.25, 0.5, ..., 15 and only this state (m=1).
set -euo pipefail
cd "/home/jek354/research/UCC_Hubbard/.claude/worktrees/excited-states/XDiag ED pipeline"
export ED_DATA_DIR="/home/jek354/research/data/new_data/data_h5_fixed/N=(5, 4)_4x3_excited_level4"
mkdir -p "$ED_DATA_DIR/Raw"
/usr/bin/time -v julia --project=.. TrackHubbardED_XDiag.jl 4,3 5 4 2 --level 4 --nlowest 40
julia --project=.. ExtractHubbardBasis.jl 4,3 5 4 "$ED_DATA_DIR"
python3 consolidateConvertHubbardED_XDiag.py 4,3 5 4 1 --level 4 --sectors 2 --tracked --force-complex
