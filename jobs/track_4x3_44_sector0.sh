#!/bin/bash
#SBATCH --job-name=ED_4x3_44_tracked_sector0
#SBATCH -p kim
#SBATCH --time=7-00:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=NONE
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/ED_4x3_44_tracked_sector0.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/ED_4x3_44_tracked_sector0.err
# Very excited 4x3 (4,4) target that is followed continuously in u: sector 0 (q=(0,0)),
# seeded at u=0.25 with the smallest non-zero |E| eigenstate, tracked with substeps <= 0.05
# (overlap >= 0.95 per step), stored ONLY at u = 0.25, 0.5, ..., 15 and only this state (m=1).
# Pipeline: UCC_Hubbard TrackHubbardED_XDiag.jl -> ExtractHubbardBasis.jl -> consolidation --tracked.
set -euo pipefail
cd "/home/jek354/research/UCC_Hubbard/.claude/worktrees/excited-states/XDiag ED pipeline"
export ED_DATA_DIR="/home/jek354/research/data/new_data/data_h5_fixed/N=(4, 4)_4x3_excited_tracked"
mkdir -p "$ED_DATA_DIR/Raw"
/usr/bin/time -v julia --project=.. TrackHubbardED_XDiag.jl 4,3 4 4 0 --sigma 0
julia --project=.. ExtractHubbardBasis.jl 4,3 4 4 "$ED_DATA_DIR"
python3 consolidateConvertHubbardED_XDiag.py 4,3 4 4 1 --sigma 0 --sectors 0 --tracked --force-complex
