#!/bin/bash
#SBATCH --job-name=ED_4x3_54_tracked_sector2
#SBATCH -p kim
#SBATCH --time=7-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=NONE
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/ED_4x3_54_tracked_sector2.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/ED_4x3_54_tracked_sector2.err
# Very excited 4x3 (5,4) target followed continuously in u: sector 2 (q=(0,2)), the sector that
# load_ED_data selects for this system's ground state (the paper's optimization runs; q=(0,1) is
# its exactly degenerate mirror). Seeded at u=0.25 with the smallest non-zero |E| eigenstate,
# tracked with substeps <= 0.05 (overlap >= 0.95 per step), stored ONLY at u = 0.25, 0.5, ..., 15
# and only this state (m=1). Same procedure as track_4x3_44_sector0.sh.
set -euo pipefail
cd "/home/jek354/research/UCC_Hubbard/.claude/worktrees/excited-states/XDiag ED pipeline"
export ED_DATA_DIR="/home/jek354/research/data/new_data/data_h5_fixed/N=(5, 4)_4x3_excited_tracked"
mkdir -p "$ED_DATA_DIR/Raw"
/usr/bin/time -v julia --project=.. TrackHubbardED_XDiag.jl 4,3 5 4 2 --sigma 0
julia --project=.. ExtractHubbardBasis.jl 4,3 5 4 "$ED_DATA_DIR"
python3 consolidateConvertHubbardED_XDiag.py 4,3 5 4 1 --sigma 0 --sectors 2 --tracked --force-complex
