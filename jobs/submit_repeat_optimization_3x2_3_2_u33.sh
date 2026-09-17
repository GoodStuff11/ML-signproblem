#!/bin/bash
#SBATCH --job-name=repeat_opt_3x2_33
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/repeat_opt_3x2_u33-%A_%a.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/repeat_opt_3x2_u33-%A_%a.err
#SBATCH --mem=8g
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00
#SBATCH --partition=kim
#SBATCH --array=0-39

# 100 independent random-restart optimizations of the N=(3,2) 3x2 Hubbard system at
# U = 8.0 (u index 33; load_ED_data prepends U=0 so U=8.0 is the 33rd entry), for each of the four (ansatz, loss) combinations:
#   trotter/overlap, trotter/energy, exact/overlap, exact/energy
#
# The 4 x 100 = 400 runs are split over 40 array tasks: 4 configurations x 10 blocks
# of 10 runs each. Run index r always uses seed = BASE_SEED + r regardless of which
# task executes it, so the per-task CSVs concatenate into one consistent table.

set -euo pipefail

FOLDER="N=(3, 2)_3x2"
U_IDX=33
RUNS=100
BLOCK=10
MAXITERS=1000
BASE_SEED=20260915
OUTDIR=/home/jek354/research/ML-signproblem/experimenting/ed/repeat_optimization_3x2_3_2_u33

mkdir -p "$OUTDIR"

CONFIGS=("trotter overlap" "trotter energy" "exact overlap" "exact energy")
NBLOCKS=$((RUNS / BLOCK))

CFG_IDX=$((SLURM_ARRAY_TASK_ID / NBLOCKS))
BLK_IDX=$((SLURM_ARRAY_TASK_ID % NBLOCKS))

read -r ANSATZ LOSS <<< "${CONFIGS[$CFG_IDX]}"
RUN_START=$((BLK_IDX * BLOCK + 1))
RUN_END=$((RUN_START + BLOCK - 1))

OUT="$OUTDIR/${ANSATZ}_${LOSS}_runs_${RUN_START}_${RUN_END}.csv"
echo "task $SLURM_ARRAY_TASK_ID: ansatz=$ANSATZ loss=$LOSS runs=$RUN_START..$RUN_END -> $OUT"

cd /home/jek354/research/ML-signproblem/experimenting/ed
julia --project=.. run_repeat_optimization_experiment.jl "$FOLDER" "$U_IDX" \
    --runs="$RUNS" --run_start="$RUN_START" --run_end="$RUN_END" \
    --maxiters="$MAXITERS" --initialization_samples=1 \
    --custom_ref_state=slater --antihermitian --num_exponentials=1 \
    --ansatz="$ANSATZ" --loss="$LOSS" \
    --seed="$BASE_SEED" --out="$OUT"
