#!/bin/bash
#SBATCH --job-name=barren_study_energy
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/barren_study_energy_%A_%a.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/barren_study_energy_%A_%a.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --partition=aimi
#SBATCH --array=0-19

# Generates the energy-loss counterparts of the existing overlap-loss "barren_study"
# Trotter runs, for the 10 dimH-sweep systems used by plot_dimH_and_barren_analysis.jl,
# at num_exponentials in {1 (UCC), 4 (VQE-like, P_vqe)} -- the two depths figure (c)
# (barren_plateau_vs_dimH) needs. Single U point (u_idx=33, U=8.0), matching the
# existing overlap-loss barren_study files (confirmed via their shared.jld2
# "instructions" dict: u_range=33:33). --antihermitian/--custom_ref_state=slater match
# REF_STATE_ARG/ANTIHERMITIAN in plot_dimH_and_barren_analysis.jl; --run_label=barren_study
# keeps this a freshly-initialized run under its own filename, matching the existing
# overlap-loss files' naming. maxiters left at the script default (100), matching the
# existing files' recorded convergence_info. initialization_samples=10 is hardcoded in
# run_trotter_scan_optimization.jl itself (not CLI-configurable), so it automatically
# matches the existing overlap-loss runs' 10 initial-gradient samples.

SYSTEMS=(
  "N=(2, 2)_3x2"
  "N=(3, 2)_3x2"
  "N=(3, 2)_3x3"
  "N=(3, 3)_3x2"
  "N=(3, 3)_3x3"
  "N=(3, 3)_4x2"
  "N=(4, 3)_3x3"
  "N=(4, 4)_3x3"
  "N=(5, 4)_3x3"
  "N=(5, 4)_4x3"
)
PVALS=(1 4)

n_pvals=${#PVALS[@]}
sys_idx=$(( SLURM_ARRAY_TASK_ID / n_pvals ))
p_idx=$(( SLURM_ARRAY_TASK_ID % n_pvals ))

FOLDER="${SYSTEMS[$sys_idx]}"
P="${PVALS[$p_idx]}"

echo "=== System: $FOLDER, num_exponentials=$P ==="

cd /home/jek354/research/ML-signproblem/experimenting/ed
julia --threads=$SLURM_CPUS_PER_TASK --project=.. run_trotter_scan_optimization.jl \
  "$FOLDER" 33 33 \
  --loss=energy \
  --num_exponentials=$P \
  --antihermitian \
  --custom_ref_state=slater \
  --run_label=barren_study \
  --use_gpu=false
