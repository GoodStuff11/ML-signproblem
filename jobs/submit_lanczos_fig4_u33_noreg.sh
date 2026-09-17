#!/bin/bash
#SBATCH --job-name=lanczos_fig4_noreg
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/lanczos_fig4_noreg-%A_%a.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/lanczos_fig4_noreg-%A_%a.err
#SBATCH --mem=60g
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --partition=kim
#SBATCH --array=0-9

# Re-run the exact-exponential (UCC) optimization that feeds figure 4's panels (a) and
# (a-alt), with the L2 penalty on the loss switched OFF (--regularization=0), for all 10
# systems of DIMH_SWEEP_SYSTEMS at U = 8.0.
#
# u index 33 is U = 8.0: load_ED_data prepends a U = 0 reference entry, so its U_values
# vector has 61 entries and U = 8.0 sits at index 33, matching figure4.ipynb's own
# findmin(abs.(U_values .- 8.0)) and the existing *_u_33.jld2 coefficient files.
#
# Everything else matches the runs that produced the current figure (taken from the
# "Command run:" lines of the original logs):
#   <folder> 33 33 --antihermitian --custom_ref_state=slater --loss=overlap
#            --maxiters=300 --use-gpu=false
#
# --run_label=noreg keeps the output in its own
#   unitary_map_energy_symmetry=false_N=(nu, nd)_ref_slater_antihermitian_noreg_u_33.jld2
# files, so the original (regularized) coefficients the current figure was built from are
# neither overwritten nor resumed from.

set -euo pipefail

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

SYS="${SYSTEMS[$SLURM_ARRAY_TASK_ID]}"
echo "task $SLURM_ARRAY_TASK_ID: system=$SYS"

cd /home/jek354/research/ML-signproblem/experimenting/ed
julia --project=.. run_lanczos_scan_optimization.jl "$SYS" 33 33 \
    --antihermitian --custom_ref_state=slater --loss=overlap \
    --maxiters=300 --use-gpu=false \
    --regularization=0 --run_label=noreg
