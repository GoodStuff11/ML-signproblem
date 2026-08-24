#!/bin/bash
#SBATCH --job-name=rerun_5_4_3x3_slater
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/rerun_5_4_3x3_slater_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/rerun_5_4_3x3_slater_%j.err
#SBATCH --mem=20g
#SBATCH --cpus-per-task=20
#SBATCH --exclude=kim-compute-01
#SBATCH --time=7-00:00:00
#SBATCH --partition=kim

# Reruns u_idx 2-15 for the N=(5,4)_3x3 Slater antihermitian overlap scan.
# These files had negative losses (due to expv norm drift) which were incorrectly
# clamped to 0.0 in the database. The loss formula has been corrected to use
# the normalized overlap: 1 - |<psi|phi>|^2 / <psi|psi>, which is always >= 0.
# The scan warm-starts from the existing u_idx=16 file.

cd /home/jek354/research/ML-signproblem/experimenting/ed
export JULIA_NUM_THREADS=20
julia --project=.. run_trotter_scan_optimization.jl \
    "/home/jek354/research/data/new_data/data_h5_fixed/N=(5, 4)_3x3" \
    15 2 \
    --antihermitian \
    --loss=overlap \
    --custom_ref_state=slater \
    --maxiters=300 \
    --use_gpu=false
