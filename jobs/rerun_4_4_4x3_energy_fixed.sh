#!/bin/bash
#SBATCH --job-name=rerun_4_4_4x3_energy
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/rerun_4_4_4x3_energy_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/rerun_4_4_4x3_energy_%j.err
#SBATCH --mem=30g
#SBATCH --cpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --partition=kim

# Reruns the full u_idx 2-60 energy loss scan for N=(4,4)_4x3.
# The original files were corrupted by a database script that incorrectly
# clamped the energy loss (which is the energy expectation value and can be
# legitimately negative) to 0.0. The shared.jld2 confirms the original run
# used hermitian generators (no --antihermitian) with --loss=energy.

cd /home/jek354/research/ML-signproblem/experimenting/ed
julia --project=.. run_trotter_scan_optimization.jl \
    "data/N=(4, 4)_4x3" \
    60 2 \
    --loss=energy \
    --maxiters=300
