#!/bin/bash
#SBATCH --job-name=trotter_4x3_5_4_level4_slater
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/trotter_4x3_5_4_level4_slater_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/trotter_4x3_5_4_level4_slater_%j.err
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --exclude=kim-compute-01
#SBATCH --partition=kim
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=NONE
# Excited-state optimization, 4x3 (5,4): the 4th excited level of the ground-state sector 2, q=(0,2),
# tracked continuously in U (data: N=(5, 4)_4x3_excited_level4, jobs/track_4x3_54_sector2_level4.sh).
# Settings reproduce the paper (main.tex) runs (submit_trotter_4x3_5_4.sh): one layer, overlap loss,
# antihermitian generators, Slater reference, maxiters=300, U indices 2..61 (U = 0.25..15) forward with
# warm starts, multi-start (5 x 30 iterations) at the first U, and the pre-2026-09-23 optimizer
# behaviour: stages LBFGS,GD,LBFGS each to maxiters, no stall stop, L-BFGS memory 10.
# Reference = the Slater determinant with the largest weight in the tracked state at U=0.25
# (--reference_target=weak_connected; index 5219). Output prefix in the data folder:
# trotter_N=12_ref_slater_antihermitian_target=tracked_reftarget=weak_connected.
cd /home/jek354/research/ML-signproblem/experimenting/ed
/usr/bin/time -v julia --project=.. run_trotter_scan_optimization.jl "N=(5, 4)_4x3_excited_level4" 2 61 \
    --loss=overlap --antihermitian --custom_ref_state=slater --maxiters=300 \
    --stages=LBFGS,GD,LBFGS --stall_window=0 --lbfgs_memory=10 \
    --target_state=tracked --reference_target=weak_connected --use_gpu
