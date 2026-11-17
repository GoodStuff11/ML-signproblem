#!/bin/bash
#SBATCH --job-name=trotter_4x3_4_4_sector7_slater
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/trotter_4x3_4_4_sector7_slater_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/trotter_4x3_4_4_sector7_slater_%j.err
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --exclude=kim-compute-01
#SBATCH --partition=kim
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=NONE
# Slightly excited state of 4x3 (4,4): lowest state of momentum sector 7, q=(2,1).
# Settings reproduce the paper (main.tex) runs, e.g. submit_trotter_4x3_5_4.sh:
#   one layer, overlap (fidelity) loss, antihermitian generators, Slater reference,
#   maxiters=300, U indices 2..61 (U = 0.25..15) scanned forward with warm starts,
#   multi-start (5 x 30 iterations) at the first U.
# Optimizer flags restore the pre-2026-09-23 behaviour those runs used (commit c8c8e1b
# changed the defaults afterwards): stages LBFGS,GD,LBFGS each to maxiters, no stall stop,
# L-BFGS memory 10 (Optim default). Output prefix: trotter_N=12_ref_slater_antihermitian_sector=7.
cd /home/jek354/research/ML-signproblem/.claude/worktrees/excited-states/experimenting/ed
/usr/bin/time -v julia --project=.. run_trotter_scan_optimization.jl "N=(4, 4)_4x3" 2 61 \
    --loss=overlap --antihermitian --custom_ref_state=slater --maxiters=300 \
    --stages=LBFGS,GD,LBFGS --stall_window=0 --lbfgs_memory=10 \
    --sector=7 --use_gpu
