#!/bin/bash
#SBATCH --job-name=trotter_3x3_4_4_slater
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/trotter_3x3_4_4_slater.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/trotter_3x3_4_4_slater.err
#SBATCH --mem=30g
#SBATCH --cpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --partition=kim

cd /home/jek354/research/ML-signproblem/experimenting/ed
julia --project=.. run_trotter_scan_optimization.jl data/N=\(4,\ 4\)_3x3 37 60 --loss=overlap --antihermitian --custom_ref_state=slater --maxiters=300
