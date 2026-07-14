#!/bin/bash
#SBATCH --job-name=trotter_4x3_4_4_default
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/trotter_4x3_4_4_default.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/trotter_4x3_4_4_default.err
#SBATCH --mem=30g
#SBATCH --cpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --partition=kim

cd /home/jek354/research/ML-signproblem/experimenting/ed
julia --project=.. run_trotter_scan_optimization.jl data/N=\(4,\ 4\)_4x3 2 60 --loss=overlap --antihermitian --maxiters=300
