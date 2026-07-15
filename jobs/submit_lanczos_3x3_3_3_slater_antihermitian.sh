#!/bin/bash
#SBATCH --job-name=lanczos_3x3_3_3_slater_antihermitian
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/lanczos_3x3_3_3_slater_antihermitian.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/lanczos_3x3_3_3_slater_antihermitian.err
#SBATCH --mem=40g
#SBATCH --cpus-per-task=20
#SBATCH --time=7-00:00:00
#SBATCH --partition=kim

cd /home/jek354/research/ML-signproblem/experimenting/ed
julia -t 20 --project=.. run_lanczos_scan_optimization.jl data/N=\(3,\ 3\)_3x3 2 60 --custom_ref_state=slater --antihermitian --maxiters=300 --use-gpu=false
