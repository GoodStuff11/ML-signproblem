#!/bin/bash
#SBATCH --job-name=lanczos_3x3_3_3_slater
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/lanczos_3x3_3_3_slater.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/lanczos_3x3_3_3_slater.err
#SBATCH --mem=30g
#SBATCH --cpus-per-task=80
#SBATCH --time=7-00:00:00
#SBATCH --partition=aimi

cd /home/jek354/research/ML-signproblem/experimenting/ed
julia -t 20 --project=.. run_lanczos_scan_optimization.jl N=\(5,\ 4\)_4x3 2 61 --loss=overlap --custom_ref_state=slater --maxiters=300 --use-gpu=false --antihermitian
