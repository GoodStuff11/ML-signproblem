#!/bin/bash
#SBATCH --job-name=trotter_3x2_3_2_slater_antihermitian
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/trotter_3x2_3_2_slater_antihermitian.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/trotter_3x2_3_2_slater_antihermitian.err
#SBATCH --mem=30g
#SBATCH --cpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --partition=kim

cd /home/jek354/research/ML-signproblem/experimenting/ed
julia --project=.. run_trotter_scan_optimization.jl "N=(3, 2)_3x2" 2 60 --loss=overlap --antihermitian --custom_ref_state=slater --num_exponentials=1 --maxiters=300
