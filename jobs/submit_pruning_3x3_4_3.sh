#!/bin/bash
#SBATCH --job-name=pruning_3x3_4_3
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/logs/pruning_3x3_4_3_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/logs/pruning_3x3_4_3_%j.err
#SBATCH --mem=20g
#SBATCH --cpus-per-task=20
#SBATCH --exclude=kim-compute-01
#SBATCH --time=7-00:00:00
#SBATCH --partition=kim

cd /home/jek354/research/ML-signproblem/experimenting/ed
/usr/bin/time julia --project=.. run_pruning_analysis.jl "N=(4, 3)_3x3" --type=trotter --custom_ref_state=slater --antihermitian
