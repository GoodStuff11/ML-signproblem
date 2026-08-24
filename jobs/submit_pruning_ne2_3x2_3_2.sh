#!/bin/bash
#SBATCH --job-name=pruning_ne2_3x2_3_2
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/logs/pruning_ne2_3x2_3_2_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/logs/pruning_ne2_3x2_3_2_%j.err
#SBATCH --mem=20g
#SBATCH --cpus-per-task=20
#SBATCH --exclude=kim-compute-01
#SBATCH --time=1-00:00:00
#SBATCH --partition=kim

cd /home/jek354/research/ML-signproblem/experimenting/ed
/usr/bin/time julia --project=.. run_pruning_analysis.jl "N=(3, 2)_3x2" --type=trotter --antihermitian --custom_ref_state=slater --num_exponentials=2
