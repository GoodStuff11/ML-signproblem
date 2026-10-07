#!/bin/bash
#SBATCH --job-name=pruning_ne2_4x3_5_4
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/logs/pruning_ne2_4x3_5_4_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/logs/pruning_ne2_4x3_5_4_%j.err
#SBATCH --mem=20g
#SBATCH --cpus-per-task=20
#SBATCH --exclude=kim-compute-01
#SBATCH --time=7-00:00:00
#SBATCH --partition=kim

cd /home/jek354/research/ML-signproblem/experimenting/ed
/usr/bin/time julia --project=.. run_pruning_analysis.jl "N=(5, 4)_4x3" --type=trotter --antihermitian --custom_ref_state=slater --num_exponentials=2
