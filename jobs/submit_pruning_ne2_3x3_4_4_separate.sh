#!/bin/bash
#SBATCH --job-name=pruning_ne2_3x3_4_4_separate
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/logs/pruning_ne2_3x3_4_4_separate_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/logs/pruning_ne2_3x3_4_4_separate_%j.err
#SBATCH --mem=20g
#SBATCH --cpus-per-task=20
#SBATCH --exclude=kim-compute-01
#SBATCH --time=1-00:00:00
#SBATCH --partition=kim

cd /home/jek354/research/ML-signproblem/experimenting/ed
/usr/bin/time julia --project=.. run_pruning_analysis.jl "N=(4, 4)_3x3_separate" --type=trotter --antihermitian --custom_ref_state=slater --num_exponentials=2
