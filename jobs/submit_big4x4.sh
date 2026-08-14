#!/bin/bash
#SBATCH --job-name=N=(4,4)_3x3_separate
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/logs/N=(4,4)_3x3_separate_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/logs/N=(4,4)_3x3_separate_%j.err
#SBATCH --mem=20g
#SBATCH --cpus-per-task=20
#SBATCH --exclude=kim-compute-01
#SBATCH --time=99-00:00:00
#SBATCH --partition=kim

cd /home/jek354/research/ML-signproblem/experimenting/ed
/usr/bin/time julia --project=.. run_trotter_scan_optimization.jl "N=(4, 3)_3x3" 2 61 --loss=overlap --antihermitian --custom_ref_state=slater --maxiters=300
