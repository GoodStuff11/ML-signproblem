#!/bin/bash
#SBATCH --job-name=trotter_4x3_5_4
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/logs/trotter_4x3_5_4_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/logs/trotter_4x3_5_4_%j.err
#SBATCH --mem=20g
#SBATCH --cpus-per-task=20
#SBATCH --exclude=kim-compute-01
#SBATCH --time=99-00:00:00
#SBATCH --partition=kim

cd /home/jek354/research/ML-signproblem/experimenting/ed
/usr/bin/time julia --project=.. run_trotter_scan_optimization.jl "N=(5, 4)_4x3" 2 61 --loss=overlap --antihermitian --custom_ref_state=slater --maxiters=300
