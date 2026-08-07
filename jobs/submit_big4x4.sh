#!/bin/bash
#SBATCH --job-name=big4x4
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/logs/big_4x4.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/logs/big_4x4.err
#SBATCH --mem=600g
#SBATCH --cpus-per-task=60
#SBATCH --time=99-00:00:00
#SBATCH --partition=kim

cd /home/jek354/research/ML-signproblem/experimenting/ed
/usr/bin/time julia --project=.. run_trotter_scan_optimization.jl "N=(6, 6)_4x4" 33 33 --loss=overlap --antihermitian --custom_ref_state=slater --maxiters=300
