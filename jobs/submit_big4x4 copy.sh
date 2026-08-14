#!/bin/bash
#SBATCH --job-name=(6,6)_4x4
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/logs/N=(6,6)_4x4_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/logs/N=(6,6)_4x4_%j.err
#SBATCH --mem=600g
#SBATCH --exclude=kim-compute-01
#SBATCH --time=99-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=kim

cd /home/jek354/research/ML-signproblem/experimenting/ed
/usr/bin/time julia --project=.. run_trotter_scan_optimization.jl "N=(6, 6)_4x4" 33 33 --loss=overlap --antihermitian --custom_ref_state=slater --maxiters=500
