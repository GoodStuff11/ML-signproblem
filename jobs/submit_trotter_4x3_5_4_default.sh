#!/bin/bash
#SBATCH --job-name=trotter_4x3_5_4_default
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/trotter_4x3_5_4_default_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/trotter_4x3_5_4_default_%j.err
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --exclude=kim-compute-01
#SBATCH --time=7-00:00:00
#SBATCH --partition=kim

cd /home/jek354/research/ML-signproblem/experimenting/ed
julia --project=.. run_trotter_scan_optimization.jl "N=(5, 4)_4x3" 2 61 --loss=overlap --antihermitian --maxiters=300 --use_gpu
