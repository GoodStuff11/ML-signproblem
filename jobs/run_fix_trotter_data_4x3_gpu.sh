#!/bin/bash
#SBATCH --job-name=fix_trotter_4x3_gpu
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/logs/fix_trotter_4x3_gpu_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/logs/fix_trotter_4x3_gpu_%j.err
#SBATCH --mem=30g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --exclude=kim-compute-01
#SBATCH --time=7-00:00:00
#SBATCH --partition=kim

cd /home/jek354/research/ML-signproblem/experimenting/ed
/usr/bin/time julia --project=.. fix_trotter_data_4x3_gpu.jl
