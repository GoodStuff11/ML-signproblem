#!/bin/bash
#SBATCH --job-name=test_trotter_datatype_gpu
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/logs/test_trotter_datatype_gpu_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/logs/test_trotter_datatype_gpu_%j.err
#SBATCH --mem=20g
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=kim-compute-01
#SBATCH --time=7-00:00:00
#SBATCH --partition=kim

cd /home/jek354/research/ML-signproblem/experimenting/ed
julia --project=.. testing/test_trotter_datatype_gpu.jl --use_gpu
