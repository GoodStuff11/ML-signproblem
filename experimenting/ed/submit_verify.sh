#!/usr/bin/env bash
#SBATCH --mem=20g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --partition=kim
#SBATCH --job-name=verify_barren_plateau_temp
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/verify_temp_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/verify_temp_%j.err

cd /home/jek354/research/ML-signproblem/experimenting/ed
julia --project=.. test_barren_plateau_temp.jl --use_gpu=true
