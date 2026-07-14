#!/bin/bash
#SBATCH --mem=20g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:30:00
#SBATCH --partition=kim
#SBATCH --job-name=test_scaling_run
#SBATCH -o /home/jek354/research/ML-signproblem/jobs/test_scaling_run.out
#SBATCH -e /home/jek354/research/ML-signproblem/jobs/test_scaling_run.err

cd /home/jek354/research/ML-signproblem/experimenting/ed
julia --project=.. run_optimization_experiments.jl pure_mild_mid_narrow --u-idx=53 --init=nn --loss=energy --maxiters=1 --use-gpu=false
