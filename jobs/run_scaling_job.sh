#!/bin/bash
#SBATCH --job-name=system-scaling-eval
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/system_scaling_eval_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/system_scaling_eval_%j.err
#SBATCH --partition=kim
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=20g

cd /home/jek354/research/ML-signproblem/experimenting/ed
julia --project=.. system_scaling.jl --strategy=neural --folder-set=2x2_only --name=pure_mild_mid_narrow --u-range=2:3 --use-gpu=false
