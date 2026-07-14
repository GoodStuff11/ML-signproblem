#!/bin/bash
#SBATCH --job-name=eval_scaling_pure_03
#SBATCH --partition=kim
#SBATCH --gres=gpu:1
#SBATCH --mem=20g
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --exclude=kim-compute-01
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/eval_scaling_pure_03-%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/eval_scaling_pure_03-%j.err

mkdir -p /home/jek354/research/ML-signproblem/jobs
julia --project=.. system_scaling.jl --strategy=neural --name=pure_mid_narrow_one_minus_loss_power_03 --weighting=one_minus_loss_power_03 --u-range=35:53 --folder-set=square_pure --base-hidden=96,96 --embed-dim=48 --context-hidden=48,24 --scale-hidden=24,12
