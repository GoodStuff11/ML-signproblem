#!/bin/bash
#SBATCH --job-name=target_fidelity_4x3_5_4_gpu
#SBATCH --array=2-61%8
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/logs/target_fidelity_4x3_5_4_gpu_%A_%a.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/logs/target_fidelity_4x3_5_4_gpu_%A_%a.err
#SBATCH --mem=20g
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=kim-compute-01
#SBATCH --time=1-00:00:00
#SBATCH --partition=kim

cd /home/jek354/research/ML-signproblem/experimenting/ed
/usr/bin/time julia --project=.. run_trotter_scan_optimization.jl "N=(5, 4)_4x3" $SLURM_ARRAY_TASK_ID --num_exponentials=1 --loss=overlap --antihermitian --custom_ref_state=slater --target_fidelity=0.94 --maxiters=200 --use_gpu
