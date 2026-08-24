#!/bin/bash
#SBATCH --job-name=target_fidelity_3x2_3_2
#SBATCH --array=2-61%8
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/logs/target_fidelity_3x2_3_2_%A_%a.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/logs/target_fidelity_3x2_3_2_%A_%a.err
#SBATCH --mem=20g
#SBATCH --cpus-per-task=20
#SBATCH --exclude=kim-compute-01
#SBATCH --time=1-00:00:00
#SBATCH --partition=kim

cd /home/jek354/research/ML-signproblem/experimenting/ed
/usr/bin/time julia --project=.. run_trotter_scan_optimization.jl "N=(3, 2)_3x2" $SLURM_ARRAY_TASK_ID --num_exponentials=1 --loss=overlap --antihermitian --custom_ref_state=slater --target_fidelity=0.995 --maxiters=200
