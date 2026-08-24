#!/bin/bash
#SBATCH --job-name=(6,6)_4x4_cpu_growth2
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/logs/N=(6,6)_4x4_cpu_growth2_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/logs/N=(6,6)_4x4_cpu_growth2_%j.err
#SBATCH --mem=600g
#SBATCH --cpus-per-task=20
#SBATCH --time=99-00:00:00
#SBATCH --partition=kim

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /home/jek354/research/ML-signproblem/experimenting/ed
/usr/bin/time julia --project=.. run_trotter_scan_optimization.jl "N=(6, 6)_4x4" 33 33 --loss=overlap --antihermitian --custom_ref_state=slater --num_exponentials=2 --grow_from_exponentials=1 --grow_mode=chain --maxiters=300 --use_gpu=false
