#!/bin/bash
#SBATCH --job-name=trotter_3x3_5_4_default
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/trotter_default_N_5__4_3x3_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/trotter_default_N_5__4_3x3_%j.err
#SBATCH --mem=30g
#SBATCH --cpus-per-task=20
#SBATCH --exclude=kim-compute-01
#SBATCH --time=7-00:00:00
#SBATCH --partition=kim

cd /home/jek354/research/ML-signproblem/experimenting/ed
export JULIA_NUM_THREADS=20
julia --project=.. run_trotter_scan_optimization.jl "/home/jek354/research/data/new_data/data_h5_fixed/N=(5, 4)_3x3" 60 2 --antihermitian --loss=overlap --use_gpu=false
