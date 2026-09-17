#!/bin/bash
#SBATCH --job-name=(6,6)_4x4_gpu
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/logs/N=(6,6)_4x4_gpu_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/logs/N=(6,6)_4x4_gpu_%j.err
#SBATCH --mem=500g
#SBATCH --exclude=kim-compute-01
#SBATCH --time=99-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=kim

cd /home/jek354/research/ML-signproblem/experimenting/ed
/usr/bin/time julia --project=.. run_trotter_scan_optimization.jl "N=(6, 6)_4x4" $1 $2 --loss=overlap --antihermitian --custom_ref_state=slater --maxiters=500 --use_gpu --datatype=Float32 "${@:3}"
