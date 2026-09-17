#!/bin/bash
#SBATCH --job-name=benchmark_timings_trotter_gpu
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/logs/benchmark_timings_trotter_gpu_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/logs/benchmark_timings_trotter_gpu_%j.err
#SBATCH --partition=kim
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=60g
#SBATCH --exclude=kim-compute-01
#SBATCH --time=7-00:00:00

# Forward-pass / gradient timings for the trotter code on one GPU.
# Writes ONLY experimenting/ed/benchmarks/timings_trotter_gpu.csv (plus the tee'd stdout log).
cd /home/jek354/research/ML-signproblem/experimenting/ed
julia --project=.. benchmark_timings.jl \
    --code=trotter \
    --use_gpu \
    --losses=overlap,energy \
    --exponentials=1,2,4,8 \
    --reps=5 --warmup=1 \
    --antihermitian=true --custom_ref_state=slater \
    --tag=gpu
