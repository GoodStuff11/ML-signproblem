#!/bin/bash
#SBATCH --job-name=benchmark_timings_trotter_cpu80
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/logs/benchmark_timings_trotter_cpu80_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/logs/benchmark_timings_trotter_cpu80_%j.err
#SBATCH --partition=aimi
#SBATCH --cpus-per-task=80
#SBATCH --mem=120g
#SBATCH --time=7-00:00:00

# Forward-pass / gradient timings for the trotter code on 80 CPU cores.
# CUDA is deliberately NOT loaded here: @safe_threads falls back to serial
# execution whenever the CUDA package is present, which would silently make
# this a 1-core run.
# Writes ONLY experimenting/ed/benchmarks/timings_trotter_cpu80.csv (plus the tee'd stdout log).
cd /home/jek354/research/ML-signproblem/experimenting/ed
julia -t 80 --project=.. benchmark_timings.jl \
    --code=trotter \
    --use_gpu=false \
    --losses=overlap,energy \
    --exponentials=1,2,4,8 \
    --reps=5 --warmup=1 \
    --antihermitian=true --custom_ref_state=slater \
    --tag=cpu80
