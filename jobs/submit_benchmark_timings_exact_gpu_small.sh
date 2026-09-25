#!/bin/bash
#SBATCH --job-name=benchmark_timings_exact_gpu_small
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/logs/benchmark_timings_exact_gpu_small_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/logs/benchmark_timings_exact_gpu_small_%j.err
#SBATCH --partition=kim
#SBATCH --nodelist=kim-compute-02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=60g
#SBATCH --time=7-00:00:00

# Exact-code forward/gradient timings on one GPU for the small added systems.
# Same settings as submit_benchmark_timings_exact_gpu.sh, pinned to kim-compute-02 (Tesla T4),
# the GPU the existing rows were measured on. No optimization is run and nothing but CSV
# rows + the log is written. Appends to experimenting/ed/benchmarks/timings_exact_gpu.csv.
cd /home/jek354/research/ML-signproblem/experimenting/ed
julia --project=.. benchmarks/benchmark_timings.jl \
    --code=exact \
    --use_gpu \
    --systems="N=(3, 2)_3x2;N=(3, 3)_4x2" \
    --losses=overlap,energy \
    --reps=5 --warmup=1 \
    --antihermitian=true --custom_ref_state=slater \
    --tag=gpu_nodiag
