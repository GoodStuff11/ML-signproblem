#!/bin/bash
#SBATCH --job-name=benchmark_timings_trotter_gpu_4x4
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/logs/benchmark_timings_trotter_gpu_4x4_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/logs/benchmark_timings_trotter_gpu_4x4_%j.err
#SBATCH --partition=kim
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=500g
#SBATCH --exclude=kim-compute-01,kim-compute-02
#SBATCH --time=7-00:00:00

# Trotter-code forward/gradient timings on one GPU (A100 80GB) for the 4x4 systems.
# Minimal evaluation count: --reps=1 --warmup=0, i.e. per (loss, P) one first (JIT) call and
# one timed call of the forward pass and of the gradient. No optimization is run and nothing
# but CSV rows + the log is written. A configuration that runs out of GPU memory is logged
# and skipped (benchmark_timings.jl run_config), without a row.
# Appends to experimenting/ed/benchmarks/timings_trotter_gpu.csv.
cd /home/jek354/research/ML-signproblem/experimenting/ed
julia --project=.. benchmarks/benchmark_timings.jl \
    --code=trotter \
    --use_gpu \
    --systems="N=(5, 4)_4x4;N=(6, 6)_4x4" \
    --losses=overlap,energy \
    --exponentials=1 \
    --reps=1 --warmup=0 \
    --antihermitian=true --custom_ref_state=slater \
    --tag=gpu
