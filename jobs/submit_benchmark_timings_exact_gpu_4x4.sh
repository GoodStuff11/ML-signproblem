#!/bin/bash
#SBATCH --job-name=benchmark_timings_exact_gpu_4x4
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/logs/benchmark_timings_exact_gpu_4x4_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/logs/benchmark_timings_exact_gpu_4x4_%j.err
#SBATCH --partition=kim
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=500g
#SBATCH --exclude=kim-compute-01
#SBATCH --time=7-00:00:00

# Exact-code forward/gradient timings on one GPU for the 4x4 systems.
# 500 GB host memory only fits the A100-80GB nodes (kim-compute-03/04/05), so these rows are
# on a different GPU from the T4 rows (recorded in the CSV's `gpu` column).
# Minimal evaluation count: --reps=1 --warmup=0, i.e. per loss one first (JIT) call and one
# timed call of the forward pass and of the gradient. No optimization is run and nothing
# but CSV rows + the log is written. N=(5, 4)_4x4 runs first so its rows are flushed to the
# CSV even if N=(6, 6)_4x4 runs out of (GPU) memory.
# Appends to experimenting/ed/benchmarks/timings_exact_gpu.csv.
# Override the system list with e.g.: SYSTEMS="N=(6, 6)_4x4" sbatch <this script>
cd /home/jek354/research/ML-signproblem/experimenting/ed
julia --project=.. benchmarks/benchmark_timings.jl \
    --code=exact \
    --use_gpu \
    --systems="${SYSTEMS:-N=(5, 4)_4x4;N=(6, 6)_4x4}" \
    --losses=overlap,energy \
    --reps=1 --warmup=0 \
    --antihermitian=true --custom_ref_state=slater \
    --tag=gpu_nodiag
