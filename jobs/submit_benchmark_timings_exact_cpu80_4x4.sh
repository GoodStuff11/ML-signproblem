#!/bin/bash
#SBATCH --job-name=benchmark_timings_exact_cpu80_4x4
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/logs/benchmark_timings_exact_cpu80_4x4_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/logs/benchmark_timings_exact_cpu80_4x4_%j.err
#SBATCH --partition=aimi
#SBATCH --nodelist=aimi-cpu-01
#SBATCH --cpus-per-task=80
#SBATCH --mem=500g
#SBATCH --time=7-00:00:00

# Exact-code forward/gradient timings on 80 CPU cores for the 4x4 systems.
# Minimal evaluation count: --reps=1 --warmup=0, i.e. per loss one first (JIT) call and one
# timed call of the forward pass and of the gradient. No optimization is run and nothing
# but CSV rows + the log is written. N=(5, 4)_4x4 runs first so its rows are flushed to the
# CSV even if N=(6, 6)_4x4 runs out of memory.
# Appends to experimenting/ed/benchmarks/timings_exact_cpu80.csv.
# Override the system list with e.g.: SYSTEMS="N=(6, 6)_4x4" sbatch <this script>
cd /home/jek354/research/ML-signproblem/experimenting/ed
julia -t 80 --project=.. benchmarks/benchmark_timings.jl \
    --code=exact \
    --use_gpu=false \
    --systems="${SYSTEMS:-N=(5, 4)_4x4;N=(6, 6)_4x4}" \
    --losses=overlap,energy \
    --reps=1 --warmup=0 \
    --antihermitian=true --custom_ref_state=slater \
    --tag=cpu80_nodiag
