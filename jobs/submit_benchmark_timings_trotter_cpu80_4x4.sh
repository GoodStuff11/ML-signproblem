#!/bin/bash
#SBATCH --job-name=benchmark_timings_trotter_cpu80_4x4
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/logs/benchmark_timings_trotter_cpu80_4x4_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/logs/benchmark_timings_trotter_cpu80_4x4_%j.err
#SBATCH --partition=aimi
#SBATCH --nodelist=aimi-cpu-01
#SBATCH --cpus-per-task=80
#SBATCH --mem=500g
#SBATCH --time=7-00:00:00

# Trotter-code forward/gradient timings on 80 CPU cores for the 4x4 systems, P=1 only.
# Minimal evaluation count: --reps=1 --warmup=0, i.e. per loss one first (JIT) call and one
# timed call of the forward pass and of the gradient. No optimization is run and nothing
# but CSV rows + the log is written. Appends to experimenting/ed/benchmarks/timings_trotter_cpu80.csv.
#
# The CPU gradient keeps one state vector per gate (2712 vectors of 64 MB for N=(6, 6)_4x4),
# which may exceed the memory limit. A host out-of-memory kill cannot be caught inside julia,
# so each system runs in its own process: an N=(6, 6)_4x4 failure cannot lose the
# N=(5, 4)_4x4 rows.
cd /home/jek354/research/ML-signproblem/experimenting/ed
for system in "N=(5, 4)_4x4" "N=(6, 6)_4x4"; do
    echo "=== $system  $(date) ==="
    julia -t 80 --project=.. benchmarks/benchmark_timings.jl \
        --code=trotter \
        --use_gpu=false \
        --systems="$system" \
        --losses=overlap,energy \
        --exponentials=1 \
        --reps=1 --warmup=0 \
        --antihermitian=true --custom_ref_state=slater \
        --tag=cpu80
    echo "=== $system finished with exit status $? ==="
done
