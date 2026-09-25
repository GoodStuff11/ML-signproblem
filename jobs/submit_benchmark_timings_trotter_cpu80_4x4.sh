#!/bin/bash
#SBATCH --job-name=benchmark_timings_trotter_cpu80_4x4
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/logs/benchmark_timings_trotter_cpu80_4x4_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/logs/benchmark_timings_trotter_cpu80_4x4_%j.err
#SBATCH --partition=aimi
#SBATCH --nodelist=aimi-cpu-01
#SBATCH --cpus-per-task=80
#SBATCH --mem=500g
#SBATCH --time=7-00:00:00

# Trotter-code forward/gradient timings on 80 CPU cores for the 4x4 systems.
# Minimal evaluation count: --reps=1 --warmup=0, i.e. per (loss, P) one first (JIT) call and
# one timed call of the forward pass and of the gradient. No optimization is run and nothing
# but CSV rows + the log is written. Appends to experimenting/ed/benchmarks/timings_trotter_cpu80.csv.
#
# The CPU gradient keeps one state vector per gate per layer (2712*P vectors; 64 MB each for
# N=(6, 6)_4x4), so large P can exceed the memory limit. A host out-of-memory kill cannot be
# caught inside julia, so each (system, P) runs in its own process, in increasing P, and a
# system stops at its first failure (larger P would need even more memory).
cd /home/jek354/research/ML-signproblem/experimenting/ed
for system in "N=(5, 4)_4x4" "N=(6, 6)_4x4"; do
    for P in 1 2 4 8; do
        echo "=== $system  P=$P  $(date) ==="
        julia -t 80 --project=.. benchmarks/benchmark_timings.jl \
            --code=trotter \
            --use_gpu=false \
            --systems="$system" \
            --losses=overlap,energy \
            --exponentials=$P \
            --reps=1 --warmup=0 \
            --antihermitian=true --custom_ref_state=slater \
            --tag=cpu80
        status=$?
        if [ $status -ne 0 ]; then
            echo "=== $system  P=$P failed (exit $status); skipping larger P for this system ==="
            break
        fi
    done
done
