#!/bin/bash
#SBATCH --job-name=benchmark_timings_trotter_gpu_small
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/logs/benchmark_timings_trotter_gpu_small_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/logs/benchmark_timings_trotter_gpu_small_%j.err
#SBATCH --partition=kim
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=60g
#SBATCH --exclude=kim-compute-01,kim-compute-02
#SBATCH --time=7-00:00:00

# Trotter-code forward/gradient timings on one GPU for the small added systems.
# Same settings as submit_benchmark_timings_trotter_gpu.sh; kim-compute-01/02 (V100/T4) are
# excluded so this runs on an A100 80GB like the existing rows. No optimization is run and
# nothing but CSV rows + the log is written.
# Appends to experimenting/ed/benchmarks/timings_trotter_gpu.csv.
cd /home/jek354/research/ML-signproblem/experimenting/ed
julia --project=.. benchmarks/benchmark_timings.jl \
    --code=trotter \
    --use_gpu \
    --systems="N=(3, 2)_3x2;N=(3, 3)_4x2" \
    --losses=overlap,energy \
    --exponentials=1 \
    --reps=5 --warmup=1 \
    --antihermitian=true --custom_ref_state=slater \
    --tag=gpu
