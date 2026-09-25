#!/bin/bash
#SBATCH --job-name=benchmark_timings_trotter_cpu80_small
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/logs/benchmark_timings_trotter_cpu80_small_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/logs/benchmark_timings_trotter_cpu80_small_%j.err
#SBATCH --partition=aimi
#SBATCH --nodelist=aimi-cpu-01
#SBATCH --cpus-per-task=80
#SBATCH --mem=120g
#SBATCH --time=7-00:00:00

# Trotter-code forward/gradient timings on 80 CPU cores for the small added systems.
# Same settings as submit_benchmark_timings_trotter_cpu80.sh (pinned to the node the existing
# rows came from). No optimization is run and nothing but CSV rows + the log is written.
# Appends to experimenting/ed/benchmarks/timings_trotter_cpu80.csv.
# Submit with --dependency on any other benchmark job using aimi-cpu-01, so timings are not
# measured while another 80-thread job shares the node.
cd /home/jek354/research/ML-signproblem/experimenting/ed
julia -t 80 --project=.. benchmarks/benchmark_timings.jl \
    --code=trotter \
    --use_gpu=false \
    --systems="N=(3, 2)_3x2;N=(3, 3)_4x2" \
    --losses=overlap,energy \
    --exponentials=1,2,4,8 \
    --reps=5 --warmup=1 \
    --antihermitian=true --custom_ref_state=slater \
    --tag=cpu80
