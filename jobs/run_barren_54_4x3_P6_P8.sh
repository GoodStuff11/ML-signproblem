#!/bin/bash
#SBATCH --job-name=barren_54_4x3
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/barren_54_4x3_P%a_%A.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/barren_54_4x3_P%a_%A.err
#SBATCH --mem=160G
#SBATCH --cpus-per-task=32
#SBATCH --time=7-00:00:00
#SBATCH --partition=aimi,kim
#SBATCH --exclude=kim-compute-01
#SBATCH --array=6,8

P=$SLURM_ARRAY_TASK_ID
echo "=== System: N=(5, 4)_4x3, num_exponentials=$P ==="

cd /home/jek354/research/ML-signproblem/experimenting/ed
julia --threads=$SLURM_CPUS_PER_TASK --project=.. run_trotter_scan_optimization.jl \
  "N=(5, 4)_4x3" 33 33 \
  --num_exponentials=$P \
  --antihermitian \
  --custom_ref_state=slater \
  --run_label=barren_study \
  --use_gpu=false
