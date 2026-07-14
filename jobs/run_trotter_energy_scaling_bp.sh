#!/bin/bash
#SBATCH --job-name=trotter-energy-scaling-bp-single-per
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/trotter_energy_scaling_bp_single_per_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/trotter_energy_scaling_bp_single_per_%j.err
#SBATCH --partition=kim
#SBATCH --exclude=kim-compute-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30g
#SBATCH --time=7-00:00:00

cd /home/jek354/research/ML-signproblem/experimenting/ed

echo "=== Starting NN Training (Energy Loss) ==="
julia --project=.. system_scaling.jl \
  --strategy=neural \
  --weighting=low_u_mild \
  --u-range=35:53 \
  --folder-set=square_pure \
  --base-hidden=96,96 \
  --embed-dim=48 \
  --context-hidden=48,24 \
  --scale-hidden=24,12 \
  --name=pure_mild_mid_narrow_trotter_energy \
  --is-trotter=true \
  --loss-type=energy \
  --use-gpu=true

echo "=== Starting Barren Plateau Study (Energy Loss NN) ==="
julia --threads=auto --project=.. barren_plateau.jl \
  --system_set=single_per \
  --spin_conserved=true \
  --momentum_conserved=true \
  --num_exponentials="1|3|5" \
  --random_std="0.01|0.1|1|10" \
  --nn=pure_mild_mid_narrow_trotter_energy \
  --use_gpu=false
