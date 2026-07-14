#!/bin/bash
#SBATCH --job-name=custom_ref_random_energy
#SBATCH --partition=kim
#SBATCH --gres=gpu:1
#SBATCH --mem=20g
#SBATCH --cpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --exclude=kim-compute-01
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/custom_ref_random_energy-%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/custom_ref_random_energy-%j.err

cd /home/jek354/research/ML-signproblem/experimenting/ed/
julia --project=.. run_custom_ref_experiments.jl --ref-type=random --loss-type=energy
