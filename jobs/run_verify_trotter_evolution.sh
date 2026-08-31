#!/bin/bash
#SBATCH --job-name=verify_trotter_evolution
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/logs/verify_trotter_evolution_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/logs/verify_trotter_evolution_%j.err
#SBATCH --mem=20g
#SBATCH --cpus-per-task=2
#SBATCH --exclude=kim-compute-01
#SBATCH --time=7-00:00:00
#SBATCH --partition=kim

cd /home/jek354/research/ML-signproblem/experimenting/ed
/usr/bin/time julia --project=.. verify_trotter_state_evolution.jl
