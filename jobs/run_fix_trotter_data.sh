#!/bin/bash
#SBATCH --job-name=fix_trotter_data
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/logs/fix_trotter_data_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/logs/fix_trotter_data_%j.err
#SBATCH --mem=40g
#SBATCH --cpus-per-task=4
#SBATCH --exclude=kim-compute-01
#SBATCH --time=7-00:00:00
#SBATCH --partition=kim

cd /home/jek354/research/ML-signproblem/experimenting/ed
/usr/bin/time julia --project=.. fix_trotter_data.jl
