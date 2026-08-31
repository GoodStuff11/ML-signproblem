#!/bin/bash
#SBATCH --job-name=test_load_4x3
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/logs/test_load_4x3_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/logs/test_load_4x3_%j.err
#SBATCH --mem=64g
#SBATCH --cpus-per-task=4
#SBATCH --exclude=kim-compute-01
#SBATCH --time=1:00:00
#SBATCH --partition=kim

cd /home/jek354/research/ML-signproblem/experimenting/ed
/usr/bin/time julia --project=.. test_load_4x3_slurm.jl
