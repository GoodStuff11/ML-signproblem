#!/bin/bash
#SBATCH --job-name=test_5_4_3x3
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/logs/test_5_4_3x3_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/logs/test_5_4_3x3_%j.err
#SBATCH --mem=20g
#SBATCH --cpus-per-task=2
#SBATCH --exclude=kim-compute-01
#SBATCH --time=7-00:00:00
#SBATCH --partition=kim

cd /home/jek354/research/ML-signproblem/experimenting/ed
/usr/bin/time julia --project=.. test_5_4_3x3.jl
