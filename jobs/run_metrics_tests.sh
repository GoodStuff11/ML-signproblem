#!/bin/bash
#SBATCH --job-name=trotter_metrics_tests
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/logs/trotter_metrics_tests_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/logs/trotter_metrics_tests_%j.err
#SBATCH --mem=20g
#SBATCH --cpus-per-task=4
#SBATCH --exclude=kim-compute-01
#SBATCH --time=7-00:00:00
#SBATCH --partition=kim

cd /home/jek354/research/ML-signproblem/experimenting/ed
/usr/bin/time julia --project=.. testing/test_overlap_loss_and_metrics_saving.jl
