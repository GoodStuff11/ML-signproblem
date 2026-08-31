#!/bin/bash
#SBATCH --job-name=trotter_files_audit
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/logs/trotter_files_audit_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/logs/trotter_files_audit_%j.err
#SBATCH --mem=40g
#SBATCH --cpus-per-task=4
#SBATCH --exclude=kim-compute-01
#SBATCH --time=7-00:00:00
#SBATCH --partition=kim

cd /home/jek354/research/ML-signproblem/experimenting/ed
/usr/bin/time julia --project=.. audit_trotter_files.jl
