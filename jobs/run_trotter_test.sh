#!/bin/bash
#SBATCH --job-name=trotter_slater_test
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/trotter_slater_test.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/trotter_slater_test.err
#SBATCH --mem=20g
#SBATCH --cpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --partition=kim

cd /home/jek354/research/ML-signproblem/experimenting/ed
julia --project=.. run_trotter_scan_optimization.jl data_new_sign/N=\(2,\ 2\)_3x2 2 3 --maxiters=2 --custom_ref_state=slater
