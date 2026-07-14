#!/bin/bash
#SBATCH --job-name=test_3x3_4_4
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/test_3x3_4_4.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/test_3x3_4_4.err
#SBATCH --mem=30g
#SBATCH --cpus-per-task=20
#SBATCH --time=7-00:00:00
#SBATCH --partition=kim

cd /home/jek354/research/ML-signproblem/experimenting/ed
julia -t 20 --project=.. trotter_exp_testing.jl data/N=\(4,\ 4\)_3x3 --n_up=4 --n_dn=4 --lvec=3x3 --output=trotter_order_comparison
