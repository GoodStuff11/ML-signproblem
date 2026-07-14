#!/bin/bash
#SBATCH --job-name=test_3x3_3_3_slater_antihermitian
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/test_3x3_3_3_slater_antihermitian.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/test_3x3_3_3_slater_antihermitian.err
#SBATCH --mem=30g
#SBATCH --cpus-per-task=20
#SBATCH --time=7-00:00:00
#SBATCH --partition=kim

cd /home/jek354/research/ML-signproblem/experimenting/ed
julia -t 20 --project=.. trotter_exp_testing.jl data/N=\(3,\ 3\)_3x3 --n_up=3 --n_dn=3 --lvec=3x3 --antihermitian=true --custom_ref_state=slater --loss=overlap --output=trotter_order_comparison_ref_slater_antihermitian_loss_overlap
