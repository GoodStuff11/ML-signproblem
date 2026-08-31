#!/bin/bash
#SBATCH --job-name=lanczos_3x3_4_4_slater_antihermitian
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/lanczos_3x3_4_4_slater_antihermitian_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/lanczos_3x3_4_4_slater_antihermitian_%j.err
#SBATCH --mem=30g
#SBATCH --time=7-00:00:00
#SBATCH --partition=aimi
#SBATBH --cpus-per-task=80

cd /home/jek354/research/ML-signproblem/experimenting/ed
julia --project=.. run_lanczos_scan_optimization.jl /home/jek354/research/data/new_data/data_h5_fixed/N=\(4,\ 4\)_3x3 2 61 --antihermitian --custom_ref_state=slater --loss=overlap --use-gpu=false
