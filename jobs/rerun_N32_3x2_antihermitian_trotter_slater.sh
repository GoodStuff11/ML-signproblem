#!/bin/bash
#SBATCH --job-name=rerun_N32_3x2_trotter_slater
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/rerun_N32_3x2_trotter_slater-%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/rerun_N32_3x2_trotter_slater-%j.err
#SBATCH --mem=20g
#SBATCH --cpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --partition=kim

cd /home/jek354/research/ML-signproblem/experimenting/ed
julia --project=.. run_trotter_scan_optimization.jl "N=(3, 2)_3x2" 2 60 --antihermitian --custom_ref_state=slater --loss=overlap
