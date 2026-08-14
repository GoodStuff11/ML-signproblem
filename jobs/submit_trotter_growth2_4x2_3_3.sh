#!/bin/bash
#SBATCH --job-name=trotter_growth2_4x2_3_3
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/logs/trotter_growth2_4x2_3_3_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/logs/trotter_growth2_4x2_3_3_%j.err
#SBATCH --mem=20g
#SBATCH --cpus-per-task=20
#SBATCH --exclude=kim-compute-01
#SBATCH --time=99-00:00:00
#SBATCH --partition=kim

cd /home/jek354/research/ML-signproblem/experimenting/ed
/usr/bin/time julia --project=.. run_trotter_scan_optimization.jl "N=(3, 3)_4x2" 2 61 --loss=overlap --antihermitian --custom_ref_state=slater --num_exponentials=2 --grow_from_exponentials=1 --grow_mode=per_u --maxiters=200
