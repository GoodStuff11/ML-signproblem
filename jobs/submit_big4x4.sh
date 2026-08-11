#!/bin/bash
#SBATCH --job-name=big4x4
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/logs/big_4x4_optionB.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/logs/big_4x4_optionB.err
#SBATCH --mem=600g
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=kim-compute-01
#SBATCH --time=99-00:00:00
#SBATCH --partition=kim

cd /home/jek354/research/ML-signproblem/experimenting/ed
/usr/bin/time julia --project=.. run_trotter_scan_optimization.jl "N=(6, 6)_4x4" 33 33 --loss=overlap --antihermitian --custom_ref_state=slater --maxiters=300 --use_gpu --datatype=Float32
