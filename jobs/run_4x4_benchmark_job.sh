#!/bin/bash
#SBATCH -J benchmark_4x4
#SBATCH -o /home/jek354/research/ML-signproblem/jobs/4x4_benchmark_%j.out
#SBATCH -e /home/jek354/research/ML-signproblem/jobs/4x4_benchmark_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jek354@cornell.edu
#SBATCH -N 1
#SBATCH -n 30
#SBATCH --get-user-env
#SBATCH --mem=250G
#SBATCH -t 1-00:00:00
#SBATCH --partition=aimi
#SBATCH --nodelist=aimi-cpu-01

cd /home/jek354/research/ML-signproblem/experimenting/ed/
julia --threads=auto --project=.. run_4x4_benchmark.jl
