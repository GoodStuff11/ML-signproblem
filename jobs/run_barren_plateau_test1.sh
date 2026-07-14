#!/bin/bash
#SBATCH -J BP_test1
#SBATCH -o /home/jek354/research/ML-signproblem/jobs/barren_plateau_test1_%j.out
#SBATCH -e /home/jek354/research/ML-signproblem/jobs/barren_plateau_test1_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jek354@cornell.edu
#SBATCH -N 1
#SBATCH -n 60
#SBATCH --get-user-env
#SBATCH --mem=100G
#SBATCH -t 7-00:00:00
#SBATCH --partition=aimi
#SBATCH --nodelist=aimi-cpu-01

cd /home/jek354/research/ML-signproblem/experimenting/ed/
julia --threads=auto --project=.. barren_plateau.jl --use_gpu=false 
