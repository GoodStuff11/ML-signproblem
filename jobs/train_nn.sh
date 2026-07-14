#!/bin/bash
#SBATCH -J train_nn
#SBATCH -o /home/jek354/research/ML-signproblem/jobs/train_nn_%j.out
#SBATCH -e /home/jek354/research/ML-signproblem/jobs/train_nn_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jek354@cornell.edu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --get-user-env
#SBATCH --mem=20G
#SBATCH -t 1-00:00:00
#SBATCH --partition=kim
#SBATCH --gres=gpu:1

cd /home/jek354/research/ML-signproblem/experimenting/ed/
julia --project=.. system_scaling.jl --strategy=neural "$@"
