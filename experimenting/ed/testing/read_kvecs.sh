#!/bin/bash
#SBATCH --job-name=read_kvecs
#SBATCH --partition=kim
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=kvecs.log
#SBATCH --error=kvecs.log

julia --project=.. read_kvecs.jl
