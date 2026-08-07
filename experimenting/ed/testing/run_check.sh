#!/bin/bash
#SBATCH --job-name=check_indexer
#SBATCH --partition=kim
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --output=check_out.log
#SBATCH --error=check_out.log

julia --project=.. check_indexer.jl
