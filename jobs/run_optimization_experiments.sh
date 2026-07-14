#!/bin/bash
#SBATCH -J opt_sweep_experiments
#SBATCH -o /home/jek354/research/ML-signproblem/jobs/opt_sweep_experiments_%j.out
#SBATCH -e /home/jek354/research/ML-signproblem/jobs/opt_sweep_experiments_%j.err
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --mem=20G
#SBATCH -t 1-00:00:00
#SBATCH --partition=kim

cd /home/jek354/research/ML-signproblem/experimenting/ed/
julia --project=.. --threads=auto run_optimization_experiments.jl trained_neural_networks/trained_neural_network_2x2_only_one_minus_loss_power_10.jld2
