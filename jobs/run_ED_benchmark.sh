#!/bin/bash
#SBATCH -J ED                                   # Job name
#SBATCH -o unitary_optimization_4x3_%j.out       # output file (%j expands to jobID)
#SBATCH -e unitary_optimization_4x3_%j.err       # error log file (%j expands to jobID)
#SBATCH --mail-type=ALL                      # Request status by email 
#SBATCH --mail-user=jek354@cornell.edu        # Email address to send results to.
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 1                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=65G                             # server memory requested (per node)
#SBATCH -t 24:00:00                           # Time limit (hh:mm:ss)
#SBATCH --partition=kim                       # Request partition

cd /home/jek354/research/ML-signproblem/experimenting/ed/
julia --project=/home/jek354/research/ML-signproblem/experimenting benchmark_krylov_ed.jl
