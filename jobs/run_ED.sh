#!/bin/bash
#SBATCH -J ED                                   # Job name
#SBATCH -o ed_%j.out       # output file (%j expands to jobID)
#SBATCH -e ed_%j.err       # error log file (%j expands to jobID)
#SBATCH --mail-type=ALL                      # Request status by email 
#SBATCH --mail-user=jek354@cornell.edu        # Email address to send results to.
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 2                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=20G                             # server memory requested (per node)
#SBATCH -t 1-00:00:00                           # Time limit (hh:mm:ss)
#SBATCH --partition=kim                       # Request partition
#SBATCH --nodelist=kim-cpu-01

cd /home/jek354/research/ML-signproblem/experimenting/ed/
julia --threads=auto --project=/home/jek354/research/ML-signproblem/experimenting run_ed_lanczos_momentum.jl "$@"
