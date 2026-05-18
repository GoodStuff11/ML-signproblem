#!/bin/bash
#SBATCH -J 3x3                                   # Job name
#SBATCH -o 3x3_%j.out       # output file (%j expands to jobID)
#SBATCH -e 3x3_%j.err       # error log file (%j expands to jobID)
#SBATCH --mail-type=ALL                      # Request status by email 
#SBATCH --mail-user=jek354@cornell.edu        # Email address to send results to.
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 40                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=30G                             # server memory requested (per node)
#SBATCH -t 7-00:00:00                           # Time limit (hh:mm:ss)
#SBATCH --partition=aimi                       # Request partition


cd /home/jek354/research/ML-signproblem/experimenting/ed/
julia --threads=auto --project=/home/jek354/research/ML-signproblem/experimenting run_lanczos_scan_optimization.jl "$@"
