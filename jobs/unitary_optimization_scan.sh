#!/bin/bash
#SBATCH -J 4x4_big                                   # Job name
#SBATCH -o 4x4_%j.out       # output file (%j expands to jobID)
#SBATCH -e 4x4_%j.err       # error log file (%j expands to jobID)
#SBATCH --mail-type=ALL                      # Request status by email 
#SBATCH --mail-user=jek354@cornell.edu        # Email address to send results to.
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 40                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=250G                             # server memory requested (per node)
#SBATCH --gres=gpu:1
#SBATCH -t 7-00:00:00                           # Time limit (hh:mm:ss)
#SBATCH --partition=kim                       # Request partition


cd /home/jek354/research/ML-signproblem/experimenting/ed/
julia --threads=auto --project=/home/jek354/research/ML-signproblem/experimenting run_lanczos_scan_optimization.jl "$@"
