#!/bin/bash
#SBATCH -J 4x4                                   # Job name
#SBATCH -o scan_unitary_optimization_4x4_%j.out       # output file (%j expands to jobID)
#SBATCH -e scan_unitary_optimization_4x4_%j.err       # error log file (%j expands to jobID)
#SBATCH --mail-type=ALL                      # Request status by email 
#SBATCH --mail-user=jek354@cornell.edu        # Email address to send results to.
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 80                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=120G                             # server memory requested (per node)
#SBATCH -t 48:00:00                           # Time limit (hh:mm:ss)
#SBATCH --partition=aimi                       # Request partition
#SBATCH --nodelist=aimi-cpu-01

cd /home/jek354/research/ML-signproblem/experimenting/ed/
if [ "$#" -eq 2 ]; then
    julia --threads=80 --project=/home/jek354/research/ML-signproblem/experimenting run_lanczos_scan_optimization.jl "$1" "$2"
elif [ "$#" -eq 1 ]; then
    julia --threads=80 --project=/home/jek354/research/ML-signproblem/experimenting run_lanczos_scan_optimization.jl "$1"
else
    julia --threads=80 --project=/home/jek354/research/ML-signproblem/experimenting run_lanczos_scan_optimization.jl
fi