#!/bin/bash
#SBATCH --job-name=ffsim_6_3
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/ffsim_6_3_%j.log
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/ffsim_6_3_%j.log
#SBATCH --partition=kim
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=20g

cd /home/jek354/research/ML-signproblem/experimenting/ucc
conda run -n lit_env python barren_plateau_ffsim.py --norb=6 --nocc=3 --num_samples=20 --hamiltonian_type=hubbard
