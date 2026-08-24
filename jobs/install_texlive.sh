#!/bin/bash
#SBATCH --job-name=texlive_install
#SBATCH --partition=kim
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/texlive_install_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/texlive_install_%j.err

/home/jek354/.local/src_texlive/install-tl-20260814/install-tl --profile=/home/jek354/.local/src_texlive/texlive.profile --location https://mirror.ctan.org/systems/texlive/tlnet
