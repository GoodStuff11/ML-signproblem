#!/bin/bash
#SBATCH --partition=aimi
#SBATCH --time=01:00:00
#SBATCH --mem=10G

export JULIA_CPU_TARGET="generic"
cd /home/jek354/research/ML-signproblem/experimenting/ed/
julia --project=/home/jek354/research/ML-signproblem/experimenting -e "import Pkg; Pkg.precompile(); using CairoMakie; println(\"done\")"