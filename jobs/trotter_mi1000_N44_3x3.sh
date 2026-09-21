#!/bin/bash
#SBATCH --job-name=trotter_mi1000_N44_3x3
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/trotter_mi1000_N44_3x3-%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/trotter_mi1000_N44_3x3-%j.err
#SBATCH --mem=32g
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --partition=kim

# Re-optimize the directly-optimized Trotter ansatz (num_exponentials=1, overlap loss)
# at u_idx=33 (U=8.0) with --maxiters=1000.
#
# The existing "barren_study" runs used run_trotter_scan_optimization.jl's DEFAULT
# --maxiters=100 which, with the 3-stage [:LBFGS, :GradientDescent, :LBFGS] chain, capped
# them at 303 iterations. Nine of the ten systems stopped at exactly that cap with terminal
# |grad| ~ 1e-3..4e-3 -- truncated mid-descent, not converged -- while the exact-exponential
# curve they are compared against in figure4.ipynb ran to |grad| ~ 3e-5. Written under a
# separate --run_label so the original barren_study files (which also feed figure 4
# panel (c)) are left untouched.

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd /home/jek354/research/ML-signproblem/experimenting/ed
julia --project=.. run_trotter_scan_optimization.jl "N=(4, 4)_3x3" 33 33 \
    --maxiters=1000 --loss=overlap --num_exponentials=1 \
    --antihermitian --custom_ref_state=slater \
    --run_label=barren_study_mi1000
