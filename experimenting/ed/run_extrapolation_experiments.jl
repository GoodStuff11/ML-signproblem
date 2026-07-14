# run_extrapolation_experiments.jl
# Experiments targeting best extrapolation at U ≈ 8.
#
# All "_nonsplit" folder sets are DISJOINT from the test_folders in system_scaling.jl:
#   - small_nonsplit  : N=(3,3)_3x2_2, N=(3,3)_3x2_3
#   - medium_nonsplit : N=(3,3)_3x2_2, N=(3,3)_3x2_3, N=(4,4)_4x2_2
#   - mixed_nonsplit  : N=(3,3)_3x2_2, N=(3,3)_3x2_3, N=(3,3)_4x2, N=(4,4)_4x2_2
#
# Weighting schemes:
#   uniform       — equal weight for all U values
#   high_u        — exp(+0.5 * log10(U)), gentle favour of large U
#   high_u_strong — exp(+1.5 * log10(U)), aggressive large-U focus
#   focus_u8      — Gaussian centred at log10(8) ≈ 0.903, σ=0.4, peaks near U=8
#
# u-range 2:50 covers U ≈ 0.001..9.4 (the range of interest for U≈8 extrapolation).

experiments = [
    # --- Small training set (2 folders, replications of N=(3,3)_3x2) ---
    "--weighting=uniform       --u-range=2:50 --folder-set=small_nonsplit  --name=small_uniform",
    "--weighting=high_u        --u-range=2:50 --folder-set=small_nonsplit  --name=small_high_u",
    "--weighting=focus_u8      --u-range=2:50 --folder-set=small_nonsplit  --name=small_focus_u8",

    # --- Medium training set (3 folders, adds N=(4,4)_4x2_2) ---
    "--weighting=uniform       --u-range=2:50 --folder-set=medium_nonsplit --name=medium_uniform",
    "--weighting=high_u_strong --u-range=2:50 --folder-set=medium_nonsplit --name=medium_high_u_strong",
    "--weighting=focus_u8      --u-range=2:50 --folder-set=medium_nonsplit --name=medium_focus_u8",

    # --- Mixed training set (4 folders, adds N=(3,3)_4x2) ---
    "--weighting=focus_u8      --u-range=2:50 --folder-set=mixed_nonsplit  --name=mixed_focus_u8",
]

n_total = length(experiments)
for (i, exp) in enumerate(experiments)
    println("\n" * "="^60)
    println("STARTING EXPERIMENT $i/$n_total: $exp")
    println("="^60 * "\n")

    args = split(strip(exp), r"\s+")
    cmd = `julia --project=.. run_scaling.jl --strategy=neural $args`

    try
        run(cmd)
    catch e
        println("ERROR during Experiment $i: $e")
    end
end
