# run_extrapolation_experiments2.jl
# Second round of experiments targeting U ≈ 8 extrapolation.
#
# Key changes vs round 1:
#   - u-range 35:53 (U ≈ 2.8 to ~17), concentrating capacity near the target region
#   - New aspect-ratio folder sets: training on same-geometry (3x3) different fillings
#     based on the insight that N=(3,3)_3x3 + N=(2,2)_2x2 extrapolates well to N=(4,4)_3x3
#   - low_u and low_u_strong weightings (user found low-U weighting works better)
#
# Training folder sets (all disjoint from test_folders):
#   square_nonsplit  : N=(3,3)_3x3_newsign, N=(4,4)_3x3 (clamped), N=(4,5)_3x3
#   square_and_rect  : above + N=(3,3)_3x2_2, N=(3,3)_3x2_3
#   multiscale_3x3   : square_nonsplit + N=(4,4)_4x2_2, N=(3,3)_4x2
#   mixed_nonsplit   : N=(3,3)_3x2_2/3, N=(3,3)_4x2, N=(4,4)_4x2_2  (round 1 best)
#
# Weighting schemes:
#   low_u        : exp(-0.5 * log10(U))  — gentle low-U preference
#   low_u_strong : exp(-1.5 * log10(U))  — aggressive low-U preference
#   uniform      : equal weight

experiments = [
    # --- Square-geometry sets (same lattice shape as test targets) ---
    "--weighting=low_u        --u-range=35:53 --folder-set=square_nonsplit  --name=sq_low_u",
    "--weighting=low_u_strong --u-range=35:53 --folder-set=square_nonsplit  --name=sq_low_u_strong",
    "--weighting=uniform      --u-range=35:53 --folder-set=square_nonsplit  --name=sq_uniform",

    # --- Square + rectangular mix ---
    "--weighting=low_u        --u-range=35:53 --folder-set=square_and_rect  --name=sqrect_low_u",
    "--weighting=low_u_strong --u-range=35:53 --folder-set=square_and_rect  --name=sqrect_low_u_strong",

    # --- Multiscale: 3x3 geometry + 4x2 rectangles ---
    "--weighting=low_u        --u-range=35:53 --folder-set=multiscale_3x3   --name=multi3x3_low_u",

    # --- Rerun of round-1 best with low_u weighting ---
    "--weighting=low_u        --u-range=35:53 --folder-set=mixed_nonsplit   --name=mixed_low_u",
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
