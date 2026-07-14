# run_extrapolation_experiments3.jl
# Third round of experiments targeting U ≈ 8 extrapolation.
#
# Key changes vs round 2:
#   - Focus exclusively on clean square-geometry (3x3 / 2x2) training sets,
#     dropping duplicate datasets (_2, _3 variants of the same filling)
#     that risk confusing training with near-identical but non-identical data.
#   - Three new folder sets:
#       square_pure     : N=(3,3)_3x3_newsign + N=(4,5)_3x3 (2 clean fillings)
#       square_extended : above + N=(4,5)_3x3_3 (3 distinct fillings, all full)
#       square_with_2x2 : square_pure + N=(2,2)_2x2 (adds a small-system anchor)
#     Note: square_nonsplit (from round 2) also included the sparse N=(4,4)_3x3
#     (only 3 u-files), which these sets drop in favour of cleaner data.
#   - Broader sweep of low-U weighting exponents:
#       low_u_mild        : α = 0.25  (gentler than low_u)
#       low_u             : α = 0.50  (round 2 best)
#       low_u_strong      : α = 1.50  (round 2 runner-up)
#       low_u_very_strong : α = 3.00  (new extreme)
#   - u-range 35:53 retained (U ≈ 2.8–17), same as round 2.
#
# Training folder contents (files with u-data):
#   N=(3,3)_3x3_newsign : 63 files  — distinct sign convention
#   N=(4,5)_3x3         : 64 files  — full
#   N=(4,5)_3x3_3       : 63 files  — full, distinct filling from N=(4,5)_3x3
#   N=(2,2)_2x2         : 61 files  — small square anchor (also a test folder)

experiments = [
    # --- square_pure: minimal, no duplicates ---
    "--weighting=low_u_mild        --u-range=35:53 --folder-set=square_pure      --name=pure_low_u_mild",
    "--weighting=low_u             --u-range=35:53 --folder-set=square_pure      --name=pure_low_u",
    "--weighting=low_u_strong      --u-range=35:53 --folder-set=square_pure      --name=pure_low_u_strong",
    "--weighting=low_u_very_strong --u-range=35:53 --folder-set=square_pure      --name=pure_low_u_very_strong",

    # --- square_extended: adds a third distinct square filling ---
    "--weighting=low_u_mild        --u-range=35:53 --folder-set=square_extended  --name=ext_low_u_mild",
    "--weighting=low_u             --u-range=35:53 --folder-set=square_extended  --name=ext_low_u",
    "--weighting=low_u_strong      --u-range=35:53 --folder-set=square_extended  --name=ext_low_u_strong",

    # --- square_with_2x2: adds small-system 2x2 square anchor ---
    "--weighting=low_u_mild        --u-range=35:53 --folder-set=square_with_2x2  --name=sq2x2_low_u_mild",
    "--weighting=low_u             --u-range=35:53 --folder-set=square_with_2x2  --name=sq2x2_low_u",
    "--weighting=low_u_strong      --u-range=35:53 --folder-set=square_with_2x2  --name=sq2x2_low_u_strong",
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
