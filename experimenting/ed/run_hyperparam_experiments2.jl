# run_hyperparam_experiments2.jl
#
# Further investigation of smaller neural network architectures for
# low_u_mild weighting on square_pure.
#
# Prior sweep results:
#   small (64-width, 2-layer):  5/5 non-trivial success, but weaker pred/rand
#   std   (128-width, 2-layer): 5/5 non-trivial success, best overall pred/rand
#   large (256-width):          4/5 — overfitting
#
# This sweep explores finer gradations below the standard size:
#   1. tiny          : base=[32,32], embed=16, ctx=[16,8], scale=[8,4]
#   2. small_shallow : base=[64], embed=32, ctx=[32], scale=[16]        (1 hidden layer)
#   3. small_deep    : base=[64,64,64], embed=32, ctx=[32,16], scale=[16,8]
#   4. mid_narrow    : base=[96,96], embed=48, ctx=[48,24], scale=[24,12]
#   5. mid_tapered   : base=[128,64], embed=48, ctx=[48,24], scale=[24,12]
#   6. std_shallow   : base=[128], embed=64, ctx=[64], scale=[32]       (1 hidden layer)

experiments = [
    "--weighting=low_u_mild --u-range=35:53 --folder-set=square_pure --base-hidden=32,32 --embed-dim=16 --context-hidden=16,8 --scale-hidden=8,4 --name=pure_mild_tiny",

    "--weighting=low_u_mild --u-range=35:53 --folder-set=square_pure --base-hidden=64 --embed-dim=32 --context-hidden=32 --scale-hidden=16 --name=pure_mild_small_shallow",

    "--weighting=low_u_mild --u-range=35:53 --folder-set=square_pure --base-hidden=64,64,64 --embed-dim=32 --context-hidden=32,16 --scale-hidden=16,8 --name=pure_mild_small_deep",

    "--weighting=low_u_mild --u-range=35:53 --folder-set=square_pure --base-hidden=96,96 --embed-dim=48 --context-hidden=48,24 --scale-hidden=24,12 --name=pure_mild_mid_narrow",

    "--weighting=low_u_mild --u-range=35:53 --folder-set=square_pure --base-hidden=128,64 --embed-dim=48 --context-hidden=48,24 --scale-hidden=24,12 --name=pure_mild_mid_tapered",

    "--weighting=low_u_mild --u-range=35:53 --folder-set=square_pure --base-hidden=128 --embed-dim=64 --context-hidden=64 --scale-hidden=32 --name=pure_mild_std_shallow",
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
