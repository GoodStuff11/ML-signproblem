# run_hyperparam_experiments.jl
#
# Investigating different neural network sizes for low_u_mild weighting scheme
# on the square_pure folder set.
#
# Configured experiments:
#   1. small  : base=[64, 64], embed=32, ctx=[32, 16], scale=[16, 8]
#   2. std    : base=[128, 128], embed=64, ctx=[64, 32], scale=[32, 16] (baseline)
#   3. large  : base=[256, 256], embed=128, ctx=[128, 64], scale=[64, 32]
#   4. deep   : base=[256, 256, 256], embed=128, ctx=[128, 64], scale=[64, 32]
#

experiments = [
    "--weighting=low_u_mild --u-range=35:53 --folder-set=square_pure --base-hidden=64,64 --embed-dim=32 --context-hidden=32,16 --scale-hidden=16,8 --name=pure_mild_small",
    
    "--weighting=low_u_mild --u-range=35:53 --folder-set=square_pure --base-hidden=128,128 --embed-dim=64 --context-hidden=64,32 --scale-hidden=32,16 --name=pure_mild_std",
    
    "--weighting=low_u_mild --u-range=35:53 --folder-set=square_pure --base-hidden=256,256 --embed-dim=128 --context-hidden=128,64 --scale-hidden=64,32 --name=pure_mild_large",
    
    "--weighting=low_u_mild --u-range=35:53 --folder-set=square_pure --base-hidden=256,256,256 --embed-dim=128 --context-hidden=128,64 --scale-hidden=64,32 --name=pure_mild_deep",
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
