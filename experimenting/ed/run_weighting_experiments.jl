# run_weighting_experiments.jl
# Launch the remaining 3 alternative weighting scheme experiments in parallel on the SLURM cluster.

experiments = [
    ("loss_mild", "--weighting=loss_mild --u-range=35:53 --folder-set=square_pure --base-hidden=96,96 --embed-dim=48 --context-hidden=48,24 --scale-hidden=24,12 --name=pure_mid_narrow_loss_mild"),
    ("loss_std", "--weighting=loss_std --u-range=35:53 --folder-set=square_pure --base-hidden=96,96 --embed-dim=48 --context-hidden=48,24 --scale-hidden=24,12 --name=pure_mid_narrow_loss_std"),
    ("loss_power_std", "--weighting=loss_power_std --u-range=35:53 --folder-set=square_pure --base-hidden=96,96 --embed-dim=48 --context-hidden=48,24 --scale-hidden=24,12 --name=pure_mid_narrow_loss_power_std")
]

processes = []
println("Launching SLURM jobs in parallel on partition 'kim'...")

for (name, exp_args) in experiments
    args_list = split(strip(exp_args), r"\s+")
    
    # We use srun to submit a GPU job on partition kim
    # Command: srun --mem=20g --gres=gpu:1 --cpus-per-task=4 --time=1:00:00 --partition=kim julia --project=.. run_scaling.jl --strategy=neural <args>
    cmd = `srun --mem=20g --gres=gpu:1 --cpus-per-task=4 --time=1:00:00 --partition=kim julia --project=.. run_scaling.jl --strategy=neural $args_list`
    
    println("Starting job for $name...")
    # Run in the background (wait=false)
    proc = run(cmd, wait=false)
    push!(processes, (name, proc))
    
    # Stagger launches to prevent lock contention
    sleep(15)
end

println("All jobs submitted. Waiting for completion...")

for (name, proc) in processes
    try
        wait(proc)
        println("Job '$name' completed.")
    catch e
        println("ERROR: Job '$name' failed with: $e")
    end
end

println("All parallel experiments completed!")
