#=
submit_forward_trotter_3x3.jl

Submit Slurm sbatch Trotter optimization jobs for 3x3 (4, 4) and (4, 5) systems
scanning forward from U index 2 to 60 using the Slater determinant reference state.

Target folders:
  - N=(4, 4)_3x3
  - N=(4, 4)_3x3_2
  - N=(4, 5)_3x3
  - N=(4, 5)_3x3_2
  - N=(4, 5)_3x3_3

Parameters:
  u_start = 2
  u_end = 60
  --antihermitian
  --custom_ref_state=slater
  --loss=overlap
  --cpus-per-task=20
  --mem=20G
  --time=7-00:00:00
  --partition=kim
=#

using Dates
using Lattices
using LinearAlgebra
using SparseArrays
using Combinatorics
using HDF5
using JLD2

include("data_path.jl")
include("utility_functions.jl")
include("ed_objects.jl")
include("ed_functions.jl")
include("logging.jl")

function sanitize_name(str::String)::String
    return replace(str, " " => "_", "=" => "_", "(" => "", ")" => "", "," => "_")
end

function submit_forward_jobs()
    root_data = "/home/jek354/research/data/new_data/data"
    jobs_dir = "/home/jek354/research/ML-signproblem/jobs"
    mkpath(jobs_dir)

    target_folders = [
        "N=(4, 4)_3x3",
        "N=(4, 4)_3x3_2",
        "N=(4, 5)_3x3",
        "N=(4, 5)_3x3_2",
        "N=(4, 5)_3x3_3"
    ]

    exp_dir = "/home/jek354/research/ML-signproblem/experimenting/ed"
    submitted_jobs = Tuple{String, String}[]

    println("Job log files will be written to: $jobs_dir")

    for folder in target_folders
        full_folder_path = joinpath(root_data, folder)
        if !isdir(full_folder_path)
            println("Warning: Target folder does not exist: $full_folder_path")
            continue
        end

        safe_folder = sanitize_name(folder)
        job_name = "trotter_fwd_slater_$(safe_folder)"
        out_log = joinpath(jobs_dir, "$(job_name).out")
        err_log = joinpath(jobs_dir, "$(job_name).err")

        cmd_args = ["\"$(full_folder_path)\"", "2", "60", "--antihermitian", "--custom_ref_state=slater", "--loss=overlap"]
        cmd_str = "cd $(exp_dir) && julia --project=.. run_trotter_scan_optimization.jl " * join(cmd_args, " ")

        sbatch_cmd = `sbatch --mem=20G --cpus-per-task=20 --time=7-00:00:00 --partition=kim --job-name=$(job_name) --output=$(out_log) --error=$(err_log) --wrap=$(cmd_str)`

        println("Submitting Trotter forward job for '$folder'...")
        output_str = read(sbatch_cmd, String)
        m = match(r"Submitted batch job (\d+)", output_str)
        job_id = !isnothing(m) ? m.captures[1] : "UNKNOWN"
        println("   -> Job ID: $job_id ($job_name)")
        push!(submitted_jobs, (job_id, job_name))
    end

    println("\n==================================================")
    println("Summary:")
    println("  Submitted forward Trotter jobs: $(length(submitted_jobs))")
    println("==================================================")

    return submitted_jobs
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "submit_forward_trotter_3x3")
    with_logging(log_path) do
        submit_forward_jobs()
        return 0
    end
end
