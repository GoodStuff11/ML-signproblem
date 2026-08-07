#=
submit_trotter_pruning_filtered.jl

Submit Slurm sbatch jobs for pruning analysis on Trotter Anti-Hermitian optimizations
for all systems excluding 4x3 and 4x4 lattices. Runs for both Slater and Non-Slater reference states.

Target exclusions:
  - Excludes 4x3 (dim == [4, 3] or [3, 4])
  - Excludes 4x4 (dim == [4, 4])

Slurm parameters:
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

function submit_filtered_pruning_jobs()
    root_data = "/home/jek354/research/data/new_data/data"
    root_data_new_sign = "/home/jek354/research/data/new_data/data_new_sign"

    roots = [("data", root_data), ("data_new_sign", root_data_new_sign)]

    jobs_dir = "/home/jek354/research/ML-signproblem/jobs"
    mkpath(jobs_dir)

    submitted_jobs = Tuple{String, String}[]
    skipped_no_data = 0

    println("Pruning job log files will be written to: $jobs_dir")

    for (root_label, root_path) in roots
        !isdir(root_path) && continue
        folders = readdir(root_path)

        for folder in folders
            full_folder_path = joinpath(root_path, folder)
            !isdir(full_folder_path) && continue
            (folder == "tmp" || occursin("copy", folder)) && continue

            dim = parse_lattice_dimension(folder)
            isnothing(dim) && continue

            # Filter out 4x3 and 4x4
            if dim == [4, 3] || dim == [3, 4] || dim == [4, 4]
                println("Skipping excluded folder '$folder' (dim = $dim)")
                continue
            end

            sites = prod(dim)
            N_elec = parse_electron_count(folder)
            safe_folder = sanitize_name(folder)
            exp_dir = "/home/jek354/research/ML-signproblem/experimenting/ed"

            # 2 Trotter tasks (Slater ref & Non-Slater ref)
            tasks = [
                (
                    label = "trotter_slater",
                    prefix = build_save_name_prefix(:trotter; sites=sites, custom_ref_state_arg="slater", antihermitian=true, loss_type=:overlap),
                    cli_extra = ["--type=trotter", "--custom_ref_state=slater", "--antihermitian", "--loss=overlap"]
                ),
                (
                    label = "trotter_nonslater",
                    prefix = build_save_name_prefix(:trotter; sites=sites, custom_ref_state_arg=nothing, antihermitian=true, loss_type=:overlap),
                    cli_extra = ["--type=trotter", "--antihermitian", "--loss=overlap"]
                )
            ]

            files = readdir(full_folder_path)

            for t in tasks
                prefix = t.prefix
                shared_data_path = joinpath(full_folder_path, "$(prefix)_shared.jld2")
                u_files = filter(x -> startswith(x, "$(prefix)_u_") && endswith(x, ".jld2"), files)

                if !isfile(shared_data_path) || isempty(u_files)
                    println("Skipping $(t.label) for $folder ($root_label): No optimization data (u_files: $(length(u_files)), shared: $(isfile(shared_data_path)))")
                    skipped_no_data += 1
                    continue
                end

                job_name = "pruning_$(t.label)_$(root_label)_$(safe_folder)"
                out_log = joinpath(jobs_dir, "$(job_name).out")
                err_log = joinpath(jobs_dir, "$(job_name).err")

                cmd_args = ["\"$(full_folder_path)\""]
                append!(cmd_args, t.cli_extra)

                cmd_str = "cd $(exp_dir) && julia --project=.. run_pruning_analysis.jl " * join(cmd_args, " ")

                sbatch_cmd = `sbatch --mem=20G --cpus-per-task=20 --time=7-00:00:00 --partition=kim --job-name=$(job_name) --output=$(out_log) --error=$(err_log) --wrap=$(cmd_str)`

                println("Submitting Slurm pruning job: $job_name ...")
                output_str = read(sbatch_cmd, String)
                m = match(r"Submitted batch job (\d+)", output_str)
                job_id = !isnothing(m) ? m.captures[1] : "UNKNOWN"
                println("   -> Job ID: $job_id")
                push!(submitted_jobs, (job_id, job_name))
            end
        end
    end

    println("\n==================================================")
    println("Summary:")
    println("  Submitted Trotter pruning jobs: $(length(submitted_jobs))")
    println("  Skipped (no optimization data): $skipped_no_data")
    println("==================================================")

    return submitted_jobs
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "submit_trotter_pruning_filtered")
    with_logging(log_path) do
        submit_filtered_pruning_jobs()
        return 0
    end
end
