#=
submit_all_pruning_jobs.jl

Programmatically submit Slurm sbatch jobs for pruning analysis on all finished
optimization runs across `data` and `data_new_sign` for system sizes < 12 sites.

Usage:
  julia --project=.. submit_all_pruning_jobs.jl [--force]

Options:
  --force (optional): Force resubmission of pruning analysis even if `pruning_analysis_\$(prefix).jld2` exists.
                      Default: false (skips already completed pruning analysis files).

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

"""
    parse_arguments(args::Vector{String})

Parse command line options for submit_all_pruning_jobs.jl.
"""
function parse_arguments(args::Vector{String})
    force = false
    for arg in args
        if arg == "--force"
            force = true
        end
    end
    return force
end

"""
    sanitize_name(str::String) -> String

Sanitize a folder or job name for clean filename generation.
"""
function sanitize_name(str::String)::String
    return replace(str, " " => "_", "=" => "_", "(" => "", ")" => "", "," => "_")
end

"""
    submit_pruning_jobs(force::Bool)

Scan data and data_new_sign directories for finished optimizations and submit sbatch pruning jobs.
"""
function submit_pruning_jobs(force::Bool)
    root_data = "/home/jek354/research/data/new_data/data"
    root_data_new_sign = "/home/jek354/research/data/new_data/data_new_sign"

    roots = [("data", root_data), ("data_new_sign", root_data_new_sign)]

    jobs_dir = "/home/jek354/research/ML-signproblem/jobs"
    mkpath(jobs_dir)

    submitted_jobs = Tuple{String, String}[]
    skipped_existing = 0
    skipped_no_data = 0

    println("Pruning job logs will be written to: $jobs_dir")

    for (root_label, root_path) in roots
        !isdir(root_path) && continue
        folders = readdir(root_path)

        for folder in folders
            full_folder_path = joinpath(root_path, folder)
            !isdir(full_folder_path) && continue
            (folder == "tmp" || occursin("copy", folder)) && continue

            dim = parse_lattice_dimension(folder)
            isnothing(dim) && continue
            sites = prod(dim)

            # System size filter: strictly less than 12 sites
            if sites >= 12
                println("Skipping folder '$folder' (sites = $sites >= 12)")
                continue
            end

            N_elec = parse_electron_count(folder)
            
            use_symmetry = false
            try
                _, _, _, _, _, _, sym, _ = load_ED_data(full_folder_path; verbose=false)
                use_symmetry = sym
            catch
                jld2_path = joinpath(full_folder_path, "meta_data_and_E.jld2")
                if isfile(jld2_path)
                    d = load_saved_dict(jld2_path)
                    use_symmetry = get(d["meta_data"], "use_symmetry", false)
                end
            end

            safe_folder = sanitize_name(folder)
            exp_dir = "/home/jek354/research/ML-signproblem/experimenting/ed"

            # 4 combinations
            tasks = [
                (
                    type = :exact,
                    type_str = "exact",
                    ref_arg = "slater",
                    label = "exact_slater",
                    prefix = build_save_name_prefix(:exact; electrons=N_elec, use_symmetry=use_symmetry, custom_ref_state_arg="slater", antihermitian=true, loss_type=:overlap),
                    cli_extra = ["--type=exact", "--custom_ref_state=slater", "--antihermitian", "--loss=overlap"]
                ),
                (
                    type = :exact,
                    type_str = "exact",
                    ref_arg = nothing,
                    label = "exact_nonslater",
                    prefix = build_save_name_prefix(:exact; electrons=N_elec, use_symmetry=use_symmetry, custom_ref_state_arg=nothing, antihermitian=true, loss_type=:overlap),
                    cli_extra = ["--type=exact", "--antihermitian", "--loss=overlap"]
                ),
                (
                    type = :trotter,
                    type_str = "trotter",
                    ref_arg = "slater",
                    label = "trotter_slater",
                    prefix = build_save_name_prefix(:trotter; sites=sites, custom_ref_state_arg="slater", antihermitian=true, loss_type=:overlap),
                    cli_extra = ["--type=trotter", "--custom_ref_state=slater", "--antihermitian", "--loss=overlap"]
                ),
                (
                    type = :trotter,
                    type_str = "trotter",
                    ref_arg = nothing,
                    label = "trotter_nonslater",
                    prefix = build_save_name_prefix(:trotter; sites=sites, custom_ref_state_arg=nothing, antihermitian=true, loss_type=:overlap),
                    cli_extra = ["--type=trotter", "--antihermitian", "--loss=overlap"]
                )
            ]

            files = readdir(full_folder_path)

            for t in tasks
                prefix = t.prefix
                shared_data_path = joinpath(full_folder_path, "$(prefix)_shared.jld2")
                pruning_output_path = joinpath(full_folder_path, "pruning_analysis_$(prefix).jld2")
                u_files = filter(x -> startswith(x, "$(prefix)_u_") && endswith(x, ".jld2"), files)

                # Check if optimization has runs available to prune
                if !isfile(shared_data_path) || isempty(u_files)
                    println("Skipping $(t.label) for $folder: No optimization data found (u_files: $(length(u_files)), shared: $(isfile(shared_data_path)))")
                    skipped_no_data += 1
                    continue
                end

                # Check if pruning analysis output already exists
                if !force && isfile(pruning_output_path)
                    println("Skipping existing pruning output: $pruning_output_path")
                    skipped_existing += 1
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
    println("  Submitted pruning jobs: $(length(submitted_jobs))")
    println("  Skipped (pruning output already exists): $skipped_existing")
    println("  Skipped (no optimization data): $skipped_no_data")
    println("==================================================")

    return submitted_jobs
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "submit_all_pruning_jobs")
    with_logging(log_path) do
        force = parse_arguments(ARGS)
        submit_pruning_jobs(force)
        return 0
    end
end
