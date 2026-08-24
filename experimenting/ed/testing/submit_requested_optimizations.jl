#=
submit_requested_optimizations.jl

Submit optimization jobs for the 4 systems with 4 options each:
Systems:
  1. N=(5, 4)_3x3
  2. N=(4, 4)_3x3
  3. N=(3, 3)_3x2
  4. N=(3, 2)_3x2

Options:
  1. Trotter with Slater reference
  2. Trotter with default reference (eigenstate)
  3. Exact with Slater reference
  4. Exact with default reference (eigenstate)

Usage:
  julia --project=.. submit_requested_optimizations.jl [--force] [--dry-run]
=#

using Dates
using Lattices
using LinearAlgebra
using SparseArrays
using Combinatorics
using HDF5
using JLD2

include("../data_path.jl")
include("../utility_functions.jl")
include("../ed_objects.jl")
include("../ed_functions.jl")
include("../logging.jl")

function parse_cli(args::Vector{String})
    force = false
    dry_run = false
    for arg in args
        if arg == "--force"
            force = true
        elseif arg == "--dry-run"
            dry_run = true
        end
    end
    return force, dry_run
end

function sanitize_name(str::String)::String
    return replace(str, " " => "_", "=" => "_", "(" => "", ")" => "", "," => "_")
end

function get_requested_runs()
    data_root = "/home/jek354/research/data/new_data/data_h5_fixed"
    systems = [
        "N=(5, 4)_3x3",
        "N=(4, 4)_3x3",
        "N=(3, 3)_3x2",
        "N=(3, 2)_3x2"
    ]

    runs = []
    for sys in systems
        full_path = joinpath(data_root, sys)
        dim = parse_lattice_dimension(sys)
        sites = prod(dim)
        N_elec = parse_electron_count(sys)

        use_symmetry = false
        try
            _, _, _, _, _, _, sym, _ = load_ED_data(full_path; verbose=false)
            use_symmetry = sym
        catch e
            @warn "Could not read symmetry for $sys: $e"
        end

        # 4 options per system
        options = [
            (
                mode = :trotter_slater,
                script = "run_trotter_scan_optimization.jl",
                label = "trotter_slater",
                prefix = build_save_name_prefix(:trotter; sites=sites, custom_ref_state_arg="slater", antihermitian=true, loss_type=:overlap),
                cli_extra = ["--antihermitian", "--custom_ref_state=slater", "--loss=overlap", "--use_gpu=false"]
            ),
            (
                mode = :trotter_default,
                script = "run_trotter_scan_optimization.jl",
                label = "trotter_default",
                prefix = build_save_name_prefix(:trotter; sites=sites, custom_ref_state_arg=nothing, antihermitian=true, loss_type=:overlap),
                cli_extra = ["--antihermitian", "--loss=overlap", "--use_gpu=false"]
            ),
            (
                mode = :exact_slater,
                script = "run_lanczos_scan_optimization.jl",
                label = "exact_slater",
                prefix = build_save_name_prefix(:exact; electrons=N_elec, use_symmetry=use_symmetry, custom_ref_state_arg="slater", antihermitian=true, loss_type=:overlap),
                cli_extra = ["--antihermitian", "--custom_ref_state=slater", "--loss=overlap", "--use-gpu=false"]
            ),
            (
                mode = :exact_default,
                script = "run_lanczos_scan_optimization.jl",
                label = "exact_default",
                prefix = build_save_name_prefix(:exact; electrons=N_elec, use_symmetry=use_symmetry, custom_ref_state_arg=nothing, antihermitian=true, loss_type=:overlap),
                cli_extra = ["--antihermitian", "--loss=overlap", "--use-gpu=false"]
            )
        ]

        for opt in options
            push!(runs, (
                system = sys,
                full_path = full_path,
                sites = sites,
                electrons = N_elec,
                opt...
            ))
        end
    end
    return runs
end

function submit_all(force::Bool, dry_run::Bool)
    jobs_dir = "/home/jek354/research/ML-signproblem/jobs"
    mkpath(jobs_dir)
    exp_dir = "/home/jek354/research/ML-signproblem/experimenting/ed"

    runs = get_requested_runs()
    results = []

    for r in runs
        shared_file = joinpath(r.full_path, "$(r.prefix)_shared.jld2")
        safe_sys = sanitize_name(r.system)
        job_name = "$(r.label)_$(safe_sys)"
        out_log = joinpath(jobs_dir, "$(job_name).out")
        err_log = joinpath(jobs_dir, "$(job_name).err")

        cmd_args = ["\"$(r.full_path)\"", "60", "2"]
        append!(cmd_args, r.cli_extra)
        cmd_str = "cd $(exp_dir) && julia --project=.. $(r.script) " * join(cmd_args, " ")

        sbatch_cmd = `sbatch --mem=20G --cpus-per-task=20 --time=7-00:00:00 --partition=kim --job-name=$(job_name) --output=$(out_log) --error=$(err_log) --wrap=$(cmd_str)`

        if !force && isfile(shared_file)
            println("Skipping existing completed run: $shared_file")
            push!(results, (system=r.system, option=r.label, status="SKIPPED (already exists)", job_id="N/A", cmd=string(sbatch_cmd)))
            continue
        end

        if dry_run
            println("[DRY-RUN] Would submit: $job_name")
            println("          Command: $sbatch_cmd")
            push!(results, (system=r.system, option=r.label, status="DRY-RUN", job_id="DRY-RUN", cmd=string(sbatch_cmd)))
        else
            println("Submitting Slurm job: $job_name ...")
            output_str = read(sbatch_cmd, String)
            m = match(r"Submitted batch job (\d+)", output_str)
            job_id = !isnothing(m) ? m.captures[1] : "UNKNOWN"
            println("   -> Job ID: $job_id")
            push!(results, (system=r.system, option=r.label, status="SUBMITTED", job_id=job_id, cmd=string(sbatch_cmd)))
        end
    end

    println("\n==================================================")
    println("Submission Summary:")
    for res in results
        println("  [$(res.status)] $(res.system) | $(res.option) -> Job ID: $(res.job_id)")
    end
    println("==================================================")
    return results
end

function (@main)(ARGS)
    force, dry_run = parse_cli(ARGS)
    submit_all(force, dry_run)
    return 0
end
