#=
run_trotter_scan_optimization.jl

Run Trotter optimization over a range of U interaction parameters using unitaries mapped from Lanczos ED data,
represented in the momentum basis.

Usage:
  julia --project=.. run_trotter_scan_optimization.jl [folder] [u_start] [u_end] [--maxiters=<number>] [--loss=<type>] [--num_exponentials=<number>] [--antihermitian] [--custom_ref_state=<value>] [--use_gpu=<bool>] [--datatype=<type>] [--grow_from_exponentials=<number>] [--grow_mode=<mode>] [--target_fidelity=<value>]

Arguments:
  folder (required): Path to the ED data folder (e.g., "N=(2, 2)_2x2").
  u_start (optional): Start index of U values, or direction. Default: "25".
                     Valid options:
                     - "forward": Scan forward from index 26 to the end of U values.
                     - "backward": Scan backward from index 18 down to 1.
                     - [integer]: Run a single specific U-index (if u_end is omitted or equal to u_start) or start of a range.
  u_end (optional): End index of U values (if specifying a range). Default: nothing.
                     Valid options:
                     - [integer]: End of index range. If equal to u_start or omitted, runs a single U index.
                     - omitted/nothing: Optimizes a single U value specified by u_start.
  --maxiters=<number> (optional): Maximum number of iterations for optimization. Default: 100.
  --loss=<type> (optional): The loss function to optimize. Default: "overlap".
                     Valid options:
                     - "overlap": Optimize overlap loss (1 - |<ψ'|U|ψ>|^2).
                     - "energy": Optimize energy loss (<ψ|U^† H U|ψ>).
  --num_exponentials=<number> (optional): Number of Trotter layers/steps. Default: 1.
  --antihermitian (optional): Use real-antihermitian generators instead of i * real-hermitian. Default: false.
  --custom_ref_state=<value> (optional): Use a custom reference state as a Slater determinant.
                     Valid options:
                     - "slater": The Slater determinant ground state of the tight-binding model.
                     - [integer]: Use the Slater determinant at this specific 1-based basis index.
  --use_gpu=<bool> (optional): Enable GPU acceleration for overlap loss and gradient calculations.
                     Uses dynamic strided gradient checkpointing (rematerialization) to automatically
                     fit computations within available GPU VRAM without risking Out-Of-Memory (OOM) crashes.
                     Default: false.
                     Valid options:
                     - "--use_gpu" or "--use_gpu=true": Enable CUDA GPU acceleration.
                     - "--use_gpu=false": Disable GPU acceleration.
  --datatype=<type> (optional): Data type for vector and matrix operations. Default: ComplexF64.
                     Valid options: ComplexF64, ComplexF32, Float64, Float32.
  --grow_from_exponentials=<number> (optional): Bootstrap this run's (larger) --num_exponentials ansatz
                     from an already-optimized run with this smaller num_exponentials value.
  --grow_mode=<mode> (optional): "chain" or "per_u". Default: "chain".
  --target_fidelity=<value> (optional): Prune parameter vector to reproduce target fidelity.
  --run_label=<string> (optional): Append this label to the saved-file prefix, so the run is always
                     freshly randomly initialized under a distinct filename rather than resuming from
                     (or overwriting) any existing coefficients already saved under the standard prefix.
=#

# Pre-scan ARGS for GPU flag before loading CUDA package
_use_gpu_prescan = let val = false
    for arg in ARGS
        if arg == "--use_gpu" || arg == "--use_gpu=true"
            val = true
        elseif arg == "--use_gpu=false"
            val = false
        end
    end
    val
end

if _use_gpu_prescan
    ENV["JULIA_CUDA_USE_COMPAT"] = "true"
    using CUDA
end

using Lattices
using LinearAlgebra
using Combinatorics
using SparseArrays
using Statistics
using Random
using JLD2
using HDF5
using Zygote

include("data_path.jl")
include("logging.jl")
include("utility_functions.jl")
using .UtilityFunctions
include("trotter.jl")
using .Trotter

include("ed_objects.jl")
include("ed_functions.jl")

"""
    parse_arguments(args::Vector{String})

Parse command line arguments for running Trotter scan optimization.
"""
function parse_arguments(args::Vector{String})
    maxiters = 100
    loss_type = :overlap
    num_exponentials = 1
    antihermitian = false
    custom_ref_state_arg = nothing
    use_gpu = false
    datatype = ComplexF64
    grow_from_exponentials = nothing
    grow_mode = :chain
    target_fidelity = nothing
    run_label = nothing
    filtered_args = String[]

    for arg in args
        if startswith(arg, "--maxiters=")
            maxiters = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--loss=")
            val = String(split(arg, "=", limit=2)[2])
            if val == "overlap"
                loss_type = :overlap
            elseif val == "energy"
                loss_type = :energy
            else
                error("Invalid --loss option: '$val'. Valid options are: 'overlap', 'energy'.")
            end
        elseif startswith(arg, "--num_exponentials=")
            num_exponentials = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--antihermitian" || startswith(arg, "--antihermitian=")
            if occursin("=", arg)
                antihermitian = parse(Bool, split(arg, "=", limit=2)[2])
            else
                antihermitian = true
            end
        elseif startswith(arg, "--custom_ref_state=")
            custom_ref_state_arg = split(arg, "=", limit=2)[2]
        elseif arg == "--use_gpu" || arg == "--use_gpu=true"
            use_gpu = true
        elseif arg == "--use_gpu=false"
            use_gpu = false
        elseif startswith(arg, "--datatype=")
            val = String(split(arg, "=", limit=2)[2])
            if val == "ComplexF64"
                datatype = ComplexF64
            elseif val == "ComplexF32"
                datatype = ComplexF32
            elseif val == "Float64"
                datatype = Float64
            elseif val == "Float32"
                datatype = Float32
            else
                error("Invalid --datatype option: '$val'. Valid options are: 'ComplexF64', 'ComplexF32', 'Float64', 'Float32'.")
            end
        elseif startswith(arg, "--grow_from_exponentials=")
            grow_from_exponentials = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--grow_mode=")
            val = String(split(arg, "=", limit=2)[2])
            if val == "chain"
                grow_mode = :chain
            elseif val == "per_u"
                grow_mode = :per_u
            else
                error("Invalid --grow_mode option: '$val'. Valid options are: 'chain', 'per_u'.")
            end
        elseif startswith(arg, "--target_fidelity=")
            target_fidelity = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--run_label=")
            run_label = String(split(arg, "=", limit=2)[2])
        else
            push!(filtered_args, arg)
        end
    end

    if isempty(filtered_args)
        error("Usage: julia run_trotter_scan_optimization.jl <folder> [u_start] [u_end] [options]")
    end

    if !isnothing(grow_from_exponentials) && grow_from_exponentials >= num_exponentials
        error("--grow_from_exponentials=$grow_from_exponentials must be less than --num_exponentials=$num_exponentials")
    end

    folder = data_folder(filtered_args[1])
    u_start = length(filtered_args) >= 2 ? filtered_args[2] : "25"
    u_end = length(filtered_args) >= 3 ? filtered_args[3] : nothing

    if !antihermitian && datatype <: Real
        datatype = ComplexF64
    end

    return folder, u_start, u_end, maxiters, loss_type, num_exponentials, antihermitian, custom_ref_state_arg, use_gpu, datatype, grow_from_exponentials, grow_mode, target_fidelity, run_label
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "run_trotter_scan_optimization")
    with_logging(log_path) do
        folder, u_start, u_end, maxiters, loss_type, num_exponentials, antihermitian, custom_ref_state_arg, use_gpu, datatype, grow_from_exponentials, grow_mode, target_fidelity, run_label = parse_arguments(ARGS)

        println("Number of threads: $(Threads.nthreads())")
        println("Use GPU: $use_gpu")
        println("Data Type: $datatype")

        # 1. Load ED data (loads indexer if JLD2, or we can use it to build the sector basis)
        U_values, state_vecs, indexer, _, N_elec, spin_conserved, _, sign_convention =
            load_ED_data(folder; verbose=true, sign_convention=:spin_first, use_slater_reference=custom_ref_state_arg == "slater")

        n_up, n_dn = N_elec

        # Parse dimension from folder name, default to (3, 3) if fails
        Lvec = parse_lattice_dimension(folder)
        N_sites = prod(Lvec)

        # 2. Computing the basis
        basis_sector = Trotter.get_basis_sector(indexer, Lvec, N_sites)

        # 3. Find the Hamiltonian
        @time H_hop_sector, basis_dict_sector, _ = Trotter.TamFermion.HubbardMomentumBasis(
            1.0, 0.0, Lvec, (n_up, n_dn); indexer=indexer
        )
        @time H_int_sector, _, _ = Trotter.TamFermion.HubbardMomentumBasis(
            0.0, 1.0, Lvec, (n_up, n_dn); indexer=indexer
        )

        # 4. Enumerate Trotter gates and tau terms
        @time gates = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=!antihermitian)
        @time tau_terms = Trotter.fgateToTauSector(gates, N_sites, basis_sector; antihermitian=antihermitian)

        # 5. Set up scan range
        scan_instructions = Dict{String,Any}(
            "starting level" => 1,
            "ending level" => 1,
            "num_exponentials" => num_exponentials,
            "antihermitian" => antihermitian
        )

        # u_build suffix only appears when increasing num_exponentials via --grow_from_exponentials;
        # run_label (if given) is always appended so this run never resumes from, or overwrites,
        # any existing coefficients saved under the standard (label-less) prefix.
        suffix_parts = String[]
        !isnothing(grow_from_exponentials) && push!(suffix_parts, "u_build")
        !isnothing(run_label) && push!(suffix_parts, run_label)
        suffix = isempty(suffix_parts) ? nothing : join(suffix_parts, "_")

        save_name_prefix = build_save_name_prefix(
            :trotter;
            sites=N_sites,
            electrons=N_elec,
            custom_ref_state_arg=custom_ref_state_arg,
            antihermitian=antihermitian,
            loss_type=loss_type,
            num_exponentials=num_exponentials,
            suffix=suffix
        )

        grow_from_save_name = if !isnothing(grow_from_exponentials)
            build_save_name_prefix(
                :trotter;
                sites=N_sites,
                electrons=N_elec,
                custom_ref_state_arg=custom_ref_state_arg,
                antihermitian=antihermitian,
                loss_type=loss_type,
                num_exponentials=grow_from_exponentials
            )
        else
            nothing
        end

        # Pruned (--target_fidelity) runs must not overwrite the full, unpruned coefficients
        # they are pruned from, so their output goes under a distinctly-named prefix; the
        # unpruned save_name_prefix computed above keeps pointing at the un-pruned file.
        pruned_suffix = if !isnothing(grow_from_exponentials)
            "u_build_target_fidelity=$(target_fidelity)"
        else
            "target_fidelity=$(target_fidelity)"
        end
        output_name_prefix = isnothing(target_fidelity) ? save_name_prefix : build_save_name_prefix(
            :trotter;
            sites=N_sites,
            electrons=N_elec,
            custom_ref_state_arg=custom_ref_state_arg,
            antihermitian=antihermitian,
            loss_type=loss_type,
            num_exponentials=num_exponentials,
            suffix=pruned_suffix
        )

        # Helper to find existing U index file, checking both current output_name_prefix and alternate u_build suffix
        function find_existing_u_file(prefix::String, u_val::Int)
            f = joinpath(folder, "$(prefix)_u_$(u_val).jld2")
            if isfile(f)
                return f
            end
            # Check opposite u_build presence
            alt_suffix = occursin("u_build", prefix) ? nothing : "u_build"
            alt_prefix = if isnothing(target_fidelity)
                build_save_name_prefix(
                    :trotter;
                    sites=N_sites,
                    electrons=N_elec,
                    custom_ref_state_arg=custom_ref_state_arg,
                    antihermitian=antihermitian,
                    loss_type=loss_type,
                    num_exponentials=num_exponentials,
                    suffix=alt_suffix
                )
            else
                alt_pruned_suffix = isnothing(alt_suffix) ? "target_fidelity=$(target_fidelity)" : "u_build_target_fidelity=$(target_fidelity)"
                build_save_name_prefix(
                    :trotter;
                    sites=N_sites,
                    electrons=N_elec,
                    custom_ref_state_arg=custom_ref_state_arg,
                    antihermitian=antihermitian,
                    loss_type=loss_type,
                    num_exponentials=num_exponentials,
                    suffix=alt_pruned_suffix
                )
            end
            alt_file = joinpath(folder, "$(alt_prefix)_u_$(u_val).jld2")
            if isfile(alt_file)
                return alt_file
            end
            return nothing
        end

        if u_end === nothing
            v1 = tryparse(Int, u_start)
            if isnothing(v1)
                if u_start == "forward"
                    println("Forward scan")
                    scan_instructions["u_range"] = 26:length(U_values)
                else
                    println("Backward scan")
                    scan_instructions["u_range"] = 18:-1:1
                end
                load_file = find_existing_u_file(output_name_prefix, 19)
                if !isnothing(load_file)
                    scan_instructions["load_file"] = load_file
                    println("Load: $(scan_instructions["load_file"])")
                end
            else
                println("Optimizing single U index: $v1 (U = $(U_values[v1]))")
                scan_instructions["u_range"] = v1:v1
                current_u_file = find_existing_u_file(output_name_prefix, v1)
                if !isnothing(current_u_file)
                    scan_instructions["load_file"] = current_u_file
                    println("Loading existing coefficients and loss history for current U value (u_idx = $v1): $current_u_file")
                end
            end
        else
            v1 = parse(Int, u_start)
            v2 = parse(Int, u_end)
            if v1 > v2
                scan_instructions["u_range"] = v1:-1:v2
                adjacent_file = find_existing_u_file(output_name_prefix, v1 + 1)
                if !isnothing(adjacent_file)
                    scan_instructions["load_file"] = adjacent_file
                end
            elseif v1 < v2
                scan_instructions["u_range"] = v1:v2
                adjacent_file = find_existing_u_file(output_name_prefix, v1 - 1)
                if !isnothing(adjacent_file)
                    scan_instructions["load_file"] = adjacent_file
                end
            else
                # v1 == v2 (single U value)
                scan_instructions["u_range"] = v1:v1
                current_u_file = find_existing_u_file(output_name_prefix, v1)
                if !isnothing(current_u_file)
                    scan_instructions["load_file"] = current_u_file
                    println("Loading existing coefficients and loss history for current U value (u_idx = $v1): $current_u_file")
                end
            end
        end

        # Pruning: restrict optimization to the subset of parameters that reproduces
        # --target_fidelity, per the pruning analysis file for this num_exponentials.
        active_indices = nothing
        pruning_threshold = nothing
        if !isnothing(target_fidelity)
            if length(scan_instructions["u_range"]) != 1
                error("--target_fidelity requires a single U index (u_start == u_end, not a range/forward/backward scan), so it can be (and always should be) run concurrently across different U indices. Got u_range=$(scan_instructions["u_range"]).")
            end
            u_idx = first(scan_instructions["u_range"])

            base_file = find_existing_u_file(save_name_prefix, u_idx)
            if isnothing(base_file)
                error("--target_fidelity requires an existing fully-optimized (unpruned) coefficients file for num_exponentials=$(num_exponentials) at U index $u_idx: $(joinpath(folder, "$(save_name_prefix)_u_$(u_idx).jld2")). Run the normal (non-pruned) scan for this U index first.")
            end
            base_coeffs = load_saved_dict(base_file)["coefficients"]

            pruning_prefix = build_save_name_prefix(
                "pruning_analysis_trotter";
                sites=N_sites,
                custom_ref_state_arg=custom_ref_state_arg,
                antihermitian=antihermitian,
                loss_type=loss_type,
                num_exponentials=num_exponentials
            )
            pruning_file = joinpath(folder, "$(pruning_prefix).jld2")
            if !isfile(pruning_file)
                error("--target_fidelity requires a pruning analysis file for num_exponentials=$(num_exponentials) (matching loss/antihermitian/custom_ref_state settings): $pruning_file. Run run_pruning_analysis.jl (with --num_exponentials=$(num_exponentials)) first.")
            end
            pruning_data = load(pruning_file)
            thresholds = pruning_data["thresholds"]
            error_data = pruning_data["error_data"]
            target_infidelity = 1 - target_fidelity
            best_idx = argmin(abs.(error_data[:, u_idx] .- target_infidelity))
            pruning_threshold = thresholds[best_idx]

            active_indices = findall(abs.(base_coeffs) .>= pruning_threshold)
            if isempty(active_indices)
                error("target_fidelity=$target_fidelity (threshold=$pruning_threshold) prunes all $(length(base_coeffs)) parameters at U index $u_idx -- nothing left to optimize.")
            end
            println("Target fidelity $target_fidelity => pruning threshold $pruning_threshold ($(length(base_coeffs) - length(active_indices))/$(length(base_coeffs)) parameters pruned)")
        end

        # 6. Run scan optimization
        Trotter.interaction_scan_map_to_state(
            state_vecs, scan_instructions, gates, tau_terms, basis_sector, N_sites;
            maxiters=maxiters,
            optimizer=[:LBFGS, :GradientDescent, :LBFGS],
            initialization_samples=10,
            H_hopping=H_hop_sector, H_interaction=H_int_sector,
            save_folder=folder, save_name=output_name_prefix,
            loss_type=loss_type,
            U_values=U_values,
            antihermitian=antihermitian,
            use_gpu=use_gpu,
            datatype=datatype,
            grow_from_num_exponentials=grow_from_exponentials,
            grow_from_save_name=grow_from_save_name,
            grow_mode=grow_mode,
            active_indices=active_indices,
            target_fidelity=target_fidelity,
            pruning_threshold=pruning_threshold
        )

        return 0
    end
end
