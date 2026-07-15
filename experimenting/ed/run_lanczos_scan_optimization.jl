#=
run_lanczos_scan_optimization.jl

Run optimization over a range of U interaction parameters using unitaries mapped from Lanczos ED data.

Usage:
  julia --project=.. run_lanczos_scan_optimization.jl [folder] [u_start] [u_end] [--nn=<nn_strategy>] [--maxiters=<number>] [--loss=<type>] [--use-gpu=<true|false>]

Arguments:
  folder (required): Path to the ED data folder (e.g., "data/N=(2, 2)_2x2").
  u_start (optional): Start index of U values, or direction. Default: 25.
                     Valid options:
                     - "forward": Scan forward from index 26 to the end of U values.
                     - "backward": Scan backward from index 18 down to 1.
                     - [integer]: Run a single specific U-index or the start of a range.
  u_end (optional): End index of U values (if specifying a range). Default: nothing.
  --nn=<nn_strategy> (optional): Name or path of neural network strategy file to load.
  --maxiters=<number> (optional): Maximum number of iterations for optimization. Default: 200.
  --loss=<type> (optional): The loss function to optimize. Default: "overlap".
                     Valid options:
                     - "overlap": Optimize overlap loss (1 - |<ψ'|U|ψ>|^2).
                     - "energy": Optimize energy loss (<ψ|U^† H U|ψ>).
  --use-gpu=<true|false> (optional): Whether to use GPU acceleration (CUDA). If set to false, it runs entirely on CPU (using multiple threads if julia is started with threads) without loading the CUDA package. Default: true (or auto-detect if GPU is available).

Examples:
  julia --project=.. run_lanczos_scan_optimization.jl "data/N=(2, 2)_2x2" 25
  julia --project=.. run_lanczos_scan_optimization.jl "data/N=(2, 2)_2x2" 25 35
  julia --project=.. run_lanczos_scan_optimization.jl "data/N=(2, 2)_2x2" forward --nn="3x3_(3, 3)_and_2x2_(2,2)"
  julia --project=.. run_lanczos_scan_optimization.jl "data/N=(2, 2)_2x2" 25 --loss=energy
  julia --project=.. run_lanczos_scan_optimization.jl "data/N=(2, 2)_2x2" 25 --use-gpu=false
=#

using Lattices
using LinearAlgebra
using Combinatorics
using SparseArrays
using Plots
import Graphs
using LaTeXStrings
using Statistics
using Random
using Zygote
using Optimization, OptimizationOptimisers
using JSON
using OptimizationOptimJL
using JLD2
using HDF5
using KrylovKit

# Pre-scan ARGS for --use-gpu before loading CUDA
_use_gpu = let val = nothing
    for arg in ARGS
        if startswith(arg, "--use-gpu=")
            val = parse(Bool, split(arg, "=", limit=2)[2])
        end
    end
    val
end

if _use_gpu !== false
    try
        ENV["JULIA_CUDA_USE_COMPAT"] = "true"
        using CUDA
        if CUDA.functional()
            @info "GPU available: $(CUDA.name(CUDA.CuDevice(0)))"
        else
            @info "CUDA not functional yet — trying local_toolkit mode"
            CUDA.set_runtime_version!(local_toolkit=true)
            if CUDA.functional()
                @info "GPU available: $(CUDA.name(CUDA.CuDevice(0)))"
            else
                @info "CUDA not functional. CPU fallback will be used."
            end
        end
    catch e
        @warn "CUDA loading or initialization failed: $e. CPU fallback will be used."
    end
end

# LinearAlgebra.BLAS.set_num_threads(1)

using Dates


include("utility_functions.jl")
include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("nn_strategy.jl")
include("logging.jl")


"""
    parse_arguments(args::Vector{String})

Parse command line arguments for running Lanczos scan optimization.
Expected arguments:
1. folder (String): The directory containing exact diagonalization data.
2. u_start (String): The starting U-index or direction ("forward", "backward"). Default: "25".
3. u_end (String): The ending U-index (optional).
4. --nn=<path> (String): Optional neural network strategy parameter.
5. --maxiters=<number> (Int): Optional maximum iterations parameter. Default: 200.
6. --loss=<type> (String): The loss function to optimize ("overlap", "energy"). Default: "overlap".
7. --use-gpu=<true|false> (Bool): Optional flag to enable/disable GPU. Handled by pre-scan.
8. --custom_ref_state=<value> (String): Use a custom reference state as a Slater determinant.
"""
function parse_arguments(args::Vector{String})
    nn_strategy_file = nothing
    maxiters = 200
    loss_type = :overlap
    custom_ref_state_arg = nothing
    antihermitian = false
    filtered_args = String[]
    for arg in args
        if startswith(arg, "--nn=")
            val = String(split(arg, "=", limit=2)[2])
            if isfile(val)
                nn_strategy_file = val
            else
                resolved = joinpath("trained_neural_networks", "trained_neural_network_$(val).jld2")
                if isfile(resolved)
                    nn_strategy_file = resolved
                else
                    error("Neural network strategy file not found: '$(val)' (also tried '$(resolved)')")
                end
            end
        elseif startswith(arg, "--maxiters=")
            val = String(split(arg, "=", limit=2)[2])
            maxiters = parse(Int, val)
        elseif startswith(arg, "--loss=")
            val = String(split(arg, "=", limit=2)[2])
            if val == "overlap"
                loss_type = :overlap
            elseif val == "energy"
                loss_type = :energy
            else
                error("Invalid --loss option: '$val'. Valid options are: 'overlap', 'energy'.")
            end
        elseif startswith(arg, "--custom_ref_state=")
            custom_ref_state_arg = String(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--antihermitian")
            if occursin("=", arg)
                antihermitian = parse(Bool, split(arg, "=", limit=2)[2])
            else
                antihermitian = true
            end
        elseif startswith(arg, "--use-gpu=")
            # Pre-scanned at top level, skip in standard argument processing
            continue
        else
            push!(filtered_args, arg)
        end
    end

    if length(filtered_args) < 1
        error("Please input a folder. Ex: data/N=(2, 2)_2x2")
    end
    folder = filtered_args[1]

    u_start = length(filtered_args) >= 2 ? filtered_args[2] : "25"
    u_end = length(filtered_args) >= 3 ? filtered_args[3] : nothing

    return folder, u_start, u_end, nn_strategy_file, maxiters, loss_type, custom_ref_state_arg, antihermitian
end


"""
    construct_custom_ref_state(custom_ref_state_arg::Union{String,Nothing}, folder::String, H_dim::Int, U_values::Vector{Float64})

Construct a custom Slater determinant reference state vector of length `H_dim` if requested by the command line argument.
Returns a `Vector{ComplexF64}` or `nothing`.
"""
function construct_custom_ref_state(custom_ref_state_arg::Union{String,Nothing}, folder::String, H_dim::Int, U_values::Vector{Float64})
    if isnothing(custom_ref_state_arg)
        return nothing
    end

    local slater_idx
    if custom_ref_state_arg == "slater"
        jld2_path = joinpath(folder, "meta_data_and_E.jld2")
        if isfile(jld2_path)
            println("Finding Slater ground state index from JLD2 file...")
            dic = load_saved_dict(jld2_path)
            all_E = dic["E"]
            U_values_for_sector = dic["meta_data"]["U_values"]
            k_min = find_best_energy_sector(all_E, U_values_for_sector; data=dic)
            slater_idx = get_slater_ground_state(dic, k_min)
        else
            println("Finding Slater ground state index from HDF5 file...")
            valid_files = [f for f in readdir(folder) if occursin("HubbardED", f)]
            if isempty(valid_files)
                error("No HubbardED HDF5 file found in folder: $folder")
            end
            h5_file = joinpath(folder, valid_files[1])
            slater_idx = h5open(h5_file, "r") do data
                key_labels = [parse(Int, k) for k in keys(data["data/energies"])]
                all_E = [real.(read(data, "data/energies/$(k)"))[:, 1] for k in key_labels]
                k_min = find_best_energy_sector(all_E, U_values; labels=key_labels)
                return get_slater_ground_state(data, k_min)
            end
        end
        println("Slater ground state index found: $slater_idx")
    else
        try
            slater_idx = parse(Int, custom_ref_state_arg)
        catch e
            error("Invalid --custom_ref_state value: '$custom_ref_state_arg'. Must be 'slater' or an integer index.")
        end
        if slater_idx < 1 || slater_idx > H_dim
            error("Parsed Slater index $slater_idx is out of bounds (1 to $H_dim).")
        end
        println("Using user-specified Slater index: $slater_idx")
    end

    if slater_idx == -1
        error("No Slater ground state could be found in the current sector.")
    end

    custom_ref = zeros(ComplexF64, H_dim)
    custom_ref[slater_idx] = 1.0
    return custom_ref
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "run_lanczos_scan_optimization")
    with_logging(log_path) do

        folder, u_start, u_end, nn_strategy_file, maxiters, loss_type, custom_ref_state_arg, antihermitian = parse_arguments(ARGS)

        # Parse electrons and dimension from the folder name
        electrons_parsed = (2, 2)
        dim_parsed = [2, 2]
        try
            electrons_parsed = parse_electron_count(folder)
            dim_parsed = parse_lattice_dimension(folder)
        catch e
            @warn "Could not parse electrons or dimensions from folder path, using defaults: (2, 2) and [2, 2]"
        end

        if isnothing(custom_ref_state_arg)
            U_values, target_vecs, indexer, precomputed_structures, N, spin_conserved, use_symmetry, sign_convention =
                load_ED_data(folder; verbose=true)
        else
            U_values, target_vecs, indexer, precomputed_structures, N, spin_conserved, use_symmetry, sign_convention =
                load_ED_data(folder; verbose=true, use_slater_reference=false)
        end

        scan_instructions = Dict(
            "starting level" => 1,
            "ending level" => 1, # level index for targets
            "optimization_scheme" => [2],
            "use symmetry" => use_symmetry,
            "multi_start_iters" => 50, # 30
            "multi_start_samples" => 5, #5
            "initialization_samples" => 20,#20
            "sign_convention" => sign_convention,
            "U_values" => U_values,
            "antihermitian" => antihermitian
        )

        if nn_strategy_file !== nothing
            println("Using neural network found in $(nn_strategy_file)")
        end

        save_name_prefix = "unitary_map_energy_symmetry=$(use_symmetry)_N=$N"
        if !isnothing(custom_ref_state_arg)
            save_name_prefix *= "_ref_$(custom_ref_state_arg)"
        end
        if antihermitian
            save_name_prefix *= "_antihermitian"
        end
        if loss_type == :energy
            save_name_prefix *= "_loss_energy"
        end
        if nn_strategy_file !== nothing
            nn_name = replace(basename(nn_strategy_file), "trained_neural_network_" => "", ".jld2" => "")
            save_name_prefix *= "_nn_$(nn_name)"
        end

        if u_end === nothing
            v1 = tryparse(Int, u_start)
            if isnothing(v1)
                if u_start == "forward"
                    println("Forward")
                    scan_instructions["u_range"] = 26:length(U_values)
                else
                    u_start == "backward"
                    println("backward")
                    scan_instructions["u_range"] = 18:-1:1
                end
                scan_instructions["load_file"] = joinpath(folder, "$(save_name_prefix)_u_19.jld2")
                println("Load: $(scan_instructions["load_file"])")
            else
                println("doing: $v1 (U = $(U_values[v1]))")
                scan_instructions["u_range"] = v1:v1
            end
        else
            v1 = parse(Int, u_start)
            v2 = parse(Int, u_end)
            if v1 > v2
                scan_instructions["u_range"] = v1:-1:v2
                if isfile(joinpath(folder, "$(save_name_prefix)_u_$(v1+1).jld2"))
                    scan_instructions["load_file"] = joinpath(folder, "$(save_name_prefix)_u_$(v1+1).jld2")
                end
            else
                scan_instructions["u_range"] = v1:v2
                if isfile(joinpath(folder, "$(save_name_prefix)_u_$(v1-1).jld2"))
                    scan_instructions["load_file"] = joinpath(folder, "$(save_name_prefix)_u_$(v1-1).jld2")
                end
            end
        end

        H_dim = !isnothing(indexer) ? length(indexer.inv_comb_dict) : (size(target_vecs, 1) == length(U_values) ? size(target_vecs, 2) : size(target_vecs, 1))
        custom_ref = construct_custom_ref_state(custom_ref_state_arg, folder, H_dim, U_values)

        interaction_scan_map_to_state(target_vecs, scan_instructions, indexer,
            spin_conserved;
            maxiters=maxiters, gradient=:adjoint_gradient,
            perturb_optimization=0.0,
            optimizer=[:GradientDescent, :LBFGs, :GradientDescent, :LBFGs],
            save_folder=folder, save_name=save_name_prefix,
            precomputed_structures=precomputed_structures,
            max_time_ratio=50.0,
            nn_strategy_file=nn_strategy_file,
            nn_electrons=electrons_parsed,
            nn_dim=dim_parsed,
            nn_U_values=U_values,
            U_values=U_values,
            loss_type=loss_type,
            use_gpu=_use_gpu,
            custom_ref_state=custom_ref)

        return 0
    end # with_logging
end
