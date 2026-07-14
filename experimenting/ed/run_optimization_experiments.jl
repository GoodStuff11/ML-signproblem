#=
run_optimization_experiments.jl

Runs a single optimization experiment configuration (1 U-value × 1 initialization × 1 loss function)
on a dataset using a trained neural network strategy.

Usage:
  julia --project=.. run_optimization_experiments.jl <nn_strategy_path> --u-idx=<idx> --init=<random|nn> --loss=<overlap|energy> [options]

Arguments:
  nn_strategy_path (required): Path to the trained JLD2 neural network strategy file.

Options:
  --u-idx=<number> (required): The U index to optimize on (e.g. 25, 39, 49, 53).
  --init=<random|nn> (required): Initialization mode (random multi-start vs neural network).
  --loss=<overlap|energy> (required): Loss function to minimize.
  --folder=<path> (optional): The dataset folder to use. Default: "nn_test_data/N=(4, 5)_3x3_2".
  --maxiters=<number> (optional): Maximum iterations for optimization. Default: 200.
  --use-gpu=<true|false> (optional): Whether to use GPU acceleration (CUDA). If set to false, it runs entirely on CPU (using multiple threads if julia is started with threads) without loading the CUDA package. Default: true.
=#

using Lattices
using LinearAlgebra
using Combinatorics
using SparseArrays
using Statistics
using Random
using JLD2
using Dates
using Zygote
using Optimization
using OptimizationOptimisers
using OptimizationOptimJL
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

include("utility_functions.jl")
include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("nn_strategy.jl")
include("logging.jl")

"""
    parse_arguments(args::Vector{String})

Parses the command-line arguments for the optimization experiment.
Returns (nn_strategy_path, u_idx, init_type, loss_type, maxiters, folder).
"""
function parse_arguments(args::Vector{String})
    nn_strategy_path = nothing
    u_idx = nothing
    init_type = nothing
    loss_type = nothing
    maxiters = 200
    folder = "nn_test_data/N=(4, 4)_3x3"
    filtered_args = String[]

    for arg in args
        if startswith(arg, "--maxiters=")
            val = String(split(arg, "=", limit=2)[2])
            maxiters = parse(Int, val)
        elseif startswith(arg, "--u-idx=")
            val = String(split(arg, "=", limit=2)[2])
            u_idx = parse(Int, val)
        elseif startswith(arg, "--folder=")
            folder = String(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--init=")
            val = String(split(arg, "=", limit=2)[2])
            if val == "random"
                init_type = :random
            elseif val == "nn"
                init_type = :nn
            else
                error("Invalid --init option: '$val'. Valid options are: 'random', 'nn'")
            end
        elseif startswith(arg, "--loss=")
            val = String(split(arg, "=", limit=2)[2])
            if val == "overlap"
                loss_type = :overlap
            elseif val == "energy"
                loss_type = :energy
            else
                error("Invalid --loss option: '$val'. Valid options are: 'overlap', 'energy'")
            end
        elseif startswith(arg, "--use-gpu=")
            # Pre-scanned at top level, skip in standard argument processing
            continue
        else
            push!(filtered_args, arg)
        end
    end

    if length(filtered_args) < 1
        error("Please provide the path to the trained neural network strategy file.")
    end

    nn_strategy_path = filtered_args[1]
    if !isfile(nn_strategy_path)
        resolved = joinpath("trained_neural_networks", "trained_neural_network_$(nn_strategy_path).jld2")
        if isfile(resolved)
            nn_strategy_path = resolved
        else
            error("Neural network strategy file not found: '$nn_strategy_path' (also tried '$resolved')")
        end
    end

    if isnothing(u_idx)
        error("Missing required option: --u-idx=<number>")
    end
    if isnothing(init_type)
        error("Missing required option: --init=<random|nn>")
    end
    if isnothing(loss_type)
        error("Missing required option: --loss=<overlap|energy>")
    end

    return nn_strategy_path, u_idx, init_type, loss_type, maxiters, folder
end

function (@main)(ARGS)
    # Generate unique log suffix for this specific experiment combination
    nn_strategy_path, u_idx, init_type, loss_type, maxiters, folder = parse_arguments(ARGS)
    suffix = "u_$(u_idx)_init_$(init_type)_loss_$(loss_type)"
    log_path = make_log_path(@__DIR__, "run_optimization_experiments_$(suffix)")

    with_logging(log_path) do
        println("Loading dataset from: $folder")
        U_values, target_vecs, indexer, precomputed_structures, N, spin_conserved, use_symmetry, sign_convention = load_ED_data(folder; verbose=true)

        # Parse electrons and dimension dynamically from JLD2 meta data
        meta_dict = JLD2.load(joinpath(folder, "meta_data_and_E.jld2"))
        if haskey(meta_dict, "dict")
            meta_dict = meta_dict["dict"]
        end
        meta_data = meta_dict["meta_data"]
        electrons_parsed = meta_data["electron count"]
        sites_str = meta_data["sites"]
        dim_parsed = parse.(Int, split(sites_str, "x"))

        println("\n" * "="^80)
        println("RUNNING EXPERIMENT: U index = $u_idx (U = $(U_values[u_idx])), Init = $init_type, Loss = $loss_type")
        println("="^80 * "\n")

        # Setup scan instructions
        scan_instructions = Dict(
            "starting level" => 1,
            "ending level" => 1,
            "optimization_scheme" => [2],
            "use symmetry" => use_symmetry,
            "multi_start_iters" => 50,
            "multi_start_samples" => 5,
            "initialization_samples" => 20,
            "sign_convention" => sign_convention,
            "U_values" => U_values,
            "u_range" => u_idx:u_idx
        )

        # Determine NN strategy file argument
        current_nn_strategy = (init_type == :nn) ? nn_strategy_path : nothing

        # Construct unique save prefix
        save_name_prefix = "unitary_map_energy_symmetry=$(use_symmetry)_N=$N"
        if loss_type == :energy
            save_name_prefix *= "_loss_energy"
        end
        if init_type == :nn
            nn_name = replace(basename(nn_strategy_path), "trained_neural_network_" => "", ".jld2" => "")
            save_name_prefix *= "_nn_$(nn_name)"
        else
            save_name_prefix *= "_random_multistart"
        end

        println("Saving results with prefix: $save_name_prefix")

        # Execute scan
        interaction_scan_map_to_state(target_vecs, scan_instructions, indexer,
            spin_conserved;
            maxiters=maxiters, gradient=:adjoint_gradient,
            perturb_optimization=0.0,
            optimizer=[:GradientDescent, :LBFGs, :GradientDescent, :LBFGs],
            save_folder=folder, save_name=save_name_prefix,
            precomputed_structures=precomputed_structures,
            max_time_ratio=50.0,
            nn_strategy_file=current_nn_strategy,
            nn_electrons=electrons_parsed,
            nn_dim=dim_parsed,
            nn_U_values=U_values,
            U_values=U_values,
            loss_type=loss_type,
            use_gpu=_use_gpu)

        println("\nExperiment completed successfully!")
        return 0
    end
end
