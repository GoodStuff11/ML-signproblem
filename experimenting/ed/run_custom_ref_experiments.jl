#=
run_custom_ref_experiments.jl

Runs a single custom reference state optimization experiment configuration
on the "nn_test_data/N=(4, 5)_3x3_2" dataset using random initialization.

Usage:
  julia --project=.. run_custom_ref_experiments.jl --ref-type=<random|first_one> --loss-type=<overlap|energy> [options]

Arguments (Options):
  --ref-type=<random|first_one> (required):
    Specifies the type of custom reference state used during optimization:
      - "random": A completely random normalized state (complex elements drawn from a standard normal distribution and normalized).
      - "first_one": A state with a 1 for the first element and 0 everywhere else.

  --loss-type=<overlap|energy> (required):
    Specifies the loss function to minimize:
      - "overlap": Minimize overlap loss (1 - |<psi'|U|psi>|^2).
      - "energy": Minimize energy loss (<psi|U^\dagger H U|psi>).

  --use-gpu=<true|false> (optional):
    Whether to use GPU acceleration (CUDA). If set to false, it runs entirely on CPU (using multiple threads if julia is started with threads) without loading the CUDA package. Default: true.

  --maxiters=<number> (optional):
    Maximum iterations for optimization. Default: 200.
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
include("logging.jl")

"""
    parse_arguments(args::Vector{String})

Parses the command-line arguments for the custom reference state optimization experiment.
Returns (ref_type, loss_type, maxiters).
"""
function parse_arguments(args::Vector{String})
    ref_type = nothing
    loss_type = nothing
    maxiters = 200

    for arg in args
        if startswith(arg, "--maxiters=")
            val = String(split(arg, "=", limit=2)[2])
            maxiters = parse(Int, val)
        elseif startswith(arg, "--ref-type=")
            val = String(split(arg, "=", limit=2)[2])
            if val == "random"
                ref_type = :random
            elseif val == "first_one"
                ref_type = :first_one
            else
                error("Invalid --ref-type option: '$val'. Valid options are: 'random', 'first_one'")
            end
        elseif startswith(arg, "--loss-type=")
            val = String(split(arg, "=", limit=2)[2])
            if val == "overlap"
                loss_type = :overlap
            elseif val == "energy"
                loss_type = :energy
            else
                error("Invalid --loss-type option: '$val'. Valid options are: 'overlap', 'energy'")
            end
        elseif startswith(arg, "--use-gpu=")
            # Pre-scanned at top level, skip in standard argument processing
            continue
        end
    end

    if isnothing(ref_type)
        error("Missing required option: --ref-type=<random|first_one>")
    end
    if isnothing(loss_type)
        error("Missing required option: --loss-type=<overlap|energy>")
    end

    return ref_type, loss_type, maxiters
end

function (@main)(ARGS)
    ref_type, loss_type, maxiters = parse_arguments(ARGS)

    # Generate unique log suffix for this experiment
    suffix = "ref_$(ref_type)_loss_$(loss_type)"
    log_path = make_log_path(@__DIR__, "run_custom_ref_experiments_$(suffix)")

    with_logging(log_path) do
        folder = "nn_test_data/N=(4, 5)_3x3_2"
        println("Loading dataset from: $folder")
        U_values, target_vecs, indexer, precomputed_structures, N, spin_conserved, use_symmetry, sign_convention = load_ED_data(folder; verbose=true)

        # Parse electrons and dimension
        electrons_parsed = (4, 5)
        dim_parsed = [3, 3]

        dim = size(target_vecs, 2)
        println("Hilbert space dimension: $dim")

        # Set up custom reference state
        # We seed the RNG to ensure reproducibility of the random reference state across runs
        Random.seed!(1234)
        if ref_type == :random
            custom_ref_state = randn(ComplexF64, dim)
            custom_ref_state ./= norm(custom_ref_state)
            println("Created a completely random normalized state for the reference.")
        elseif ref_type == :first_one
            custom_ref_state = zeros(ComplexF64, dim)
            custom_ref_state[1] = 1.0 + 0.0im
            println("Created a first-element-only (1 for the first element, 0 elsewhere) state for the reference.")
        else
            error("Unknown ref_type: $ref_type")
        end

        # We want to run optimization for standard U indices: 25, 39, 49, 53
        u_indices = [25, 39, 49, 53]

        println("\n" * "="^80)
        println("RUNNING SWEEP: Ref = $ref_type, Loss = $loss_type, U indices = $u_indices")
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
            "u_range" => u_indices
        )

        # Construct unique save prefix
        save_name_prefix = "unitary_map_energy_symmetry=$(use_symmetry)_N=$N"
        if loss_type == :energy
            save_name_prefix *= "_loss_energy"
        end
        save_name_prefix *= "_ref_$(ref_type)_random_multistart"

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
            nn_strategy_file=nothing, # No neural network strategy for random init
            nn_electrons=electrons_parsed,
            nn_dim=dim_parsed,
            nn_U_values=U_values,
            U_values=U_values,
            loss_type=loss_type,
            use_gpu=_use_gpu,
            custom_ref_state=custom_ref_state)

        println("\nSweep completed successfully!")
        return 0
    end
end
