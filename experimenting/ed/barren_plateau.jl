using Lattices
using LinearAlgebra
using Combinatorics
using SparseArrays
using Statistics
using Random
using Zygote
using Optimization, OptimizationOptimisers
using OptimizationOptimJL
using JLD2
using HDF5
using KrylovKit
using ExponentialUtilities
using Printf
using Dates

# Load CUDA package stably
using CUDA
try
    if !CUDA.functional()
        @info "CUDA not functional yet — trying local_toolkit mode"
        CUDA.set_runtime_version!(local_toolkit=true)
    end
    if CUDA.functional()
        @info "GPU available: $(CUDA.name(CUDA.CuDevice(0)))"
    end
catch e
    @warn "CUDA setup warning: $e"
end

# Include source files
include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("utility_functions.jl")
include("nn_strategy.jl")
include("logging.jl")

function parse_args(args)
    U = 4.0
    epsilon = 0.05
    nn_file = "trained_neural_network_3x3_(3, 3)_and_2x2_(2,2).jld2"
    use_gpu = CUDA.functional()

    for arg in args
        if startswith(arg, "--U=")
            U = parse(Float64, split(arg, "=")[2])
        elseif startswith(arg, "--epsilon=")
            epsilon = parse(Float64, split(arg, "=")[2])
        elseif startswith(arg, "--nn=")
            nn_file = split(arg, "=")[2]
        elseif startswith(arg, "--use_gpu=")
            use_gpu = parse(Bool, split(arg, "=")[2])
        end
    end
    return U, epsilon, nn_file, use_gpu
end

"""
Example command:
```bash
julia --project=.. barren_plateau.jl --U=4.0 --epsilon=0.05
```
"""
function (@main)(ARGS)
    # Prevent redundant execution on multi-task Slurm allocations
    if haskey(ENV, "SLURM_PROCID") && ENV["SLURM_PROCID"] != "0"
        return 0
    end

    log_path = make_log_path(@__DIR__, "barren_plateau")
    with_logging(log_path) do

    U, epsilon, nn_file, use_gpu = parse_args(ARGS)

    println("=========================================================================")
    println("              BARREN PLATEAUS GRADIENT VARIANCE SCALING STUDY            ")
    println("=========================================================================")
    println("Parameters selected:")
    println("  - U (interaction strength): $U")
    println("  - Epsilon (perturbation SD): $epsilon")
    println("  - Neural Network Strategy: $nn_file")
    println("  - Using GPU: $use_gpu")
    println("=========================================================================\n")

    # Load neural network strategy
    strategy = load_neural_network(nn_file)

    # Defined list of target systems to study scaling behavior
    # ((N_up, N_down), (Lx, Ly), folder_name)
    systems = [
        ((2, 2), (2, 2), "N=(2, 2)_2x2"),
        ((2, 2), (3, 2), "N=(2, 2)_3x2"),
        ((3, 3), (3, 2), "N=(3, 3)_3x2"),
        ((3, 3), (3, 2), "N=(3, 3)_3x2_2"),
        ((3, 3), (3, 2), "N=(3, 3)_3x2_3"),
        ((3, 3), (4, 2), "N=(3, 3)_4x2"),
        ((3, 3), (3, 3), "N=(3, 3)_3x3"),
        ((4, 4), (4, 2), "N=(4, 4)_4x2"),
        ((4, 4), (4, 2), "N=(4, 4)_4x2_2"),
        ((4, 4), (3, 3), "N=(4, 4)_3x3"),
        ((4, 4), (3, 3), "N=(4, 4)_3x3_2"),
    ]

    results = []

    for (electrons, dims, folder_name) in systems
        system_size = collect(dims)
        folder = joinpath("data", folder_name)
        if !isdir(folder)
            @warn "Directory $folder does not exist. Skipping."
            continue
        end

        println("Evaluating system from folder: $folder_name...")

        # 1. Load ED data to obtain the reference state at U=10^-5 (u_idx=1) and the correct basis indexer
        local_data = try
            load_ED_data(folder)
        catch e
            @warn "Failed to load ED data from $folder: $e. Skipping this system size."
            continue
        end
        U_values, target_vecs, indexer, precomputed_structures, N, spin_conserved, use_symmetry, sign_convention = local_data

        # Robustly extract the first eigenvector (at u_idx = 1) with size equal to the subspace dimension
        if target_vecs isa AbstractVector && eltype(target_vecs) <: AbstractVector
            ref = target_vecs[1]
        elseif size(target_vecs, 1) == length(U_values)
            ref = target_vecs[1, :]
        else
            ref = target_vecs[:, 1]
        end
        dim = length(ref)

        # 2. Initialize square periodic lattice and Hubbard subspace with the correct momentum sector
        lattice = Square(dims, Periodic())
        subspace = HubbardSubspace(electrons[1], electrons[2], lattice; k=indexer.k)

        # 3. Construct H_hopping and H_interaction using standard conventions
        H_hopping, H_interaction = create_hubbard_matrices(subspace; get_indexer=false, sign_convention=:coordinate_first)

        # 4. Generate operator structures (order 2)
        precomputed_structures = precompute_n_body_structures(indexer, 2; spin_conserved=true, sign_convention=:coordinate_first)
        struct_cache = precomputed_structures[(2, false)]
        t_keys = struct_cache[:t_keys]

        # 5. Predict initial variational coefficients using Neural Network
        ctx = NeuralNetContext(U, electrons, strategy.U_max)
        t_vals_init = interpolate_coefficients(strategy, ctx, t_keys, system_size)

        # 6. Execute gradient variance barren plateaus study using the loaded reference state
        _, mean_var = test_barren_plateaus(
            (H_hopping, H_interaction),
            subspace,
            indexer,
            U,
            t_vals_init,
            epsilon,
            ref;
            num_samples=40,
            use_gpu=use_gpu,
            sign_convention=:coordinate_first
        )

        # 7. Execute random sampling study centered at 0 with a larger epsilon standard deviation
        epsilon_large = 10.0 * epsilon
        _, mean_var_random = test_barren_plateaus(
            (H_hopping, H_interaction),
            subspace,
            indexer,
            U,
            zeros(length(t_keys)),
            epsilon_large,
            ref;
            num_samples=40,
            use_gpu=use_gpu,
            sign_convention=:coordinate_first
        )

        push!(results, (dim=dim, num_params=length(t_keys), mean_var=mean_var, mean_var_random=mean_var_random))
    end

    # Print summary scaling table comparing NN prediction and random sampling
    println("\n=================================================================================================")
    println("                                  SCALING STUDY SUMMARY RESULTS                                  ")
    println("=================================================================================================")
    @printf("%-15s | %-15s | %-30s | %-30s\n", "Hilbert Dim (D)", "Param Count (K)", "Mean Var (NN, ϵ=$epsilon)", "Mean Var (Rand, ϵ=$(10*epsilon))")
    println("-"^102)
    for res in results
        @printf("%-15d | %-15d | %-30.10e | %-30.10e\n", res.dim, res.num_params, res.mean_var, res.mean_var_random)
    end
    println("=================================================================================================\n")

    return 0
    end # with_logging
end
