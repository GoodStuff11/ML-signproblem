#=
test_barren_plateau_temp.jl

Temporary test script to verify barren plateau scaling changes.
=#

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

# Pre-scan ARGS for --use_gpu before loading any CUDA package.
_pre_scan_use_gpu = let val = true
    for arg in ARGS
        if startswith(arg, "--use_gpu=")
            val = parse(Bool, split(arg, "=", limit=2)[2])
        end
    end
    val
end

if _pre_scan_use_gpu !== false
    using CUDA
    try
        if !CUDA.functional()
            CUDA.set_runtime_version!(local_toolkit=true)
        end
    catch e
        @warn "CUDA setup warning: $e"
    end
end

# Include source files
include("utility_functions.jl")
include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("nn_strategy.jl")
include("logging.jl")

function parse_arguments(args::Vector{String})
    U = 4.0
    epsilon = 0.005
    use_gpu = @isdefined(CUDA) && CUDA.functional()
    test_idx = nothing

    for arg in args
        if startswith(arg, "--U=")
            U = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--epsilon=")
            epsilon = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--use_gpu=")
            use_gpu = parse(Bool, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--test=")
            test_idx = parse(Int, split(arg, "=", limit=2)[2])
        end
    end
    return U, epsilon, use_gpu, test_idx
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "test_barren_plateau_temp")
    with_logging(log_path) do
        U, epsilon, use_gpu, test_idx = parse_arguments(ARGS)
        large_epsilon = 10.0

        # Set test-specific options
        num_exponentials = 1
        test_spin_conserved = true

        tests = [
            (label="0s basis, ϵ=$large_epsilon", init=:zeros, eps=large_epsilon, ref=:basis),
            (label="0s basis, ϵ=0.0", init=:zeros, eps=0.0, ref=:basis),
        ]

        println("=========================================================================")
        println("         TEST BARREN PLATEAUS GRADIENT VARIANCE SCALING STUDY            ")
        println("=========================================================================")
        println("Parameters selected:")
        println("  - U (interaction strength): $U")
        println("  - Epsilon (perturbation SD): $epsilon")
        println("  - Using GPU: $use_gpu")
        println("=========================================================================\n")

        systems = [
            ((2, 2), (2, 2), "N=(2, 2)_2x2"),
        ]

        results = []

        for (electrons, dims, folder_name) in systems
            folder = joinpath("data", folder_name)
            if !isdir(folder)
                @warn "Directory $folder does not exist. Skipping."
                continue
            end

            println("Evaluating system from folder: $folder_name...")
            local_data = load_ED_data(folder)
            U_values, target_vecs, indexer, precomputed_structures, N, spin_conserved, use_symmetry, sign_convention = local_data

            lattice = Square(dims, Periodic())
            subspace = HubbardSubspace(electrons[1], electrons[2], lattice; k=indexer.k)
            my_indexer = indexer
            dim = get_subspace_dimension(subspace)

            H_hopping, H_interaction = create_hubbard_matrices(subspace; indexer=my_indexer, get_indexer=false, sign_convention=:coordinate_first)

            precomputed_structures = precompute_n_body_structures(my_indexer, [2]; use_symmetry=[false], spin_conserved=test_spin_conserved, sign_convention=:coordinate_first)
            struct_cache = precomputed_structures[(2, false)]
            t_keys = struct_cache[:t_keys]

            num_samples = 5
            test_results = []
            for test in tests
                # Resolve initial coefficients
                c_init = zeros(length(t_keys) * num_exponentials)

                # Resolve reference state
                ref_state = zeros(ComplexF64, dim)
                ref_state[1] = 1.0

                println("Running test: $(test.label) (Hilbert Dim: $dim, Params: $(length(c_init)))...")
                @time variances_t, max_var_t, loss_var_t, precomputed_structures, all_grads, all_losses = test_barren_plateaus(
                    (H_hopping, H_interaction),
                    subspace,
                    my_indexer,
                    U,
                    c_init,
                    test.eps,
                    ref_state;
                    num_samples=num_samples,
                    use_gpu=use_gpu,
                    sign_convention=:coordinate_first,
                    precomputed_structures=precomputed_structures,
                    verbose=true,
                    num_exponentials=num_exponentials,
                    spin_conserved=test_spin_conserved,
                )

                push!(test_results, (
                    eps=test.eps,
                    label=test.label,
                    variances=variances_t,
                    max_var=max_var_t,
                    loss_var=loss_var_t,
                    grads=all_grads,
                    losses=all_losses
                ))
            end

            push!(results, (
                dim=dim,
                num_params=length(t_keys) * num_exponentials,
                test_results=test_results,
            ))
        end

        # Print summary scaling table
        headers = ["Hilbert Dim (D)", "Param Count (K)", "Param Index"]
        for test in tests
            if test.eps == 0.0
                push!(headers, "Grad (ϵ=0.0)")
                push!(headers, "Loss (ϵ=0.0)")
            else
                push!(headers, "Var (ϵ=$(test.eps))")
                push!(headers, "E[g^2] (ϵ=$(test.eps))")
                push!(headers, "Mean(L) (ϵ=$(test.eps))")
                push!(headers, "Var(L) (ϵ=$(test.eps))")
            end
        end

        col_widths = [15, 15, 12]
        for test in tests
            if test.eps == 0.0
                push!(col_widths, 18)
                push!(col_widths, 18)
            else
                push!(col_widths, 18)
                push!(col_widths, 18)
                push!(col_widths, 18)
                push!(col_widths, 18)
            end
        end

        total_width = sum(col_widths) + 3 * (length(col_widths) - 1)
        half_title_padding = max(0, div(total_width - length("SCALING STUDY SUMMARY RESULTS"), 2))

        println("\n" * "="^total_width)
        println(" "^half_title_padding * "SCALING STUDY SUMMARY RESULTS")
        println("="^total_width)

        # Print header
        for (i, h) in enumerate(headers)
            print(rpad(h, col_widths[i]))
            if i < length(headers)
                print(" | ")
            end
        end
        println()

        println("-"^total_width)

        # Print rows
        for res in results
            # Find the ranking of parameter indices
            ranking_test_idx = findfirst(tr -> tr.eps > 0.0, res.test_results)
            sort_indices = if ranking_test_idx !== nothing
                vars = res.test_results[ranking_test_idx].variances
                sortperm(vars, rev=true)
            else
                first_tr = res.test_results[1]
                grads_avg = mean(first_tr.grads, dims=1)[1, :]
                sortperm(abs.(grads_avg), rev=true)
            end

            # Display the top N parameters. Let's show top 5 parameters.
            top_n = min(5, res.num_params)
            for rank_idx in 1:top_n
                p_idx = sort_indices[rank_idx]
                
                print(rpad(string(res.dim), 15), " | ", rpad(string(res.num_params), 15), " | ", rpad(string(p_idx), 12))
                
                for (t_idx, tr) in enumerate(res.test_results)
                    print(" | ")
                    if tr.eps == 0.0
                        grad_val = tr.grads[1, p_idx]
                        loss_val = tr.losses[1]
                        @printf("%-18.6e", grad_val)
                        print(" | ")
                        @printf("%-18.6e", loss_val)
                    else
                        var_val = tr.variances[p_idx]
                        eg2_val = mean(tr.grads[:, p_idx] .^ 2)
                        mean_loss = mean(tr.losses)
                        var_loss = tr.loss_var
                        @printf("%-18.6e", var_val)
                        print(" | ")
                        @printf("%-18.6e", eg2_val)
                        print(" | ")
                        @printf("%-18.6e", mean_loss)
                        print(" | ")
                        @printf("%-18.6e", var_loss)
                    end
                end
                println()
            end
            println("-"^total_width)
        end
        println("="^total_width * "\n")

        return 0
    end
end
