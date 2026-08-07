using IJulia
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
using ExponentialUtilities
ENV["JULIA_CUDA_USE_COMPAT"] = "true"
using CUDA
using Printf
using Flux
using Dates

include("logging.jl")
include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("utility_functions.jl")
include("nn_strategy.jl")
include("data_path.jl")

function run_benchmark()
    test_folders = [
        data_folder("N=(6, 6)_4x3"),
        data_folder("N=(4, 5)_4x4"),
        data_folder("N=(5, 5)_4x4")
    ]

    strategies = [
        ("pure_mild_mid_narrow", "trained_neural_networks/trained_neural_network_pure_mild_mid_narrow.jld2"),
        ("loss_power_neg03", "trained_neural_networks/trained_neural_network_pure_mid_narrow_loss_power_neg03.jld2"),
        ("loss_power_neg04", "trained_neural_networks/trained_neural_network_pure_mid_narrow_loss_power_neg04.jld2"),
        ("loss_power_neg05", "trained_neural_networks/trained_neural_network_pure_mid_narrow_loss_power_neg05.jld2"),
        ("one_minus_loss_power_03", "trained_neural_networks/trained_neural_network_pure_mid_narrow_one_minus_loss_power_03.jld2"),
        ("one_minus_loss_power_05", "trained_neural_networks/trained_neural_network_pure_mid_narrow_one_minus_loss_power_05.jld2"),
        ("one_minus_loss_power_07", "trained_neural_networks/trained_neural_network_pure_mid_narrow_one_minus_loss_power_07.jld2"),
        ("one_minus_loss_power_15", "trained_neural_networks/trained_neural_network_pure_mid_narrow_one_minus_loss_power_15.jld2"),
        ("one_minus_loss_power_20", "trained_neural_networks/trained_neural_network_pure_mid_narrow_one_minus_loss_power_20.jld2"),
        ("one_minus_loss_power_25", "trained_neural_networks/trained_neural_network_pure_mid_narrow_one_minus_loss_power_25.jld2")
    ]

    order = 2

    # Load all neural network strategies first
    loaded_strategies = []
    for (strat_name, strat_path) in strategies
        if isfile(strat_path)
            println("Loading strategy: $strat_name from $strat_path")
            push!(loaded_strategies, (strat_name, load_neural_network(strat_path)))
        else
            println("Strategy file not found: $strat_path")
        end
    end

    if isempty(loaded_strategies)
        println("No strategies loaded. Exiting benchmark.")
        return
    end

    for large_folder in test_folders
        if !isdir(large_folder)
            println("Folder not found: $large_folder")
            continue
        end

        large_electrons = parse_electron_count(large_folder)

        println("\n" * "="^80)
        println("EVALUATING SYSTEM: $large_folder (electrons=$large_electrons)")
        println("="^80 * "\n")

        println("=== Loading large system ED data ===")
        # Use verbose=true to see JLD2 loading progress
        U_values_large, target_vecs_large, indexer_large, precomputed_structures_large,
        N_large, spin_conserved_large, use_symmetry_large, sign_convention_large = load_ED_data(large_folder; verbose=true)

        dim_large = length(indexer_large.inv_comb_dict)
        println("Large system Hilbert space dim: $dim_large")

        momentum_basis = false
        initial_loss = 1.0

        # Check if precomputed structure exists for this order and symmetry branch
        has_precomputed = haskey(precomputed_structures_large, (order, use_symmetry_large))

        t_keys_large, rows_large, cols_large, signs_large, ops_list_large, param_index_map_large = if has_precomputed
            println("Found precomputed operators structure in cache! Loading...")
            struct_cache = precomputed_structures_large[(order, use_symmetry_large)]
            tk = struct_cache["t_keys"]
            r = struct_cache["rows"]
            c = struct_cache["cols"]
            s = struct_cache["signs"]
            ol = struct_cache["ops_list"]
            pim = build_param_index_map(ol, tk)
            tk, r, c, s, ol, pim
        else
            println("No precomputed structure found in cache. Building operators structure...")
            println("\n=== Building operator structure for large system (order=$order) ===")
            @time t_dict_large, t_keys_large = create_randomized_nth_order_operator(
                order, indexer_large, true;
                magnitude=initial_loss * 100,
                omit_H_conj=!use_symmetry_large,
                conserve_spin=spin_conserved_large,
                normalize_coefficients=false,
                conserve_momentum=momentum_basis
            )
            @time rows_large, cols_large, signs_large, ops_list_large = build_n_body_structure_from_keys(
                t_keys_large, indexer_large, typeof(t_dict_large[t_keys_large[1]]);
                sign_convention=sign_convention_large
            )
            pim = build_param_index_map(ops_list_large, t_keys_large)
            t_keys_large, rows_large, cols_large, signs_large, ops_list_large, pim
        end
        ops_large = []

        dim_large_vec = [parse(Int, x) for x in split(
            load_saved_dict(joinpath(large_folder, "meta_data_and_E.jld2"))["meta_data"]["sites"], "x")]

        # Filter u_indices from the meta_data_and_E.jld2 eigenvectors.
        # Not all U values may have been computed, so we check if the eigenvector is non-zero.
        u_indices = Int[]
        num_vecs = size(target_vecs_large, 1)
        for ui in 1:min(length(U_values_large), num_vecs)
            state_check = target_vecs_large[ui, :]
            if norm(state_check) > 0.0
                push!(u_indices, ui)
            end
        end

        println("Found $(length(u_indices)) valid U-values in ED data.")
        if isempty(u_indices)
            println("No valid U values to test.")
            continue
        end

        ref_u_idx = u_indices[1]
        state1 = target_vecs_large[ref_u_idx, :]

        # Evaluate all strategies on this folder
        for (strat_name, strategy) in loaded_strategies
            println("\n" * "-"^60)
            println("STRATEGY: $strat_name")
            println("-"^60 * "\n")

            println("\n=== Evaluating accuracy (pred/baseline/random) per U-index ===")
            println(@sprintf("%-8.4s  %-14s  %-12s  %-18s  %-14s  %-15s  %-15s  %-15s  %-15s",
                "U-value", "Baseline loss", "Pred loss", "Best Rand loss", "pred/baseline",
                "MeanAbs Stored", "MeanAbs Pred", "RMS Stored", "RMS Pred"))
            println("-"^140)

            for u_idx in u_indices
                if u_idx == ref_u_idx
                    continue # skip the reference point itself, as overlap loss is trivially 0
                end

                ctx = NeuralNetContext(U_values_large[u_idx], large_electrons, strategy.U_max)
                new_coeffs = interpolate_coefficients(strategy, ctx, t_keys_large, dim_large_vec)

                state2 = target_vecs_large[u_idx, :]
                pred_loss = adjoint_loss(
                    real.(new_coeffs), ops_large,
                    rows_large, cols_large, signs_large,
                    param_index_map_large, nothing, nothing,
                    dim_large, state1, state2, nothing,
                    !use_symmetry_large, false
                )

                # Recalculate baseline loss (no unitary mapping, i.e. zero coefficients)
                zero_coeffs = zeros(length(t_keys_large))
                stored_loss_recalc = adjoint_loss(
                    zero_coeffs, ops_large,
                    rows_large, cols_large, signs_large,
                    param_index_map_large, nothing, nothing,
                    dim_large, state1, state2, nothing,
                    !use_symmetry_large, false
                )

                # Sample random coefficients and pick the best achieving loss
                best_rand_loss = Inf
                best_a = 0.0
                for a in (0.5, 0.1, 0.02)
                    for trial in 1:100
                        rand_coeffs = clamp.(a .* randn(length(t_keys_large)), -a, a)
                        rand_loss = adjoint_loss(
                            rand_coeffs, ops_large,
                            rows_large, cols_large, signs_large,
                            param_index_map_large, nothing, nothing,
                            dim_large, state1, state2, nothing,
                            !use_symmetry_large, false
                        )
                        if rand_loss < best_rand_loss
                            best_rand_loss = rand_loss
                            best_a = a
                        end
                    end
                end

                mean_abs_pred = mean(abs.(new_coeffs))
                mean_abs_stored = 0.0
                rms_pred = sqrt(mean(new_coeffs .^ 2))
                rms_stored = 0.0

                ratio = stored_loss_recalc == 0.0 ? NaN : pred_loss / stored_loss_recalc

                best_rand_str = @sprintf("%.6g (a=%.2g)", best_rand_loss, best_a)
                println(@sprintf(
                    "%-8.4g  %-14.6g  %-12.6g  %-18.18s  %-14.4g  %-15.6g  %-15.6g  %-15.6g  %-15.6g",
                    U_values_large[u_idx], stored_loss_recalc, pred_loss, best_rand_str, ratio,
                    mean_abs_stored, mean_abs_pred, rms_stored, rms_pred
                ))
            end
            println("\nDone testing strategy $strat_name on $large_folder.")
        end
    end
end

log_path = make_log_path(@__DIR__, "run_4x4_benchmark")
println("Logging output to: $log_path")
with_logging(log_path) do
    run_benchmark()
end
