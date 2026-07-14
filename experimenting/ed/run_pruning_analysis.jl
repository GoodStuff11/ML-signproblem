#=
run_pruning_analysis.jl

Run pruning analysis on a set of optimized unitary parameter mapping files.

Usage:
  julia --project=.. run_pruning_analysis.jl [folder]

Arguments:
  folder (optional): The path to the folder containing optimization files. Default: "data/N=(4, 4)_3x3_2".

Examples:
  julia --project=.. run_pruning_analysis.jl data/N=(4, 4)_3x3_2
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
using JSON
using JLD2
using Zygote
using Optimization, OptimizationOptimisers
using OptimizationOptimJL
using ExponentialUtilities
# using CUDA
using Dates

include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("utility_functions.jl")
include("logging.jl")


"""
    parse_arguments(args::Vector{String})

Parse command line arguments for running pruning analysis.
Expected arguments:
1. folder (String): Path to the folder containing unitary optimization files. Default: "data/N=(4, 4)_3x3_2"
"""
function parse_arguments(args::Vector{String})
    folder = "data/N=(4, 4)_3x3_2"
    if length(args) >= 1
        folder = args[1]
    end
    return folder
end



function run_pruning_analysis(folder)
    U_values, target_vecs, indexer, _, N, _, _, sign_convention = load_ED_data(folder)

    dim = length(indexer.inv_comb_dict)

    # Load shared data
    shared_data = load_saved_dict(joinpath(folder, "unitary_map_energy_symmetry=false_N=$(N)_shared.jld2"))

    coefficient_labels = shared_data["coefficient_labels"]
    param_mapping = shared_data["param_mapping"]
    parities = shared_data["parities"]
    instructions = shared_data["instructions"]

    # Find coefficient iteration files
    iter_files = filter(f -> occursin("_u_", basename(f)) && endswith(basename(f), ".jld2"), readdir(folder, join=true))

    thresholds = 10.0 .^ (-8:0.1:0)
    removed_terms = zeros(Int, length(thresholds), length(U_values))
    error_data = zeros(Float64, length(thresholds), length(U_values))

    num_maps = length(iter_files)
    println("Analyzing $num_maps unitary mappings against $(length(thresholds)) thresholds")

    # PRE-COMPUTE Sparse Matrix Mappings
    cached_structures = Dict()
    for o_idx in eachindex(coefficient_labels)
        if isnothing(coefficient_labels[o_idx]) || isempty(coefficient_labels[o_idx])
            continue
        end

        sorted_labels = sort(coefficient_labels[o_idx], order=sign_convention == :spin_first ? ColSnake() : RowSnake())
        rows, cols, signs, ops_list = build_n_body_structure_from_keys(sorted_labels, indexer, Float64; sign_convention=sign_convention)
        param_index_map = build_param_index_map(ops_list, coefficient_labels[o_idx])

        cached_structures[o_idx] = (rows, cols, signs, param_index_map)
    end

    total_params = 0

    @safe_threads for k in 1:num_maps
        iter_data = load_saved_dict(iter_files[k])
        ending_U_index = iter_data["u_idx"]
        ending_U_level = get(instructions, "ending level", 1) # target level 

        starting_U_index = 1 # ref_u_idx defined in the interaction mapping
        starting_U_level = 1 # ref_level

        # Exact Overlap test
        state1 = target_vecs[starting_U_index, :]
        state2 = target_vecs[ending_U_index, :]

        coeffs = iter_data["coefficients"]
        total_params = sum(isnothing(c) ? 0 : length(c) for c in coeffs)

        println("Processing Map $k / $num_maps (U index: $ending_U_index)")
        for (l, threshold) in enumerate(thresholds)

            # Form Pruned Matrix
            pruned_mats = []
            removed_count = 0

            if isnothing(coeffs)
                push!(pruned_mats, sparse(zeros(ComplexF64, dim, dim)))
            else
                for o_idx in eachindex(coeffs)
                    if isnothing(coeffs[o_idx]) || isempty(coeffs[o_idx])
                        continue
                    end

                    if o_idx > length(coefficient_labels) || isnothing(coefficient_labels[o_idx])
                        continue
                    end

                    c_vals = coeffs[o_idx] isa AbstractArray ? copy(coeffs[o_idx]) : [coeffs[o_idx]]

                    # Apply threshold
                    removed_mask = abs.(c_vals) .< threshold
                    removed_count += sum(removed_mask)

                    for idx in eachindex(c_vals)
                        if removed_mask[idx]
                            c_vals[idx] = 0.0
                        end
                    end

                    # Load matrix topology mapping variables directly from pre-computed cache
                    rows, cols, signs, param_index_map = cached_structures[o_idx]
                    vals = update_values(signs, param_index_map, c_vals, param_mapping[o_idx], parities[o_idx])

                    mat = sparse(rows, cols, vals, dim, dim)
                    mat = make_hermitian(mat)

                    push!(pruned_mats, mat)
                end
            end

            # Combine across orders
            summed_mat = isempty(pruned_mats) ? sparse(zeros(ComplexF64, dim, dim)) : sum(pruned_mats)

            # Overlap Calculation with mapped state
            mapped_state = expv(1im, summed_mat, state1)
            mapped_overlap = state2' * mapped_state
            true_loss = 1 - abs2(mapped_overlap)

            removed_terms[l, ending_U_index] = removed_count
            error_data[l, ending_U_index] = true_loss
        end
    end

    println("Done evaluating Overlaps. Saving metrics to disk...")
    # Optionally save results out
    # JLD2.jldsave(joinpath(folder, "pruning_analysis.jld2"); error_data=error_data, removed_terms=removed_terms, thresholds=thresholds)

    display(error_data)
    println("\n=== SUMMARY STATISTICS ===")
    println("Largest threshold applied: ", thresholds[end])
    println("Total optimizable parameters per mapped unitary sequence: ", total_params)
    println("Max parameters removed at max threshold: ", maximum(removed_terms[end, :]))

    # Evaluate a generic baseline error for reference
    ref_overlap = abs2(target_vecs[1, :]' * target_vecs[33, :])
    println("Unmapped (Baseline) Error between State U=$(round(U_values[1], digits=2)) and State U=$(round(U_values[33], digits=2)): ", 1 - ref_overlap)
    println("Mapped Error at max threshold (U=$(round(U_values[33], digits=2))): ", error_data[end, 33])
    println("Mapped Error at zero threshold (U=$(round(U_values[33], digits=2))): ", error_data[1, 33])
    println("Mapped Error at zero threshold (U=$(round(U_values[15], digits=2))): ", error_data[1, 15])
    println("==========================\n")

    # Simple plots
    # plot tracking pruning impact on maps globally
    # display(plot(thresholds, mean(error_data, dims=2), xscale=:log10, yscale=:log10, xlabel="Threshold Cutoff", ylabel="Mean Unitary Misfit Error (Loss)", title="Pruning Deficit Error Tracking"))
end

function @main(ARGS)
    log_path = make_log_path(@__DIR__, "run_pruning_analysis")
    with_logging(log_path) do
        folder = parse_arguments(ARGS)
        run_pruning_analysis(folder)
    end # with_logging
end
