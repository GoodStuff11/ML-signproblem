#=
run_pruning_analysis.jl

Run pruning analysis on a set of optimized unitary parameter mapping files.
Supports both exact exponential (unitary map) and Trotterized optimizations.

Usage:
  julia --project=.. run_pruning_analysis.jl [folder] [--type=<exact|trotter>] [--custom_ref_state=<value>] [--antihermitian] [--loss=<overlap|energy>]

Arguments:
  folder (optional): The path to the folder containing optimization files. Default: "N=(4, 4)_3x3_2".
  --type (optional): Whether to use trotter or exact exponential coefficients. Default: "exact".
  --custom_ref_state (optional): The custom reference state to use (e.g. "slater" or an integer index). Default: nothing.
  --antihermitian (optional): Whether antihermitian generators were used. Default: false.
  --loss (optional): The loss function used during optimization. Default: "overlap".
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
using Dates
using HDF5

include("trotter.jl")
using .Trotter
import .Trotter: @safe_threads, load_saved_dict

include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("logging.jl")
include("data_path.jl")


"""
    parse_arguments(args::Vector{String})

Parse command line arguments for running pruning analysis.
"""
function parse_arguments(args::Vector{String})
    folder = data_folder("N=(4, 4)_3x3_2")
    type = :exact
    custom_ref_state_arg = nothing
    antihermitian = false
    loss_type = :overlap
    positional = String[]

    for arg in args
        if startswith(arg, "--type=")
            val = String(split(arg, "=", limit=2)[2])
            if val == "trotter"
                type = :trotter
            elseif val == "exact"
                type = :exact
            else
                error("Invalid --type: '$val'. Valid options: 'trotter', 'exact'")
            end
        elseif startswith(arg, "--custom_ref_state=")
            custom_ref_state_arg = String(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--antihermitian")
            if occursin("=", arg)
                antihermitian = parse(Bool, split(arg, "=", limit=2)[2])
            else
                antihermitian = true
            end
        elseif startswith(arg, "--loss=")
            val = String(split(arg, "=", limit=2)[2])
            if val == "overlap"
                loss_type = :overlap
            elseif val == "energy"
                loss_type = :energy
            else
                error("Invalid --loss: '$val'. Valid options: 'overlap', 'energy'")
            end
        elseif startswith(arg, "--")
            error("Unknown option: $arg")
        else
            push!(positional, arg)
        end
    end

    if length(positional) >= 1
        folder = data_folder(positional[1])
    end

    return folder, type, custom_ref_state_arg, antihermitian, loss_type
end

function get_file_prefix(type, N_sites, custom_ref_state_arg, use_symmetry, N, antihermitian, loss_type)
    local prefix
    if type == :trotter
        prefix = "trotter_N=$N_sites"
        if !isnothing(custom_ref_state_arg)
            prefix *= "_ref_$(custom_ref_state_arg)"
        end
        if antihermitian
            prefix *= "_antihermitian"
        end
        if loss_type == :energy
            prefix *= "_loss_energy"
        end
    else
        prefix = "unitary_map_energy_symmetry=$(use_symmetry)_N=$N"
        if !isnothing(custom_ref_state_arg)
            prefix *= "_ref_$(custom_ref_state_arg)"
        end
        if antihermitian
            prefix *= "_antihermitian"
        end
        if loss_type == :energy
            prefix *= "_loss_energy"
        end
    end
    return prefix
end

function run_pruning_analysis(folder, type, custom_ref_state_arg, antihermitian, loss_type)
    sign_convention = type == :trotter ? :spin_first : :coordinate_first
    use_slater_ref = custom_ref_state_arg == "slater"
    U_values, target_vecs, indexer, _, N, _, use_symmetry, sign_convention =
        load_ED_data(folder; verbose=true, use_slater_reference=use_slater_ref, sign_convention=sign_convention)

    dim = length(indexer.inv_comb_dict)

    # Parse Lvec and N_sites for Trotter
    dim_parsed = parse_lattice_dimension(folder)
    N_sites = prod(dim_parsed)

    prefix = get_file_prefix(type, N_sites, custom_ref_state_arg, use_symmetry, N, antihermitian, loss_type)

    # Load shared data
    shared_data_path = joinpath(folder, "$(prefix)_shared.jld2")
    if !isfile(shared_data_path)
        error("Shared data file not found at: $shared_data_path")
    end
    shared_data = load_saved_dict(shared_data_path)

    # Find coefficient iteration files
    iter_files = filter(f -> startswith(basename(f), "$(prefix)_u_") && endswith(basename(f), ".jld2"), readdir(folder, join=true))
    num_maps = length(iter_files)
    if num_maps == 0
        error("No iteration files found matching prefix: $(prefix)_u_")
    end

    state1 = target_vecs[1, :]

    local gates, basis_ints
    # Shared keys/structures for Exact Exp
    local coefficient_labels, param_mapping, parities, cached_structures
    if type == :trotter
        println("Reconstructing basis sector and gates for Trotter...")
        basis_ints = Trotter.get_basis_sector(indexer, dim_parsed, N_sites)
        gates = Trotter.enumerate_ferm_excitations(2, dim_parsed; conserve_mom=true, conserve_sz=true, include_diagonal=!antihermitian)
    else#if type == :exact
        coefficient_labels = shared_data["coefficient_labels"]
        param_mapping = shared_data["param_mapping"]
        parities = shared_data["parities"]

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
    end

    thresholds = 10.0 .^ (-8:0.1:0)
    removed_terms = zeros(Int, length(thresholds), length(U_values))
    error_data = zeros(Float64, length(thresholds), length(U_values))

    println("Analyzing $num_maps unitary mappings against $(length(thresholds)) thresholds")

    total_params = 0

    @safe_threads for k in 1:num_maps
        iter_data = load_saved_dict(iter_files[k])
        ending_U_index = iter_data["u_idx"]
        state2 = target_vecs[ending_U_index+(use_slater_ref ? 1 : 0), :]

        if type == :exact
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
                        if antihermitian
                            mat = make_antihermitian(mat)
                        else
                            mat = make_hermitian(mat)
                        end

                        push!(pruned_mats, mat)
                    end
                end

                # Combine across orders
                summed_mat = isempty(pruned_mats) ? sparse(zeros(ComplexF64, dim, dim)) : sum(pruned_mats)

                # Overlap Calculation with mapped state
                if antihermitian
                    mapped_state = expv(1.0, summed_mat, state1)
                else
                    mapped_state = expv(1im, summed_mat, state1)
                end
                mapped_overlap = state2' * mapped_state
                true_loss = 1 - abs2(mapped_overlap)

                removed_terms[l, ending_U_index] = removed_count
                error_data[l, ending_U_index] = true_loss
            end
        else # type == :trotter
            A_base = iter_data["coefficients"]
            total_params = length(A_base)
            num_gates = length(gates)
            stored_num_exp = length(A_base) ÷ num_gates

            println("Processing Trotter Map $k / $num_maps (U index: $ending_U_index)")
            for (l, threshold) in enumerate(thresholds)
                # Apply threshold to coefficients
                A_pruned = copy(A_base)

                removed_mask = abs.(A_pruned) .< threshold
                removed_count = sum(removed_mask)

                A_pruned[removed_mask] .= 0.0

                # Apply unitary sequence
                psi = TrotterOptimization.apply_unitary(
                    A_pruned, gates, state1, basis_ints, N_sites, stored_num_exp;
                    antihermitian=antihermitian
                )

                mapped_overlap = state2' * psi
                true_loss = 1 - abs2(mapped_overlap)

                removed_terms[l, ending_U_index] = removed_count
                error_data[l, ending_U_index] = true_loss
            end
        end
    end

    println("Done evaluating Overlaps. Saving metrics to disk...")
    save_path = joinpath(folder, "pruning_analysis_$(prefix).jld2")
    println("Saving results to: ", save_path)
    JLD2.jldsave(save_path; error_data=error_data, removed_terms=removed_terms, thresholds=thresholds)
    println(size(error_data))
    display(error_data[1:50, 50])
    println("\n=== SUMMARY STATISTICS ===")
    println("Largest threshold applied: ", thresholds[end])
    println("Total optimizable parameters per mapped unitary sequence: ", total_params)
    println("Max parameters removed at max threshold: ", maximum(removed_terms[end, :]))

    # making sure that the printed output is actually meaningful.
    computed_u_indices = Int[]
    for f in iter_files
        push!(computed_u_indices, load_saved_dict(f)["u_idx"])
    end
    sort!(computed_u_indices)

    if !isempty(computed_u_indices)
        u_idx_test = computed_u_indices[end] # Largest U index computed
        println("U=I Error at max threshold (max truncation) (U=$(round(U_values[u_idx_test], digits=2))): ", error_data[end, u_idx_test])
        println("U≠I Error at zero threshold (no truncation) (U=$(round(U_values[u_idx_test], digits=2))): ", error_data[1, u_idx_test])

        u_idx_mid = computed_u_indices[1] # Smallest U index computed
        println("U≠I Error at zero threshold (U=$(round(U_values[u_idx_mid], digits=2))): ", error_data[1, u_idx_mid])
    end
    println("==========================\n")
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "run_pruning_analysis")
    with_logging(log_path) do
        folder, type, custom_ref_state_arg, antihermitian, loss_type = parse_arguments(ARGS)
        run_pruning_analysis(folder, type, custom_ref_state_arg, antihermitian, loss_type)
    end # with_logging
end
