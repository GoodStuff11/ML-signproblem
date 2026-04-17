# DON'T MODIFY THIS CELL
using Lattices
using ChainRulesCore
using ExponentialUtilities
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

include("ed_objects.jl")
include("task.jl")


function make_hermitian(A::SparseMatrixCSC)
    # acts similar to Hermitian(A) but is when only one of A[i,j] and A[j,i] are non-zero
    # This function doesn't override non-zero values with zero values like Hermitian(A) can
    I, J, V = findnz(A)
    return sparse(
        vcat(I, J),
        vcat(J, I),
        vcat(V, conj.(V)),
        size(A, 1), size(A, 2)
    )
end
function make_antihermitian(A::SparseMatrixCSC)
    # acts similar to Hermitian(A) but is when only one of A[i,j] and A[j,i] are non-zero
    # This function doesn't override non-zero values with zero values like Hermitian(A) can

    I, J, V = findnz(A)
    return sparse(
        vcat(I, J),
        vcat(J, I),
        vcat(V, -conj.(V)),
        size(A, 1), size(A, 2)
    )
end

"""
Computes the Jordan-Wigner sign for a given configuration and set of operators.

THIS FUNCTION IS USED FOR SANITY CHECKS BUT IT DOESN'T NEED TO BE USED FOR OPTIMIZATION IF INEFFICIENT
"""
function compute_jw_sign(
    conf::Tuple{Set{T},Set{T}},
    sorted_sites::Vector{T},
    ops::Vector{Tuple{T,Int,Symbol}}
) where T
    # computes the sign for the term given by ops (in second quantized), associated with the 
    # configuration conf.

    # Full JW order over sites and spins
    jw_order = [(s, σ) for s in sorted_sites for σ in (1, 2)]

    # Map each mode to its index in JW order
    jw_index = Dict{Tuple{T,Int},Int}((sσ, i) for (i, sσ) in enumerate(jw_order))

    # Initial occupation vector (ordered)
    occupied_modes = Set{Tuple{T,Int}}()
    for (s, σ) in jw_order
        if s ∈ conf[σ]
            push!(occupied_modes, (s, σ))
        end
    end

    # Compute sign by counting how many occupied modes come before each operator
    sign = 1

    # Process from right to left (annihilate first)
    for (site, spin, op) in reverse(ops)
        mode = (site, spin)
        idx = jw_index[mode]

        # Count how many occupied modes come *before* this mode
        n_occupied_before = count(m -> jw_index[m] < idx, occupied_modes)
        sign *= (-1)^n_occupied_before

        # Update occupation based on op
        if op == :annihilate
            # Fermion is removed — no longer present
            delete!(occupied_modes, mode)
        elseif op == :create
            # Fermion is added — affects future operators
            push!(occupied_modes, mode)
        else
            error("Unknown operator: $op")
        end
    end

    return sign
end

"""
function which takes in a set of keys corresponding to different operators, and outputs the 
    information required to construct a sparse matrix, without specifying what the coefficient
    values are. This includes row, column indices, plus the sign at that element, and the operator
    the matrix element is associated with.


THIS FUNCTION IS USED FOR SANITY CHECKS BUT IT DOESN'T NEED TO BE USED FOR OPTIMIZATION IF INEFFICIENT
"""
function build_nth_order_sparse(
    t_keys::AbstractVector,
    indexer::CombinationIndexer{T},
    ::Type{U}=Float64;
    skip_lower_triangular::Bool=false
) where {T,U<:Number}
    sorted_sites = sort(indexer.a)
    rows = Int[]
    cols = Int[]
    signs = U[]
    ops_list = Vector{Vector{Tuple{T,Int,Symbol}}}()

    for (i1, conf) in enumerate(indexer.inv_comb_dict)
        for ops in t_keys
            # Clone the config
            conf_new = [copy(conf[1]), copy(conf[2])]
            valid = true

            # Apply operators
            for (site, spin, op) in reverse(ops)
                if op == :annihilate
                    if site ∉ conf_new[spin]
                        valid = false
                        break
                    end
                    delete!(conf_new[spin], site)
                elseif op == :create
                    if site ∈ conf_new[spin]
                        valid = false
                        break
                    end
                    push!(conf_new[spin], site)
                else
                    error("Invalid operator symbol: $op")
                end
            end

            if !valid
                continue
            end

            i2 = index(indexer, conf_new[1], conf_new[2])
            if skip_lower_triangular && i1 > i2 #only considering upper diagonal so ensure hermiticity
                continue
            end
            s = compute_jw_sign(conf, sorted_sites, ops)
            push!(rows, i1)
            push!(cols, i2)
            push!(signs, s)
            push!(ops_list, ops)
        end
    end

    return rows, cols, signs, ops_list
end

"""
based on the indexer defining the basis, this function creates a set of keys (not ordered in any particular way) 
    associated list all possible operators that can be put in the exponential.

THIS FUNCTION IS USED FOR SANITY CHECKS BUT IT DOESN'T NEED TO BE USED FOR OPTIMIZATION IF INEFFICIENT
"""
function build_nth_order_operator(n::Int, indexer::CombinationIndexer; omit_H_conj::Bool=false,
    conserve_spin::Bool=false, conserve_momentum::Bool=false)
    # function creates a dictionary of free parameters in the form of a dictionary. 
    # when spin is conserved, the Hilbert space is smaller, so a restricted number of coefficients are possible. The rest aren't filled in
    # When hermiticity is forced, we only need to worry about upper diagonal elements. The rest can be filled in afterward

    t_keys = Set{Vector{Tuple{Coordinate{2,Int64},Int,Symbol}}}()
    site_list = sort(indexer.a) #ensuring normal ordering
    all_ops(label) = combinations([(s, σ, label) for s in site_list for σ in 1:2], n)
    equal_spin(create, annihilate) = sum((σ * 2 - 3) for (s, σ, _) in create) == sum((σ * 2 - 3) for (s, σ, _) in annihilate)
    geq_ops(create, annihilate) = [(s.coordinates..., σ) for (s, σ, _) in create] <= [(s.coordinates..., σ) for (s, σ, _) in annihilate]

    for (ops_create, ops_annihilate) in Iterators.product(all_ops(:create), all_ops(:annihilate))
        key = [ops_create; ops_annihilate]

        # We must conserve momentum if the user specified conserve_momentum=true
        # OR if the indexer is explicitly restricted to a momentum sector (which means non-conserving ops will jump out of the Hilbert space)
        must_conserve_momentum = conserve_momentum || (!isnothing(indexer.k) && !isnothing(indexer.lattice_dims))

        if must_conserve_momentum
            tot_k = zeros(Int, length(indexer.lattice_dims))
            for (s, σ, _) in ops_create
                tot_k .+= (s.coordinates .- 1)
            end
            for (s, σ, _) in ops_annihilate
                tot_k .-= (s.coordinates .- 1)
            end
            tot_k = tot_k .% indexer.lattice_dims
            is_momentum_conserved = all(tot_k .== 0)
        else
            is_momentum_conserved = true
        end

        if (!omit_H_conj || geq_ops(ops_create, ops_annihilate)) && (!conserve_spin || equal_spin(ops_create, ops_annihilate)) && is_momentum_conserved
            if key ∉ t_keys
                push!(t_keys, key)
            end
        end
    end
    return collect(t_keys)
end


"""
    load_saved_dict(filename::AbstractString) -> Dict

Load the dictionary saved using `save_with_metadata`.
Returns the dictionary stored under the key `"dict"`.
"""
function load_saved_dict(filename::AbstractString)
    return JLD2.jldopen(filename, "r") do file
        file["dict"]
    end
end

"""
Wrapper function for optimize_unitary that will be used at the highest level. 
"""
function perform_optimization(degen_rm_U::Union{AbstractMatrix,Vector}, indexer::CombinationIndexer,
    spin_conserved::Bool=false; optimization_scheme::Vector{Int}=[1, 2], antihermitian::Bool=false,
    maxiters=100, gradient::Symbol=:gradient, metric_functions::Dict{String,Function}=Dict{String,Function}(), optimizer::Union{Symbol,Vector{Symbol}}=:LBFGS,
    perturb_optimization::Float64=0.1,
    save_folder::Union{String,Nothing}=nothing, save_name::String="data"
)
    # println("\n--- Optimizing between U indices: $u_idx1 and $u_idx2 ---")

    state1 = degen_rm_U[1, :]
    state2 = degen_rm_U[end, :]

    args = optimize_unitary(state1, state2, indexer;
        spin_conserved=spin_conserved,
        maxiters=maxiters, optimization_scheme=optimization_scheme, gradient=gradient,
        metric_functions=metric_functions, antihermitian=antihermitian, optimizer=optimizer,
        initial_coefficients=Any[], perturb_optimization=perturb_optimization)

    computed_matrices, coefficient_labels, coefficient_values, metrics, operator_cache = args

    data_dict = Dict{String,Any}(
        "all_matrices" => computed_matrices,
        "coefficients" => coefficient_values,
        "coefficient_labels" => coefficient_labels,
        "metrics" => metrics,
        "states" => Dict(
            "starting state" => state1,
            "ending state" => state2,
        ),
        "operators" => operator_cache
    )

    # if !isnothing(save_folder)
    #     mkpath(save_folder)
    #     JLD2.jldsave(joinpath(save_folder, "$(save_name).jld2"); dict=data_dict)
    # end

    return data_dict
end

function setup(folder)
    max_iters = 200
    optimization_scheme = [2, 1]
    optimizers = [:GradientDescent, :LBFGS, :GradientDescent, :LBFGS]

    # collect necessary data from file
    file_path = joinpath(folder, "meta_data_and_E.jld2")
    dic = load_saved_dict(file_path)
    meta_data = dic["meta_data"]
    all_full_eig_vecs = dic["all_full_eig_vecs"]
    all_E = dic["E"] # Needed for energy selection
    indexer = dic["indexer"]

    # display meta data for logging
    println("Meta data:")
    display(meta_data)

    # Extract N for saving
    N = meta_data["electron count"]
    spin_conserved = !isa(meta_data["electron count"], Number) # True if tuple (N_up, N_down)

    # find lowest energy momentum sector and select it. 
    min_E = Inf
    k_min = 1
    for (k, E_vec) in enumerate(all_E)
        if !isempty(E_vec)
            E_ground = E_vec[1]
            if E_ground < min_E
                min_E = E_ground
                k_min = k
            end
        end
    end
    println("Selected lowest energy symmetry sector: $k_min with Energy $(min_E)")

    # Select the eigenvectors for this sector
    # all_full_eig_vecs is a list of sectors. each sector is a list of vectors (per U).
    target_vecs = all_full_eig_vecs[k_min]
    if indexer isa Vector
        indexer = indexer[k_min]
    end

    return target_vecs, indexer, spin_conserved, max_iters, optimization_scheme, optimizers
end

"""
sanity_check()

Sanity check optimization code on smaller system. The point of this is not to be used as the time
or final loss benchmark, but just to verify that the code is working as intended, for a smaller system
where testing this is easier.
"""
function sanity_check()
    # perform optimization on simpler problem
    target_vecs, indexer, spin_conserved, max_iters, optimization_scheme, optimizers = setup("data/N=(3, 3)_3x2")
    data_dict = perform_optimization(target_vecs, indexer,
        spin_conserved; optimization_scheme=optimization_scheme,
        maxiters=10, gradient=:adjoint_gradient,
        optimizer=optimizers,
        save_folder=nothing, save_name="unitary_map_energy_N=$N")

    state1 = data_dict["states"]["starting state"]
    state2 = data_dict["states"]["ending state"]
    U_state1 = expv(1im, sum(data_dict["all_matrices"]), state1)
    @assert norm(U_state1) ≈ 1
    @assert data_dict["metrics"]["loss"][end] ≈ 1 - abs2(state2' * U_state1)
    for order = 1:2
        t_keys = build_nth_order_operator(order, indexer; omit_H_conj=true, conserve_spin=spin_conserved, conserve_momentum=true)
        dim = length(indexer.inv_comb_dict)
        # create matrix operators to make gradient computation faster
        ops = []
        for k in collect(t_keys)
            _rows, _cols, _signs, _ = build_nth_order_sparse([k], indexer)
            push!(ops, make_hermitian(sparse(_rows, _cols, _signs, dim, dim)))
        end

        @assert sum(ops[k] * data_dict["coefficients"][order][k] for k in eachindex(data_dict["coefficients"][order])) ≈ data_dict["all_matrices"][order]
    end

end
function (@main)(ARGS)
    sanity_check()

    target_vecs, indexer, spin_conserved, max_iters, optimization_scheme, optimizers = setup("data/N=(4, 4)_4x2")
    # performs the optimization
    time_elapsed = @elapsed begin
        data_dict = perform_optimization(target_vecs, indexer,
            spin_conserved; optimization_scheme=optimization_scheme,
            maxiters=max_iters, gradient=:adjoint_gradient,
            optimizer=optimizers,
            save_folder=nothing, save_name="unitary_map_energy_N=$N")
    end

    # verifying accuracy (sanity checks)
    state1 = data_dict["states"]["starting state"]
    state2 = data_dict["states"]["ending state"]
    U_state1 = expv(1im, sum(data_dict["all_matrices"]), state1)
    @assert norm(U_state1) ≈ 1
    @assert data_dict["metrics"]["loss"][end] ≈ 1 - abs2(state2' * U_state1)

    println("print(f'Candidate metric to be minimized: $((data_dict["metrics"]["loss"][end]*1e5 + time_elapsed))")

end
