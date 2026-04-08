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
include("utility_functions.jl")
include("task.jl")



# DO NOT MODIFY THIS CELL
"""
Wrapper function for optimize_unitary that will be used at the highest level. 
"""
function perform_optimization(degen_rm_U::Union{AbstractMatrix,Vector}, indexer::CombinationIndexer,
    spin_conserved::Bool=false; optimization_scheme::Vector{Int}=[1,2],antihermitian::Bool=false,
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
            "starting state"=>state1,
            "ending state"=>state2,
        ),
        "operators" => operator_cache
    )

    # if !isnothing(save_folder)
    #     mkpath(save_folder)
    #     JLD2.jldsave(joinpath(save_folder, "$(save_name).jld2"); dict=data_dict)
    # end

    return data_dict
end

function (@main)(ARGS)
    max_iters = 200
    optimization_scheme = [2, 1]
    folder = "data/N=(6, 6)_4x3" # options: "data/N=(6, 6)_4x3", "data/N=(3, 3)_3x2", "data/N=(4, 5)_3x3"
    optimizers = [:GradientDescent, :LBFGS, :GradientDescent, :LBFGS]

    # collect necessary data from file
    # folder = "/home/jek354/research/ML-signproblem/experimenting/ed/data/N=(3, 3)_3x2"
    file_path = joinpath(folder, "meta_data_and_E.jld2")
    dic = load_saved_dict(file_path)
    meta_data = dic["meta_data"]
    U_values = meta_data["U_values"]
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
    U_state1 = expv(1im,sum(data_dict["all_matrices"]),state1)
    @assert norm(U_state1) ≈ 1
    @assert data_dict["metrics"]["loss"][end] ≈ 1 - abs2(state2'*U_state1)
    # for order=1:2
    #     t_keys = build_nth_order_operator(order, indexer; omit_H_conj=true, conserve_spin=spin_conserved, conserve_momentum=true)
    #     dim = length(indexer.inv_comb_dict)
    #     # create matrix operators to make gradient computation faster
    #     ops = []
    #     for k in collect(t_keys)
    #         _rows, _cols, _signs, _ = build_nth_order_sparse([k], indexer)
    #         push!(ops, make_hermitian(sparse(_rows, _cols, _signs, dim, dim)))
    #     end

    #     @assert sum(ops[k] * data_dict["coefficients"][order][k] for k in eachindex(data_dict["coefficients"][order])) ≈ data_dict["all_matrices"][order]
    # end

    # the goal is make the loss as small as it can be with minimal time. For each different system size
    # println("performance:")
    # println("Time: $time_elapsed")
    # println("Loss: $(data_dict["metrics"]["loss"][end])")
    # println("System size: $folder")

    println(-(data_dict["metrics"]["loss"][end]*1e5 + time_elapsed))

end
