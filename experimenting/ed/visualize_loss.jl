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
using KrylovKit

include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("utility_functions.jl")

function setup_optimization_problem(state1, state2, indexer;
    order=1, use_symmetry=false, spin_conserved=false, antihermitian=false,
    magnitude=1.0)

    # Create operators
    t_dict = create_randomized_nth_order_operator(order, indexer; magnitude=magnitude, omit_H_conj=!use_symmetry, conserve_spin=spin_conserved, normalize_coefficients=true)
    rows, cols, signs, ops_list = build_n_body_structure(t_dict, indexer)
    t_keys = collect(keys(t_dict))
    param_index_map = build_param_index_map(ops_list, t_keys)

    dim = length(indexer.inv_comb_dict)

    ops = []
    if use_symmetry
        try
            inv_param_map, parameter_mapping, parity = find_symmetry_groups(t_keys, maximum(indexer.a).coordinates...,
                hermitian=!antihermitian, antihermitian=antihermitian, trans_x=true, trans_y=true, spin_symmetry=true)
        catch e
            println("Symmetry finding failed: $e. Falling back to no symmetry.")
            inv_param_map = [[i] for i in 1:length(t_keys)]
            parameter_mapping = 1:length(t_keys)
            parity = ones(Int, length(t_keys))
        end

        for key_idcs in inv_param_map
            tmp_t_dict::Dict{Array{Tuple{Coordinate{2,Int64},Int64,Symbol},1},Float64} = Dict()
            for key_idx in key_idcs
                tmp_t_dict[t_keys[key_idx]] = parity[key_idx]
            end
            _rows, _cols, _signs, _ops_list = build_n_body_structure(tmp_t_dict, indexer)
            _param_index_map = build_param_index_map(_ops_list, collect(keys(tmp_t_dict)))
            _vals = update_values(_signs, _param_index_map, collect(values(tmp_t_dict)))
            push!(ops, sparse(_rows, _cols, _vals, dim, dim))
        end
        # Initial random guess
        t_vals = rand(typeof(signs[1]), length(inv_param_map)) * magnitude
    else
        for k in collect(keys(t_dict))
            _rows, _cols, _signs, _ = build_n_body_structure(Dict(k => 1.0), indexer)
            if antihermitian
                push!(ops, make_antihermitian(sparse(_rows, _cols, _signs, dim, dim)))
            else
                push!(ops, make_hermitian(sparse(_rows, _cols, _signs, dim, dim)))
            end
        end
        t_vals = collect(values(t_dict))
        inv_param_map = nothing
        parameter_mapping = nothing
        parity = nothing
    end

    return t_vals, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, ops
end

function (@main)(ARGS)
    folder = joinpath(@__DIR__, "data/N=(3, 3)_3x2")
    file_path = joinpath(folder, "meta_data_and_E.jld2")
    if !isfile(file_path)
        println("Error: File not found at $file_path")
        return
    end

    dic = load_saved_dict(file_path)
    println("Keys in dic: $(keys(dic))")

    meta_data = dic["meta_data"]
    U_values = meta_data["U_values"]
    all_full_eig_vecs = dic["all_full_eig_vecs"]
    all_E = dic["E"]
    indexer = dic["indexer"]

    # Handle Tuple vs Number for electron count
    if isa(meta_data["electron count"], Tuple)
        spin_conserved = true
    else
        spin_conserved = false
    end

    use_symmetry = false

    # Select lowest energy sector
    min_E = Inf
    k_min = 1
    for (k, E_vec) in enumerate(all_E)
        if !isempty(E_vec)           E_ground = E_vec[1]
            if E_ground < min_E
                min_E = E_ground
                k_min = k
            end
        end
    end
    u_start_idx = 1
    u_end_idx = length(U_values) ÷2

    target_vecs = all_full_eig_vecs[k_min]

    println("Visualizing loss between U=$(U_values[u_start_idx]) and U=$(U_values[u_end_idx])")

    state1 = target_vecs[u_start_idx, :] # Ground state at U_start
    state2 = target_vecs[u_end_idx, :]   # Ground state at U_end

    selected_order = 2
    use_antihermitian = false

    # Setup optimization problem (extracting parameters structure)
    t_vals_init, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, ops = setup_optimization_problem(
        state1, state2, indexer;
        order=selected_order, # Use 2nd order operators
        use_symmetry=use_symmetry,
        spin_conserved=spin_conserved,
        antihermitian=use_antihermitian
    )

    println("Number of parameters: $(length(t_vals_init))")
    println("Hilbert space dimension: $(length(indexer.inv_comb_dict))")

    # Define the loss function wrapper
    # fast_loss(t_vals, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state1, state2, use_symmetry, antihermitian; p=nothing)
    function loss_func(t_vals, p=nothing)
        # Using simplified loss to avoid excessive printing from fast_loss
        vals = update_values(signs, param_index_map, t_vals, parameter_mapping, parity)
        mat = sparse(rows, cols, vals, dim, dim)
        if !use_symmetry
            if use_antihermitian
                mat = make_antihermitian(mat)
            else
                mat = make_hermitian(mat)
            end
        end
        if p isa AbstractMatrix
            mat += p
        end

        if !use_antihermitian
            loss = 1 - abs2(state2' * expv(1.0im, mat, state1))
        else
            loss = 1 - abs2(state2' * expv(1.0, mat, state1))
        end
        return loss
    end

    # --- 1D Cut Visualization ---
    args = optimize_unitary(state1, state2, indexer;
        spin_conserved=spin_conserved, use_symmetry=use_symmetry,
        maxiters=300, optimization_scheme=[1, 2], optimization=:adjoint_gradient,
        antihermitian=use_antihermitian)
    # computed_matrices, _, coefficient_values, _, _, metrics = args

    # println("Generating Hessian eigenvectors for directions...")

    # inv_H = metrics["other"][end].trace[end].metadata["~inv(H)"]
    # eg = eigvals(inv_H)
    # println(eg)
    # for e in eg
    #     if real.(e) < 0
    #         println("eigenvalue of $(1/e)")
    #     end
    # end

    # # Target point for Hessian computation
    # x0 = coefficient_values[selected_order]
    # p_offset = computed_matrices[3-selected_order]

    # # Wrapper for adjoint_loss to use with Zygote
    # # adjoint_loss(t_vals, ops, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, v1, v2, p, do_hermitian, antihermitian)
    # f_hess(x) = adjoint_loss(x, ops, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state2, state1, p_offset, !use_symmetry, use_antihermitian)

    # # Gradient function
    # grad_f(x) = Zygote.gradient(f_hess, x)[1]

    # # Hessian-vector product via finite-difference of gradient (secant/BFGS-like)
    # g0 = grad_f(x0)
    # function Hv(v)
    #     ε = 1e-4
    #     g_plus = grad_f(x0 + ε * v)
    #     return (g_plus .- g0) / ε
    # end

    # n_params = length(x0)
    # println("Computing max curvature eigenvector...")
    # # Find largest eigenvalue (max curvature)
    # vals_l, vecs_l, info_l = KrylovKit.eigsolve(Hv, randn(n_params), 1, :LR, ishermitian=true)
    # dir1 = real.(vecs_l[1])
    # dir1 /= norm(dir1)

    # println("Computing min curvature eigenvector...")
    # # Find smallest eigenvalue (min curvature)
    # vals_s, vecs_s, info_s = KrylovKit.eigsolve(Hv, randn(n_params), 1, :SR, ishermitian=true)
    # dir2 = real.(vecs_s[1])
    # # Orthogonalize against dir1 
    # dir2 -= dot(dir1, dir2) * dir1
    # dir2 /= norm(dir2)

    # println("Eigenvalues: Largest = $(vals_l[1]), Smallest = $(vals_s[1])")
    # println("Component of grad in max eig: $(dot(normalize(g0), dir1))")
    # println("Component of grad in min eig: $(dot(normalize(g0), dir2))")

    # println("Generating 1D cut...")
    # alphas = range(-10.0, 10.0, length=500)
    # # println(loss_func(coefficient_values[selected_order], computed_matrices[3-selected_order]))
    # losses_1d = [loss_func(alpha * dir1 + x0, p_offset) for alpha in alphas]

    # p1 = plot(alphas, losses_1d, title="Loss Landscape 1D Cut (Max Curvature)", dpi=200, xlabel="Alpha", ylabel="Loss", label="Loss", yscale=:log10)
    # savefig(p1, "loss_landscape_max_1d.png")
    # println("Saved loss_landscape_1d.png")

    # losses_1d = [loss_func(alpha * dir2 + x0, p_offset) for alpha in alphas]

    # p1 = plot(alphas, losses_1d, title="Loss Landscape 1D Cut (Min Curvature)", dpi=200, xlabel="Alpha", ylabel="Loss", label="Loss", yscale=:log10)
    # savefig(p1, "loss_landscape_min_1d.png")
    # println("Saved loss_landscape_1d.png")

    # # --- 2D Contour Visualization ---
    # println("Generating 2D contour...")
    # x_range = range(-5.0, 5.0, length=50)
    # y_range = range(-5.0, 5.0, length=50)

    # losses_2d = [loss_func(x * dir1 + y * dir2 + x0, p_offset) for x in x_range, y in y_range]

    # p2 = contour(x_range, y_range, losses_2d, title="Loss Landscape 2D (Hessian Eigenplane)", xlabel="Max Curvature Dir", ylabel="Min Curvature Dir", fill=true)
    # savefig(p2, "loss_landscape_2d.png")
    # println("Saved loss_landscape_2d.png")

end
