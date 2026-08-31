#=
test_regularized_loss_matching.jl

Verification test script to validate the alignment between regularized loss functions
and adjoint pullback gradients in `ed_optimization.jl`.

Usage:
  julia --project=.. testing/test_regularized_loss_matching.jl [options]

Command line options:
  --atol=<float>       Absolute tolerance for finite-difference gradient check (default: 1e-4)
  --dim=<int>          Dimension of mock Hilbert space (default: 16)
  --num_params=<int>   Number of parameters (default: 5)
=#

using Lattices
using LinearAlgebra
using Random
using SparseArrays
using Zygote

include("../utility_functions.jl")
include("../logging.jl")
include("../ed_objects.jl")
include("../ed_functions.jl")
include("../ed_optimization.jl")

"""
    parse_arguments(args) -> Dict{Symbol, Any}

Parse command-line arguments into a options dictionary.
"""
function parse_arguments(args)
    opts = Dict{Symbol, Any}(
        :atol => 1e-4,
        :dim => 16,
        :num_params => 5
    )
    for arg in args
        if startswith(arg, "--atol=")
            opts[:atol] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--dim=")
            opts[:dim] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--num_params=")
            opts[:num_params] = parse(Int, split(arg, "=", limit=2)[2])
        end
    end
    return opts
end

"""
    finite_difference_grad(f, x; h=1e-6) -> Vector{Float64}

Compute central finite difference gradient for scalar function `f` at `x`.
"""
function finite_difference_grad(f, x::Vector{Float64}; h=1e-6)
    g = zeros(Float64, length(x))
    for i in eachindex(x)
        x_plus = copy(x)
        x_minus = copy(x)
        x_plus[i] += h
        x_minus[i] -= h
        g[i] = (f(x_plus) - f(x_minus)) / (2h)
    end
    return g
end

function run_tests(opts)
    println("="^60)
    println("Running Regularized Loss Alignment Test Suite")
    println("="^60)

    Random.seed!(42)
    dim = opts[:dim]
    num_params = opts[:num_params]
    atol = opts[:atol]

    state1 = normalize!(randn(ComplexF64, dim))
    state2 = normalize!(randn(ComplexF64, dim))
    t_vals = randn(Float64, num_params) .* 0.2

    num_nonzeros = 40
    rows = rand(1:dim, num_nonzeros)
    cols = rand(1:dim, num_nonzeros)
    signs = randn(ComplexF64, num_nonzeros)
    param_index_map = rand(1:num_params, num_nonzeros)
    parameter_mapping = nothing
    parity = nothing
    use_symmetry = false
    antihermitian = false

    # Build ops structure for adjoint_loss
    ops_cpu = []
    indices_by_param = [Int[] for _ in 1:num_params]
    for k in eachindex(param_index_map)
        push!(indices_by_param[param_index_map[k]], k)
    end
    for i in 1:num_params
        idx = indices_by_param[i]
        rows_sub = Int[]
        cols_sub = Int[]
        vals_sub = ComplexF64[]
        for j in idx
            r = rows[j]
            c = cols[j]
            s = signs[j]
            push!(rows_sub, r)
            push!(cols_sub, c)
            push!(vals_sub, s)
            push!(rows_sub, c)
            push!(cols_sub, r)
            push!(vals_sub, antihermitian ? -conj(s) : conj(s))
        end
        push!(ops_cpu, (rows_sub, cols_sub, vals_sub))
    end

    # Test 1: Check adjoint_loss value formula
    println("\n[Test 1] Regularized vs Unregularized Loss Value Check...")
    loss_reg_val, back = Zygote.pullback(
        t -> adjoint_loss(t, ops_cpu, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state2, state1, nothing, !use_symmetry, antihermitian),
        t_vals
    )
    reg_penalty = 0.5 * 1e-3 * sum(abs2, t_vals)
    
    # Calculate unregularized loss directly
    vals_l = update_values(signs, param_index_map, t_vals, parameter_mapping, parity)
    A_l = sparse(rows, cols, vals_l, dim, dim)
    A_l = make_hermitian(A_l)
    psi = expv(1.0im, A_l, state1)
    overlap = dot(state2, psi)
    loss_unreg_direct = 1.0 - abs2(overlap) / real(dot(psi, psi))

    println("Direct Unregularized Loss : $loss_unreg_direct")
    println("Calculated Regularized Loss: $loss_reg_val")
    println("Expected Regularized Loss  : $(loss_unreg_direct + reg_penalty)")

    if isapprox(loss_reg_val, loss_unreg_direct + reg_penalty; atol=1e-12)
        println("PASSED: Forward loss function correctly includes regularizer +0.5 * 1e-3 * sum(t^2).")
    else
        error("FAILED: Loss regularizer term mismatch!")
    end

    # Test 2: Check pullback gradient matches finite-difference gradient of regularized loss
    println("\n[Test 2] Pullback Gradient vs Finite-Difference Gradient Check...")
    grad_pullback = back(1.0)[1]
    
    loss_func = t -> adjoint_loss(t, ops_cpu, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state2, state1, nothing, !use_symmetry, antihermitian)
    grad_fd = finite_difference_grad(loss_func, t_vals)

    grad_diff = norm(grad_pullback .- grad_fd)
    println("Pullback Gradient (first 5) : $(grad_pullback[1:min(5, end)])")
    println("Finite-Diff Grad (first 5)  : $(grad_fd[1:min(5, end)])")
    println("L2 Difference               : $grad_diff")

    if grad_diff < atol
        println("PASSED: Pullback gradient matches finite difference gradient of regularized loss within atol=$atol.")
    else
        error("FAILED: Gradient mismatch between pullback and finite difference!")
    end

    # Test 3: Verify fast_loss and zygote_loss
    println("\n[Test 3] Fast Loss & Zygote Loss Regularization Check...")
    l_fast = fast_loss(t_vals, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state1, state2, use_symmetry, antihermitian)
    l_zygote = zygote_loss(t_vals, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state1, state2, use_symmetry, antihermitian)

    println("Fast Loss   : $l_fast")
    println("Zygote Loss : $l_zygote")

    if isapprox(l_fast, loss_reg_val; atol=1e-10) && isapprox(l_zygote, loss_reg_val; atol=1e-10)
        println("PASSED: fast_loss and zygote_loss produce regularized loss matching adjoint_loss.")
    else
        error("FAILED: Mismatch in fast_loss or zygote_loss!")
    end

    println("\n" * "="^60)
    println("ALL TESTS PASSED SUCCESSFULLY!")
    println("="^60)
end

function (@main)(ARGS)
    opts = parse_arguments(ARGS)
    log_path = make_log_path(@__DIR__, "test_regularized_loss_matching")
    with_logging(log_path) do
        run_tests(opts)
    end
end
