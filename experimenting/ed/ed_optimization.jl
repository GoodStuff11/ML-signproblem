
"""
Apply exp(α M) * v efficiently.
Uses expv if M is large & sparse, otherwise falls back to exp(M).
"""
function apply_exp(M, v, α)
    n = size(M, 1)
    if issparse(M) && n > 128
        return expv(α, M, v)
    else
        return exp(α * M) * v
    end
end

function approximate_trotter_grad_loss(grad, t_vals, ops, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state1, state2, use_symmetry, antihermitian; p=nothing)
    # Type assertions for globals to ensure performance (though arguments are not globals here, keeping structure)
    local M_typed::Vector{SparseMatrixCSC{Float64,Int}} = ops
    local v1_typed::Vector{Complex{Float64}} = state1
    local v2_typed::Vector{Complex{Float64}} = state2
    local N_typed::Int = 100 # determines the accuracy of the method
    local Hs_dim_typed::Int = dim
    local DIM_typed::Int = length(ops)

    # Reconstruct X = Σ a_i M_i (dense)
    vals = update_values(signs, param_index_map, t_vals, parameter_mapping, parity)
    if !use_symmetry
        if antihermitian
            X = sparse(rows, cols, vals, dim, dim)
            X = Matrix(make_antihermitian(X))
        else
            X = sparse(rows, cols, 1im * vals, dim, dim)
            X = Matrix(make_hermitian(X))
        end
    else
        X = zeros(ComplexF64, dim, dim)
        for k in eachindex(vals)
            if antihermitian
                X[rows[k], cols[k]] = vals[k]
            else
                X[rows[k], cols[k]] = 1im * vals[k]
            end
        end
    end

    # Pre-compute Operator A = I + X/N
    invN = 1.0 / N_typed
    A = Matrix{Complex{Float64}}(I, Hs_dim_typed, Hs_dim_typed)
    @. A += X * invN

    # Forward Pass
    # Store r[k] = A^(k-1) * v_1
    # We need r corresponding to A^1 ... A^N.
    # Let's allocate N+1 slots, r[k] = A^(k-1) * v_1
    r = Vector{Vector{Complex{Float64}}}(undef, N_typed + 1)
    r[1] = v1_typed
    for k in 1:N_typed
        r[k+1] = A * r[k]
    end

    # Overlap computation
    overlap = dot(v2_typed, r[N_typed+1])
    loss = 1 - abs2(overlap)
    # println(loss)
    # Return early if gradient is not required
    if grad === nothing
        return loss
    end

    # Backward Pass
    grads = zeros(ComplexF64, DIM_typed)
    # l calculates v_2' * A^(N-i)
    # i goes N down to 1.
    l = copy(v2_typed) # Left vector

    # Temporary buffer for l update
    l_next = similar(l)

    for i in N_typed:-1:1
        # Current term: v_2' * A^(N-i) * M[j] * A^i * v_1
        # l holds (A^T)^(N-i) * v_2. So l' corresponds to v_2' A^(N-i).
        # r_vec should be A^i * v_1.
        # r index mapping: r[k] = A^(k-1) v_1. So we need r[i+1].

        r_current = r[i+1]

        # Accumulate gradients for each j
        for j in 1:DIM_typed
            # Compute scalar: l' * M[j] * r_current
            # Efficiently using sparse structure of M[j]
            m = M_typed[j]
            _rows = rowvals(m)
            _vals = nonzeros(m)
            val = 0.0

            # Iterate columns of sparse matrix
            for c in 1:size(m, 2)
                rc = r_current[c]
                # If rc is 0, we can skip? No, dense vector usually not 0.
                for idx in nzrange(m, c)
                    # M[row, c] * r[c] * l[row]
                    val += l[_rows[idx]] * _vals[idx] * rc
                end
            end
            grads[j] += val
        end

        # Update l for next step (next i is smaller, so N-i is larger by 1 -> multiply by A')
        # l = A' * l
        if i > 1
            mul!(l_next, A', l)
            l .= l_next
        end
    end

    # Loss L = 1 - |overlap|^2
    # dL/da = - (d(overlap)/da * conj(overlap) + overlap * conj(d(overlap)/da))
    #       = - 2 * Real( d(overlap)/da * conj(overlap) )

    # Compute scale factor
    # We computed grads for d(overlap), so we need to multiply by -2 * conj(overlap)
    # and take the real part (if our parameters are real).
    # If t_vals are real (which they usually are in this context), dL/dt must be real.

    # Note: grads is currently d_overlap * N (accumulated sum).
    # So d_overlap = grads / N

    scale_factor = -2 * conj(overlap) / N_typed

    if antihermitian
        # dL/da = -2 * Real( d(overlap)/da * conj(overlap) )
        # d(overlap)/da = <v2 | d/da exp(X) | v1>
        # X = sum a_i M_i (M_i are real anti-hermitian)
        # d(overlap)/da = <v2 | ... M_i ... | v1>
        grad .= real.(grads .* scale_factor)
    else
        # X = sum a_i (i M_i) (M_i are hermitian)
        # factor of i comes out
        grad .= real.(grads .* scale_factor .* 1im)
    end

    return loss
end

function fast_loss(t_vals, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state1, state2, use_symmetry, antihermitian; p=nothing)
    t = @elapsed begin
    vals = update_values(signs, param_index_map, t_vals, parameter_mapping, parity)
    mat = sparse(rows, cols, vals, dim, dim)
    if !use_symmetry
        if antihermitian
            mat = make_antihermitian(mat)
        else
            mat = make_hermitian(mat)
        end
    end
    if p isa AbstractMatrix
        mat += p
    end
    if antihermitian
        loss = 1 - abs2(state2' * expv(1.0, mat, state1))
    else
        loss = 1 - abs2(state2' * expv(1im, mat, state1))
    end
end
println("time=$t loss=$loss")
    return loss
end

function zygote_loss(t_vals, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state1, state2, use_symmetry, antihermitian; p=nothing)
    vals = update_values(signs, param_index_map, t_vals, parameter_mapping, parity)
    mat = sparse(rows, cols, vals, dim, dim)
    if !use_symmetry
        if antihermitian
            mat = make_antihermitian(mat)
        else
            mat = make_hermitian(mat)
        end
    end
    if p isa AbstractMatrix
        mat += p
    end
    if antihermitian
        new_state = exp(Matrix(mat)) * state1
    else
        new_state = exp(1im * Matrix(mat)) * state1
    end
    loss = 1 - abs2(state2' * new_state)
    # println(state1)
    # println(new_state)
    @Zygote.ignore println("$loss $(sum(mat))")
    return loss
end

function optimize_unitary(state1::Vector, state2::Vector, indexer::CombinationIndexer;
    maxiters=10, ϵ=1e-5, optimization_scheme::Vector=[1,2], spin_conserved::Bool=false, use_symmetry::Bool=false,
    optimization=:gradient, metric_functions::Dict{String,Function}=Dict{String,Function}(),
    antihermitian::Bool=false
)
    # spin_conserved is only true when using (N↑, N↓) and not N
    computed_matrices = []
    computed_coefficients = []
    parameter_mappings = []
    parities = []
    coefficient_labels = []
    dim = length(indexer.inv_comb_dict)

    metrics = Dict{String,Vector{Any}}()
    loss = 1 - abs2(state1' * state2)
    metrics["loss"] = Float64[loss]
    metrics["other"] = []
    metrics["loss_std"] = Float64[0.0]
    for k in keys(metric_functions)
        metrics[k] = Any[]
    end
    if loss < 1e-8
        println("States are already equal")
        return computed_matrices, coefficient_labels, computed_coefficients, parameter_mappings, parities, metrics
    end

    for order ∈ optimization_scheme
        @time t_dict = create_randomized_nth_order_operator(order, indexer; magnitude=1.0, omit_H_conj=!use_symmetry, conserve_spin=spin_conserved, normalize_coefficients=true)
        @time rows, cols, signs, ops_list = build_n_body_structure(t_dict, indexer)
        t_keys = collect(keys(t_dict))
        param_index_map = build_param_index_map(ops_list, t_keys)

        magnitude_esimate = loss/length(t_keys)
        println("magnitude: $magnitude_esimate")


        ops = []
        if use_symmetry
            inv_param_map, parameter_mapping, parity = find_symmetry_groups(t_keys, maximum(indexer.a).coordinates...,
                hermitian=!antihermitian, antihermitian=antihermitian, trans_x=true, trans_y=true, spin_symmetry=true)

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
            t_vals = rand(typeof(signs[1]), length(inv_param_map)) * magnitude_esimate

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
        push!(coefficient_labels, t_keys)

        # for exp(sum_i t_i A_i), find A_i
        # if !isnothing(parameter_mapping)
        #     trotter_matrices = []
        #     for p in parameter_mapping
        #         push!(trotter_matrices, sparse(rows[p], cols[p], signs[p], dim, dim))
        #     end
        # end

        tmp_losses = []
        function callback(state, loss_val)
            # state.gradient
            N = 20
            push!(tmp_losses, loss_val)
            if length(tmp_losses) > N && std(tmp_losses[end-N:end]) < 1e-8
                println("std: $(std(tmp_losses[end-N:end]))")
                return true
            end
            return false
        end

        function fg!(grad, t_vals, p=nothing)
            return approximate_trotter_grad_loss(grad, t_vals, ops, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state1, state2, use_symmetry, antihermitian, p=p)
        end

        function f_nongradient(t_vals, p=nothing)
            return fast_loss(t_vals, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state1, state2, use_symmetry, antihermitian, p=p)
        end

        function f(t_vals, p=nothing)
            return zygote_loss(t_vals, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state1, state2, use_symmetry, antihermitian, p=p)
        end

        function f_adjoint(t_vals, p=nothing)
            return adjoint_loss(t_vals, ops, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state2, state1, p, !use_symmetry, antihermitian)
        end

        if optimization == :gradient
            optf = Optimization.OptimizationFunction(f, Optimization.AutoZygote())
        elseif optimization == :manualgradient
            optf = Optimization.OptimizationFunction(f_nongradient, grad=fg!)
        elseif optimization == :adjoint_gradient
            optf = Optimization.OptimizationFunction(f_adjoint, Optimization.AutoZygote())
        else
            optf = Optimization.OptimizationFunction(f_nongradient)
        end

        if length(computed_matrices) > 0
            prob = Optimization.OptimizationProblem(optf, t_vals, sum(computed_matrices))
        else
            prob = Optimization.OptimizationProblem(optf, t_vals)
        end

        if optimization == :gradient || optimization == :manualgradient || optimization == :adjoint_gradient
            # opt = OptimizationOptimisers.Adam(learning_rate)
            # BFGS is faster than LBFGS, which both converge faster than adam
            @time sol = Optimization.solve(prob, Optim.BFGS(), maxiters=maxiters)
            loss = sol.objective
            metric = sol.original
            coefficients = sol.u
        elseif optimization == :finite_differences
            @time sol = Optim.optimize(f_nongradient, t_vals, Optim.BFGS())
            coefficients = sol.minimizer
            loss = sol.minimum
            metric = sol
            println("loss=$loss")
        else
            function prob_func(prob, i, repeat)
                remake(prob, u0=t_vals)
            end

            ensembleproblem = Optimization.EnsembleProblem(prob; prob_func)
            @time sol = Optimization.solve(ensembleproblem, OptimizationOptimJL.ParticleSwarm(), EnsembleThreads(), trajectories=Threads.nthreads(), maxiters=maxiters, callback=callback)
            sol = sol[argmin([s.objective for s in sol])]
        end

        vals = update_values(signs, param_index_map, coefficients, parameter_mapping, parity)

        # loss = f(new_tvals, if length(computed_matrices) > 0 sum(computed_matrices) else nothing end)
        push!(metrics["other"], metric)
        if !use_symmetry
            if antihermitian
                push!(computed_matrices, make_antihermitian(sparse(rows, cols, vals, dim, dim)))
            else
                push!(computed_matrices, make_hermitian(sparse(rows, cols, vals, dim, dim)))
            end
        else
            push!(computed_matrices, sparse(rows, cols, vals, dim, dim))
        end
        println("Finished order $order")
        push!(metrics["loss"], loss)
        # push!(metrics["loss_std"], std(last(tmp_losses, 20)))
        push!(computed_coefficients, coefficients)
        push!(parameter_mappings, parameter_mapping)
        push!(parities, parity)
        for (k, func) in metric_functions
            push!(metrics[k], func(state1, state2, computed_matrices, tmp_losses))
        end
        # println("loss std: $(metrics["loss_std"][end])")
        # if loss < ϵ
        #     break
        # end
    end
    return computed_matrices, coefficient_labels, computed_coefficients, parameter_mappings, parities, metrics
end


function test_map_to_state(degen_rm_U::Union{AbstractMatrix,Vector}, instructions::Dict{String,Any}, indexer::CombinationIndexer,
    spin_conserved::Bool=false;
    maxiters=100, optimization=:gradient, metric_functions::Dict{String,Function}=Dict{String,Function}()
)
    # spin_conserved is only true when using (N↑, N↓) and not N.

    # meta_data = Dict("starting state"=>Dict("U index"=>1, "levels"=>1:5),
    #             "ending state"=>Dict("U index"=>5, "levels"=>1),
    #             "electron count"=>3, "sites"=>"2x3", "bc"=>"periodic", "basis"=>"adiabatic", 
    #             "U_values"=>U_values)
    data_dict = Dict{String,Any}("norm1_metrics" => [], "norm2_metrics" => [],
        "loss_metrics" => [], "labels" => [], "loss_std_metrics" => [], "all_matrices" => [],
        "coefficients" => [], "coefficient_labels" => nothing, "param_mapping" => nothing, "parities" => nothing)

    finish_early = false
    for i in instructions["starting state"]["levels"]
        for j in instructions["ending state"]["levels"]
            if degen_rm_U isa AbstractMatrix
                state1 = degen_rm_U[instructions["starting state"]["U index"], :]
                state2 = degen_rm_U[instructions["ending state"]["U index"], :]
                finish_early = true
            else
                state1 = degen_rm_U[instructions["starting state"]["U index"]][:, i]
                state2 = degen_rm_U[instructions["ending state"]["U index"]][:, j]
            end
            args = optimize_unitary(state1, state2, indexer;
                spin_conserved=spin_conserved, use_symmetry=get!(instructions, "use symmetry", false),
                maxiters=maxiters, optimization_scheme=get!(instructions, "optimization_scheme", [1,2]), optimization=optimization,
                metric_functions=metric_functions, antihermitian=get!(instructions, "antihermitian", false))
            computed_matrices, coefficient_labels, coefficient_values, param_mapping, parities, metrics = args
            push!(data_dict["norm1_metrics"], [norm(cm, 1) for cm in computed_matrices])
            push!(data_dict["norm2_metrics"], [norm(cm, 2) for cm in computed_matrices])
            push!(data_dict["all_matrices"], computed_matrices)
            push!(data_dict["coefficients"], coefficient_values)
            if isnothing(data_dict["coefficient_labels"])
                data_dict["coefficient_labels"] = coefficient_labels
                data_dict["param_mapping"] = param_mapping
                data_dict["parities"] = parities
            end

            for (k, val) in metrics
                if k * "_metrics" ∉ keys(data_dict)
                    data_dict[k*"_metrics"] = [val]
                else
                    push!(data_dict[k*"_metrics"], val)
                end
            end
            push!(data_dict["labels"], Dict(
                "starting state" => Dict("level" => i, "U index" => instructions["starting state"]["U index"]),
                "ending state" => Dict("level" => j, "U index" => instructions["ending state"]["U index"]))
            )

            if finish_early
                return data_dict
            end
        end
    end


    return data_dict
end

function test_map_sd_sum(degen_rm_U::Vector, instructions::Dict{String,Any}, indexer::CombinationIndexer; maxiters=maxiters)
    data_dict = Dict{String,Any}("norm1_metrics" => [], "norm2_metrics" => [],
        "loss_metrics" => [], "labels" => [], "coefficients" => [])
    for j in instructions["goal state"]["levels"]
        goal_state = degen_rm_U[instructions["goal state"]["U index"]][:, j]
        computed_matrices, coefficients, losses = optimize_sd_sum_2(goal_state, indexer; maxiters=maxiters)
        push!(data_dict["norm1_metrics"], [norm(cm, 1) for cm in computed_matrices])
        push!(data_dict["norm2_metrics"], [norm(cm, 2) for cm in computed_matrices])
        push!(data_dict["coefficients"], coefficients)
        push!(data_dict["loss_metrics"], losses)
        push!(data_dict["labels"], Dict(
            "goal state" => Dict("level" => j, "U index" => instructions["goal state"]["U index"]))
        )
    end
    return data_dict
end
# -----------------------------------------------------------------------------
# Optimized Adjoint Gradient Logic
# -----------------------------------------------------------------------------

using ChainRulesCore
using ExponentialUtilities
using LinearAlgebra
using SparseArrays

"""
    adjoint_loss(t_vals, ops, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, v1, v2, p)

Computes the loss L = 1 - |<v1 | exp(im * (A(t) + p)) | v2>|^2 efficiently using expv
Custom rrule provides efficient gradients w.r.t t_vals.
"""
function adjoint_loss(t_vals, ops, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, v1, v2, p, do_hermitian, antihermitian=false)
    # 1. Construct A(t) efficiently
    vals = update_values(signs, param_index_map, t_vals, parameter_mapping, parity)
    A = sparse(rows, cols, vals, dim, dim)
    if do_hermitian
        if antihermitian
            A = make_antihermitian(A)
        else
            A = make_hermitian(A)
        end
    end

    # 2. Add offset p if present
    if !isnothing(p) && !(p isa SciMLBase.NullParameters)
        B = A + p
    else
        B = A
    end

    if antihermitian
        # psi, _ = exponentiate(A, 1.0, v2)
        psi = expv(1.0, B, v2)
    else
        # psi, _ = exponentiate(A, 1.0im, v2)
        psi = expv(1.0im, B, v2)
    end

    # 4. Overlap
    # ov = <v1 | psi>
    overlap = dot(v1, psi)
    loss = 1 - abs2(overlap)
    @Zygote.ignore println("$loss $(sum(abs.(A)))")
    return loss
end

function ChainRulesCore.rrule(::typeof(adjoint_loss), t_vals, ops, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, v1, v2, p, do_hermitian, antihermitian)
    # --- Reconstruct A ---
    vals = update_values(signs, param_index_map, t_vals, parameter_mapping, parity)
    A = sparse(rows, cols, vals, dim, dim)
    if do_hermitian
        if antihermitian
            A = make_antihermitian(A)
        else
            A = make_hermitian(A)
        end
    end
    if !isnothing(p) && !(p isa SciMLBase.NullParameters)
        A = A + p
    end

    # --- Exact Gradient via Block Matrix ---
    # We want to compute: <v1 | \nabla(exp(im*A) * v2)
    # The gradient vector for parameter M_i is given by the top block of exp(im * [A M_i; 0 A]) * [0; v2].
    # We compute this for each parameter in parallel.

    # First, compute forward state and overlap for the primal return
    # Use tighter tolerance 1e-12 typically for unitarity, though default is ~1e-12.
    if antihermitian
        # psi, _ = exponentiate(A, 1.0, v2, tol=1e-12)
        psi = expv(1.0, A, v2)
    else
        # psi, _ = exponentiate(A, 1.0im, v2, tol=1e-12)
        psi = expv(1.0im, A, v2)
    end
    overlap = dot(v1, psi)
    y = 1 - abs2(overlap)

    function adjoint_loss_pullback(ȳ)
        # ȳ is sensitivity of loss (scalar)

        # dL/dt = -2 Re( <psi|v1> * <v1| dpsi/dt > )
        # We need <v1 | dpsi/dt > for each parameter.

        grad_t = Vector{Float64}(undef, length(t_vals))

        # Quadrature Gradient Implementation (N=50)
        N_steps = 50
        dt = 1.0 / N_steps

        # Forward Checkpoints: phis[k] ~ exp(i A t_k) v2
        # phis[1] corresponding to t=0 is v2.
        phis = Vector{Vector{ComplexF64}}(undef, N_steps + 1)
        phis[1] = v2

        for k in 1:N_steps
            if antihermitian
                # phis[k+1], _ = exponentiate(A, dt, phis[k], tol=1e-12)
                phis[k+1] = expv(dt, A, phis[k])
            else
                # phis[k+1], _ = exponentiate(A, dt * 1.0im, phis[k], tol=1e-12)
                phis[k+1] = expv(dt * 1.0im, A, phis[k])
            end
        end
        # Note: phis[end] should match 'psi' captured from primal pass.

        # Backward Checkpoints: chis[k] ~ exp(-i A (1-t_k)) v1
        # chis[N+1] corresponding to t=1 is v1.
        chis = Vector{Vector{ComplexF64}}(undef, N_steps + 1)
        chis[N_steps+1] = v1

        for k in N_steps:-1:1
            if antihermitian
                # chis[k], _ = exponentiate(A, -dt, chis[k+1], tol=1e-12)
                chis[k] = expv(-dt, A, chis[k+1])
            else
                # chis[k], _ = exponentiate(A, -dt * 1.0im, chis[k+1], tol=1e-12)
                chis[k] = expv(-dt * 1.0im, A, chis[k+1])
            end
        end

        # Simpson's Rule Weights
        weights = ones(N_steps + 1)
        weights[2:2:end-1] .= 4.0
        weights[3:2:end-2] .= 2.0
        weights[1] = 1.0
        weights[end] = 1.0
        weights .*= (dt / 3.0)

        # Pre-compute overlap factor (using captured 'overlap' from primal)
        conj_overlap_factor = conj(overlap) * ȳ

        # Accumulate gradients parallelized over parameters
        Threads.@threads for i in eachindex(grad_t)
            M = ops[i]
            val = 0.0 + 0.0im
            for k in 1:(N_steps+1)
                term = dot(chis[k], M, phis[k])
                val += term * weights[k]
            end

            # dO/dt = i * integral
            # dO/dt = int
            if antihermitian
                dO_dt = val
            else
                dO_dt = val * 1.0im
            end
            grad_t[i] = -2 * real(conj_overlap_factor * dO_dt) + 1e-10*t_vals[i] # regularization
        end

        return NoTangent(), grad_t, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return y, adjoint_loss_pullback
end



