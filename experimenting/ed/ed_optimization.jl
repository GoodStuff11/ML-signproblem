


# using CombDiff
# using LinearAlgebra
# using SparseArrays
# import Statistics
# using Statistics: std, mean
# using InteractiveUtils
# using Dates
using ExponentialUtilities
using Statistics

# Auxiliary Matrix Type for Trotter Gradient
struct AuxiliaryMatrix{T,A,B} <: AbstractMatrix{T}
    H::A
    Hk::B
    n::Int
end

AuxiliaryMatrix(H::A, Hk::B, n::Int) where {A,B} = AuxiliaryMatrix{eltype(A),A,B}(H, Hk, n)

Base.size(M::AuxiliaryMatrix) = (2 * M.n, 2 * M.n)
Base.eltype(::AuxiliaryMatrix{T}) where T = T

function LinearAlgebra.opnorm(M::AuxiliaryMatrix, p::Real=Inf)
    return opnorm(M.H, p) + opnorm(M.Hk, p)
end

LinearAlgebra.ishermitian(M::AuxiliaryMatrix) = false
LinearAlgebra.issymmetric(M::AuxiliaryMatrix) = false

function _mul_aux!(y, M, x, alpha, beta)
    n = M.n
    x1 = selectdim(x, 1, 1:n)
    x2 = selectdim(x, 1, n+1:2n)
    y1 = selectdim(y, 1, 1:n)
    y2 = selectdim(y, 1, n+1:2n)

    mul!(y1, M.H, x1, alpha, beta)
    mul!(y2, M.H, x2, alpha, beta)
    mul!(y2, M.Hk, x1, alpha, 1)
    return y
end

# Multi-dispatch to avoid ambiguity with LinearAlgebra
function LinearAlgebra.mul!(y::AbstractVector, M::AuxiliaryMatrix, x::AbstractVector, alpha::Number, beta::Number)
    return _mul_aux!(y, M, x, alpha, beta)
end
function LinearAlgebra.mul!(y::AbstractMatrix, M::AuxiliaryMatrix, x::AbstractMatrix, alpha::Number, beta::Number)
    return _mul_aux!(y, M, x, alpha, beta)
end
function LinearAlgebra.mul!(y::AbstractVector, M::AuxiliaryMatrix, x::AbstractMatrix, alpha::Number, beta::Number)
    return _mul_aux!(y, M, x, alpha, beta)
end
function LinearAlgebra.mul!(y::AbstractMatrix, M::AuxiliaryMatrix, x::AbstractVector, alpha::Number, beta::Number)
    return _mul_aux!(y, M, x, alpha, beta)
end

LinearAlgebra.mul!(y::AbstractVector, M::AuxiliaryMatrix, x::AbstractVector) = mul!(y, M, x, 1, 0)
LinearAlgebra.mul!(y::AbstractMatrix, M::AuxiliaryMatrix, x::AbstractMatrix) = mul!(y, M, x, 1, 0)

# --- Suzuki-Trotter Utilities ---

function get_suzuki_trotter_sequence(num_ops, order)
    if order == 1
        return [(i, 1.0) for i in 1:num_ops]
    elseif order == 2
        # S2 = exp(0.5 dt H1) ... exp(dt Hm) ... exp(0.5 dt H1)
        seq = []
        for i in 1:num_ops
            push!(seq, (i, 0.5))
        end
        # Reverse and combine middle
        for i in num_ops:-1:1
            push!(seq, (i, 0.5))
        end
        return seq
    elseif order == 4
        # S4(dt) = S2(p*dt) S2(p*dt) S2((1-4p)*dt) S2(p*dt) S2(p*dt)
        p = 1.0 / (4.0 - 4.0^(1.0 / 3.0))
        s2 = get_suzuki_trotter_sequence(num_ops, 2)
        seq = []
        for w in [p, p, 1 - 4p, p, p]
            for (idx, weight) in s2
                push!(seq, (idx, weight * w))
            end
        end
        return seq
    else
        error("Suzuki-Trotter order $order not supported. Use 1, 2, or 4.")
    end
end

function trotter_evolve(psi, ops, t_vals, order, steps)
    dt = 1.0 / steps
    seq_base = get_suzuki_trotter_sequence(length(ops), order)
    curr_psi = copy(psi)
    for _ in 1:steps
        for (idx, w) in seq_base
            curr_psi = expv(1im * dt * w * t_vals[idx], ops[idx], curr_psi)
        end
    end
    return curr_psi
end

function trotter_gradient_adjoint(state1, state2, ops, t_vals, order, steps)
    dim = length(state1)
    num_ops = length(ops)
    dt = 1.0 / steps
    seq_base = get_suzuki_trotter_sequence(num_ops, order)

    # Build full sequence for all steps
    full_seq = []
    for _ in 1:steps
        append!(full_seq, seq_base)
    end

    L = length(full_seq)
    phis = Vector{Vector{ComplexF64}}(undef, L + 1)
    phis[1] = state1
    for j in 1:L
        idx, w = full_seq[j]
        phis[j+1] = expv(1im * dt * w * t_vals[idx], ops[idx], phis[j])
    end

    overlap = state2' * phis[L+1]
    grad = zeros(Float64, num_ops)
    chi = state2

    for j in L:-1:1
        idx, w = full_seq[j]
        # dOverlap/dt_idx = <chi_j | i * dt * w * ops[idx] | phis[j+1]>
        contrib = 1im * dt * w * (chi' * (ops[idx] * phis[j+1]))
        grad[idx] += -2 * real(conj(overlap) * contrib)

        # Pull back chi: chi_{j-1} = exp(-i * dt * w * t_vals[idx] * ops[idx]) * chi_j
        chi = expv(-1im * dt * w * t_vals[idx], ops[idx], chi)
    end

    return grad
end
LinearAlgebra.mul!(y::AbstractVector, M::AuxiliaryMatrix, x::AbstractMatrix) = mul!(y, M, x, 1, 0)
LinearAlgebra.mul!(y::AbstractMatrix, M::AuxiliaryMatrix, x::AbstractVector) = mul!(y, M, x, 1, 0)


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

function optimize_unitary(state1::Vector, state2::Vector, indexer::CombinationIndexer;
    maxiters=10, ϵ=1e-5, max_order=2, spin_conserved::Bool=false, use_symmetry::Bool=false,
    optimization=:gradient, metric_functions::Dict{String,Function}=Dict{String,Function}(),
    trotter_order=0, trotter_steps=1
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

    for order = 1:max_order
        magnitude_esimate = loss / 2
        learning_rate = loss / 10
        println("magnitude: $magnitude_esimate")
        println("learning rate: $learning_rate")
        @time t_dict = create_randomized_nth_order_operator(order, indexer; magnitude=magnitude_esimate, omit_H_conj=!use_symmetry, conserve_spin=spin_conserved)
        @time rows, cols, signs, ops_list = build_n_body_structure(t_dict, indexer)
        t_keys = collect(keys(t_dict))
        param_index_map = build_param_index_map(ops_list, t_keys)

        ops = []
        if use_symmetry
            inv_param_map, parameter_mapping, parity = find_symmetry_groups(t_keys, maximum(indexer.a).coordinates...,
                hermitian=true, trans_x=true, trans_y=true, spin_symmetry=true)

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
                rows, cols, signs, ops_list = build_n_body_structure(Dict(k => 1), indexer)
                push!(ops, make_hermitian(sparse(rows, cols, signs, dim, dim)))
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



        function trotter_grad!(grad, t_vals, p=nothing)
            if trotter_order > 0
                # Use Adjoint method for Suzuki-Trotter
                grad_adj = trotter_gradient_adjoint(state1, state2, ops, t_vals, trotter_order, trotter_steps)
                grad .= grad_adj
                return grad
            end

            # Reconstruct Full Hamiltonian H
            vals = update_values(signs, param_index_map, t_vals, parameter_mapping, parity)
            H = sparse(rows, cols, vals, dim, dim)
            if !use_symmetry
                H = make_hermitian(H)
            end

            # Compute baseline forward evolution U|ψ1> once for overlap term
            psi1_U = expv(1im, H, state1)
            overlap = state2' * psi1_U

            # Parallel loop over parameters to compute gradient components
            Threads.@threads for k in 1:length(t_vals)
                Hk = ops[k]

                # Construct Auxiliary Operator (Implicit M)
                # M = [H 0; Hk H]
                M = AuxiliaryMatrix(H, Hk, dim)

                # Compute derivative via expv
                # exp(iM) * [psi1; 0] = [U*psi1; i * (dU/dt_k)*psi1] (because M is lower tri)
                v_in = zeros(ComplexF64, 2 * dim)
                v_in[1:dim] = state1

                # Increased Krylov subspace size m=100 to ensure convergence for large Norm(H)
                v_out = expv(1im, M, v_in; m=100)

                # The second block is i * dU/dt_k * psi1
                # This IS d(exp(iH))/dt * psi1.
                d_psi = v_out[dim+1:end]

                # Gradient of Loss L = 1 - |<ψ2|U|ψ1>|^2
                d_overlap = state2' * d_psi
                grad[k] = -2 * real(conj(overlap) * d_overlap)
            end
            return grad
        end

        function f_nongradient(t_vals, p=nothing)
            if trotter_order > 0
                psi_evolved = trotter_evolve(state1, ops, t_vals, trotter_order, trotter_steps)
                loss = 1 - abs2(state2' * psi_evolved)
                println(loss)
                return loss
            end

            vals = update_values(signs, param_index_map, t_vals, parameter_mapping, parity)
            mat = sparse(rows, cols, vals, dim, dim)
            if !use_symmetry
                mat = make_hermitian(mat)
            end
            if p isa AbstractMatrix
                mat += p
            end
            loss = 1 - abs2(state2' * expv(1im, mat, state1))
            println(loss)
            return loss
        end
        function f(t_vals, p=nothing)
            vals = update_values(signs, param_index_map, t_vals, parameter_mapping, parity)
            mat = sparse(rows, cols, vals, dim, dim)
            if !use_symmetry
                mat = make_hermitian(mat)
            end
            if p isa AbstractMatrix
                mat += p
            end
            loss = 1 - abs2(state2' * exp(1im * Matrix(mat)) * state1)
            println(loss)
            return loss
        end

        if optimization == :gradient
            optf = Optimization.OptimizationFunction(f, Optimization.AutoZygote())
        elseif optimization == :manualgradient
            optf = Optimization.OptimizationFunction(f_nongradient, grad=trotter_grad!)
        elseif optimization == :combdiff
            # Construct M tensor: M[k, p, q] = signs[k] * (p == rows[k] && q == cols[k])
            # We represent this as a dense array for CombDiff.

            D = dim
            num_params = length(t_vals)
            M_tensor = zeros(ComplexF64, num_params, D, D)

            # Helper to map op_index to param_index
            function get_param_idx(op_idx)
                raw_idx = param_index_map[op_idx]
                if !isnothing(parameter_mapping)
                    return parameter_mapping[raw_idx]
                else
                    return raw_idx
                end
            end

            function get_parity_sign(op_idx)
                if !isnothing(parity) && !isnothing(parameter_mapping)
                    raw_idx = param_index_map[op_idx]
                    return parity[raw_idx]
                end
                return 1.0
            end

            for (i, (r, c, s)) in enumerate(zip(rows, cols, signs))
                k = get_param_idx(i)
                sign_val = s * get_parity_sign(i)
                M_tensor[k, r, c] += sign_val
                if !use_symmetry
                    if r != c
                        M_tensor[k, c, r] += conj(sign_val)
                    else
                        M_tensor[k, c, r] += conj(sign_val)
                    end
                end
            end

            v1_vec = Vector(conj(state2))
            v2_vec = Vector(state1)

            # Instantiate the specific gradient function for this M, v1, v2
            # COMBDIF_GRAD_GEN takes (M, v1, v2, matexp)
            my_matexp = (x) -> exp(1im * x)
            grad_func = COMBDIF_GRAD_GEN(M_tensor, v1_vec, v2_vec, my_matexp)

            function combdiff_gradient!(G, x, p)
                # grad_func(x, seed) -> result_map
                # seed is 1.0 for scalar output
                res_map = grad_func(x, 1.0 + 0.0im)
                for i in 1:length(x)
                    G[i] = real(res_map(i))
                end
            end

            optf = Optimization.OptimizationFunction(f_nongradient, grad=combdiff_gradient!)
        else
            optf = Optimization.OptimizationFunction(f_nongradient)
        end

        if length(computed_matrices) > 0
            prob = Optimization.OptimizationProblem(optf, t_vals, sum(computed_matrices))
        else
            prob = Optimization.OptimizationProblem(optf, t_vals)
        end

        if optimization == :gradient || optimization == :manualgradient || optimization == :combdiff
            # opt = OptimizationOptimisers.Adam(learning_rate)
            # BFGS is faster than LBFGS, which both converge faster than adam
            @time sol = Optimization.solve(prob, Optim.BFGS(), maxiters=maxiters, callback=callback)
            s = sol
        else
            function prob_func(prob, i, repeat)
                remake(prob, u0=t_vals)
            end

            ensembleproblem = Optimization.EnsembleProblem(prob; prob_func)
            @time sol = Optimization.solve(ensembleproblem, OptimizationOptimJL.ParticleSwarm(), EnsembleThreads(), trajectories=Threads.nthreads(), maxiters=maxiters, callback=callback)
            s = sol[argmin([s.objectives for s in sol])]
        end

        vals = update_values(signs, param_index_map, sol.u, parameter_mapping, parity)

        # loss = f(new_tvals, if length(computed_matrices) > 0 sum(computed_matrices) else nothing end)
        loss = sol.objective
        push!(metrics["other"], sol.original)
        if !use_symmetry
            push!(computed_matrices, make_hermitian(sparse(rows, cols, vals, dim, dim)))
        else
            push!(computed_matrices, sparse(rows, cols, vals, dim, dim))
        end
        println("Finished order $order")
        push!(metrics["loss"], loss)
        push!(metrics["loss_std"], std(last(tmp_losses, 20)))
        push!(computed_coefficients, sol.u)
        push!(parameter_mappings, parameter_mapping)
        push!(parities, parity)
        for (k, func) in metric_functions
            push!(metrics[k], func(state1, state2, computed_matrices, tmp_losses))
        end
        println("loss std: $(metrics["loss_std"][end])")
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
                maxiters=maxiters, max_order=get!(instructions, "max_order", 2), optimization=optimization,
                metric_functions=metric_functions)
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