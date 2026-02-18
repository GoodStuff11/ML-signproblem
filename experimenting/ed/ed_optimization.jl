include("optimization_helpers.jl")

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
    Zygote.@ignore println("$loss $(sum(mat))")
    return loss
end


# -----------------------------------------------------------------------------
# Riemannian Optimization Helpers
# -----------------------------------------------------------------------------


"""
    project_to_tangent_rank1!(G, U, ops, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, antihermitian, use_symmetry, v1, v2, overlap)

Computes the projection of the Euclidean gradient E corresponding to Loss = 1 - |<v2|U|v1>|^2 onto the tangent space at U spanned by {U * A_k}.
The gradient E is rank-1: E = - conj(overlap) * v2 * v1'.
The unconstrained tangent direction is Xi = U' * E = - conj(overlap) * (U' v2) * v1'.
Let w2 = U' v2. Then Xi = - conj(overlap) * w2 * v1'.
We project Xi onto the basis {A_k} (skew-Hermitian generators).
c_k = <A_k, Xi> / <A_k, A_k> = Re(tr(A_k' Xi)) / norm(A_k)^2.
tr(A_k' Xi) = - conj(overlap) * tr(A_k' w2 v1') = - conj(overlap) * (v1' A_k' w2).
This involves only sparse matrix-vector products and dot products.
Complexity: O(N_ops * nnz(A_k)) per step, vs O(dim^3) for dense.
"""
function project_to_tangent_rank1!(G, U, ops, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, antihermitian, use_symmetry, v1, v2, overlap)

    t1 = @elapsed begin
        # 1. Compute w2 = U' * v2
        # Since U is unitary, U' = inv(U). If U is stored as matrix.
        w2 = U' * v2

        conj_overlap = conj(overlap)
        # Factor for projection: - conj(overlap)
        factor = -conj_overlap

        n_ops = length(ops)
        coeffs = zeros(Float64, n_ops)
    end
    # Re-use buffer for A_k' * w2?
    # Actually A_k is sparse. 
    # If antihermitian=false (Hermitian M_k, A_k = i M_k), A_k' = -i M_k = -A_k.
    # If antihermitian=true (AntiHermitian M_k, A_k = M_k), A_k' = -M_k = -A_k.
    # So A_k' = -A_k always for generators of Unitary group?
    # Yes, elements of u(N) are skew-Hermitian.

    # Compute v1' * A_k' * w2
    # = - (v1' * A_k * w2)
    # = - (w2' * A_k' * v1)' = - (w2' * (-A_k) * v1)' ...
    # Just compute z = A_k * w2 (sparse mv), then dot(v1, z).

    # Pre-allocate if possible, but Ops are different sparse matrices.
    # We can perform the dot product without allocating full vector z if we iterate non-zeros.

    t2 = @elapsed begin
        for k in 1:n_ops
            op = ops[k]
            # Calculate term = v1' * A_k' * w2
            # If !antihermitian: A_k = i * op. A_k' = -i * op.
            # term = v1' * (-i op) * w2 = -i * (v1' * op * w2).
            # We need val = v1' * op * w2.

            # Sparse dot: sum_{r,c} v1[r]^* op[r,c] w2[c]
            val = 0.0 + 0.0im
            rows_op = rowvals(op)
            vals_op = nonzeros(op)
            for col in 1:size(op, 2)
                # w2_c = w2[col]
                # Optimization: check if w2[col] is zero? Unlikely for dense vector.
                w2_c = w2[col]
                for idx in nzrange(op, col)
                    row = rows_op[idx]
                    v = vals_op[idx]
                    # term: v1[row]' * v * w2[col]
                    val += conj(v1[row]) * v * w2_c
                end
            end

            if !antihermitian
                # A_k = i op.
                # tr(A_k' Xi) = factor * (-i * val)
                # c_k = Re( factor * -i * val ) / norm_sq
                scalar_term = factor * (-1im * val)
            else
                # A_k = op. A_k' = -op (if op is skew).
                # Wait, if 'op' passed is the generator itself (antihermitian=true means op is skew).
                # Then A_k' = op' = -op.
                # term = v1' * (-op) * w2 = -val.
                scalar_term = factor * (-val)
            end

            norm_sq = dim # approximate
            coeffs[k] = real(scalar_term) / norm_sq
        end
    end

    t3 = @elapsed begin
        # Reconstruct Xi_proj
        vals = update_values(signs, param_index_map, coeffs, parameter_mapping, parity)
        Xi_new = sparse(rows, cols, vals, dim, dim)

        if !use_symmetry
            if antihermitian
                Xi_new = make_antihermitian(Xi_new)
            else
                Xi_new = make_hermitian(Xi_new)
                Xi_new = 1im * Xi_new
            end
        else
            if !antihermitian
                Xi_new = 1im * Xi_new
            end
        end
    end

    t4 = @elapsed begin
        # G = U * Xi_new
        mul!(G, U, Xi_new)
    end
    # if rand() < 0.05
    println("Profiling project_to_tangent_rank1!: Prep=$(round(t1, digits=6)) Loop=$(round(t2, digits=6)) Recon=$(round(t3, digits=6)) Mul=$(round(t4, digits=6)) Total=$(round(t1+t2+t3+t4, digits=6))")
    # end
    return G
end


function optimize_unitary(state1::Vector, state2::Vector, indexer::CombinationIndexer;
    maxiters=10, ϵ=1e-5, optimization_scheme::Vector=[1, 2], spin_conserved::Bool=false, use_symmetry::Bool=false,
    gradient=:adjoint_gradient, metric_functions::Dict{String,Function}=Dict{String,Function}(),
    antihermitian::Bool=false, optimizer::Union{Symbol,Vector{Symbol}}=:LBFGS, perturb_optimization::Float64=2.0,
    operator_cache::Dict{Int,Dict{Symbol,Any}}=Dict{Int,Dict{Symbol,Any}}()
)
    # spin_conserved is only true when using (N↑, N↓) and not N
    max_order_scheme = isempty(optimization_scheme) ? 0 : maximum(optimization_scheme)
    max_order_cache = isempty(operator_cache) ? 0 : maximum(keys(operator_cache))
    max_order = max(max_order_scheme, max_order_cache)

    # Initialize return vectors with nothing
    computed_matrices = Vector{Any}(nothing, max_order)
    computed_coefficients = Vector{Any}(nothing, max_order)
    parameter_mappings = Vector{Any}(nothing, max_order)
    parities = Vector{Any}(nothing, max_order)
    coefficient_labels = Vector{Any}(nothing, max_order)

    # helper for p_args (sum of all OTHER matrices)
    function get_p_args(current_order)
        mats = [m for (i, m) in enumerate(computed_matrices) if i != current_order && !isnothing(m)]
        return isempty(mats) ? nothing : sum(mats)
    end

    dim = length(indexer.inv_comb_dict)
    metrics = Dict{String,Vector{Any}}()
    loss = 1 - abs2(state1' * state2)
    metrics["loss"] = Float64[loss]
    metrics["other"] = []
    metrics["loss_std"] = Float64[0.0]
    for k in keys(metric_functions)
        metrics[k] = Any[]
    end

    println("Initial loss: $loss")
    println("Dimension: $dim")
    if loss < 1e-15
        println("States are already equal")
        return computed_matrices, coefficient_labels, computed_coefficients, parameter_mappings, parities, metrics, operator_cache
    end

    # Define a function to ensure operator structure is computed and cached
    function ensure_operator_structure!(order)
        if haskey(operator_cache, order)
            return operator_cache[order]
        end
        # compute operator structure, initial coefficients and operators
        @time t_dict = create_randomized_nth_order_operator(order, indexer; magnitude=loss, omit_H_conj=!use_symmetry, conserve_spin=spin_conserved, normalize_coefficients=true)
        @time rows, cols, signs, ops_list = build_n_body_structure(t_dict, indexer)
        t_keys = collect(keys(t_dict))
        param_index_map = build_param_index_map(ops_list, t_keys)

        ops = []
        sym_data = nothing
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
            sym_data = (inv_param_map, parameter_mapping, parity)
        else
            for k in collect(keys(t_dict))
                _rows, _cols, _signs, _ = build_n_body_structure(Dict(k => 1.0), indexer)
                if antihermitian
                    push!(ops, make_antihermitian(sparse(_rows, _cols, _signs, dim, dim)))
                else
                    push!(ops, make_hermitian(sparse(_rows, _cols, _signs, dim, dim)))
                end
            end
            parameter_mapping = nothing
            parity = nothing
        end

        cache_entry = Dict(
            :t_dict => t_dict,
            :rows => rows,
            :cols => cols,
            :signs => signs,
            :ops_list => ops_list,
            :t_keys => t_keys,
            :param_index_map => param_index_map,
            :sym_data => sym_data,
            :ops => ops,
            :parameter_mapping => parameter_mapping,
            :parity => parity,
            :coefficients => nothing # Initialize as nothing
        )
        operator_cache[order] = cache_entry
        return cache_entry
    end

    # 1. Pre-population: realize matrices for any coefficients provided in the cache
    for (order_idx, struct_data) in operator_cache
        coeffs = get(struct_data, :coefficients, nothing)
        if !isnothing(coeffs)
            # Assign labels and mappings
            coefficient_labels[order_idx] = struct_data[:t_keys]
            parameter_mappings[order_idx] = struct_data[:parameter_mapping]
            parities[order_idx] = struct_data[:parity]
            computed_coefficients[order_idx] = coeffs

            # Compute the matrix
            vals = update_values(struct_data[:signs], struct_data[:param_index_map], coeffs, struct_data[:parameter_mapping], struct_data[:parity])
            if !use_symmetry
                if antihermitian
                    computed_matrices[order_idx] = make_antihermitian(sparse(struct_data[:rows], struct_data[:cols], vals, dim, dim))
                else
                    computed_matrices[order_idx] = make_hermitian(sparse(struct_data[:rows], struct_data[:cols], vals, dim, dim))
                end
            else
                computed_matrices[order_idx] = sparse(struct_data[:rows], struct_data[:cols], vals, dim, dim)
            end
        end
    end

    # 2. Main Optimization Scheme Loop
    for order ∈ optimization_scheme
        struct_data = ensure_operator_structure!(order)

        # Extract variables for convenience from struct_data
        t_dict = struct_data[:t_dict]
        rows = struct_data[:rows]
        cols = struct_data[:cols]
        signs = struct_data[:signs]
        ops_list = struct_data[:ops_list]
        t_keys = struct_data[:t_keys]
        param_index_map = struct_data[:param_index_map]
        sym_data = struct_data[:sym_data]
        ops = struct_data[:ops]
        parameter_mapping = struct_data[:parameter_mapping]
        parity = struct_data[:parity]
        existing_coeffs = struct_data[:coefficients]

        coefficient_labels[order] = t_keys

        # Determine Initial Coefficients (t_vals) for this optimization order
        magnitude_esimate = loss / length(t_keys)

        if !isnothing(existing_coeffs)
            t_vals = existing_coeffs
        else
            if use_symmetry
                # sym_data is (inv_param_map, ...)
                t_vals = real(rand(typeof(signs[1]), length(sym_data[1])) * magnitude_esimate)
            else
                t_vals = real(collect(values(t_dict)))
            end
        end

        println("Parameter count: $(length(t_vals))")
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
            # push!(loss_tracker, loss_val)
            N = 20

            grad_msg = if isnothing(state.grad)
                "gradient=N/A relative-change=N/A curvature=N/A"
            else
                "gradient=$(sum(abs, state.grad)) relative-change=$(sum(state.grad ./ state.u)) curvature=$(sum(state.grad .* state.u))"
            end

            # println("loss=$loss_val state=$(sum(abs, state.u)/length(state.u)) $grad_msg")
            println("loss=$loss_val state=$(norm(state.u)) $grad_msg")
            # if optimizer == :riemann
            #     unitarity_err = norm(state.u' * state.u - I)
            #     println("loss=$loss_val unitarity_err=$unitarity_err")
            # else
            #     println("loss=$loss_val")
            # end

            push!(tmp_losses, loss_val)
            if length(tmp_losses) > N && std(tmp_losses[end-N:end]) < 1e-8
                # println("std: $(std(tmp_losses[end-N:end]))")
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

        if gradient == :gradient
            optf = Optimization.OptimizationFunction(f, Optimization.AutoZygote())
        elseif gradient == :manualgradient
            optf = Optimization.OptimizationFunction(f_nongradient, grad=fg!)
        elseif gradient == :adjoint_gradient
            optf = Optimization.OptimizationFunction(f_adjoint, Optimization.AutoZygote())
        else
            optf = Optimization.OptimizationFunction(f_nongradient)
        end


        optimizers = optimizer isa Vector ? optimizer : [optimizer]

        sol = nothing
        for optimizer_sym in optimizers
            # 1. Setup Phase
            if length(optimizers) > 1 && perturb_optimization > 1e-9
                t_vals = t_vals + perturb_optimization * mean(abs.(t_vals)) * (2 * rand(length(t_vals)) .- 1)
            end
            if optimizer_sym == :riemann
                # --- Riemannian Setup ---
                p_args = get_p_args(order)
                if !isnothing(p_args)
                    # If we have background matrices, we need to fold them into the starting point?
                    # params_to_unitary handles computed_matrices.
                    # But here we only pass 'computed_matrices' which includes 'order' if it was computed?
                    # No, get_p_args excludes 'order'.
                    # Actually params_to_unitary just sums them.
                    # Let's constructing a temp list for params_to_unitary.
                    temp_mats = [m for (i, m) in enumerate(computed_matrices) if i != order && !isnothing(m)]
                else
                    temp_mats = []
                end

                U0 = params_to_unitary(t_vals, rows, cols, signs, param_index_map, parameter_mapping, parity, temp_mats, dim, antihermitian, use_symmetry)
                manifold = UnitaryMatrices(dim)

                function cost_riemann(U, p)
                    return 1 - abs2(dot(state2, U * state1))
                end
                function grad_riemann(G, U, p)
                    overlap = dot(state2, U * state1)
                    project_to_tangent_rank1!(G, U, ops, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, antihermitian, use_symmetry, state1, state2, overlap)
                    return G
                end

                current_optf = OptimizationFunction(cost_riemann, grad=grad_riemann)
                prob = OptimizationProblem(current_optf, U0, manifold=manifold)
                opt_algo = OptimizationManopt.GradientDescentOptimizer()

            else
                # --- Euclidean Setup ---
                p_args = get_p_args(order)

                # Create Problem using global optf (defined outside loop)
                prob = OptimizationProblem(optf, t_vals, p_args)

                # Determine Algorithm
                if optimizer_sym == :LBFGS
                    opt_algo = Optim.LBFGS()
                elseif optimizer_sym == :BFGS
                    opt_algo = OptimizationOptimJL.BFGS()
                elseif optimizer_sym == :GradientDescent
                    opt_algo = OptimizationOptimJL.GradientDescent()
                else
                    opt_algo = OptimizationOptimJL.LBFGS()
                end
            end

            # 2. Execution Phase: Single Solve Call
            println("Solving with $optimizer_sym...")
            @time sol = Optimization.solve(prob, opt_algo, maxiters=maxiters, callback=callback)

            # 3. Post-Process Phase
            if optimizer_sym == :riemann
                t_vals = unitary_to_params(sol.u, ops, dim, antihermitian)
            else
                t_vals = sol.u
            end
        end

        # After loop, final coefficients are in t_vals
        coefficients = t_vals

        loss = f_nongradient(t_vals, get_p_args(order))
        metric = sol # approx?


        vals = update_values(signs, param_index_map, coefficients, parameter_mapping, parity)

        # loss = f(new_tvals, if length(computed_matrices) > 0 sum(computed_matrices) else nothing end)
        push!(metrics["other"], metric)

        # Construct and Store Matrix
        if !use_symmetry
            if antihermitian
                computed_matrices[order] = make_antihermitian(sparse(rows, cols, vals, dim, dim))
            else
                computed_matrices[order] = make_hermitian(sparse(rows, cols, vals, dim, dim))
            end
        else
            computed_matrices[order] = sparse(rows, cols, vals, dim, dim)
        end

        println("Finished order $order")
        push!(metrics["loss"], loss)
        # push!(metrics["loss_std"], std(last(tmp_losses, 20)))
        computed_coefficients[order] = coefficients
        parameter_mappings[order] = parameter_mapping
        parities[order] = parity
        # Store back into cache
        operator_cache[order][:coefficients] = coefficients

        # Metrics using all computed matrices
        # Need to clean list for metrics?
        clean_matrices = [m for m in computed_matrices if !isnothing(m)]
        for (k, func) in metric_functions
            push!(metrics[k], func(state1, state2, clean_matrices, tmp_losses))
        end
    end
    # display(plot(loss_tracker, yscale=:log10))
    # println("hey")
    return computed_matrices, coefficient_labels, computed_coefficients, parameter_mappings, parities, metrics, operator_cache
end


function test_map_to_state(degen_rm_U::Union{AbstractMatrix,Vector}, instructions::Dict{String,Any}, indexer::CombinationIndexer,
    spin_conserved::Bool=false;
    maxiters=100, gradient::Symbol=:gradient, metric_functions::Dict{String,Function}=Dict{String,Function}(), optimizer::Union{Symbol,Vector{Symbol}}=:LBFGS
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
                maxiters=maxiters, optimization_scheme=get!(instructions, "optimization_scheme", [1, 2]), gradient=gradient,
                metric_functions=metric_functions, antihermitian=get!(instructions, "antihermitian", false), optimizer=optimizer)
            computed_matrices, coefficient_labels, coefficient_values, param_mapping, parities, metrics, _ = args
            push!(data_dict["norm1_metrics"], [isnothing(cm) ? 0.0 : norm(cm, 1) for cm in computed_matrices])
            push!(data_dict["norm2_metrics"], [isnothing(cm) ? 0.0 : norm(cm, 2) for cm in computed_matrices])
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

function interaction_scan_map_to_state(degen_rm_U::Union{AbstractMatrix,Vector}, instructions::Dict{String,Any}, indexer::CombinationIndexer,
    spin_conserved::Bool=false;
    maxiters=100, gradient::Symbol=:gradient, metric_functions::Dict{String,Function}=Dict{String,Function}(), optimizer::Union{Symbol,Vector{Symbol}}=:LBFGS,
    save_folder::Union{String,Nothing}=nothing, save_name::String="scan_data"
)
    # instructions["u_range"] should be a range of indices, e.g., 1:10
    # instructions["starting state"] should define the fixed reference state (state1)

    data_dict = Dict{String,Any}("norm1_metrics" => [], "norm2_metrics" => [],
        "loss_metrics" => [], "labels" => [], "loss_std_metrics" => [], "all_matrices" => [],
        "coefficients" => [], "coefficient_labels" => nothing, "param_mapping" => nothing, "parities" => nothing)

    shared_cache = Dict{Int,Dict{Symbol,Any}}()

    u_indices = instructions["u_range"]

    if !isnothing(save_folder)
        mkpath(save_folder)
    end
    shared_data_saved = false

    # Define state1 (fixed reference)
    ref_u_idx = instructions["starting state"]["U index"]
    ref_level = instructions["starting state"]["levels"][1] # assuming single level for reference

    for u_idx in u_indices
        println("\n--- Scanning U index: $u_idx ---")

        if degen_rm_U isa AbstractMatrix
            state1 = degen_rm_U[ref_u_idx, :]
            state2 = degen_rm_U[u_idx, :]
        else
            state1 = degen_rm_U[ref_u_idx][:, ref_level]
            state2 = degen_rm_U[u_idx][:, instructions["ending state"]["levels"][1]]
        end

        args = optimize_unitary(state1, state2, indexer;
            spin_conserved=spin_conserved, use_symmetry=get!(instructions, "use symmetry", false),
            maxiters=maxiters, optimization_scheme=get!(instructions, "optimization_scheme", [1, 2]), gradient=gradient,
            metric_functions=metric_functions, antihermitian=get!(instructions, "antihermitian", false), optimizer=optimizer,
            operator_cache=shared_cache)

        computed_matrices, coefficient_labels, coefficient_values, param_mapping, parities, metrics, shared_cache = args

        # Store results for this U
        push!(data_dict["norm1_metrics"], [isnothing(cm) ? 0.0 : norm(cm, 1) for cm in computed_matrices])
        push!(data_dict["norm2_metrics"], [isnothing(cm) ? 0.0 : norm(cm, 2) for cm in computed_matrices])
        push!(data_dict["all_matrices"], computed_matrices)
        push!(data_dict["coefficients"], coefficient_values)
        if isnothing(data_dict["coefficient_labels"])
            data_dict["coefficient_labels"] = coefficient_labels
            data_dict["param_mapping"] = param_mapping
            data_dict["parities"] = parities
        end

        # Save shared data once we have it
        if !isnothing(save_folder) && !shared_data_saved && !isnothing(coefficient_labels)
            shared_dict = Dict(
                "coefficient_labels" => coefficient_labels,
                "param_mapping" => param_mapping,
                "parities" => parities,
                "instructions" => instructions,
                "u_range" => u_indices
            )
            JLD2.jldsave(joinpath(save_folder, "$(save_name)_shared.jld2"); dict=shared_dict)
            shared_data_saved = true
        end

        # Save iteration data
        if !isnothing(save_folder)
            iter_dict = Dict(
                "u_idx" => u_idx,
                "coefficient_values" => coefficient_values,
                "metrics" => metrics,
                "norm1" => [isnothing(cm) ? 0.0 : norm(cm, 1) for cm in computed_matrices],
                "norm2" => [isnothing(cm) ? 0.0 : norm(cm, 2) for cm in computed_matrices]
            )
            JLD2.jldsave(joinpath(save_folder, "$(save_name)_u_$u_idx.jld2"); dict=iter_dict)
        end

        for (k, val) in metrics
            if k * "_metrics" ∉ keys(data_dict)
                data_dict[k*"_metrics"] = [val]
            else
                push!(data_dict[k*"_metrics"], val)
            end
        end
        push!(data_dict["labels"], Dict(
            "starting state" => Dict("level" => ref_level, "U index" => ref_u_idx),
            "ending state" => Dict("level" => instructions["ending state"]["levels"][1], "U index" => u_idx))
        )
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
    # Zygote.@ignore println("loss=$loss coeff-mag=$(sum(abs.(t_vals)))")
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
            grad_t[i] = -2 * real(conj_overlap_factor * dO_dt) + 1e-3 * t_vals[i] # regularization
        end

        return NoTangent(), grad_t, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return y, adjoint_loss_pullback
end



