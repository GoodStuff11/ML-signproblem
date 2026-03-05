function optimize_unitary(state1::Vector, state2::Vector, indexer::CombinationIndexer;
    maxiters=10, ϵ=1e-5, optimization_scheme::Vector=[1, 2], spin_conserved::Bool=false,
    gradient=:adjoint_gradient, metric_functions::Dict{String,Function}=Dict{String,Function}(),
    antihermitian::Bool=false, optimizer::Union{Symbol,Vector{Symbol}}=:LBFGS, perturb_optimization::Float64=0.001,
    initial_coefficients::Vector{Any}=Any[], initialization_samples::Int=20,
    operator_cache::Dict{Int,Dict{Symbol,Any}}=Dict{Int,Dict{Symbol,Any}}(),
    momentum_basis::Bool=false
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
    prev_loss = loss
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

    # Define a function to ensure operator structure is pre-computed and cached. This operator structure doesn't store any coefficient values.
    function ensure_operator_structure!(order)
        if haskey(operator_cache, order)
            return operator_cache[order]
        end
        # compute operator structure, initial coefficients and operators
        @time t_dict = create_randomized_nth_order_operator(order, indexer; magnitude=loss * 100, omit_H_conj=true, conserve_spin=spin_conserved, normalize_coefficients=false, conserve_momentum=momentum_basis)
        @time rows, cols, signs, ops_list = build_n_body_structure(t_dict, indexer)
        t_keys = collect(keys(t_dict))
        param_index_map = build_param_index_map(ops_list, t_keys)

        # create matrix operators to make gradient computation faster
        ops = []
        for k in collect(keys(t_dict))
            _rows, _cols, _signs, _ = build_n_body_structure(Dict(k => 1.0), indexer)
            if antihermitian
                push!(ops, make_antihermitian(sparse(_rows, _cols, _signs, dim, dim)))
            else
                push!(ops, make_hermitian(sparse(_rows, _cols, _signs, dim, dim)))
            end
        end

        cache_entry = Dict(
            :t_dict => t_dict,
            :rows => rows,
            :cols => cols,
            :signs => signs,
            :ops_list => ops_list,
            :t_keys => t_keys,
            :param_index_map => param_index_map,
            :ops => ops
        )
        operator_cache[order] = cache_entry
        return cache_entry
    end

    # 1. Pre-population: realize matrices for any coefficients provided externally
    # println(isempty(initial_coefficients), " ", initial_coefficients)
    if !isempty(initial_coefficients)
        println("COMPUTED MATRICES")
        for order_idx in eachindex(initial_coefficients)
            coeffs = initial_coefficients[order_idx]
            if isnothing(coeffs) || isempty(coeffs)
                continue
            end

            struct_data = ensure_operator_structure!(order_idx)

            # Assign labels and mappings
            coefficient_labels[order_idx] = struct_data[:t_keys]
            parameter_mappings[order_idx] = struct_data[:parameter_mapping]
            parities[order_idx] = struct_data[:parity]
            computed_coefficients[order_idx] = coeffs

            # Compute the matrix
            vals = update_values(struct_data[:signs], struct_data[:param_index_map], coeffs, struct_data[:parameter_mapping], struct_data[:parity])
            if antihermitian
                computed_matrices[order_idx] = make_antihermitian(sparse(struct_data[:rows], struct_data[:cols], vals, dim, dim))
            else
                computed_matrices[order_idx] = make_hermitian(sparse(struct_data[:rows], struct_data[:cols], vals, dim, dim))
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
        coefficient_labels[order] = t_keys

        function f_adjoint(t_vals, p=nothing)
            return adjoint_loss(t_vals, ops, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state2, state1, p, antihermitian)
        end

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

            println("loss=$loss_val avg_coef=$(mean(abs.(state.u))) $grad_msg")

            push!(tmp_losses, loss_val)
            if length(tmp_losses) > N && std(tmp_losses[end-N:end]) < 1e-8
                return true
            end
            return false
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

        # runs the optimization for the selection initial coefficients. 
        function execute_optimization(current_t_vals, current_maxiters, _optimizers)
            local_t_vals = copy(current_t_vals)
            local_loss = loss
            local_sol = nothing

            for (optimizer_idx, optimizer_sym) in enumerate(_optimizers)
                # 1. Setup Phase
                # Perturbs the coefficients if the previous optimization reached a bad local minimum, also helps with regularization.
                if optimizer_idx > 1 && perturb_optimization > 1e-9 && mean(abs.(local_t_vals)) > 1e-1
                    local_t_vals = local_t_vals * (1 - perturb_optimization) + perturb_optimization * mean(abs.(local_t_vals)) * (2 * rand(length(local_t_vals)) .- 1)
                end


                # --- Euclidean Setup ---
                p_args = get_p_args(order)

                # Create Problem using global optf (defined outside loop)
                prob = OptimizationProblem(optf, local_t_vals, p_args)

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

                # 2. Execution Phase: Single Solve Call
                empty!(tmp_losses)
                println("Solving with $optimizer_sym...")
                @time local_sol = Optimization.solve(prob, opt_algo, maxiters=current_maxiters, callback=callback)

                # 3. Post-Process Phase
                local_t_vals = local_sol.u
                local_loss = local_sol.objective
            end

            return local_sol, local_t_vals, local_loss
        end

        # find values for t_vals. If they don't already exist, they are sampled, and selected based on having a large enough gradient and a low enough loss.
        if length(initial_coefficients) >= order && !isnothing(initial_coefficients[order])
            t_vals = initial_coefficients[order]
        elseif initialization_samples > 0
            println("Sampling $initialization_samples initial configurations over a range of magnitudes...")
            p_args = get_p_args(order)

            good_samples = []
            initial_loss = loss

            log_min = log10(1e-7)
            log_max = log10(1e-1)

            for s in 1:initialization_samples
                mag = 10^(log_min + (log_max - log_min) * rand())

                t_sample = (2 * rand(length(t_dict)) .- 1) * mag


                res = Zygote.withgradient(t -> f_adjoint(t, p_args), t_sample)
                l_tmp = res.val
                g_tmp = res.grad[1]
                gnorm = norm(g_tmp)

                is_good = (gnorm > 1e-8) && (l_tmp < initial_loss * 10)

                if is_good
                    push!(good_samples, (gnorm, l_tmp, copy(t_sample)))
                end

                if initialization_samples <= 50
                    println("Sample $s (mag=$(round(mag, sigdigits=3))): loss=$l_tmp grad_norm=$gnorm (GOOD: $is_good)")
                end
            end

            sort!(good_samples, by=x -> x[1], rev=true)
            top_n = min(5, length(good_samples))

            if top_n == 0
                println("No good samples found, falling back to random initialization")
                t_vals = real(collect(values(t_dict)))

            else
                best_t = nothing
                best_loss = Inf
                println("Performing quick optimization on top $top_n candidates...")
                quick_maxiters = min(30, maxiters)
                for i in 1:top_n
                    candidate_t = good_samples[i][3]
                    _, opt_t, opt_loss = execute_optimization(candidate_t, quick_maxiters, [:GradientDescent, :LBFGS])
                    println("Candidate $i quick opt loss: $opt_loss")
                    if opt_loss < best_loss
                        best_loss = opt_loss
                        best_t = opt_t
                    end
                end
                t_vals = best_t
                println("Selected best candidate with loss=$best_loss")
            end
        else
            error("DON'T DO THIS EVER")
        end

        println("Parameter count: $(length(t_vals))")

        _, t_vals, loss = execute_optimization(t_vals, maxiters, optimizers)
        coefficients = t_vals


        vals = update_values(signs, param_index_map, coefficients, parameter_mapping, parity)

        # Construct and Store Matrix
        if antihermitian
            computed_matrices[order] = make_antihermitian(sparse(rows, cols, vals, dim, dim))
        else
            computed_matrices[order] = make_hermitian(sparse(rows, cols, vals, dim, dim))
        end


        println("Finished order $order")
        push!(metrics["loss"], loss)
        # push!(metrics["loss_std"], std(last(tmp_losses, 20)))
        computed_coefficients[order] = coefficients
        parameter_mappings[order] = parameter_mapping
        parities[order] = parity
        # Store back into cache
        # operator_cache[order][:coefficients] = coefficients

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



"""
Wrapper function for optimize_unitary that will be used at the highest level. 
"""
function perform_optimization(degen_rm_U::Union{AbstractMatrix,Vector}, u_idx1::Int, u_idx2::Int, instructions::Dict{String,Any}, indexer::CombinationIndexer,
    spin_conserved::Bool=false;
    maxiters=100, gradient::Symbol=:gradient, metric_functions::Dict{String,Function}=Dict{String,Function}(), optimizer::Union{Symbol,Vector{Symbol}}=:LBFGS,
    perturb_optimization::Float64=0.1,
    save_folder::Union{String,Nothing}=nothing, save_name::String="data"
)
    println("\n--- Optimizing between U indices: $u_idx1 and $u_idx2 ---")

    state1 = degen_rm_U[u_idx1, :]
    state2 = degen_rm_U[u_idx2, :]

    args = optimize_unitary(state1, state2, indexer;
        spin_conserved=spin_conserved,
        maxiters=maxiters, optimization_scheme=get!(instructions, "optimization_scheme", [1, 2]), gradient=gradient,
        metric_functions=metric_functions, antihermitian=get!(instructions, "antihermitian", false), optimizer=optimizer,
        initial_coefficients=Any[], perturb_optimization=perturb_optimization)

    computed_matrices, coefficient_labels, coefficient_values, param_mapping, parities, metrics, _ = args

    data_dict = Dict{String,Any}(
        "all_matrices" => computed_matrices,
        "coefficients" => coefficient_values,
        "coefficient_labels" => coefficient_labels,
        "param_mapping" => param_mapping,
        "parities" => parities,
        "metrics" => metrics,
        "labels" => Dict(
            "starting state" => Dict("U index" => u_idx1),
            "ending state" => Dict("U index" => u_idx2)
        )
    )

    if !isnothing(save_folder)
        mkpath(save_folder)
        JLD2.jldsave(joinpath(save_folder, "$(save_name).jld2"); dict=data_dict)
    end

    return data_dict
end


# -----------------------------------------------------------------------------
# Optimized Adjoint Gradient Logic
# -----------------------------------------------------------------------------

"""
    adjoint_loss(t_vals, ops, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, v1, v2, p)

Computes the loss L = 1 - |<v1 | exp(im * (A(t) + p)) | v2>|^2 efficiently using expv
Custom rrule provides efficient gradients w.r.t t_vals.
"""
function adjoint_loss(t_vals, ops, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, v1, v2, p, antihermitian=false)
    # 1. Construct A(t) efficiently
    vals = update_values(signs, param_index_map, t_vals, parameter_mapping, parity)
    A = sparse(rows, cols, vals, dim, dim)
    if antihermitian
        A = make_antihermitian(A)
    else
        A = make_hermitian(A)
    end

    # 2. Add offset p if present
    if !isnothing(p) && !(p isa SciMLBase.NullParameters)
        B = A + p
    else
        B = A
    end

    if antihermitian
        psi = expv(1.0, B, v2)
    else
        psi = expv(1.0im, B, v2)
    end

    # 4. Overlap
    # ov = <v1 | psi>
    overlap = dot(v1, psi)
    loss = 1 - abs2(overlap)
    # Zygote.@ignore println("loss=$loss coeff-mag=$(sum(abs.(t_vals)))")
    return loss
end

function ChainRulesCore.rrule(::typeof(adjoint_loss), t_vals, ops, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, v1, v2, p, antihermitian)
    # --- Reconstruct A ---
    vals = update_values(signs, param_index_map, t_vals, parameter_mapping, parity)
    A = sparse(rows, cols, vals, dim, dim)
    if antihermitian
        A = make_antihermitian(A)
    else
        A = make_hermitian(A)
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


