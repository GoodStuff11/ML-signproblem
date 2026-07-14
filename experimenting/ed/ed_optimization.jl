"""
Apply exp(α M) * v efficiently.
Uses expv if M is large & sparse, otherwise falls back to exp(M).
"""
function apply_exp(M, v, α)
    n = size(M, 1)
    if issparse(M) && n > 128
        return expv(α, M, v)
    else
        dense_M = issparse(M) ? Matrix(M) : M
        return exp(α * dense_M) * v
    end
end

function approximate_trotter_grad_loss(grad, t_vals, ops, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state1, state2, use_symmetry, antihermitian; p=nothing)
    # Type assertions for globals to ensure performance (though arguments are not globals here, keeping structure)
    local ops_data_typed = ops
    local v1_typed::Vector{Complex{Float64}} = state1
    local v2_typed::Vector{Complex{Float64}} = state2
    local N_typed::Int = 100 # determines the accuracy of the method
    local Hs_dim_typed::Int = dim
    local DIM_typed::Int = length(t_vals)

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
            I, J, V = ops_data_typed[j]
            m = sparse(I, J, V, Hs_dim_typed, Hs_dim_typed)
            grads[j] += dot(l, m * r_current)
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

function trotter(ops, dim, v1, v2, trotter_order)
    ## todo: wait for tamra to give better trotter implementation
    for (i, t) in enumerate(ops)
        I, J, V = ops[i]
        M = sparse(I, J, V, dim, dim)

    end


end

function fast_loss(t_vals, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state1, state2, use_symmetry, antihermitian; p=nothing, num_exponentials::Int=1)
    t = @elapsed begin
        L = num_exponentials
        P = div(length(t_vals), L)
        psi = state1
        for l in 1:L
            t_l = t_vals[((l-1)*P+1):(l*P)]
            vals = update_values(signs, param_index_map, t_l, parameter_mapping, parity)
            mat = sparse(rows, cols, vals, dim, dim)
            if !use_symmetry
                if antihermitian
                    mat = make_antihermitian(mat)
                else
                    mat = make_hermitian(mat)
                end
            end
            if l == 1 && p isa AbstractMatrix
                mat += p
            end
            if antihermitian
                psi = expv(1.0, mat, psi)
            else
                psi = expv(1.0im, mat, psi)
            end
        end
        loss = 1 - abs2(state2' * psi)
    end
    println("time=$t loss=$loss")
    return loss
end

function zygote_loss(t_vals, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state1, state2, use_symmetry, antihermitian; p=nothing, num_exponentials::Int=1)
    L = num_exponentials
    P = div(length(t_vals), L)
    psi = state1
    for l in 1:L
        t_l = t_vals[((l-1)*P+1):(l*P)]
        vals = update_values(signs, param_index_map, t_l, parameter_mapping, parity)
        mat = sparse(rows, cols, vals, dim, dim)
        if !use_symmetry
            if antihermitian
                mat = make_antihermitian(mat)
            else
                mat = make_hermitian(mat)
            end
        end
        if l == 1 && p isa AbstractMatrix
            mat += p
        end
        if antihermitian
            psi = exp(Matrix(mat)) * psi
        else
            psi = exp(1im * Matrix(mat)) * psi
        end
    end
    loss = 1 - abs2(state2' * psi)
    Zygote.@ignore println("loss=$loss")
    return loss
end

function fast_energy_loss(t_vals, H, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state1, use_symmetry, antihermitian; p=nothing)
    error("fast_energy_loss is not implemented.")
end

function zygote_energy_loss(t_vals, H, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state1, use_symmetry, antihermitian; p=nothing)
    error("zygote_energy_loss is not implemented.")
end

function format_bytes_custom(bytes)
    gb = bytes / 1024^3
    if gb >= 1.0
        return string(round(gb, digits=3), " GB")
    else
        mb = bytes / 1024^2
        return string(round(mb, digits=3), " MB")
    end
end

function print_mem_usage(msg::String=""; verbose=false)
    if !verbose
        return
    end
    gc_live = try
        Base.gc_live_bytes()
    catch
        0
    end
    rss = Sys.maxrss()
    mem_str = "GC Live: " * format_bytes_custom(gc_live) * " | MaxRSS: " * format_bytes_custom(rss)
    try
        if isdefined(Main, :CUDA) && CUDA.has_cuda_gpu()
            gpu_mem = CUDA.alloc_bytes()
            mem_str *= " | GPU: " * format_bytes_custom(gpu_mem)
        end
    catch
    end
    println("  [Memory] $msg -> $mem_str")
end

function ensure_operator_structure!(order::Int, operator_cache::Dict, indexer::CombinationIndexer,
    spin_conserved::Bool, use_symmetry::Bool, momentum_basis::Bool, sign_convention::Symbol,
    precomputed_structures::Dict, antihermitian::Bool, init_mag::Number)

    if haskey(operator_cache, order)
        return operator_cache[order]
    end

    # compute operator structure, initial coefficients and operators
    if haskey(precomputed_structures, (order, use_symmetry))
        println("Loading precomputed structure for order $order, use_symmetry=$use_symmetry")
        struct_cache = precomputed_structures[(order, use_symmetry)]
        rows, cols, signs, ops_list = struct_cache[:rows], struct_cache[:cols], struct_cache[:signs], struct_cache[:ops_list]
        t_keys, param_index_map = struct_cache[:t_keys], struct_cache[:param_index_map]
    else
        print_mem_usage("Before create_randomized_nth_order_operator")
        t_dict, t_keys = create_randomized_nth_order_operator(order, indexer, true; magnitude=init_mag, omit_H_conj=!use_symmetry, conserve_spin=spin_conserved, normalize_coefficients=false, conserve_momentum=momentum_basis)
        print_mem_usage("After create_randomized_nth_order_operator")
        rows, cols, signs, ops_list = build_n_body_structure_from_keys(t_keys, indexer, typeof(t_dict[t_keys[1]]); sign_convention=sign_convention)
        print_mem_usage("After build_n_body_structure_from_keys")
        param_index_map = build_param_index_map(ops_list, t_keys)
    end

    # create matrix operators to make gradient computation faster
    ops = []
    sym_data = nothing

    # Pre-group indices using param_index_map to avoid calling build_n_body_structure O(N) times
    indices_by_param = [Int[] for _ in 1:length(t_keys)]
    for k in eachindex(param_index_map)
        push!(indices_by_param[param_index_map[k]], k)
    end

    if use_symmetry
        inv_param_map, parameter_mapping, parity = find_symmetry_groups(t_keys, maximum(indexer.a).coordinates...,
            hermitian=!antihermitian, antihermitian=antihermitian, trans_x=true, trans_y=true, spin_symmetry=true) # have caution with this being true for N↑≠N↓

        for (param_idx, key_idcs) in enumerate(inv_param_map)
            idx = Int[]
            for key_idx in key_idcs
                append!(idx, indices_by_param[key_idx])
            end

            rows_sub = [rows[j] for j in idx]
            cols_sub = [cols[j] for j in idx]
            vals_sub = [signs[idx[k]] * parity[param_index_map[idx[k]]] for k in eachindex(idx)]
            push!(ops, (rows_sub, cols_sub, vals_sub))
        end
        sym_data = (inv_param_map, parameter_mapping, parity)
    else
        for i in 1:length(t_keys)
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
            push!(ops, (rows_sub, cols_sub, vals_sub))
        end
        parameter_mapping = nothing
        parity = nothing
    end

    cache_entry = Dict(
        :rows => rows,
        :cols => cols,
        :signs => signs,
        :ops_list => ops_list,
        :t_keys => t_keys,
        :param_index_map => param_index_map,
        :sym_data => sym_data,
        :ops => ops,
        :parameter_mapping => parameter_mapping,
        :parity => parity
    )
    operator_cache[order] = cache_entry
    print_mem_usage("After ensure_operator_structure! complete")
    return cache_entry
end

function setup_gpu_resources(use_gpu, state1, state2, H, loss_type, ops)
    use_gpu_flag = false
    ops_gpu = nothing
    state1_gpu = nothing
    state2_gpu = nothing
    H_gpu = nothing

    if (use_gpu === nothing || use_gpu == true) && @isdefined(CUDA) && CUDA.has_cuda_gpu()
        try
            test_gpu = CUDA.CuArray([1.0])
            use_gpu_flag = true
        catch e
            @warn "CUDA initialization failed, falling back to CPU." exception = e
        end
    end

    if use_gpu_flag
        try
            println("Using GPU")
            ops_gpu = ops
            state1_gpu = CUDA.CuArray(state1)
            state2_gpu = CUDA.CuArray(state2)
            if loss_type == :energy
                H_gpu = CUDA.CUSPARSE.CuSparseMatrixCSC(H)
            end
        catch e
            @warn "Failed to allocate arrays on GPU, falling back to CPU." exception = e
            use_gpu_flag = false
        end
    end
    if !use_gpu_flag
        println("Not using GPU")
    end
    return use_gpu_flag, ops_gpu, state1_gpu, state2_gpu, H_gpu
end

function setup_loss_functions(loss_type, order, ops, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state1, state2, H, use_symmetry, antihermitian, use_gpu_flag, ops_gpu, state1_gpu, state2_gpu, H_gpu, num_exponentials)
    fg! = nothing
    f_nongradient = nothing
    f = nothing
    f_adjoint = nothing
    f_adjoint_gpu = nothing

    if loss_type == :energy
        fg! = (grad, t_vals, p=nothing) -> error("manualgradient is not implemented for energy loss.")
        f_nongradient = (t_vals, p=nothing) -> error("fast_energy_loss is not implemented.")
        f = (t_vals, p=nothing) -> error("zygote_energy_loss is not implemented.")
        f_adjoint = (t_vals, p=nothing) -> adjoint_energy_loss(t_vals, ops, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state1, H, p, !use_symmetry, antihermitian; num_exponentials=num_exponentials)
        f_adjoint_gpu = (t_vals, p=nothing) -> begin
            p_gpu = p === nothing ? nothing : CUDA.CUSPARSE.CuSparseMatrixCSC(p)
            gpu_adjoint_energy_loss(t_vals, ops_gpu, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state1_gpu, H_gpu, p_gpu, !use_symmetry, antihermitian; num_exponentials=num_exponentials)
        end
    else
        fg! = (grad, t_vals, p=nothing) -> approximate_trotter_grad_loss(grad, t_vals, ops, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state1, state2, use_symmetry, antihermitian, p=p)
        f_nongradient = (t_vals, p=nothing) -> fast_loss(t_vals, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state1, state2, use_symmetry, antihermitian; p=p, num_exponentials=num_exponentials)
        f = (t_vals, p=nothing) -> zygote_loss(t_vals, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state1, state2, use_symmetry, antihermitian; p=p, num_exponentials=num_exponentials)
        f_adjoint = (t_vals, p=nothing) -> adjoint_loss(t_vals, ops, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state2, state1, p, !use_symmetry, antihermitian; num_exponentials=num_exponentials)
        f_adjoint_gpu = (t_vals, p=nothing) -> begin
            p_gpu = p === nothing ? nothing : CUDA.CUSPARSE.CuSparseMatrixCSC(p)
            gpu_adjoint_loss(t_vals, ops_gpu, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state2_gpu, state1_gpu, p_gpu, !use_symmetry, antihermitian; num_exponentials=num_exponentials)
        end
    end
    return fg!, f_nongradient, f, f_adjoint, f_adjoint_gpu
end

function setup_optimization_function(gradient, use_gpu_flag, f, fg!, f_nongradient, f_adjoint, f_adjoint_gpu)
    if gradient == :gradient
        optf = Optimization.OptimizationFunction(f, Optimization.AutoZygote())
    elseif gradient == :manualgradient
        optf = Optimization.OptimizationFunction(f_nongradient, grad=fg!)
    elseif gradient == :adjoint_gradient && use_gpu_flag
        optf = Optimization.OptimizationFunction(f_adjoint_gpu, Optimization.AutoZygote())
    elseif gradient == :adjoint_gradient
        optf = Optimization.OptimizationFunction(f_adjoint, Optimization.AutoZygote())
    else
        optf = Optimization.OptimizationFunction(f_nongradient)
    end
    return optf
end

function execute_single_optimization(optf, current_t_vals, current_maxiters, optimizers, order, get_p_args, time_tracker, max_time_ratio, perturb_optimization, initial_loss; loss_history=Float64[])
    local_t_vals = copy(current_t_vals)
    local_loss = initial_loss
    local_sol = nothing

    tmp_losses = Float64[]
    function callback(state, loss_val)
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

    for (optimizer_idx, optimizer_sym) in enumerate(optimizers)
        # 1. Setup Phase
        if optimizer_idx > 1 && perturb_optimization > 1e-9
            used_perturb_optimization = perturb_optimization^(1 + (optimizer_idx - 1) / 3)
            local_t_vals = local_t_vals * (1 - used_perturb_optimization) + used_perturb_optimization * mean(abs.(local_t_vals)) * (2 * rand(length(local_t_vals)) .- 1)
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

        # Async peak memory monitor
        monitor_done = Threads.Atomic{Bool}(false)
        monitor_task = if isdefined(Main, :CUDA) || @isdefined(CUDA)
            nothing
        else
            Threads.@spawn begin
                last_rss = Sys.maxrss()
                while !monitor_done[]
                    current_rss = Sys.maxrss()
                    if current_rss > last_rss + 50 * 1024^2 # Warn if RSS grows by > 50MB
                        println("  [Async Memory Warning] MaxRSS increased to: ", format_bytes_custom(current_rss))
                        last_rss = current_rss
                    end
                    sleep(0.01)
                end
            end
        end

        last_time = time()
        function timed_callback(state, loss_val)
            current_time = time()
            dt = current_time - last_time
            last_time = current_time

            if !haskey(time_tracker, optimizer_sym)
                time_tracker[optimizer_sym] = Float64[]
            end

            if !isnothing(max_time_ratio) && length(time_tracker[optimizer_sym]) > 5
                avg_time = mean(time_tracker[optimizer_sym])
                if dt > max_time_ratio * avg_time
                    println("Stopping optimization: Step took $(dt)s, which is > $(max_time_ratio)x the average $(avg_time)s")
                    return true
                end
            end

            push!(time_tracker[optimizer_sym], dt)
            push!(loss_history, loss_val)
            return callback(state, loss_val)
        end

        print_mem_usage("Before Optimization.solve with $optimizer_sym")
        @time local_sol = Optimization.solve(prob, opt_algo, maxiters=current_maxiters, callback=timed_callback)
        print_mem_usage("After Optimization.solve with $optimizer_sym")

        monitor_done[] = true
        if !isnothing(monitor_task)
            try
                wait(monitor_task)
            catch
            end
        end

        # 3. Post-Process Phase
        local_t_vals = local_sol.u
        local_loss = local_sol.objective
    end

    return local_sol, local_t_vals, local_loss
end

function run_multistart_initialization(
    initialization_samples::Int, multi_start_samples::Int, multi_start_iters::Int, maxiters::Int,
    order::Int, get_p_args, signs, sym_data, t_keys, use_symmetry, loss_type, initial_loss, magnitude_estimate,
    gradient, use_gpu_flag, f_adjoint, f_adjoint_gpu, optf, optimizers, time_tracker, max_time_ratio, perturb_optimization,
    num_exponentials::Int
)
    println("Sampling $initialization_samples initial configurations over a range of magnitudes...")
    p_args = get_p_args(order)

    good_samples = []
    log_min = log10(1e-7)
    log_max = log10(1e-1)

    for s in 1:initialization_samples
        mag = 10^(log_min + (log_max - log_min) * rand())
        P = use_symmetry ? length(sym_data[1]) : length(t_keys)

        if use_symmetry
            t_sample = real(rand(typeof(signs[1]), P * num_exponentials) * mag)
        else
            t_sample = (2 * rand(P * num_exponentials) .- 1) * mag
        end

        grad_func = (gradient == :adjoint_gradient && use_gpu_flag) ? f_adjoint_gpu : f_adjoint

        res = Zygote.withgradient(t -> grad_func(t, p_args), t_sample)
        l_tmp = res.val
        g_tmp = res.grad[1]
        gnorm = norm(g_tmp)

        is_good = (gnorm > 1e-8) && (loss_type == :energy ? (l_tmp < initial_loss + abs(initial_loss) / 5) : (l_tmp < initial_loss * 10))

        if is_good
            push!(good_samples, (gnorm, l_tmp, copy(t_sample)))
        end

        if initialization_samples <= 50
            println("Sample $s (mag=$(round(mag, sigdigits=3))): loss=$l_tmp grad_norm=$gnorm (GOOD: $is_good)")
        end
    end

    sort!(good_samples, by=x -> x[1], rev=true)
    top_n = min(multi_start_samples, length(good_samples))

    local_multistart_losses = Vector{Float64}[]
    local_best_start_idx = 0
    multistart_run = false
    t_vals = nothing

    if top_n == 0
        println("No good samples found, falling back to random initialization")
        P = use_symmetry ? length(sym_data[1]) : length(t_keys)
        if use_symmetry
            t_vals = real(rand(typeof(signs[1]), P * num_exponentials) * magnitude_estimate)
        else
            t_vals = (2 * rand(P * num_exponentials) .- 1) * magnitude_estimate
        end
    else
        best_t = nothing
        best_loss = Inf
        println("Performing quick optimization on top $top_n candidates...")
        quick_maxiters = min(multi_start_iters, maxiters)
        multistart_run = true
        for i in 1:top_n
            candidate_t = good_samples[i][3]
            candidate_history = Float64[]
            try
                _, opt_t, opt_loss = execute_single_optimization(optf, candidate_t, quick_maxiters, [:GradientDescent, :LBFGS], order, get_p_args, time_tracker, max_time_ratio, perturb_optimization, initial_loss; loss_history=candidate_history)
                println("Candidate $i quick opt loss: $opt_loss")
                push!(local_multistart_losses, candidate_history)
                if opt_loss < best_loss
                    best_loss = opt_loss
                    best_t = opt_t
                    local_best_start_idx = i
                end
            catch e
                if e isa ArgumentError && occursin("matrix contains Infs or NaNs", e.msg)
                    @warn "Candidate $i threw ArgumentError (matrix contains Infs or NaNs). Proceeding to next candidate." exception = e
                    continue
                else
                    rethrow(e)
                end
            end
        end
        if isnothing(best_t)
            println("All candidates failed quick optimization with Infs/NaNs, falling back to random initialization")
            P = use_symmetry ? length(sym_data[1]) : length(t_keys)
            if use_symmetry
                t_vals = real(rand(typeof(signs[1]), P * num_exponentials) * magnitude_estimate)
            else
                t_vals = (2 * rand(P * num_exponentials) .- 1) * magnitude_estimate
            end
        else
            t_vals = best_t
            println("Selected best candidate with loss=$best_loss")
        end
    end
    return t_vals, local_multistart_losses, local_best_start_idx, multistart_run
end

function optimize_unitary(state1::Vector, state2::Vector, indexer::CombinationIndexer;
    maxiters=10, ϵ=1e-5, optimization_scheme::Vector=[2], spin_conserved::Bool=false, use_symmetry::Bool=false,
    gradient=:adjoint_gradient, metric_functions::Dict{String,Function}=Dict{String,Function}(),
    antihermitian::Bool=false, optimizer::Union{Symbol,Vector{Symbol}}=:LBFGS, perturb_optimization::Float64=0.001,
    initial_coefficients::Vector{Any}=Any[], initialization_samples::Int=40,
    operator_cache::Dict{Int,Dict{Symbol,Any}}=Dict{Int,Dict{Symbol,Any}}(),
    momentum_basis::Bool=false, multi_start_samples::Int=5, multi_start_iters::Int=30,
    precomputed_structures::Dict=Dict(),
    sign_convention::Symbol=:spin_first,
    time_tracker::Dict{Symbol,Vector{Float64}}=Dict{Symbol,Vector{Float64}}(),
    max_time_ratio::Union{Float64,Nothing}=nothing,
    nn_strategy_file::Union{AbstractString,Nothing}=nothing,
    nn_ctx_u::Union{Float64,Nothing}=nothing,
    nn_electrons::Union{Tuple{Int,Int},Nothing}=nothing,
    nn_dim::Union{Vector{Int},Nothing}=nothing,
    loss_type::Symbol=:overlap,
    H::Union{AbstractMatrix,Nothing}=nothing,
    use_gpu::Union{Bool,Nothing}=nothing,
    num_exponentials::Int=1
)

    if loss_type == :energy && isnothing(H)
        error("Hamiltonian H must be provided when loss_type is :energy")
    end

    if momentum_basis
        use_symmetry = false # Disable spatial symmetries when working directly in momentum space
    end

    if !isnothing(nn_strategy_file) && !isfile(nn_strategy_file)
        resolved = joinpath("trained_neural_networks", "trained_neural_network_$(nn_strategy_file).jld2")
        if isfile(resolved)
            nn_strategy_file = resolved
        else
            error("Neural network strategy file not found: '$nn_strategy_file' (also tried '$resolved')")
        end
    end
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
        mats = []
        for (i, m) in enumerate(computed_matrices)
            if i != current_order && !isnothing(m)
                if m isa Vector
                    append!(mats, m)
                else
                    push!(mats, m)
                end
            end
        end
        return isempty(mats) ? nothing : sum(mats)
    end

    dim = length(indexer.inv_comb_dict)
    metrics = Dict{String,Vector{Any}}()
    loss = loss_type == :energy ? real(dot(state1, H * state1)) : (1 - abs2(state1' * state2))
    prev_loss = loss
    metrics["loss"] = Float64[loss]
    metrics["other"] = []
    metrics["loss_std"] = Float64[0.0]
    metrics["optimization_losses"] = Vector{Float64}[]
    metrics["multistart_losses"] = Vector{Vector{Float64}}[]
    metrics["best_start_idx"] = Int[]
    if loss_type == :overlap
        metrics["energy"] = Float64[!isnothing(H) ? real(dot(state1, H * state1)) : NaN]
    elseif loss_type == :energy
        metrics["overlap"] = Float64[1.0-abs2(dot(state1, state2))]
    end
    for k in keys(metric_functions)
        metrics[k] = Any[]
    end

    println("Initial loss: $loss")
    println("Dimension: $dim")
    if loss_type == :overlap && loss < 1e-15
        println("States are already equal")
        return computed_matrices, coefficient_labels, computed_coefficients, parameter_mappings, parities, metrics, operator_cache
    end

    # 1. Pre-population: realize matrices for any coefficients provided externally
    if !isempty(initial_coefficients)
        println("COMPUTED MATRICES")
        for order_idx in eachindex(initial_coefficients)
            coeffs = initial_coefficients[order_idx]
            if isnothing(coeffs) || isempty(coeffs)
                continue
            end

            struct_data = ensure_operator_structure!(order_idx, operator_cache, indexer, spin_conserved, use_symmetry, momentum_basis, sign_convention, precomputed_structures, antihermitian, (loss_type == :energy ? 0.01 + 0im : loss * 100))

            # Assign labels and mappings
            coefficient_labels[order_idx] = struct_data[:t_keys]
            parameter_mappings[order_idx] = struct_data[:parameter_mapping]
            parities[order_idx] = struct_data[:parity]
            computed_coefficients[order_idx] = coeffs

            # Compute the matrices
            P = use_symmetry ? length(struct_data[:sym_data][1]) : length(struct_data[:t_keys])
            L_order = div(length(coeffs), P)
            mats_order = Vector{Any}(undef, L_order)
            for l in 1:L_order
                t_l = coeffs[((l-1)*P+1):(l*P)]
                vals = update_values(struct_data[:signs], struct_data[:param_index_map], t_l, struct_data[:parameter_mapping], struct_data[:parity])
                mat_l = sparse(struct_data[:rows], struct_data[:cols], vals, dim, dim)
                if !use_symmetry
                    if antihermitian
                        mat_l = make_antihermitian(mat_l)
                    else
                        mat_l = make_hermitian(mat_l)
                    end
                end
                mats_order[l] = mat_l
            end
            computed_matrices[order_idx] = mats_order
        end
    end

    # 2. Main Optimization Scheme Loop
    for order ∈ optimization_scheme
        struct_data = ensure_operator_structure!(order, operator_cache, indexer, spin_conserved, use_symmetry, momentum_basis, sign_convention, precomputed_structures, antihermitian, (loss_type == :energy ? 0.01 + 0im : loss * 100))
        multistart_run = false
        local_multistart_losses = Vector{Float64}[]
        local_best_start_idx = 0

        # Extract variables for convenience from struct_data
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

        # GPU setup
        use_gpu_flag, ops_gpu, state1_gpu, state2_gpu, H_gpu = setup_gpu_resources(use_gpu, state1, state2, H, loss_type, ops)

        # Setup Loss and Gradient functions
        fg!, f_nongradient, f, f_adjoint, f_adjoint_gpu = setup_loss_functions(loss_type, order, ops, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state1, state2, H, use_symmetry, antihermitian, use_gpu_flag, ops_gpu, state1_gpu, state2_gpu, H_gpu, num_exponentials)

        # Optimization Function Setup
        optf = setup_optimization_function(gradient, use_gpu_flag, f, fg!, f_nongradient, f_adjoint, f_adjoint_gpu)

        optimizers = optimizer isa Vector ? optimizer : [optimizer]

        # Determine Initial Coefficients (t_vals) for this optimization order
        magnitude_estimate = loss_type == :energy ? 0.01 : (loss * 100)
        P = use_symmetry ? length(sym_data[1]) : length(t_keys)

        # find values for t_vals
        if length(initial_coefficients) >= order && !isnothing(initial_coefficients[order])
            t_vals = initial_coefficients[order]
        elseif !isnothing(nn_strategy_file) && isfile(nn_strategy_file) && order == 2
            println("Using neural network from $nn_strategy_file to initialize coefficients...")
            strategy = load_neural_network(nn_strategy_file)
            u_val = isnothing(nn_ctx_u) ? 0.001 : nn_ctx_u
            el_count = isnothing(nn_electrons) ? (2, 2) : nn_electrons
            target_dim = isnothing(nn_dim) ? [2, 2] : nn_dim

            ctx = NeuralNetContext(u_val, el_count, strategy.U_max)
            t_vals = interpolate_coefficients(strategy, ctx, t_keys, target_dim)

            if use_symmetry
                t_vals_sym = zeros(Float64, length(sym_data[1]))
                for (param_idx, key_idcs) in enumerate(sym_data[1])
                    vals_in_group = [t_vals[k] * parity[param_index_map[k]] for k in key_idcs]
                    t_vals_sym[param_idx] = mean(vals_in_group)
                end
                t_vals = t_vals_sym
            end

            if num_exponentials > 1
                t_vals_all = zeros(Float64, length(t_vals) * num_exponentials)
                t_vals_all[1:length(t_vals)] .= t_vals
                t_vals = t_vals_all
            end
        elseif initialization_samples > 0
            t_vals, local_multistart_losses, local_best_start_idx, multistart_run = run_multistart_initialization(
                initialization_samples, multi_start_samples, multi_start_iters, maxiters,
                order, get_p_args, signs, sym_data, t_keys, use_symmetry, loss_type, loss, magnitude_estimate,
                gradient, use_gpu_flag, f_adjoint, f_adjoint_gpu, optf, optimizers, time_tracker, max_time_ratio, perturb_optimization,
                num_exponentials
            )
        else
            if use_symmetry
                t_vals = real(rand(typeof(signs[1]), P * num_exponentials) * magnitude_estimate)
            else
                t_vals = (2 * rand(P * num_exponentials) .- 1) * magnitude_estimate
            end
        end

        println("Parameter count: $(length(t_vals))")

        final_history = Float64[]
        sol, t_vals, loss = execute_single_optimization(optf, t_vals, maxiters, optimizers, order, get_p_args, time_tracker, max_time_ratio, perturb_optimization, loss; loss_history=final_history)
        coefficients = t_vals

        # Construct and Store Matrices
        print_mem_usage("Before final sparse matrix construction for order $order")
        mats_order = Vector{Any}(undef, num_exponentials)
        for l in 1:num_exponentials
            t_l = coefficients[((l-1)*P+1):(l*P)]
            vals = update_values(signs, param_index_map, t_l, parameter_mapping, parity)
            mat_l = sparse(rows, cols, vals, dim, dim)
            if !use_symmetry
                if antihermitian
                    mat_l = make_antihermitian(mat_l)
                else
                    mat_l = make_hermitian(mat_l)
                end
            end
            mats_order[l] = mat_l
        end
        computed_matrices[order] = mats_order
        print_mem_usage("After final sparse matrix construction for order $order")

        println("Finished order $order")
        push!(metrics["loss"], loss)
        push!(metrics["optimization_losses"], final_history)
        if multistart_run
            push!(metrics["multistart_losses"], local_multistart_losses)
            push!(metrics["best_start_idx"], local_best_start_idx)
        else
            push!(metrics["multistart_losses"], Vector{Float64}[])
            push!(metrics["best_start_idx"], 0)
        end
        computed_coefficients[order] = coefficients
        parameter_mappings[order] = parameter_mapping
        parities[order] = parity

        # Flatten matrices for final metric checks
        clean_matrices = []
        for mats in computed_matrices
            if !isnothing(mats)
                if mats isa Vector
                    append!(clean_matrices, mats)
                else
                    push!(clean_matrices, mats)
                end
            end
        end

        psi_metric = state1
        for M in clean_matrices
            if antihermitian
                psi_metric = apply_exp(M, psi_metric, 1.0)
            else
                psi_metric = apply_exp(M, psi_metric, 1.0im)
            end
        end

        if loss_type == :overlap
            final_energy = !isnothing(H) ? real(dot(psi_metric, H * psi_metric)) : NaN
            push!(metrics["energy"], final_energy)
        elseif loss_type == :energy
            final_overlap = 1.0 - abs2(dot(psi_metric, state2))
            push!(metrics["overlap"], final_overlap)
        end

        for (k, func) in metric_functions
            push!(metrics[k], func(state1, state2, clean_matrices, final_history))
        end
    end
    # display(plot(loss_tracker, yscale=:log10))
    # println("hey")
    return computed_matrices, coefficient_labels, computed_coefficients, parameter_mappings, parities, metrics, operator_cache
end


function test_map_to_state(degen_rm_U::Union{AbstractMatrix,Vector}, instructions::Dict{String,Any}, indexer::CombinationIndexer,
    spin_conserved::Bool=false;
    maxiters=100, gradient::Symbol=:gradient, metric_functions::Dict{String,Function}=Dict{String,Function}(), optimizer::Union{Symbol,Vector{Symbol}}=:LBFGS,
    initial_coefficients::Vector{Any}=Any[], perturb_optimization::Float64=0.2, precomputed_structures::Dict=Dict(),
    time_tracker::Dict{Symbol,Vector{Float64}}=Dict{Symbol,Vector{Float64}}(),
    max_time_ratio::Union{Float64,Nothing}=nothing,
    U_values::Union{Vector{Float64},Nothing}=nothing
)
    # spin_conserved is only true when using (N↑, N↓) and not N.

    data_dict = Dict{String,Any}("norm1_metrics" => [], "norm2_metrics" => [],
        "loss_metrics" => [], "labels" => [], "loss_std_metrics" => [], "all_matrices" => [],
        "coefficients" => [], "coefficient_labels" => nothing, "param_mapping" => nothing, "parities" => nothing)

    finish_early = false
    num_exponentials = get(instructions, "num_exponentials", 1)
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
                maxiters=maxiters, optimization_scheme=get!(instructions, "optimization_scheme", [2, 1]), gradient=gradient,
                metric_functions=metric_functions, antihermitian=get!(instructions, "antihermitian", false), optimizer=optimizer,
                initial_coefficients=initial_coefficients, perturb_optimization=perturb_optimization,
                initialization_samples=get!(instructions, "initialization_samples", 50),
                multi_start_iters=get!(instructions, "multi_start_iters", 30), multi_start_samples=get!(instructions, "multi_start_samples", 5),
                precomputed_structures=precomputed_structures, sign_convention=get!(instructions, "sign_convention", :spin_first),
                time_tracker=time_tracker, max_time_ratio=max_time_ratio,
                num_exponentials=num_exponentials)
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
    perturb_optimization::Float64=0.1,
    save_folder::Union{String,Nothing}=nothing, save_name::String="scan_data",
    initial_coefficients::Vector{Any}=Any[], precomputed_structures::Dict=Dict(),
    time_tracker::Dict{Symbol,Vector{Float64}}=Dict{Symbol,Vector{Float64}}(),
    max_time_ratio::Union{Float64,Nothing}=nothing,
    nn_strategy_file::Union{AbstractString,Nothing}=nothing,
    nn_electrons::Union{Tuple{Int,Int},Nothing}=nothing,
    nn_dim::Union{Vector{Int},Nothing}=nothing,
    nn_U_values::Union{Vector{Float64},Nothing}=nothing,
    U_values::Union{Vector{Float64},Nothing}=nothing,
    loss_type::Symbol=:overlap,
    use_gpu::Union{Bool,Nothing}=nothing,
    custom_ref_state::Union{Vector,Nothing}=nothing
)
    # instructions["u_range"] should be a range of indices, e.g., 1:10
    # instructions["starting state"] should define the fixed reference state (state1)

    data_dict = Dict{String,Any}("norm1_metrics" => [], "norm2_metrics" => [],
        "loss_metrics" => [], "labels" => [], "loss_std_metrics" => [], "all_matrices" => [],
        "coefficients" => [], "coefficient_labels" => nothing, "param_mapping" => nothing, "parities" => nothing)

    shared_cache = Dict{Int,Dict{Symbol,Any}}()
    if haskey(instructions, "load_file")
        dic = load(instructions["load_file"])["dict"]
        current_coeffs = dic["coefficients"]
    else
        current_coeffs = initial_coefficients
    end

    u_indices = instructions["u_range"]

    if !isnothing(save_folder)
        mkpath(save_folder)
    end
    shared_data_saved = false

    # Define state1 (fixed reference)
    ref_u_idx = 1
    ref_level = 1

    u_vals = !isnothing(U_values) ? U_values : (haskey(instructions, "U_values") ? instructions["U_values"] : nn_U_values)

    H_hopping, H_interaction = try
        subspace = reconstruct_subspace(indexer, spin_conserved)
        create_hubbard_matrices(subspace; indexer=indexer, get_indexer=false, sign_convention=get(instructions, "sign_convention", :spin_first))
    catch e
        @warn "Failed to reconstruct Hamiltonian: $e"
        nothing, nothing
    end

    num_exponentials = get(instructions, "num_exponentials", 1)

    for u_idx in u_indices
        u_val_str = isnothing(u_vals) ? "" : " (U = $(u_vals[u_idx]))"
        println("\n--- Scanning U index: $u_idx$u_val_str ---")

        state1 = isnothing(custom_ref_state) ? degen_rm_U[ref_u_idx, :] : custom_ref_state
        state2 = degen_rm_U[u_idx, :]

        target_u = isnothing(u_vals) ? nothing : u_vals[u_idx]

        H = if !isnothing(H_hopping) && !isnothing(H_interaction) && !isnothing(target_u)
            H_hopping + target_u * H_interaction
        else
            nothing
        end

        args = optimize_unitary(state1, state2, indexer;
            spin_conserved=spin_conserved, use_symmetry=get!(instructions, "use symmetry", false),
            maxiters=maxiters, optimization_scheme=get!(instructions, "optimization_scheme", [2, 1]), gradient=gradient,
            metric_functions=metric_functions, antihermitian=get!(instructions, "antihermitian", false), optimizer=optimizer,
            initial_coefficients=current_coeffs, perturb_optimization=perturb_optimization,
            initialization_samples=get!(instructions, "initialization_samples", 50),
            multi_start_iters=get!(instructions, "multi_start_iters", 30), multi_start_samples=get!(instructions, "multi_start_samples", 5),
            precomputed_structures=precomputed_structures, sign_convention=get!(instructions, "sign_convention", :spin_first),
            time_tracker=time_tracker, max_time_ratio=max_time_ratio,
            nn_strategy_file=nn_strategy_file,
            nn_ctx_u=target_u,
            nn_electrons=nn_electrons,
            nn_dim=nn_dim,
            loss_type=loss_type,
            H=H,
            use_gpu=use_gpu,
            num_exponentials=num_exponentials)
        computed_matrices, coefficient_labels, current_coeffs, param_mapping, parities, metrics, shared_cache = args

        # Store results for this U
        push!(data_dict["norm1_metrics"], [isnothing(cm) ? 0.0 : norm(cm, 1) for cm in computed_matrices])
        push!(data_dict["norm2_metrics"], [isnothing(cm) ? 0.0 : norm(cm, 2) for cm in computed_matrices])
        push!(data_dict["all_matrices"], computed_matrices)
        push!(data_dict["coefficients"], current_coeffs)
        if isnothing(data_dict["coefficient_labels"])
            data_dict["coefficient_labels"] = coefficient_labels
            data_dict["param_mapping"] = param_mapping
            data_dict["parities"] = parities
        end
        # Save shared data once we have it
        if !isnothing(save_folder) && !shared_data_saved && !isnothing(coefficient_labels)
            println("saved")
            shared_dict = Dict(
                "coefficient_labels" => coefficient_labels,
                "param_mapping" => param_mapping,
                "parities" => parities,
                "instructions" => instructions,
                "u_range" => u_indices
            )
            println(joinpath(save_folder, "$(save_name)_shared.jld2"))
            JLD2.jldsave(joinpath(save_folder, "$(save_name)_shared.jld2"); dict=shared_dict)
            shared_data_saved = true
        end

        # Save iteration data
        if !isnothing(save_folder)# && metrics["loss"][end] < starting_loss
            iter_dict = Dict(
                "u_idx" => u_idx,
                "coefficients" => current_coeffs,
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
            "ending state" => Dict("level" => instructions["starting level"], "U index" => u_idx))
        )
    end

    return data_dict
end

# -----------------------------------------------------------------------------
# Optimized Adjoint Gradient Logic
# -----------------------------------------------------------------------------

using ChainRulesCore
using ExponentialUtilities
using KrylovKit
using Lattices
using LinearAlgebra
using SparseArrays

"""
    adjoint_loss(t_vals, ops, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, v1, v2, p)

Computes the loss L = 1 - |<v1 | exp(im * (A(t) + p)) | v2>|^2 efficiently using expv
Custom rrule provides efficient gradients w.r.t t_vals.
"""
function adjoint_loss(t_vals, ops, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, v1, v2, p, do_hermitian, antihermitian=false; num_exponentials::Int=1)
    L = num_exponentials
    P = div(length(t_vals), L)
    psi = v2
    for l in 1:L
        t_l = t_vals[((l-1)*P+1):(l*P)]
        vals_l = update_values(signs, param_index_map, t_l, parameter_mapping, parity)
        A_l = sparse(rows, cols, vals_l, dim, dim)
        if do_hermitian
            if antihermitian
                A_l = make_antihermitian(A_l)
            else
                A_l = make_hermitian(A_l)
            end
        end
        if l == 1 && !isnothing(p) && !(p isa SciMLBase.NullParameters)
            B_l = A_l + p
        else
            B_l = A_l
        end
        if antihermitian
            psi = expv(1.0, B_l, psi)
        else
            psi = expv(1.0im, B_l, psi)
        end
    end
    overlap = dot(v1, psi)
    return 1 - abs2(overlap)
end

function ChainRulesCore.rrule(::typeof(adjoint_loss), t_vals, ops, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, v1, v2, p, do_hermitian, antihermitian; num_exponentials::Int=1)
    L = num_exponentials
    P = div(length(t_vals), L)

    A_mats = Vector{Any}(undef, L)
    for l in 1:L
        t_l = t_vals[((l-1)*P+1):(l*P)]
        vals_l = update_values(signs, param_index_map, t_l, parameter_mapping, parity)
        A_l = sparse(rows, cols, vals_l, dim, dim)
        if do_hermitian
            if antihermitian
                A_l = make_antihermitian(A_l)
            else
                A_l = make_hermitian(A_l)
            end
        end
        if l == 1 && !isnothing(p) && !(p isa SciMLBase.NullParameters)
            A_l = A_l + p
        end
        A_mats[l] = A_l
    end

    phis_layers = Vector{Vector{ComplexF64}}(undef, L + 1)
    phis_layers[1] = v2
    for l in 1:L
        if antihermitian
            phis_layers[l+1] = expv(1.0, A_mats[l], phis_layers[l])
        else
            phis_layers[l+1] = expv(1.0im, A_mats[l], phis_layers[l])
        end
    end
    overlap = dot(v1, phis_layers[end])
    y = 1 - abs2(overlap)

    function adjoint_loss_pullback(ȳ)
        grad_t = Vector{Float64}(undef, length(t_vals))
        N_steps = 50
        dt = 1.0 / N_steps
        weights = ones(N_steps + 1)
        weights[2:2:end-1] .= 4.0
        weights[3:2:end-2] .= 2.0
        weights[1] = 1.0
        weights[end] = 1.0
        weights .*= (dt / 3.0)

        conj_overlap_factor = conj(overlap) * ȳ

        chis_layers = Vector{Vector{ComplexF64}}(undef, L + 1)
        chis_layers[L+1] = v1
        for l in L:-1:1
            if antihermitian
                chis_layers[l] = expv(-1.0, A_mats[l], chis_layers[l+1])
            else
                chis_layers[l] = expv(-1.0im, A_mats[l], chis_layers[l+1])
            end
        end

        for l in 1:L
            phis = Vector{Vector{ComplexF64}}(undef, N_steps + 1)
            phis[1] = phis_layers[l]
            for k in 1:N_steps
                if antihermitian
                    phis[k+1] = expv(dt, A_mats[l], phis[k])
                else
                    phis[k+1] = expv(dt * 1.0im, A_mats[l], phis[k])
                end
            end

            chis = Vector{Vector{ComplexF64}}(undef, N_steps + 1)
            chis[N_steps+1] = chis_layers[l+1]
            for k in N_steps:-1:1
                if antihermitian
                    chis[k] = expv(-dt, A_mats[l], chis[k+1])
                else
                    chis[k] = expv(-dt * 1.0im, A_mats[l], chis[k+1])
                end
            end

            grad_l = @view grad_t[((l-1)*P+1):(l*P)]
            @safe_threads for i in eachindex(grad_l)
                I, J, V = ops[i]
                M = sparse(I, J, V, dim, dim)
                if do_hermitian
                    if antihermitian
                        M = make_antihermitian(M)
                    else
                        M = make_hermitian(M)
                    end
                end
                val = 0.0 + 0.0im
                for k in 1:(N_steps+1)
                    term = dot(chis[k], M * phis[k])
                    val += term * weights[k]
                end

                if antihermitian
                    dO_dt = val
                else
                    dO_dt = val * 1.0im
                end
                t_val_idx = (l - 1) * P + i
                grad_l[i] = -2 * real(conj_overlap_factor * dO_dt) + 1e-3 * t_vals[t_val_idx]
            end
        end

        return NoTangent(), grad_t, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return y, adjoint_loss_pullback
end

# -----------------------------------------------------------------------------
# Optional GPU Logic
# -----------------------------------------------------------------------------

# Custom kernel for fast, matrix-free gradient accumulation
if @isdefined(CUDA)
    @eval begin
        function accumulate_grad_kernel!(dO_dt_real, dO_dt_imag, chi, phi, flat_rows, flat_cols, flat_vals, flat_params, weight)
            idx = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
            if idx <= length(flat_rows)
                @inbounds r = flat_rows[idx]
                @inbounds c = flat_cols[idx]
                @inbounds v = flat_vals[idx]
                @inbounds p = flat_params[idx]

                term = conj(chi[r]) * v * phi[c] * weight

                CUDA.@atomic dO_dt_real[p] += real(term)
                CUDA.@atomic dO_dt_imag[p] += imag(term)
            end
            return nothing
        end
    end
end

# To avoid errors for CPU-only environments without CUDA loaded, we can assume
# that CUDA is available when these functions are called. 
# You should `using CUDA` in the main script to make sure it is defined.

function gpu_fast_loss(t_vals, ops_gpu, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state1_gpu, state2_gpu, use_symmetry, antihermitian; p_gpu=nothing, num_exponentials::Int=1)
    t = @elapsed begin
        L = num_exponentials
        P = div(length(t_vals), L)
        psi_gpu = state1_gpu
        for l in 1:L
            t_l = t_vals[((l-1)*P+1):(l*P)]
            vals = update_values(signs, param_index_map, t_l, parameter_mapping, parity)
            mat = sparse(rows, cols, vals, dim, dim)
            if !use_symmetry
                if antihermitian
                    mat = make_antihermitian(mat)
                else
                    mat = make_hermitian(mat)
                end
            end
            mat_gpu = CUDA.CUSPARSE.CuSparseMatrixCSC(mat)
            if l == 1 && p_gpu !== nothing && !(p_gpu isa SciMLBase.NullParameters)
                mat_gpu = mat_gpu + p_gpu
            end
            if antihermitian
                psi_gpu, _ = KrylovKit.exponentiate(mat_gpu, 1.0, psi_gpu; ishermitian=false, tol=1e-12)
            else
                psi_gpu, _ = KrylovKit.exponentiate(mat_gpu, 1.0im, psi_gpu; ishermitian=true, tol=1e-12)
            end
        end
        loss = 1 - abs2(dot(state2_gpu, psi_gpu))
    end
    return loss
end

function gpu_adjoint_loss(t_vals, ops_gpu, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, v1_gpu, v2_gpu, p_gpu, do_hermitian, antihermitian=false; num_exponentials::Int=1)
    L = num_exponentials
    P = div(length(t_vals), L)
    psi_gpu = v2_gpu
    for l in 1:L
        t_l = t_vals[((l-1)*P+1):(l*P)]
        vals = update_values(signs, param_index_map, t_l, parameter_mapping, parity)
        A = sparse(rows, cols, vals, dim, dim)
        if do_hermitian
            if antihermitian
                A = make_antihermitian(A)
            else
                A = make_hermitian(A)
            end
        end
        A_gpu = CUDA.CUSPARSE.CuSparseMatrixCSC(A)
        if l == 1 && !isnothing(p_gpu) && !(p_gpu isa SciMLBase.NullParameters)
            B_gpu = A_gpu + p_gpu
        else
            B_gpu = A_gpu
        end
        if antihermitian
            psi_gpu, _ = KrylovKit.exponentiate(B_gpu, 1.0, psi_gpu; ishermitian=false, tol=1e-12)
        else
            psi_gpu, _ = KrylovKit.exponentiate(B_gpu, 1.0im, psi_gpu; ishermitian=true, tol=1e-12)
        end
    end
    overlap = dot(v1_gpu, psi_gpu)
    loss = 1 - abs2(overlap)
    return loss
end

function ChainRulesCore.rrule(::typeof(gpu_adjoint_loss), t_vals, ops_gpu, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, v1_gpu, v2_gpu, p_gpu, do_hermitian, antihermitian; num_exponentials::Int=1)
    L = num_exponentials
    P = div(length(t_vals), L)

    A_mats_gpu = Vector{Any}(undef, L)
    for l in 1:L
        t_l = t_vals[((l-1)*P+1):(l*P)]
        vals_l = update_values(signs, param_index_map, t_l, parameter_mapping, parity)
        A = sparse(rows, cols, vals_l, dim, dim)
        if do_hermitian
            if antihermitian
                A = make_antihermitian(A)
            else
                A = make_hermitian(A)
            end
        end
        A_gpu = CUDA.CUSPARSE.CuSparseMatrixCSC(A)
        if l == 1 && !isnothing(p_gpu) && !(p_gpu isa SciMLBase.NullParameters)
            A_gpu = A_gpu + p_gpu
        end
        A_mats_gpu[l] = A_gpu
    end

    phis_layers = Vector{typeof(v2_gpu)}(undef, L + 1)
    phis_layers[1] = copy(v2_gpu)
    for l in 1:L
        if antihermitian
            phis_layers[l+1], _ = KrylovKit.exponentiate(A_mats_gpu[l], 1.0, phis_layers[l]; ishermitian=false, tol=1e-12)
        else
            phis_layers[l+1], _ = KrylovKit.exponentiate(A_mats_gpu[l], 1.0im, phis_layers[l]; ishermitian=true, tol=1e-12)
        end
        CUDA.synchronize()
    end
    overlap = dot(v1_gpu, phis_layers[end])
    y = 1 - abs2(overlap)

    function gpu_adjoint_loss_pullback(ȳ)
        grad_t = Vector{Float64}(undef, length(t_vals))
        N_steps = 50
        dt = 1.0 / N_steps
        weights = ones(N_steps + 1)
        weights[2:2:end-1] .= 4.0
        weights[3:2:end-2] .= 2.0
        weights[1] = 1.0
        weights[end] = 1.0
        weights .*= (dt / 3.0)

        conj_overlap_factor = conj(overlap) * ȳ

        chis_layers = Vector{typeof(v1_gpu)}(undef, L + 1)
        chis_layers[L+1] = copy(v1_gpu)
        for l in L:-1:1
            if antihermitian
                chis_layers[l], _ = KrylovKit.exponentiate(A_mats_gpu[l], -1.0, chis_layers[l+1]; ishermitian=false, tol=1e-12)
            else
                chis_layers[l], _ = KrylovKit.exponentiate(A_mats_gpu[l], -1.0im, chis_layers[l+1]; ishermitian=true, tol=1e-12)
            end
            CUDA.synchronize()
        end

        tmp = similar(v1_gpu)

        for l in 1:L
            phis = Vector{typeof(v2_gpu)}(undef, N_steps + 1)
            phis[1] = copy(phis_layers[l])
            for k in 1:N_steps
                if antihermitian
                    phis[k+1], _ = KrylovKit.exponentiate(A_mats_gpu[l], dt, phis[k]; ishermitian=false, tol=1e-12)
                else
                    phis[k+1], _ = KrylovKit.exponentiate(A_mats_gpu[l], dt * 1.0im, phis[k]; ishermitian=true, tol=1e-12)
                end
                CUDA.synchronize()
            end

            GC.gc(true)
            CUDA.reclaim()

            chis = Vector{typeof(v1_gpu)}(undef, N_steps + 1)
            chis[N_steps+1] = copy(chis_layers[l+1])
            for k in N_steps:-1:1
                if antihermitian
                    chis[k], _ = KrylovKit.exponentiate(A_mats_gpu[l], -dt, chis[k+1]; ishermitian=false, tol=1e-12)
                else
                    chis[k], _ = KrylovKit.exponentiate(A_mats_gpu[l], -dt * 1.0im, chis[k+1]; ishermitian=true, tol=1e-12)
                end
                CUDA.synchronize()
            end

            grad_l = @view grad_t[((l-1)*P+1):(l*P)]
            for i in eachindex(grad_l)
                I, J, V = ops_gpu[i]
                M_cpu = sparse(I, J, V, dim, dim)
                if do_hermitian
                    if antihermitian
                        M_cpu = make_antihermitian(M_cpu)
                    else
                        M_cpu = make_hermitian(M_cpu)
                    end
                end

                colptr_gpu_i = CUDA.CuArray{Cint}(M_cpu.colptr)
                rowval_gpu_i = CUDA.CuArray(M_cpu.rowval)
                nzval_gpu_i = CUDA.CuArray(M_cpu.nzval)
                M_gpu = CUDA.CUSPARSE.CuSparseMatrixCSC(colptr_gpu_i, rowval_gpu_i, nzval_gpu_i, (dim, dim))

                val = 0.0 + 0.0im
                for k in 1:(N_steps+1)
                    mul!(tmp, M_gpu, phis[k])
                    CUDA.synchronize()
                    term = dot(chis[k], tmp)
                    CUDA.synchronize()
                    val += term * weights[k]
                end

                # Free per-parameter GPU temporaries immediately
                CUDA.unsafe_free!(colptr_gpu_i)
                CUDA.unsafe_free!(rowval_gpu_i)
                CUDA.unsafe_free!(nzval_gpu_i)

                if antihermitian
                    dO_dt = val
                else
                    dO_dt = val * 1.0im
                end
                t_val_idx = (l - 1) * P + i
                grad_l[i] = -2 * real(conj_overlap_factor * dO_dt) + 1e-3 * t_vals[t_val_idx]
            end

            for k in 1:(N_steps+1)
                if isassigned(phis, k)
                    CUDA.unsafe_free!(phis[k])
                end
                if isassigned(chis, k)
                    CUDA.unsafe_free!(chis[k])
                end
            end
        end

        for k in 1:(L+1)
            if isassigned(phis_layers, k)
                CUDA.unsafe_free!(phis_layers[k])
            end
            if isassigned(chis_layers, k)
                CUDA.unsafe_free!(chis_layers[k])
            end
        end
        for l in 1:L
            CUDA.unsafe_free!(A_mats_gpu[l])
        end
        CUDA.unsafe_free!(tmp)

        return NoTangent(), grad_t, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return y, gpu_adjoint_loss_pullback
end



# -----------------------------------------------------------------------------
# Energy Minimization Loss Functions & Gradients (Barren Plateaus Testing)
# -----------------------------------------------------------------------------

function adjoint_energy_loss(t_vals, ops, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, ref, H, p, do_hermitian, antihermitian=false; num_exponentials::Int=1)
    L = num_exponentials
    P = div(length(t_vals), L)
    psi = ref
    for l in 1:L
        t_l = t_vals[((l-1)*P+1):(l*P)]
        vals_l = update_values(signs, param_index_map, t_l, parameter_mapping, parity)
        A_l = sparse(rows, cols, vals_l, dim, dim)
        if do_hermitian
            if antihermitian
                A_l = make_antihermitian(A_l)
            else
                A_l = make_hermitian(A_l)
            end
        end
        if l == 1 && !isnothing(p) && !(p isa SciMLBase.NullParameters)
            B_l = A_l + p
        else
            B_l = A_l
        end
        if antihermitian
            psi = expv(1.0, B_l, psi)
        else
            psi = expv(1.0im, B_l, psi)
        end
    end
    loss = real(dot(psi, H * psi))
    return loss
end

function ChainRulesCore.rrule(::typeof(adjoint_energy_loss), t_vals, ops, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, ref, H, p, do_hermitian, antihermitian; num_exponentials::Int=1)
    L = num_exponentials
    P = div(length(t_vals), L)

    A_mats = Vector{Any}(undef, L)
    for l in 1:L
        t_l = t_vals[((l-1)*P+1):(l*P)]
        vals_l = update_values(signs, param_index_map, t_l, parameter_mapping, parity)
        A_l = sparse(rows, cols, vals_l, dim, dim)
        if do_hermitian
            if antihermitian
                A_l = make_antihermitian(A_l)
            else
                A_l = make_hermitian(A_l)
            end
        end
        if l == 1 && !isnothing(p) && !(p isa SciMLBase.NullParameters)
            A_l = A_l + p
        end
        A_mats[l] = A_l
    end

    phis_layers = Vector{Vector{ComplexF64}}(undef, L + 1)
    phis_layers[1] = ref
    for l in 1:L
        if antihermitian
            phis_layers[l+1] = expv(1.0, A_mats[l], phis_layers[l])
        else
            phis_layers[l+1] = expv(1.0im, A_mats[l], phis_layers[l])
        end
    end
    psi = phis_layers[end]
    y = real(dot(psi, H * psi))

    function adjoint_energy_loss_pullback(ȳ)
        grad_t = Vector{Float64}(undef, length(t_vals))
        N_steps = 50
        dt = 1.0 / N_steps
        weights = ones(N_steps + 1)
        weights[2:2:end-1] .= 4.0
        weights[3:2:end-2] .= 2.0
        weights[1] = 1.0
        weights[end] = 1.0
        weights .*= (dt / 3.0)

        chis_layers = Vector{Vector{ComplexF64}}(undef, L + 1)
        chis_layers[L+1] = H * psi
        for l in L:-1:1
            if antihermitian
                chis_layers[l] = expv(-1.0, A_mats[l], chis_layers[l+1])
            else
                chis_layers[l] = expv(-1.0im, A_mats[l], chis_layers[l+1])
            end
        end

        for l in 1:L
            phis = Vector{Vector{ComplexF64}}(undef, N_steps + 1)
            phis[1] = phis_layers[l]
            for k in 1:N_steps
                if antihermitian
                    phis[k+1] = expv(dt, A_mats[l], phis[k])
                else
                    phis[k+1] = expv(dt * 1.0im, A_mats[l], phis[k])
                end
            end

            chis = Vector{Vector{ComplexF64}}(undef, N_steps + 1)
            chis[N_steps+1] = chis_layers[l+1]
            for k in N_steps:-1:1
                if antihermitian
                    chis[k] = expv(-dt, A_mats[l], chis[k+1])
                else
                    chis[k] = expv(-dt * 1.0im, A_mats[l], chis[k+1])
                end
            end

            grad_l = @view grad_t[((l-1)*P+1):(l*P)]
            @safe_threads for i in eachindex(grad_l)
                I, J, V = ops[i]
                M = sparse(I, J, V, dim, dim)
                if do_hermitian
                    if antihermitian
                        M = make_antihermitian(M)
                    else
                        M = make_hermitian(M)
                    end
                end
                val = 0.0 + 0.0im
                for k in 1:(N_steps+1)
                    term = dot(chis[k], M * phis[k])
                    val += term * weights[k]
                end

                if antihermitian
                    dO_dt = val
                else
                    dO_dt = val * 1.0im
                end
                t_val_idx = (l - 1) * P + i
                grad_l[i] = 2 * real(ȳ * dO_dt) + 1e-3 * t_vals[t_val_idx]
            end
        end

        return NoTangent(), grad_t, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return y, adjoint_energy_loss_pullback
end

function gpu_adjoint_energy_loss(t_vals, ops_gpu, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, ref_gpu, H_gpu, p_gpu, do_hermitian, antihermitian=false; num_exponentials::Int=1)
    L = num_exponentials
    P = div(length(t_vals), L)
    psi_gpu = ref_gpu
    for l in 1:L
        t_l = t_vals[((l-1)*P+1):(l*P)]
        vals_l = update_values(signs, param_index_map, t_l, parameter_mapping, parity)
        A = sparse(rows, cols, vals_l, dim, dim)
        if do_hermitian
            if antihermitian
                A = make_antihermitian(A)
            else
                A = make_hermitian(A)
            end
        end
        A_gpu = CUDA.CUSPARSE.CuSparseMatrixCSC(A)
        if l == 1 && !isnothing(p_gpu) && !(p_gpu isa SciMLBase.NullParameters)
            B_gpu = A_gpu + p_gpu
        else
            B_gpu = A_gpu
        end
        if antihermitian
            psi_gpu, _ = KrylovKit.exponentiate(B_gpu, 1.0, psi_gpu; ishermitian=false, tol=1e-12)
        else
            psi_gpu, _ = KrylovKit.exponentiate(B_gpu, 1.0im, psi_gpu; ishermitian=true, tol=1e-12)
        end
    end
    loss = real(dot(psi_gpu, H_gpu * psi_gpu))
    return loss
end

function ChainRulesCore.rrule(::typeof(gpu_adjoint_energy_loss), t_vals, ops_gpu, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, ref_gpu, H_gpu, p_gpu, do_hermitian, antihermitian; num_exponentials::Int=1)
    L = num_exponentials
    P = div(length(t_vals), L)

    A_mats_gpu = Vector{Any}(undef, L)
    for l in 1:L
        t_l = t_vals[((l-1)*P+1):(l*P)]
        vals_l = update_values(signs, param_index_map, t_l, parameter_mapping, parity)
        A = sparse(rows, cols, vals_l, dim, dim)
        if do_hermitian
            if antihermitian
                A = make_antihermitian(A)
            else
                A = make_hermitian(A)
            end
        end
        A_gpu = CUDA.CUSPARSE.CuSparseMatrixCSC(A)
        if l == 1 && !isnothing(p_gpu) && !(p_gpu isa SciMLBase.NullParameters)
            A_gpu = A_gpu + p_gpu
        end
        A_mats_gpu[l] = A_gpu
    end

    phis_layers = Vector{typeof(ref_gpu)}(undef, L + 1)
    phis_layers[1] = copy(ref_gpu)
    for l in 1:L
        if antihermitian
            phis_layers[l+1], _ = KrylovKit.exponentiate(A_mats_gpu[l], 1.0, phis_layers[l]; ishermitian=false, tol=1e-12)
        else
            phis_layers[l+1], _ = KrylovKit.exponentiate(A_mats_gpu[l], 1.0im, phis_layers[l]; ishermitian=true, tol=1e-12)
        end
        CUDA.synchronize()
    end
    psi_gpu = phis_layers[end]
    y = real(dot(psi_gpu, H_gpu * psi_gpu))

    function gpu_adjoint_energy_loss_pullback(ȳ)
        grad_t = Vector{Float64}(undef, length(t_vals))
        N_steps = 50
        dt = 1.0 / N_steps
        weights = ones(N_steps + 1)
        weights[2:2:end-1] .= 4.0
        weights[3:2:end-2] .= 2.0
        weights[1] = 1.0
        weights[end] = 1.0
        weights .*= (dt / 3.0)

        chis_layers = Vector{typeof(ref_gpu)}(undef, L + 1)
        chis_layers[L+1] = H_gpu * psi_gpu
        for l in L:-1:1
            if antihermitian
                chis_layers[l], _ = KrylovKit.exponentiate(A_mats_gpu[l], -1.0, chis_layers[l+1]; ishermitian=false, tol=1e-12)
            else
                chis_layers[l], _ = KrylovKit.exponentiate(A_mats_gpu[l], -1.0im, chis_layers[l+1]; ishermitian=true, tol=1e-12)
            end
            CUDA.synchronize()
        end

        tmp = similar(ref_gpu)

        for l in 1:L
            phis = Vector{typeof(ref_gpu)}(undef, N_steps + 1)
            phis[1] = copy(phis_layers[l])
            for k in 1:N_steps
                if antihermitian
                    phis[k+1], _ = KrylovKit.exponentiate(A_mats_gpu[l], dt, phis[k]; ishermitian=false, tol=1e-12)
                else
                    phis[k+1], _ = KrylovKit.exponentiate(A_mats_gpu[l], dt * 1.0im, phis[k]; ishermitian=true, tol=1e-12)
                end
                CUDA.synchronize()
            end

            GC.gc(true)
            CUDA.reclaim()

            chis = Vector{typeof(ref_gpu)}(undef, N_steps + 1)
            chis[N_steps+1] = copy(chis_layers[l+1])
            for k in N_steps:-1:1
                if antihermitian
                    chis[k], _ = KrylovKit.exponentiate(A_mats_gpu[l], -dt, chis[k+1]; ishermitian=false, tol=1e-12)
                else
                    chis[k], _ = KrylovKit.exponentiate(A_mats_gpu[l], -dt * 1.0im, chis[k+1]; ishermitian=true, tol=1e-12)
                end
                CUDA.synchronize()
            end

            grad_l = @view grad_t[((l-1)*P+1):(l*P)]
            for i in eachindex(grad_l)
                I, J, V = ops_gpu[i]
                M_cpu = sparse(I, J, V, dim, dim)
                if do_hermitian
                    if antihermitian
                        M_cpu = make_antihermitian(M_cpu)
                    else
                        M_cpu = make_hermitian(M_cpu)
                    end
                end

                colptr_gpu_i = CUDA.CuArray{Cint}(M_cpu.colptr)
                rowval_gpu_i = CUDA.CuArray(M_cpu.rowval)
                nzval_gpu_i = CUDA.CuArray(M_cpu.nzval)
                M_gpu = CUDA.CUSPARSE.CuSparseMatrixCSC(colptr_gpu_i, rowval_gpu_i, nzval_gpu_i, (dim, dim))

                val = 0.0 + 0.0im
                for k in 1:(N_steps+1)
                    mul!(tmp, M_gpu, phis[k])
                    CUDA.synchronize()
                    term = dot(chis[k], tmp)
                    CUDA.synchronize()
                    val += term * weights[k]
                end

                CUDA.unsafe_free!(colptr_gpu_i)
                CUDA.unsafe_free!(rowval_gpu_i)
                CUDA.unsafe_free!(nzval_gpu_i)

                if antihermitian
                    dO_dt = val
                else
                    dO_dt = val * 1.0im
                end
                t_val_idx = (l - 1) * P + i
                grad_l[i] = 2 * real(ȳ * dO_dt) + 1e-3 * t_vals[t_val_idx]
            end

            for k in 1:(N_steps+1)
                if isassigned(phis, k)
                    CUDA.unsafe_free!(phis[k])
                end
                if isassigned(chis, k)
                    CUDA.unsafe_free!(chis[k])
                end
            end
        end

        for k in 1:(L+1)
            if isassigned(phis_layers, k)
                CUDA.unsafe_free!(phis_layers[k])
            end
            if isassigned(chis_layers, k)
                CUDA.unsafe_free!(chis_layers[k])
            end
        end
        for l in 1:L
            CUDA.unsafe_free!(A_mats_gpu[l])
        end
        CUDA.unsafe_free!(tmp)

        return NoTangent(), grad_t, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return y, gpu_adjoint_energy_loss_pullback
end
