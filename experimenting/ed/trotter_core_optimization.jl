#=
trotter_core_optimization.jl

Core optimization routines, multi-start initialization, and convergence diagnostics for Trotter circuits.
=#

"""
    extract_convergence_info(sol) -> Dict{String, Any}

Extract detailed convergence metrics and stopping criteria from an `OptimizationResult` (`sol`).
Returns a dictionary containing:
- `"retcode"`: String representation of the SciML return code.
- `"primary_reason"`: Human-readable explanation of why optimization stopped.
- `"g_converged"`: Bool, whether gradient norm tolerance was met (|g| <= g_tol).
- `"f_converged"`: Bool, whether function value change tolerance was met (|Δf| <= f_tol).
- `"x_converged"`: Bool, whether parameter step tolerance met (|Δx| <= x_tol).
- `"iteration_limit_reached"`: Bool, whether maxiters limit was reached.
- `"f_increased"`: Bool, whether line search failed or objective increased.
- `"g_residual"`: Float64, final gradient norm (or NaN if unavailable).
- `"iterations"`: Int, number of iterations performed.
"""
function extract_convergence_info(sol)
    retcode_str = string(sol.retcode)
    reasons = String[]

    g_conv = false
    f_conv = false
    x_conv = false
    iter_limit = false
    f_inc = false
    g_res = NaN
    iters = 0

    if hasproperty(sol, :original) && sol.original isa Optim.MultivariateOptimizationResults
        orig = sol.original
        g_conv = Optim.g_converged(orig)
        f_conv = Optim.f_converged(orig)
        x_conv = Optim.x_converged(orig)
        iter_limit = Optim.iteration_limit_reached(orig)
        f_inc = Optim.f_increased(orig)
        g_res = Optim.g_residual(orig)
        iters = Optim.iterations(orig)

        if g_conv
            push!(reasons, "Gradient tolerance met (|g| <= g_tol)")
        end
        if f_conv
            push!(reasons, "Function tolerance met (|Δf| <= f_tol)")
        end
        if x_conv
            push!(reasons, "Step size tolerance met (|Δx| <= x_tol)")
        end
        if iter_limit
            push!(reasons, "Maximum iterations reached (maxiters)")
        end
        if f_inc
            push!(reasons, "Objective increased / line search failure")
        end
    end

    if isempty(reasons)
        push!(reasons, "ReturnCode: $retcode_str")
    end

    primary_reason = join(reasons, "; ")

    return Dict{String,Any}(
        "retcode" => retcode_str,
        "primary_reason" => primary_reason,
        "g_converged" => g_conv,
        "f_converged" => f_conv,
        "x_converged" => x_conv,
        "iteration_limit_reached" => iter_limit,
        "f_increased" => f_inc,
        "g_residual" => g_res,
        "iterations" => iters
    )
end

"""
    grow_coefficients(old_coeffs, old_num_exponentials, new_num_exponentials, num_gates) -> Vector{Float64}

Extend a Trotter coefficient vector that was optimized for `old_num_exponentials` layers into
one usable for a larger `new_num_exponentials`, so the newly-added (later) layers can be
optimized starting from an already-converged shorter ansatz instead of from scratch.
"""
function grow_coefficients(old_coeffs::AbstractVector, old_num_exponentials::Int, new_num_exponentials::Int, num_gates::Int)
    if new_num_exponentials < old_num_exponentials
        error("new_num_exponentials ($new_num_exponentials) must be >= old_num_exponentials ($old_num_exponentials)")
    end
    old_len = old_num_exponentials * num_gates
    if length(old_coeffs) != old_len
        error("length(old_coeffs) = $(length(old_coeffs)) does not match old_num_exponentials * num_gates = $old_len")
    end
    new_coeffs = zeros(Float64, new_num_exponentials * num_gates)
    new_coeffs[1:old_len] .= old_coeffs
    return new_coeffs
end

"""
    embed_active_params(A_active, active_indices, M) -> Vector

Scatter the reduced coefficient vector `A_active` (one entry per index in `active_indices`)
into a zero vector of length `M`. Written with `Zygote.Buffer` so the scatter stays
differentiable when called from inside a loss function being optimized.
"""
function embed_active_params(A_active::AbstractVector{T}, active_indices::AbstractVector{Int}, M::Int) where T
    buf = Zygote.Buffer(A_active, M)
    for i in 1:M
        buf[i] = zero(T)
    end
    for (k, idx) in enumerate(active_indices)
        buf[idx] = A_active[k]
    end
    return copy(buf)
end

"""
    get_optimizer_algo(opt_sym::Symbol)

Map optimizer symbol to Optimization package algorithm instance.
"""
function get_optimizer_algo(opt_sym::Symbol)
    if opt_sym == :LBFGS
        return LBFGS()
    elseif opt_sym == :GradientDescent || opt_sym == :GD
        return GradientDescent()
    elseif opt_sym == :Adam
        return Adam(0.01)
    else
        error("Unsupported optimizer symbol: $opt_sym")
    end
end

"""
    find_multi_start_initialization(f, optf, M::Int; kwargs...) -> (A_init, local_multistart_losses, local_best_start_idx, multistart_run)

Perform multi-start initialization by sampling `initialization_samples` random configurations of size `M`.
"""
function find_multi_start_initialization(f, optf, M::Int;
    initialization_samples::Int=20,
    multi_start_samples::Int=5,
    multi_start_iters::Int=30,
    maxiters::Int=100,
    optimizer=:LBFGS,
    perturb_optimization::Float64=0.0,
    use_gpu::Bool=false)

    println("Sampling $initialization_samples initial configurations for multi-start...")
    samples_raw = Vector{Any}(undef, initialization_samples)
    log_min = log10(1e-7)
    log_max = log10(1e-1)

    @safe_threads for s in 1:initialization_samples
        mag = 10^(log_min + (log_max - log_min) * rand())
        A_sample = (2 * rand(M) .- 1) * mag
        res = Zygote.withgradient(A_sample) do x
            f(x)
        end
        loss_val = res.val
        grad = res.grad[1]
        gnorm = norm(grad)

        is_good = (gnorm > 1e-8) && (loss_val < 1.0)
        if is_good
            samples_raw[s] = (gnorm, loss_val, A_sample)
        else
            samples_raw[s] = nothing
        end
    end

    good_samples = Vector{Any}()
    for item in samples_raw
        if !isnothing(item)
            push!(good_samples, item)
        end
    end

    sort!(good_samples, by=x -> x[1], rev=true)
    top_n = min(multi_start_samples, length(good_samples))

    if top_n == 0
        println("No good samples found, falling back to random initialization.")
        fallback_A = (2 * rand(M) .- 1) * 0.01
        return fallback_A, Vector{Float64}[], 0, false
    end

    println("Performing quick optimization on top $top_n candidates...")
    candidate_results = Vector{Any}(undef, top_n)
    quick_maxiters = min(multi_start_iters, maxiters)
    optimizers = (optimizer isa AbstractVector) ? optimizer : [optimizer]

    @safe_threads for i in 1:top_n
        candidate_A = good_samples[i][3]
        curr_A = copy(candidate_A)
        curr_loss = Inf
        success = false
        candidate_history = Float64[]
        for (idx, opt) in enumerate(optimizers)
            if idx > 1 && perturb_optimization > 1e-9
                used_perturb = perturb_optimization^(1 + (idx - 1) / 3)
                curr_A = curr_A * (1 - used_perturb) + used_perturb * mean(abs.(curr_A)) * (2 * rand(length(curr_A)) .- 1)
            end
            opt_algo = (opt isa Symbol) ? get_optimizer_algo(opt) : opt
            cb = (state, loss_val) -> begin
                push!(candidate_history, loss_val)
                return false
            end
            prob = Optimization.OptimizationProblem(optf, curr_A)
            try
                sol = Optimization.solve(prob, opt_algo, maxiters=quick_maxiters, callback=cb)
                curr_A = sol.u
                curr_loss = sol.objective
                success = true
            catch e
                @warn "Candidate $i failed in quick optimization with $opt: $e"
            end
        end
        if success
            candidate_results[i] = (curr_loss, curr_A, candidate_history)
        else
            candidate_results[i] = nothing
        end
    end

    best_loss = Inf
    best_A = nothing
    local_best_start_idx = 0
    local_multistart_losses = Vector{Float64}[]
    for (i, res) in enumerate(candidate_results)
        if !isnothing(res)
            push!(local_multistart_losses, res[3])
            if res[1] < best_loss
                best_loss = res[1]
                best_A = res[2]
                local_best_start_idx = i
            end
        end
    end

    if isnothing(best_A)
        fallback_A = (2 * rand(M) .- 1) * 0.01
        return fallback_A, Vector{Float64}[], 0, false
    else
        println("Selected best candidate with loss=$best_loss")
        return best_A, local_multistart_losses, local_best_start_idx, true
    end
end

"""
    optimize_unitary(gates, tau_terms, ref, target, basis, N; kwargs...)

Optimize the parameter vector A of length `num_exponentials * length(gates)` to minimize
either overlap or energy loss. Supports multi-start initialization and GPU execution (`use_gpu=true`).
Returns `(A_opt, final_loss, metrics)`.
"""
function optimize_unitary(gates, tau_terms, ref::AbstractVector, target::Union{AbstractVector,AbstractMatrix}, basis, N::Int;
    loss_type::Symbol=:overlap,
    H::Union{AbstractMatrix,Nothing}=nothing,
    state2::Union{AbstractVector,Nothing}=nothing,
    num_exponentials::Int=1,
    active_indices::Union{Nothing,AbstractVector{Int}}=nothing,
    maxiters::Int=100,
    optimizer=:LBFGS,
    perturb_optimization::Float64=0.001,
    initialization_samples::Int=20,
    multi_start_samples::Int=5,
    multi_start_iters::Int=30,
    initial_coefficients::Union{AbstractVector,Nothing}=nothing,
    initial_history::Vector{Float64}=Float64[],
    loaded_metrics::Union{Dict,Nothing}=nothing,
    antihermitian::Bool=false,
    use_gpu::Bool=false,
    datatype::Type{<:Number}=ComplexF64,
    metric_functions::Dict{String,Function}=Dict{String,Function}())

    # Handle conversion from Complex to Real data type if specified
    ref_prep, _ = (datatype <: Real) ? strip_global_phase(ref) : (ref, 1.0)
    target_prep = if target isa AbstractVector
        (datatype <: Real) ? strip_global_phase(target)[1] : target
    else
        target
    end

    M_full = num_exponentials * length(gates)

    f = (A, p=nothing) -> begin
        A_full = isnothing(active_indices) ? A : embed_active_params(A, active_indices, M_full)
        if loss_type == :overlap
            return adjoint_loss(A_full, gates, tau_terms, ref_prep, target_prep, basis, N;
                num_exponentials=num_exponentials, antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype)
        elseif loss_type == :energy
            return energy_loss(A_full, gates, tau_terms, target_prep, ref_prep, basis, N;
                num_exponentials=num_exponentials, antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype)
        else
            error("Unknown loss_type: $loss_type")
        end
    end

    optf = Optimization.OptimizationFunction(f, Optimization.AutoZygote())
    M = isnothing(active_indices) ? M_full : length(active_indices)

    state2_vec = target isa AbstractVector ? target : state2
    state2_prep = !isnothing(state2_vec) ? ((datatype <: Real) ? strip_global_phase(state2_vec)[1] : state2_vec) : nothing
    H_mat = target isa AbstractMatrix ? target : H

    multistart_run = false
    local_multistart_losses = Vector{Float64}[]
    local_best_start_idx = 0

    if !isnothing(initial_coefficients) && length(initial_coefficients) == M
        A_init = copy(initial_coefficients)
    elseif initialization_samples > 0
        A_init, local_multistart_losses, local_best_start_idx, multistart_run = find_multi_start_initialization(f, optf, M;
            initialization_samples=initialization_samples,
            multi_start_samples=multi_start_samples,
            multi_start_iters=multi_start_iters,
            maxiters=maxiters,
            optimizer=optimizer,
            perturb_optimization=perturb_optimization,
            use_gpu=use_gpu)
    else
        A_init = (2 * rand(M) .- 1) * 0.01
    end

    initial_loss = f(A_init)

    metrics = Dict{String,Vector{Any}}()
    if !isnothing(loaded_metrics) && haskey(loaded_metrics, "loss") && !isempty(loaded_metrics["loss"])
        metrics["loss"] = copy(loaded_metrics["loss"])
    else
        # The first element of metrics["loss"] must be the loss at zero coefficients (identity unitary).
        # This represents the overlap/energy loss between the target and reference state prior to any rotation.
        zero_coeff_loss = f(zeros(eltype(A_init), M))
        metrics["loss"] = Float64[zero_coeff_loss]
    end
    metrics["other"] = []
    metrics["loss_std"] = Float64[0.0]
    metrics["optimization_losses"] = Vector{Float64}[]
    metrics["multistart_losses"] = Vector{Vector{Float64}}[]
    metrics["best_start_idx"] = Int[]
    metrics["convergence_info"] = Vector{Dict{String,Any}}[]
    metrics["stopping_reasons"] = Vector{String}[]
    if !isnothing(loaded_metrics) && haskey(loaded_metrics, "energy") && !isempty(loaded_metrics["energy"])
        metrics["energy"] = copy(loaded_metrics["energy"])
    elseif loss_type == :overlap
        metrics["energy"] = Float64[!isnothing(H_mat) ? real(dot(ref_prep, H_mat * ref_prep)) : NaN]
    end
    if !isnothing(loaded_metrics) && haskey(loaded_metrics, "overlap") && !isempty(loaded_metrics["overlap"])
        metrics["overlap"] = copy(loaded_metrics["overlap"])
    elseif loss_type == :energy
        metrics["overlap"] = Float64[!isnothing(state2_prep) ? max(0.0, 1.0 - abs2(dot(state2_prep, ref_prep))) : NaN]
    end
    for k in keys(metric_functions)
        metrics[k] = Any[]
    end

    println("Initial loss ($loss_type): $initial_loss")

    if loss_type == :overlap && 0 <= initial_loss < 1e-12
        println("States are already equal")
        push!(metrics["loss"], initial_loss)
        push!(metrics["optimization_losses"], [initial_loss])
        push!(metrics["convergence_info"], [Dict{String,Any}("optimizer" => "None", "stage" => 1, "primary_reason" => "States are already equal", "iterations" => 0, "g_residual" => 0.0)])
        push!(metrics["stopping_reasons"], ["States are already equal"])
        if haskey(metrics, "energy") && !isempty(metrics["energy"])
            push!(metrics["energy"], metrics["energy"][1])
        end
        A_zero = zeros(Float64, M_full)
        return A_zero, initial_loss, metrics
    end

    optimizers = (optimizer isa AbstractVector) ? optimizer : [optimizer]
    curr_A = copy(A_init)
    curr_loss = initial_loss
    final_history = copy(initial_history)
    stage_convergence_info = Dict{String,Any}[]

    cb = (state, loss_val) -> begin
        push!(final_history, loss_val)
        return false
    end

    for (idx, opt) in enumerate(optimizers)
        if idx > 1 && perturb_optimization > 1e-9
            used_perturb = perturb_optimization^(1 + (idx - 1) / 3)
            curr_A = curr_A * (1 - used_perturb) + used_perturb * mean(abs.(curr_A)) * (2 * rand(length(curr_A)) .- 1)
        end
        opt_algo = (opt isa Symbol) ? get_optimizer_algo(opt) : opt
        prob = Optimization.OptimizationProblem(optf, curr_A)
        println("Running main optimization step $idx with $opt (maxiters=$maxiters, use_gpu=$use_gpu)...")
        sol = Optimization.solve(prob, opt_algo, maxiters=maxiters, callback=cb)
        curr_A = sol.u
        curr_loss = sol.objective

        conv_info = extract_convergence_info(sol)
        conv_info["optimizer"] = string(opt)
        conv_info["stage"] = idx
        push!(stage_convergence_info, conv_info)
        println("    Step $idx ($opt) stopped by: $(conv_info["primary_reason"]) (Iterations: $(conv_info["iterations"]), Final |g|: $(conv_info["g_residual"]))")
    end

    curr_A = isnothing(active_indices) ? curr_A : embed_active_params(curr_A, active_indices, M_full)

    push!(metrics["loss"], curr_loss)
    push!(metrics["optimization_losses"], final_history)

    if !isnothing(loaded_metrics) && haskey(loaded_metrics, "convergence_info") && !isempty(loaded_metrics["convergence_info"])
        prev_stages = loaded_metrics["convergence_info"][1]
        all_conv = vcat(prev_stages, stage_convergence_info)
        push!(metrics["convergence_info"], all_conv)
        push!(metrics["stopping_reasons"], [info["primary_reason"] for info in all_conv])
    else
        push!(metrics["convergence_info"], stage_convergence_info)
        push!(metrics["stopping_reasons"], [info["primary_reason"] for info in stage_convergence_info])
    end

    if multistart_run
        push!(metrics["multistart_losses"], local_multistart_losses)
        push!(metrics["best_start_idx"], local_best_start_idx)
    elseif !isnothing(loaded_metrics) && haskey(loaded_metrics, "multistart_losses") && !isempty(loaded_metrics["multistart_losses"])
        push!(metrics["multistart_losses"], loaded_metrics["multistart_losses"][1])
        push!(metrics["best_start_idx"], get(loaded_metrics, "best_start_idx", [0])[1])
    else
        push!(metrics["multistart_losses"], Vector{Float64}[])
        push!(metrics["best_start_idx"], 0)
    end

    ref_evolved = apply_unitary(curr_A, gates, ref_prep, basis, N, num_exponentials; antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype)
    ref_evolved_cpu = Array(ref_evolved)
    if loss_type == :overlap
        final_energy = !isnothing(H_mat) ? real(dot(ref_evolved_cpu, H_mat * ref_evolved_cpu)) : NaN
        push!(metrics["energy"], final_energy)
    elseif loss_type == :energy
        final_overlap = !isnothing(state2_prep) ? max(0.0, 1.0 - abs2(dot(state2_prep, ref_evolved_cpu))) : NaN
        push!(metrics["overlap"], final_overlap)
    end

    for (k, func) in metric_functions
        val = try
            func(ref, target, curr_A, final_history)
        catch
            try
                func(ref, target, gates, curr_A, final_history)
            catch
                NaN
            end
        end
        push!(metrics[k], val)
    end

    return curr_A, curr_loss, metrics
end
