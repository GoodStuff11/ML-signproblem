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
    grow_coefficients(old_coeffs, old_num_exponentials, new_num_exponentials, num_gates; param_map=nothing) -> Vector{Float64}

Extend a Trotter coefficient vector that was optimized for `old_num_exponentials` layers into
one usable for a larger `new_num_exponentials`, so the newly-added (later) layers can be
optimized starting from an already-converged shorter ansatz instead of from scratch.

With `param_map !== nothing` the vector grows in units of
`num_shared_params(param_map, num_gates)` per layer rather than `num_gates`.
"""
function grow_coefficients(old_coeffs::AbstractVector, old_num_exponentials::Int, new_num_exponentials::Int, num_gates::Int;
    param_map::Union{Nothing,AbstractVector{Int}}=nothing)
    if new_num_exponentials < old_num_exponentials
        error("new_num_exponentials ($new_num_exponentials) must be >= old_num_exponentials ($old_num_exponentials)")
    end
    n_params = num_shared_params(param_map, num_gates)
    old_len = old_num_exponentials * n_params
    if length(old_coeffs) != old_len
        error("length(old_coeffs) = $(length(old_coeffs)) does not match old_num_exponentials * n_params = $old_len")
    end
    new_coeffs = zeros(Float64, new_num_exponentials * n_params)
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
    get_optimizer_algo(opt_sym::Symbol; lbfgs_memory=10)

Map optimizer symbol to Optimization package algorithm instance. `lbfgs_memory` is the
L-BFGS history length `m` (Optim's default is 10).
"""
function get_optimizer_algo(opt_sym::Symbol; lbfgs_memory::Int=10)
    if opt_sym == :LBFGS
        return LBFGS(m=lbfgs_memory)
    elseif opt_sym == :GradientDescent || opt_sym == :GD
        return GradientDescent()
    elseif opt_sym == :Adam
        # Qualified: Optim (via OptimizationOptimJL) also exports an `Adam`, so the bare name
        # is ambiguous. This is Optimisers' Adam with learning rate 0.01.
        return OptimizationOptimisers.Adam(0.01)
    else
        error("Unsupported optimizer symbol: $opt_sym")
    end
end

"""
    normalize_stages(optimizer, loss_type, maxiters) -> Vector{NamedTuple{(:opt, :loss, :maxiters)}}

Turn the `optimizer` argument of [`optimize_unitary`](@ref) into a list of stages. Each entry
may be a `Symbol` (e.g. `:LBFGS`), an Optim algorithm instance, or a `NamedTuple` with any of
`opt`, `loss` (`:overlap`/`:energy`) and `maxiters`; missing fields default to `:LBFGS`,
`loss_type` and `maxiters`. Stages whose `loss` differs from `loss_type` optimize a different
objective (e.g. energy between two overlap stages) to move the parameters out of a local
minimum of the primary loss.
"""
normalize_stages(optimizer, loss_type::Symbol, maxiters::Int) =
    [_normalize_stage(s, loss_type, maxiters) for s in (optimizer isa AbstractVector ? optimizer : [optimizer])]

_normalize_stage(s::NamedTuple, loss_type, maxiters) =
    (opt=get(s, :opt, :LBFGS), loss=get(s, :loss, loss_type), maxiters=get(s, :maxiters, maxiters))
_normalize_stage(s, loss_type, maxiters) = (opt=s, loss=loss_type, maxiters=maxiters)

"""
    parse_stage_spec(spec::AbstractString) -> NamedTuple

Parse `"OPT[:LOSS[:MAXITERS]]"`, e.g. `"LBFGS"`, `"LBFGS:energy"`, `"GD:overlap:50"`, into a
stage for [`normalize_stages`](@ref). Omitted fields are filled in later with the run defaults.
"""
function parse_stage_spec(spec::AbstractString)
    parts = split(strip(spec), ":")
    (1 <= length(parts) <= 3 && !isempty(parts[1])) || error("Invalid stage spec '$spec' (expected OPT[:LOSS[:MAXITERS]])")
    opt = Symbol(parts[1])
    opt in (:LBFGS, :GradientDescent, :GD, :Adam) || error("Invalid optimizer '$(parts[1])' in stage spec '$spec'")
    length(parts) == 1 && return (opt=opt,)
    loss = Symbol(parts[2])
    loss in (:overlap, :energy) || error("Invalid loss '$(parts[2])' in stage spec '$spec' (overlap or energy)")
    length(parts) == 2 && return (opt=opt, loss=loss)
    return (opt=opt, loss=loss, maxiters=parse(Int, parts[3]))
end

"""
    is_stalled(history, window, rtol) -> Bool

Early-stopping test on a stage's loss history. Using the running minimum `b`, the stage has
stalled when the last `window` iterations improved it by at most `rtol` times the improvement
made over the whole stage so far: `b[n-window] - b[n] <= rtol * (b[1] - b[n])`. Needs at least
`2*window + 1` entries. `window <= 0` disables it.
"""
function is_stalled(history::AbstractVector{<:Real}, window::Int, rtol::Real)
    n = length(history)
    (window > 0 && n > 2 * window) || return false
    b_now = minimum(history)
    b_then = minimum(@view history[1:(n-window)])
    return b_then - b_now <= rtol * (history[1] - b_now)
end

"""
    prepare_loss_states(ref, target, state2, H, datatype) -> (ref_prep, overlap_target, H_mat)

Phase-strip states for real datatypes (as the losses require) and split `target` into the
state used by the overlap loss and the Hamiltonian used by the energy loss. Either of the last
two may be `nothing` if it was not provided.
"""
function prepare_loss_states(ref::AbstractVector, target, state2, H, datatype::Type{<:Number})
    prep(v) = isnothing(v) ? nothing : ((datatype <: Real) ? strip_global_phase(v)[1] : v)
    ref_prep = prep(ref)
    overlap_target = target isa AbstractVector ? prep(target) : prep(state2)
    H_mat = target isa AbstractMatrix ? target : H
    return ref_prep, overlap_target, H_mat
end

"""
    make_loss_function(loss, gates, tau_terms, ref_prep, overlap_target, H_mat, basis, N; kwargs...) -> A -> loss

Loss on the full (un-pruned) coefficient vector for `loss = :overlap` or `:energy`.
"""
function make_loss_function(loss::Symbol, gates, tau_terms, ref_prep, overlap_target, H_mat, basis, N::Int;
    num_exponentials::Int=1, antihermitian::Bool=false, use_gpu::Bool=false, datatype::Type{<:Number}=ComplexF64,
    param_map::Union{Nothing,AbstractVector{Int}}=nothing)
    if loss == :overlap
        isnothing(overlap_target) && error("An overlap-loss stage needs the target state (pass state2).")
        return A -> adjoint_loss(A, gates, tau_terms, ref_prep, overlap_target, basis, N;
            num_exponentials=num_exponentials, antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype,
            param_map=param_map)
    elseif loss == :energy
        isnothing(H_mat) && error("An energy-loss stage needs the Hamiltonian (pass H).")
        return A -> energy_loss(A, gates, tau_terms, H_mat, ref_prep, basis, N;
            num_exponentials=num_exponentials, antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype,
            param_map=param_map)
    else
        error("Unknown loss_type: $loss")
    end
end

"""
    find_multi_start_initialization(f, optf, M::Int; kwargs...) -> (A_init, local_multistart_losses, local_multistart_gradients, local_best_start_idx, multistart_run, initial_gradient_samples)

Perform multi-start initialization by sampling `initialization_samples` random configurations of size `M`.
"""
function find_multi_start_initialization(f, optf, M::Int;
    initialization_samples::Int=20,
    multi_start_samples::Int=5,
    multi_start_iters::Int=30,
    maxiters::Int=100,
    optimizer=:LBFGS,
    perturb_optimization::Float64=0.0,
    use_gpu::Bool=false,
    lbfgs_memory::Int=10)
    # Returns (A_init, local_multistart_losses, local_multistart_gradients, local_best_start_idx, multistart_run, initial_gradient_samples)
    # initial_gradient_samples records (mag, gnorm, loss_val) for EVERY sampled random initialization
    # (not just the survivors that pass the is_good filter below), for barren-plateau-style analysis
    # of the initial-gradient magnitude as a function of the random-initialization scale `mag`.

    println("Sampling $initialization_samples initial configurations for multi-start...")
    samples_raw = Vector{Any}(undef, initialization_samples)
    initial_gradient_samples = Vector{NTuple{3,Float64}}(undef, initialization_samples)
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
        initial_gradient_samples[s] = (mag, gnorm, loss_val)

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
        return fallback_A, Vector{Float64}[], Vector{Vector{Float64}}[], 0, false, initial_gradient_samples
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
        candidate_gradient_history = Vector{Float64}[]
        for (idx, opt) in enumerate(optimizers)
            if idx > 1 && perturb_optimization > 1e-9
                used_perturb = perturb_optimization^(1 + (idx - 1) / 3)
                curr_A = curr_A * (1 - used_perturb) + used_perturb * mean(abs.(curr_A)) * (2 * rand(length(curr_A)) .- 1)
            end
            opt_algo = (opt isa Symbol) ? get_optimizer_algo(opt; lbfgs_memory=lbfgs_memory) : opt
            cb = (state, loss_val) -> begin
                push!(candidate_history, loss_val)
                push!(candidate_gradient_history, isnothing(state.grad) ? fill(NaN, length(state.u)) : copy(state.grad))
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
            candidate_results[i] = (curr_loss, curr_A, candidate_history, candidate_gradient_history)
        else
            candidate_results[i] = nothing
        end
    end

    best_loss = Inf
    best_A = nothing
    local_best_start_idx = 0
    local_multistart_losses = Vector{Float64}[]
    local_multistart_gradients = Vector{Vector{Float64}}[]
    for (i, res) in enumerate(candidate_results)
        if !isnothing(res)
            push!(local_multistart_losses, res[3])
            push!(local_multistart_gradients, res[4])
            if res[1] < best_loss
                best_loss = res[1]
                best_A = res[2]
                local_best_start_idx = i
            end
        end
    end

    if isnothing(best_A)
        fallback_A = (2 * rand(M) .- 1) * 0.01
        return fallback_A, Vector{Float64}[], Vector{Vector{Float64}}[], 0, false, initial_gradient_samples
    else
        println("Selected best candidate with loss=$best_loss")
        return best_A, local_multistart_losses, local_multistart_gradients, local_best_start_idx, true, initial_gradient_samples
    end
end

"""
    optimize_unitary(gates, tau_terms, ref, target, basis, N; kwargs...)

Optimize the parameter vector `A` to minimize either overlap or energy loss.
With `param_map === nothing` (the default), `A` has length
`num_exponentials * length(gates)`, one coefficient per gate per layer. With a
non-`nothing` `param_map`, `A` lives in the REDUCED space of length
`num_exponentials * num_shared_params(param_map, length(gates))`, and the
returned `A_opt` is likewise in that reduced space (see
`expand_shared_coefficients` / `trotter_shared_params.jl` for the gather it
implements).

Supports multi-start initialization and GPU execution (`use_gpu=true`).
Returns `(A_opt, final_loss, metrics)`.

Note for `metric_functions` callbacks: they receive the reduced `curr_A`
(length `num_exponentials * n_params`, not `num_exponentials * length(gates)`),
so a metric that indexes per-gate must expand it first via
`expand_shared_coefficients(curr_A, param_map, length(gates),
num_exponentials)`. These callbacks are wrapped in a double try/catch-to-NaN
below, so passing a per-gate-shaped index into a reduced-length vector is
silently swallowed as `NaN` rather than raising — get the length right.

Both this function and the `adjoint_loss`/`energy_loss` entry points it calls
are guarded against `antihermitian=true` with a diagonal gate in `gates` (see
`check_antihermitian_diagonal_gates`): such gates map to the zero operator
under that convention, so their coefficients would be silently unoptimisable.
"""
function optimize_unitary(gates, tau_terms, ref::AbstractVector, target::Union{AbstractVector,AbstractMatrix}, basis, N::Int;
    loss_type::Symbol=:overlap,
    H::Union{AbstractMatrix,Nothing}=nothing,
    state2::Union{AbstractVector,Nothing}=nothing,
    num_exponentials::Int=1,
    active_indices::Union{Nothing,AbstractVector{Int}}=nothing,
    param_map::Union{Nothing,AbstractVector{Int}}=nothing,
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
    metric_functions::Dict{String,Function}=Dict{String,Function}(),
    lbfgs_memory::Int=10,
    stall_window::Int=0,
    stall_rtol::Float64=0.005,
    basin_hops::Int=0,
    hop_scale::Float64=0.1,
    hop_mode::Symbol=:rms,
    hop_iters::Int=100,
    hop_stages=nothing,
    hop_temperature::Float64=0.0,
    hop_seed::Union{Nothing,Int}=nothing)

    loss_type in (:overlap, :energy) || error("Unknown loss_type: $loss_type")
    hop_mode in (:rms, :relative) || error("Invalid hop_mode: $hop_mode. Valid options are :rms, :relative.")

    # Handle conversion from Complex to Real data type if specified
    ref_prep, state2_prep, H_mat = prepare_loss_states(ref, target, state2, H, datatype)

    check_antihermitian_diagonal_gates(gates, antihermitian)

    n_params = num_shared_params(param_map, length(gates))
    M_full = num_exponentials * n_params

    # One loss closure per loss type, built on demand (a stage may optimize a loss other
    # than the primary `loss_type`). `f` is the primary loss.
    loss_fns = Dict{Symbol,Function}()
    loss_fn(lt::Symbol) = get!(loss_fns, lt) do
        f_full = make_loss_function(lt, gates, tau_terms, ref_prep, state2_prep, H_mat, basis, N;
            num_exponentials=num_exponentials, antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype,
            param_map=param_map)
        (A, p=nothing) -> f_full(isnothing(active_indices) ? A : embed_active_params(A, active_indices, M_full))
    end
    f = loss_fn(loss_type)

    optf = Optimization.OptimizationFunction(f, Optimization.AutoZygote())
    M = isnothing(active_indices) ? M_full : length(active_indices)

    stages = normalize_stages(optimizer, loss_type, maxiters)
    primary_opts = [s.opt for s in stages if s.loss == loss_type]
    isempty(primary_opts) && push!(primary_opts, :LBFGS)

    multistart_run = false
    local_multistart_losses = Vector{Float64}[]
    local_multistart_gradients = Vector{Vector{Float64}}[]
    local_best_start_idx = 0
    local_initial_gradient_samples = NTuple{3,Float64}[]

    if !isnothing(initial_coefficients) && length(initial_coefficients) == M
        A_init = copy(initial_coefficients)
    elseif initialization_samples > 0
        A_init, local_multistart_losses, local_multistart_gradients, local_best_start_idx, multistart_run, local_initial_gradient_samples = find_multi_start_initialization(f, optf, M;
            initialization_samples=initialization_samples,
            multi_start_samples=multi_start_samples,
            multi_start_iters=multi_start_iters,
            maxiters=maxiters,
            optimizer=primary_opts,
            perturb_optimization=perturb_optimization,
            use_gpu=use_gpu,
            lbfgs_memory=lbfgs_memory)
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
    metrics["optimization_gradients"] = Vector{Vector{Float64}}[]
    metrics["multistart_losses"] = Vector{Vector{Float64}}[]
    metrics["multistart_gradients"] = Vector{Vector{Vector{Float64}}}[]
    metrics["initial_gradient_samples"] = Vector{NTuple{3,Float64}}[]
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
    # Energy of the overlap target under the energy-loss Hamiltonian (the ED ground energy
    # when the target is the ED ground state), so the two losses can be compared on one scale.
    if !isnothing(H_mat) && !isnothing(state2_prep)
        println("Target state energy <target|H|target>: $(real(dot(state2_prep, H_mat * state2_prep)) / real(dot(state2_prep, state2_prep)))")
    end

    if loss_type == :overlap && 0 <= initial_loss < 1e-12
        println("States are already equal")
        push!(metrics["loss"], initial_loss)
        push!(metrics["optimization_losses"], [initial_loss])
        push!(metrics["optimization_gradients"], [zeros(Float64, M_full)])
        push!(metrics["convergence_info"], [Dict{String,Any}("optimizer" => "None", "stage" => 1, "primary_reason" => "States are already equal", "iterations" => 0, "g_residual" => 0.0)])
        push!(metrics["stopping_reasons"], ["States are already equal"])
        if haskey(metrics, "energy") && !isempty(metrics["energy"])
            push!(metrics["energy"], metrics["energy"][1])
        end
        A_zero = zeros(Float64, M_full)
        return A_zero, initial_loss, metrics
    end

    curr_A = copy(A_init)
    final_history = copy(initial_history)
    final_gradient_history = Vector{Float64}[]
    stage_convergence_info = Dict{String,Any}[]

    # Track the best point by the PRIMARY loss: stages on another loss, and basin hops, may
    # move away from it, and the returned coefficients must never be worse than the start.
    best_A = copy(A_init)
    best_loss = initial_loss

    # Run one stage from `A0`. Primary-loss iterations are appended to the main history
    # when `record`; every stage's own history is returned.
    function run_stage(A0, st, label; record::Bool)
        f_st = loss_fn(st.loss)
        is_primary = st.loss == loss_type
        optf_st = is_primary ? optf : Optimization.OptimizationFunction(f_st, Optimization.AutoZygote())
        stage_hist = Float64[]
        stalled = Ref(false)
        cb = (state, loss_val) -> begin
            push!(stage_hist, loss_val)
            if record && is_primary
                push!(final_history, loss_val)
                push!(final_gradient_history, isnothing(state.grad) ? fill(NaN, length(state.u)) : copy(state.grad))
            end
            if is_stalled(stage_hist, stall_window, stall_rtol)
                stalled[] = true
                return true
            end
            return false
        end
        opt_algo = (st.opt isa Symbol) ? get_optimizer_algo(st.opt; lbfgs_memory=lbfgs_memory) : st.opt
        prob = Optimization.OptimizationProblem(optf_st, A0)
        println("Running $label with $(st.opt) on $(st.loss) loss (maxiters=$(st.maxiters), use_gpu=$use_gpu)...")
        sol = Optimization.solve(prob, opt_algo, maxiters=st.maxiters, callback=cb)
        primary_val = is_primary ? Float64(sol.objective) : Float64(f(sol.u))

        conv_info = extract_convergence_info(sol)
        if conv_info["iterations"] == 0 && !isempty(stage_hist)
            # Non-Optim solvers (e.g. Optimisers' Adam) carry no Optim iteration count.
            conv_info["iterations"] = length(stage_hist)
        end
        if stalled[]
            conv_info["primary_reason"] = "Stalled (< $(stall_rtol) of the stage's improvement over the last $(stall_window) iterations)"
        end
        conv_info["optimizer"] = string(st.opt)
        conv_info["loss"] = string(st.loss)
        conv_info["stalled"] = stalled[]
        conv_info["primary_loss"] = primary_val
        println("    $label ($(st.opt), $(st.loss)) stopped by: $(conv_info["primary_reason"]) (Iterations: $(conv_info["iterations"]), Final |g|: $(conv_info["g_residual"]))" *
                (is_primary ? "" : " -> $loss_type loss $primary_val"))
        return sol.u, primary_val, conv_info, stage_hist
    end

    for (idx, st) in enumerate(stages)
        if idx > 1 && perturb_optimization > 1e-9
            used_perturb = perturb_optimization^(1 + (idx - 1) / 3)
            curr_A = curr_A * (1 - used_perturb) + used_perturb * mean(abs.(curr_A)) * (2 * rand(length(curr_A)) .- 1)
        end
        curr_A, primary_val, conv_info, _ = run_stage(curr_A, st, "main optimization step $idx"; record=true)
        conv_info["stage"] = idx
        push!(stage_convergence_info, conv_info)
        if primary_val < best_loss
            best_A, best_loss = copy(curr_A), primary_val
        end
    end

    # Basin hopping: perturb the current point, re-optimize, accept greedily (or with a
    # Metropolis rule at `hop_temperature > 0`), and always keep the best point seen.
    hop_records = Dict{String,Any}[]
    if basin_hops > 0
        rng = isnothing(hop_seed) ? Random.default_rng() : Random.MersenneTwister(hop_seed)
        hop_stage_list = isnothing(hop_stages) ?
                         [(opt=first(primary_opts), loss=loss_type, maxiters=hop_iters)] :
                         normalize_stages(hop_stages, loss_type, hop_iters)
        hop_A, hop_loss = copy(best_A), best_loss
        println("Basin hopping: $basin_hops hops from loss $hop_loss (hop_scale=$hop_scale, hop_mode=$hop_mode, " *
                "stages=$([string(s.opt, ":", s.loss, ":", s.maxiters) for s in hop_stage_list]), T=$hop_temperature)")
        for h in 1:basin_hops
            trial_A = if hop_mode == :relative
                hop_A .* (1 .+ hop_scale .* randn(rng, length(hop_A)))
            else
                rms = sqrt(mean(abs2, hop_A))
                hop_A .+ hop_scale * (rms > 0 ? rms : 1.0) .* randn(rng, length(hop_A))
            end
            start_loss = Float64(f(trial_A))
            trial_loss = start_loss
            hop_hist = Float64[]
            hop_conv = Dict{String,Any}[]
            for (s_idx, st) in enumerate(hop_stage_list)
                trial_A, trial_loss, conv_info, hist = run_stage(trial_A, st, "hop $h stage $s_idx"; record=false)
                append!(hop_hist, hist)
                push!(hop_conv, conv_info)
            end
            accepted = trial_loss < hop_loss ||
                       (hop_temperature > 0 && rand(rng) < exp(-(trial_loss - hop_loss) / hop_temperature))
            improved = trial_loss < best_loss
            improved && ((best_A, best_loss) = (copy(trial_A), trial_loss))
            accepted && ((hop_A, hop_loss) = (trial_A, trial_loss))
            println("  Hop $h/$basin_hops: perturbed loss $start_loss -> $trial_loss " *
                    "($(accepted ? "accepted" : "rejected")$(improved ? ", new best" : ""); best $best_loss)")
            push!(hop_records, Dict{String,Any}("hop" => h, "start_loss" => start_loss, "final_loss" => trial_loss,
                "accepted" => accepted, "new_best" => improved, "best_loss" => best_loss,
                "history" => hop_hist, "convergence_info" => hop_conv))
        end
    end
    metrics["basin_hops"] = Any[hop_records]

    curr_A, curr_loss = best_A, best_loss
    curr_A = isnothing(active_indices) ? curr_A : embed_active_params(curr_A, active_indices, M_full)

    push!(metrics["loss"], curr_loss)
    push!(metrics["optimization_losses"], final_history)
    push!(metrics["optimization_gradients"], final_gradient_history)

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
        push!(metrics["multistart_gradients"], local_multistart_gradients)
        push!(metrics["best_start_idx"], local_best_start_idx)
    elseif !isnothing(loaded_metrics) && haskey(loaded_metrics, "multistart_losses") && !isempty(loaded_metrics["multistart_losses"])
        push!(metrics["multistart_losses"], loaded_metrics["multistart_losses"][1])
        push!(metrics["multistart_gradients"], get(loaded_metrics, "multistart_gradients", [Vector{Vector{Float64}}[]])[1])
        push!(metrics["best_start_idx"], get(loaded_metrics, "best_start_idx", [0])[1])
    else
        push!(metrics["multistart_losses"], Vector{Float64}[])
        push!(metrics["multistart_gradients"], Vector{Vector{Float64}}[])
        push!(metrics["best_start_idx"], 0)
    end
    push!(metrics["initial_gradient_samples"], local_initial_gradient_samples)

    ref_evolved = apply_unitary(curr_A, gates, ref_prep, basis, N, num_exponentials; antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype, param_map=param_map)
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
