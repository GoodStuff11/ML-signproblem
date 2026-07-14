module TrotterOptimization

using ChainRulesCore
using LinearAlgebra
using Zygote
using Optimization
using OptimizationOptimJL
using OptimizationOptimisers
import ..TamFermion
using ..Trotter: @safe_threads
using JLD2
using Statistics

export adjoint_loss, energy_loss, optimize_unitary, interaction_scan_map_to_state

# ═══════════════════════════════════════════════════════════════════════
# SHARED HELPERS FOR FORWARD AND BACKWARD SWEEPS
# ═══════════════════════════════════════════════════════════════════════

"""
    apply_unitary_checkpoints(A, gates, ref, basis, N, num_exponentials) -> phis

Evolves the state `ref` forward through all parameters `A` and returns a list
of intermediate state checkpoints `phis`, where `phis[1]` is `ref` and `phis[end]`
is the fully evolved state.
"""
function apply_unitary_checkpoints(A::AbstractArray, gates, ref::AbstractArray, basis, N::Int, num_exponentials::Int; antihermitian::Bool=false)
    P = num_exponentials
    num_gates = length(gates)
    M = P * num_gates
    phis = Vector{typeof(ref)}(undef, M + 1)
    phis[1] = ref
    curr = 1
    for l in 1:P
        coefs = A[((l-1)*num_gates+1):(l*num_gates)]
        ops = TamFermion.fgateToExpSector(gates, coefs, N, basis; antihermitian=antihermitian)
        for op in ops
            phis[curr+1] = op * phis[curr]
            curr += 1
        end
    end
    return phis
end

"""
    backward_adjoint_propagation(A, gates, tau_terms, phis, init_adjoint_state, basis, N, num_exponentials) -> grad_A

Propagates the `adjoint_state` backward starting from `init_adjoint_state`,
computing the gradient of the parameters at each step using the forward state checkpoints `phis`.
"""
function backward_adjoint_propagation(A::AbstractArray, gates, tau_terms, phis::Vector, init_adjoint_state::AbstractVector, basis, N::Int, num_exponentials::Int; antihermitian::Bool=false)
    P = num_exponentials
    num_gates = length(gates)
    M = P * num_gates

    grad_A = Vector{Float64}(undef, M)
    adjoint_state = copy(init_adjoint_state)

    curr = M
    for l in P:-1:1
        coefs = A[((l-1)*num_gates+1):(l*num_gates)]
        ops_inv = TamFermion.fgateToExpSector(gates, -coefs, N, basis; antihermitian=antihermitian)

        for param_idx in num_gates:-1:1
            op_inv = ops_inv[param_idx]

            # Compute gradient contribution for parameter at index curr
            if antihermitian
                grad_A[curr] = -real(dot(adjoint_state, tau_terms[param_idx] * phis[curr+1]))
            else
                grad_A[curr] = imag(dot(adjoint_state, tau_terms[param_idx] * phis[curr+1]))
            end

            # Propagate adjoint state backward
            adjoint_state = op_inv * adjoint_state
            curr -= 1
        end
    end
    return grad_A
end

function apply_unitary(A::AbstractArray, gates, ref::AbstractArray, basis, N::Int, num_exponentials::Int; antihermitian::Bool=false)
    phis = apply_unitary_checkpoints(A, gates, ref, basis, N, num_exponentials; antihermitian=antihermitian)
    return phis[end]
end

# ═══════════════════════════════════════════════════════════════════════
# OVERLAP LOSS
# ═══════════════════════════════════════════════════════════════════════

function adjoint_loss(A::AbstractArray, gates, tau_terms, ref::AbstractArray, target::AbstractArray, basis, N::Int; num_exponentials::Int=1, antihermitian::Bool=false)
    ref_evolved = apply_unitary(A, gates, ref, basis, N, num_exponentials; antihermitian=antihermitian)
    return 1 - abs2(dot(target, ref_evolved))
end

function ChainRulesCore.rrule(::typeof(adjoint_loss), A::AbstractArray, gates, tau_terms, ref::AbstractArray, target::AbstractArray, basis, N::Int; num_exponentials::Int=1, antihermitian::Bool=false)
    phis = apply_unitary_checkpoints(A, gates, ref, basis, N, num_exponentials; antihermitian=antihermitian)
    evolved_ref = phis[end]
    overlap = dot(target, evolved_ref)
    loss = 1 - abs2(overlap)
    println("loss: $loss")

    function adjoint_loss_pullback(y)
        init_adjoint_state = (2 * overlap * conj(y)) * target
        grad_A = backward_adjoint_propagation(A, gates, tau_terms, phis, init_adjoint_state, basis, N, num_exponentials; antihermitian=antihermitian)
        return NoTangent(), grad_A, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return loss, adjoint_loss_pullback
end

# ═══════════════════════════════════════════════════════════════════════
# ENERGY LOSS
# ═══════════════════════════════════════════════════════════════════════

function energy_loss(A::AbstractArray, gates, tau_terms, H, ref::AbstractArray, basis, N::Int; num_exponentials::Int=1, antihermitian::Bool=false)
    ref_evolved = apply_unitary(A, gates, ref, basis, N, num_exponentials; antihermitian=antihermitian)
    return real(dot(ref_evolved, H * ref_evolved))
end

function ChainRulesCore.rrule(::typeof(energy_loss), A::AbstractArray, gates, tau_terms, H, ref::AbstractArray, basis, N::Int; num_exponentials::Int=1, antihermitian::Bool=false)
    phis = apply_unitary_checkpoints(A, gates, ref, basis, N, num_exponentials; antihermitian=antihermitian)
    evolved_ref = phis[end]
    loss = real(dot(evolved_ref, H * evolved_ref))
    println("loss: $loss")
    function energy_loss_pullback(y)
        init_adjoint_state = (-2 * conj(y)) * (H * evolved_ref)
        grad_A = backward_adjoint_propagation(A, gates, tau_terms, phis, init_adjoint_state, basis, N, num_exponentials; antihermitian=antihermitian)
        return NoTangent(), grad_A, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return loss, energy_loss_pullback
end

# ═══════════════════════════════════════════════════════════════════════
# OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════

"""
    find_multi_start_initialization(f, optf, M::Int; kwargs...) -> A_init

Perform multi-start initialization by sampling `initialization_samples` random configurations of size `M`.
Evaluate the gradient of each configuration with function `f`, select the top `multi_start_samples` candidates based on gradient norm,
run quick optimization sweeps of at most `multi_start_iters` with `optf`, and return the best overall parameter configuration.
"""
function find_multi_start_initialization(f, optf, M::Int;
    initialization_samples::Int=20,
    multi_start_samples::Int=5,
    multi_start_iters::Int=30,
    maxiters::Int=100,
    optimizer=:LBFGS,
    perturb_optimization::Float64=0.0)

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
        return (2 * rand(M) .- 1) * 0.01
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
        for (idx, opt) in enumerate(optimizers)
            if idx > 1 && perturb_optimization > 1e-9
                used_perturb = perturb_optimization^(1 + (idx - 1) / 3)
                curr_A = curr_A * (1 - used_perturb) + used_perturb * mean(abs.(curr_A)) * (2 * rand(length(curr_A)) .- 1)
            end
            opt_algo = (opt isa Symbol) ? get_optimizer_algo(opt) : opt
            prob = Optimization.OptimizationProblem(optf, curr_A)
            try
                sol = Optimization.solve(prob, opt_algo, maxiters=quick_maxiters)
                curr_A = sol.u
                curr_loss = sol.objective
                success = true
            catch e
                @warn "Candidate $i failed in quick optimization with $opt: $e"
            end
        end
        if success
            candidate_results[i] = (curr_loss, curr_A)
        else
            candidate_results[i] = nothing
        end
    end

    best_loss = Inf
    best_A = nothing
    for res in candidate_results
        if !isnothing(res) && res[1] < best_loss
            best_loss = res[1]
            best_A = res[2]
        end
    end

    if isnothing(best_A)
        return (2 * rand(M) .- 1) * 0.01
    else
        println("Selected best candidate with loss=$best_loss")
        return best_A
    end
end

"""
    optimize_unitary(gates, tau_terms, ref, target, basis, N; kwargs...)

Optimize the parameter vector A of length `num_exponentials * length(gates)` to minimize
either overlap or energy loss. Supports multi-start initialization.
"""
function optimize_unitary(gates, tau_terms, ref::AbstractVector, target::Union{AbstractVector,AbstractMatrix}, basis, N::Int;
    loss_type::Symbol=:overlap,
    num_exponentials::Int=1,
    maxiters::Int=100,
    optimizer=:LBFGS,
    perturb_optimization::Float64=0.001,
    initialization_samples::Int=20,
    multi_start_samples::Int=5,
    multi_start_iters::Int=30,
    initial_coefficients::Union{AbstractVector,Nothing}=nothing,
    antihermitian::Bool=false)

    f = (A, p=nothing) -> begin
        if loss_type == :overlap
            return adjoint_loss(A, gates, tau_terms, ref, target, basis, N; num_exponentials=num_exponentials, antihermitian=antihermitian)
        elseif loss_type == :energy
            return energy_loss(A, gates, tau_terms, target, ref, basis, N; num_exponentials=num_exponentials, antihermitian=antihermitian)
        else
            error("Unknown loss_type: $loss_type")
        end
    end

    optf = Optimization.OptimizationFunction(f, Optimization.AutoZygote())
    M = num_exponentials * length(gates)

    if !isnothing(initial_coefficients) && length(initial_coefficients) == M
        A_init = copy(initial_coefficients)
    elseif initialization_samples > 0
        A_init = find_multi_start_initialization(f, optf, M;
            initialization_samples=initialization_samples,
            multi_start_samples=multi_start_samples,
            multi_start_iters=multi_start_iters,
            maxiters=maxiters,
            optimizer=optimizer,
            perturb_optimization=perturb_optimization)
    else
        A_init = (2 * rand(M) .- 1) * 0.01
    end

    optimizers = (optimizer isa AbstractVector) ? optimizer : [optimizer]
    curr_A = copy(A_init)
    curr_loss = Inf
    local sol
    for (idx, opt) in enumerate(optimizers)
        if idx > 1 && perturb_optimization > 1e-9
            used_perturb = perturb_optimization^(1 + (idx - 1) / 3)
            curr_A = curr_A * (1 - used_perturb) + used_perturb * mean(abs.(curr_A)) * (2 * rand(length(curr_A)) .- 1)
        end
        opt_algo = (opt isa Symbol) ? get_optimizer_algo(opt) : opt
        prob = Optimization.OptimizationProblem(optf, curr_A)
        println("Running main optimization step $idx with $opt (maxiters=$maxiters)...")
        sol = Optimization.solve(prob, opt_algo, maxiters=maxiters)
        curr_A = sol.u
        curr_loss = sol.objective
    end
    return curr_A, curr_loss
end

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
    interaction_scan_map_to_state(...)

Scan over a range of U interaction parameters, optimizing Trotter parameters at each step.
Analogous to `interaction_scan_map_to_state` in `ed_optimization.jl`.
"""
function interaction_scan_map_to_state(degen_rm_U::Union{AbstractMatrix,Vector}, instructions::Dict{String,Any},
    gates, tau_terms, basis, N::Int;
    maxiters=100, optimizer=:LBFGS,
    perturb_optimization::Float64=0.001,
    save_folder::Union{String,Nothing}=nothing, save_name::String="scan_data",
    initial_coefficients::Union{AbstractVector,Nothing}=nothing,
    U_values::Union{Vector{Float64},Nothing}=nothing,
    loss_type::Symbol=:overlap,
    custom_ref_state::Union{Vector,Nothing}=nothing,
    H_hopping::Union{AbstractMatrix,Nothing}=nothing,
    H_interaction::Union{AbstractMatrix,Nothing}=nothing,
    initialization_samples::Int=20,
    multi_start_samples::Int=5,
    multi_start_iters::Int=30,
    antihermitian::Bool=get(instructions, "antihermitian", false)
)
    # instructions["u_range"] should be a range of indices, e.g., 1:10
    # instructions["starting state"] should define the fixed reference state (state1)
    instructions["antihermitian"] = antihermitian

    data_dict = Dict{String,Any}("norm1_metrics" => [], "norm2_metrics" => [],
        "loss_metrics" => [], "labels" => [], "loss_std_metrics" => [], "all_matrices" => [],
        "coefficients" => [], "coefficient_labels" => nothing, "param_mapping" => nothing, "parities" => nothing)

    if haskey(instructions, "load_file")
        dic = JLD2.load(instructions["load_file"])["dict"]
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

    u_vals = !isnothing(U_values) ? U_values : (haskey(instructions, "U_values") ? instructions["U_values"] : nothing)

    num_exponentials = get(instructions, "num_exponentials", 1)

    for u_idx in u_indices
        u_val_str = isnothing(u_vals) ? "" : " (U = $(u_vals[u_idx]))"
        println("\n--- Scanning U index: $u_idx$u_val_str ---")

        state1 = if !isnothing(custom_ref_state)
            custom_ref_state
        elseif degen_rm_U isa AbstractMatrix
            degen_rm_U[ref_u_idx, :]
        else
            degen_rm_U[ref_u_idx]
        end

        state2 = if degen_rm_U isa AbstractMatrix
            degen_rm_U[u_idx, :]
        else
            degen_rm_U[u_idx]
        end

        target_u = isnothing(u_vals) ? nothing : u_vals[u_idx]

        H = if loss_type == :energy
            if !isnothing(H_hopping) && !isnothing(H_interaction) && !isnothing(target_u)
                H_hopping + target_u * H_interaction
            else
                error("H_hopping and H_interaction must be provided for energy loss optimization.")
            end
        else
            nothing
        end

        opt_target = (loss_type == :energy) ? H : state2

        A_opt, final_loss = optimize_unitary(
            gates, tau_terms, state1, opt_target, basis, N;
            loss_type=loss_type,
            num_exponentials=num_exponentials,
            maxiters=maxiters,
            optimizer=optimizer,
            perturb_optimization=perturb_optimization,
            initialization_samples=initialization_samples,
            multi_start_samples=multi_start_samples,
            multi_start_iters=multi_start_iters,
            initial_coefficients=current_coeffs,
            antihermitian=antihermitian
        )

        current_coeffs = A_opt

        # Store results for this U
        push!(data_dict["norm1_metrics"], [norm(A_opt, 1)])
        push!(data_dict["norm2_metrics"], [norm(A_opt, 2)])
        push!(data_dict["coefficients"], A_opt)
        push!(data_dict["loss_metrics"], final_loss)

        # Construct metrics dictionary
        ref_evolved = apply_unitary(A_opt, gates, state1, basis, N, num_exponentials; antihermitian=antihermitian)
        metrics = Dict{String,Any}("loss" => [final_loss])

        # Calculate comparison metrics
        H_eval = if !isnothing(H_hopping) && !isnothing(H_interaction) && !isnothing(target_u)
            H_hopping + target_u * H_interaction
        else
            nothing
        end
        ed_energy = !isnothing(H_eval) ? real(dot(state2, H_eval * state2)) : NaN
        trotter_energy = !isnothing(H_eval) ? real(dot(ref_evolved, H_eval * ref_evolved)) : NaN
        overlap = abs2(dot(state2, ref_evolved))

        if loss_type == :overlap
            metrics["energy"] = [trotter_energy]
        elseif loss_type == :energy
            metrics["overlap"] = [1.0 - overlap]
        end

        println("  Optimization Complete:")
        println("    Final Loss ($loss_type): $final_loss")
        if !isnothing(H_eval)
            println("    Exact ED Ground Energy: $ed_energy")
            println("    Trotter Evolved Energy: $trotter_energy")
            println("    Energy Difference:      $(trotter_energy - ed_energy)")
        end
        println("    Fidelity (Overlap^2):   $overlap")

        # Save shared data once we have it
        if !isnothing(save_folder) && !shared_data_saved
            println("saving shared data...")
            shared_dict = Dict(
                "gates" => gates,
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
                "coefficients" => A_opt,
                "metrics" => metrics,
                "norm1" => [norm(A_opt, 1)],
                "norm2" => [norm(A_opt, 2)]
            )
            JLD2.jldsave(joinpath(save_folder, "$(save_name)_u_$u_idx.jld2"); dict=iter_dict)
        end

        push!(data_dict["labels"], Dict(
            "starting state" => Dict("level" => ref_level, "U index" => ref_u_idx),
            "ending state" => Dict("level" => get(instructions, "starting level", 1), "U index" => u_idx))
        )
    end

    return data_dict
end

end # module TrotterOptimization