#=
trotter_scan.jl

Interaction parameter U scan and full dataset mapping pipeline for Trotter optimization.
=#

"""
    interaction_scan_map_to_state(...)

Scan over a range of U interaction parameters, optimizing Trotter parameters at each step.
Analogous to `interaction_scan_map_to_state` in `ed_optimization.jl`.

# Growing from a smaller ansatz (`grow_from_num_exponentials`)
When increasing `num_exponentials` (via `instructions["num_exponentials"]`) beyond what was
previously optimized, pass `grow_from_num_exponentials` (the old, smaller layer count) and
`grow_from_save_name` (the `build_save_name_prefix(...)` prefix those older, per-`u_idx`
`\$(save_folder)/\$(grow_from_save_name)_u_\$(u_idx).jld2"` files were saved under) to bootstrap
the new (later) layers from the existing optimized (earlier) ones instead of starting from
scratch. [`grow_coefficients`](@ref) keeps the loaded coefficients as the first
`grow_from_num_exponentials` layers and zero-initializes the rest for optimization. This only
takes effect where there is no already-resumable file at the *current* `num_exponentials` for
that `u_idx` (an existing same-size file, e.g. from a partially-completed run, always takes
precedence). Two ways of seeding across the `u_range` are supported via `grow_mode`:
- `:chain` (default): grow only once, from `grow_from_save_name`'s file at the first `u_idx` in
  `instructions["u_range"]`. Every subsequent `u_idx` warm-starts from the *previous* `u_idx`'s
  just-optimized (already-grown) coefficients, exactly like the normal scan behavior.
- `:per_u`: grow independently at *every* `u_idx`, always reloading `grow_from_save_name`'s file
  for that same `u_idx` rather than chaining from the neighboring `u_idx`'s grown result.

# Pruning to a target fidelity (`active_indices`)
Pass `active_indices` to restrict optimization to that subset of the parameter vector's
indices (see [`embed_active_params`](@ref)/[`optimize_unitary`](@ref)); `target_fidelity` and
`pruning_threshold` are accepted purely to be recorded alongside the saved coefficients for
later inspection. Since `u_indices` is expected to hold a single index in this mode, any
loaded/warm-start coefficients are reduced to `active_indices` before being handed to
`optimize_unitary`.
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
    antihermitian::Bool=get(instructions, "antihermitian", false),
    use_gpu::Bool=false,
    datatype::Type{<:Number}=ComplexF64,
    metric_functions::Dict{String,Function}=Dict{String,Function}(),
    grow_from_num_exponentials::Union{Int,Nothing}=nothing,
    grow_from_save_name::Union{String,Nothing}=nothing,
    grow_mode::Symbol=:chain,
    active_indices::Union{Nothing,AbstractVector{Int}}=nothing,
    target_fidelity::Union{Nothing,Float64}=nothing,
    pruning_threshold::Union{Nothing,Float64}=nothing
)
    # instructions["u_range"] should be a range of indices, e.g., 1:10
    # instructions["starting state"] should define the fixed reference state (state1)
    instructions["antihermitian"] = antihermitian

    if !isnothing(grow_from_num_exponentials)
        if isnothing(grow_from_save_name)
            error("grow_from_save_name must be provided when grow_from_num_exponentials is set")
        end
        if isnothing(save_folder)
            error("save_folder must be provided when grow_from_num_exponentials is set (the grow-from files are looked up under it)")
        end
        if grow_mode ∉ (:chain, :per_u)
            error("Invalid grow_mode: $grow_mode. Valid options are :chain, :per_u.")
        end
    end

    data_dict = Dict{String,Any}("norm1_metrics" => [], "norm2_metrics" => [],
        "loss_metrics" => [], "labels" => [], "loss_std_metrics" => [], "all_matrices" => [],
        "coefficients" => [], "coefficient_labels" => nothing, "param_mapping" => nothing, "parities" => nothing)

    loaded_dict = nothing
    if haskey(instructions, "load_file") && isfile(instructions["load_file"])
        loaded_dict = JLD2.load(instructions["load_file"])["dict"]
        current_coeffs = loaded_dict["coefficients"]
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

    has_prepended_ref = !isnothing(u_vals) && (degen_rm_U isa AbstractMatrix) && (size(degen_rm_U, 1) == length(u_vals) + 1)
    target_state_idx(idx) = has_prepended_ref ? idx + 1 : idx

    num_gates = length(gates)
    grown_once = false

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
            degen_rm_U[target_state_idx(u_idx), :]
        else
            degen_rm_U[target_state_idx(u_idx)]
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

        # Check if loaded file is for the current U index (resuming/continuing optimization)
        is_current_u_resume = !isnothing(loaded_dict) && (
            (haskey(loaded_dict, "u_idx") && loaded_dict["u_idx"] == u_idx) ||
            (length(u_indices) == 1 && haskey(instructions, "load_file"))
        )

        init_history = Float64[]
        loaded_m = nothing
        if is_current_u_resume && haskey(loaded_dict, "metrics")
            loaded_m = loaded_dict["metrics"]
            if haskey(loaded_m, "optimization_losses") && !isempty(loaded_m["optimization_losses"])
                init_history = copy(loaded_m["optimization_losses"][1])
                println("  Resuming from existing optimization history ($(length(init_history)) prior iterations)")
            end
        end

        # Bootstrap a larger ansatz from an existing smaller one (see docstring above for
        # :chain vs :per_u). Only applies when there is no already-resumable file at the
        # *current* num_exponentials for this u_idx (is_current_u_resume takes precedence).
        if !is_current_u_resume && !isnothing(grow_from_num_exponentials) &&
           (grow_mode == :per_u || (grow_mode == :chain && !grown_once))
            grow_file = joinpath(save_folder, "$(grow_from_save_name)_u_$(u_idx).jld2")
            if isfile(grow_file)
                old_coeffs = JLD2.load(grow_file)["dict"]["coefficients"]
                println("  Growing initial coefficients from num_exponentials=$(grow_from_num_exponentials) to $(num_exponentials) using $grow_file")
                current_coeffs = grow_coefficients(old_coeffs, grow_from_num_exponentials, num_exponentials, num_gates)
                grown_once = true
            else
                @warn "grow_from_num_exponentials set but no file found for u_idx=$u_idx: $grow_file. Falling back to default initialization."
            end
        end

        # Reduce the warm-start vector to match optimize_unitary's active-parameter dimension;
        # current_coeffs itself stays full-length (that is the schema saved/loaded elsewhere).
        reduced_initial_coeffs = (!isnothing(active_indices) && !isnothing(current_coeffs)) ? current_coeffs[active_indices] : current_coeffs

        # Ensure valid datatype (Hermitian matrices require complex representation)
        effective_datatype = if !antihermitian && datatype <: Real
            @warn "Hermitian optimization requires complex arithmetic. Overriding datatype $datatype to ComplexF64."
            ComplexF64
        else
            datatype
        end

        A_opt, final_loss, metrics = optimize_unitary(
            gates, tau_terms, state1, opt_target, basis, N;
            loss_type=loss_type,
            H=H,
            state2=state2,
            num_exponentials=num_exponentials,
            active_indices=active_indices,
            maxiters=maxiters,
            optimizer=optimizer,
            perturb_optimization=perturb_optimization,
            initialization_samples=initialization_samples,
            multi_start_samples=multi_start_samples,
            multi_start_iters=multi_start_iters,
            initial_coefficients=reduced_initial_coeffs,
            initial_history=init_history,
            loaded_metrics=loaded_m,
            antihermitian=antihermitian,
            use_gpu=use_gpu,
            datatype=effective_datatype,
            metric_functions=metric_functions
        )

        if use_gpu && _has_cuda()
            _get_cuda().reclaim()
        end

        current_coeffs = A_opt
        loaded_dict = nothing # Consume loaded dictionary so subsequent U indices start fresh

        # Store results for this U
        push!(data_dict["norm1_metrics"], [norm(A_opt, 1)])
        push!(data_dict["norm2_metrics"], [norm(A_opt, 2)])
        push!(data_dict["coefficients"], A_opt)
        push!(data_dict["loss_metrics"], final_loss)

        # Calculate comparison metrics
        state1_prep, _ = (effective_datatype <: Real) ? strip_global_phase(state1) : (state1, 1.0)
        state2_prep, _ = (effective_datatype <: Real) ? strip_global_phase(state2) : (state2, 1.0)
        ref_evolved = apply_unitary(A_opt, gates, state1_prep, basis, N, num_exponentials; antihermitian=antihermitian, use_gpu=use_gpu, datatype=effective_datatype)
        ref_evolved_cpu = Array(ref_evolved)
        H_eval = if !isnothing(H_hopping) && !isnothing(H_interaction) && !isnothing(target_u)
            H_hopping + target_u * H_interaction
        else
            nothing
        end
        ed_energy = !isnothing(H_eval) ? real(dot(state2_prep, H_eval * state2_prep)) : NaN
        trotter_energy = !isnothing(H_eval) ? real(dot(ref_evolved_cpu, H_eval * ref_evolved_cpu)) : NaN
        overlap = abs2(dot(state2_prep, ref_evolved_cpu))

        println("  Optimization Complete:")
        println("    Final Loss ($loss_type): $final_loss")
        if haskey(metrics, "convergence_info") && !isempty(metrics["convergence_info"])
            latest_stages = metrics["convergence_info"][end]
            for info in latest_stages
                println("    Stopping Reason (Stage $(info["stage"]) - $(info["optimizer"])): $(info["primary_reason"]) (Iterations: $(info["iterations"]), Final |g|: $(info["g_residual"]))")
            end
        end
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
            if !isnothing(active_indices)
                iter_dict["active_indices"] = active_indices
                iter_dict["target_fidelity"] = target_fidelity
                iter_dict["pruning_threshold"] = pruning_threshold
            end
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
            "ending state" => Dict("level" => get(instructions, "starting level", 1), "U index" => u_idx))
        )
    end

    return data_dict
end
