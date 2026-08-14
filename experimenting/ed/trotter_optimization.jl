module TrotterOptimization

using ChainRulesCore
using LinearAlgebra
using SparseArrays
using Zygote
using Optimization
using OptimizationOptimJL
using Optim
using OptimizationOptimisers
import ..TamFermion
using ..Trotter: @safe_threads
using JLD2
using Statistics

export adjoint_loss, energy_loss, optimize_unitary, interaction_scan_map_to_state, extract_convergence_info, grow_coefficients

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

# ═══════════════════════════════════════════════════════════════════════
# SHARED HELPERS FOR FORWARD AND BACKWARD SWEEPS & DEVICE CONVERSION
# ═══════════════════════════════════════════════════════════════════════

"""
    strip_global_phase(v::AbstractVector{<:Complex}) -> (v_stripped, phase)

Strip the global complex phase from vector `v` by dividing by the phase of
the component with the largest magnitude. Returns `(real.(v_stripped), phase)`.
"""
function strip_global_phase(v::AbstractVector{<:Complex})
    idx = argmax(abs.(v))
    val = v[idx]
    phase = abs(val) > 0 ? val / abs(val) : ComplexF64(1.0)
    v_stripped = v .* conj(phase)
    return real.(v_stripped), phase
end

function strip_global_phase(v::AbstractVector{<:Real})
    return v, 1.0
end

"""
    GpuGateOps

Precomputed GPU data structures for ultra-fast gate exponentials and matrix-vector operations.
"""
function _get_cuda()
    if @isdefined(CUDA)
        return CUDA
    elseif isdefined(parentmodule(@__MODULE__), :CUDA)
        return getfield(parentmodule(@__MODULE__), :CUDA)
    elseif isdefined(Main, :CUDA)
        return getfield(Main, :CUDA)
    end
    return nothing
end

function _has_cuda()
    c = _get_cuda()
    return c !== nothing && c.has_cuda_gpu()
end

"""
    GpuGateOps

Precomputed GPU data structures for ultra-fast gate exponentials and matrix-vector operations.
"""
struct GpuGateOps
    tau_dev::Vector{Any}
    is_diag::Vector{Bool}
    sign0_val::Vector{Float64}
    w1::Any
    w2::Any
end

const _GPU_GATE_OPS_CACHE = Dict{UInt64, GpuGateOps}()

function build_direct_sparse_tau(g, N::Int, basis::AbstractVector{<:Integer}, sortOrder, spec_mask; antihermitian::Bool=false)
    d = length(basis)
    nbits = 2N
    s_I = UInt64(g.cre_up) | (UInt64(g.cre_dn) << N)
    s_J = UInt64(g.ann_up) | (UInt64(g.ann_dn) << N)
    basis64 = UInt64.(basis)

    s_Ip = s_I & ~s_J
    s_Jp = s_J & ~s_I
    Delta = s_I ⊻ s_J
    s_IJ = s_I | s_J
    p = count_ones(s_J)

    sign0 = (div(p * (p - 1), 2) % 2 == 0) ? 1.0 : -1.0
    sgn_ref = TamFermion._jw_sign_ref(s_I, s_J, nbits)
    mask = spec_mask !== nothing ? spec_mask : TamFermion._odd_spectator_mask(Delta, s_IJ, nbits)

    if s_I == s_J
        if antihermitian
            return sparse(Int[], Int[], ComplexF64[], d, d), true, sign0
        else
            idxc = findall((basis64 .& s_I) .== s_I)
            val = 2.0 * sign0
            return sparse(idxc, idxc, fill(ComplexF64(val), length(idxc)), d, d), true, sign0
        end
    end

    srcJ_mask = ((basis64 .& s_J) .== s_J) .& ((basis64 .& s_Ip) .== UInt64(0))
    isrcJ = findall(srcJ_mask)
    srcJ = basis64[isrcJ]

    sorted_basis = basis64[sortOrder]

    itgtI = Vector{Int}(undef, length(isrcJ))
    for k in eachindex(isrcJ)
        s = srcJ[k]
        t = s ⊻ Delta
        j = searchsortedfirst(sorted_basis, t)
        itgtI[k] = sortOrder[j]
    end

    signs = Vector{Float64}(undef, length(isrcJ))
    for k in eachindex(isrcJ)
        s = srcJ[k]
        spec_parity = count_ones(s & mask) & 1
        signs[k] = sgn_ref * (spec_parity == 1 ? -1.0 : 1.0)
    end

    I_vec = vcat(isrcJ, itgtI)
    J_vec = vcat(itgtI, isrcJ)
    V_vec = antihermitian ? vcat(ComplexF64.(-signs), ComplexF64.(signs)) : vcat(ComplexF64.(signs), ComplexF64.(signs))

    return sparse(I_vec, J_vec, V_vec, d, d), false, sign0
end

function _get_gpu_tau_mat(gpu_ops::GpuGateOps, k::Int)
    mat = gpu_ops.tau_dev[k]
    if mat isa SparseMatrixCSC
        CUDA_mod = _get_cuda()
        return CUDA_mod.CUSPARSE.CuSparseMatrixCSC(mat)
    else
        return mat
    end
end

function prepare_gpu_gate_ops(gates, N::Int, basis::AbstractVector{<:Integer}; antihermitian::Bool=false, datatype::Type{<:Number}=ComplexF64, stream_tau::Bool=true)
    CUDA_mod = _get_cuda()
    num_gates = length(gates)
    tau_dev = Vector{Any}(undef, num_gates)
    is_diag = Vector{Bool}(undef, num_gates)
    sign0_val = Vector{Float64}(undef, num_gates)

    if datatype <: Real && !antihermitian
        error("Real data type ($datatype) requires antihermitian=true.")
    end

    sortOrder = sortperm(UInt64.(basis))
    nbits = 2N
    spec_masks = Vector{UInt64}(undef, num_gates)
    for k in 1:num_gates
        g = gates[k]
        s_I = UInt64(g.cre_up) | (UInt64(g.cre_dn) << N)
        s_J = UInt64(g.ann_up) | (UInt64(g.ann_dn) << N)
        Delta = s_I ⊻ s_J
        s_IJ = s_I | s_J
        spec_masks[k] = TamFermion._odd_spectator_mask(Delta, s_IJ, nbits)
    end

    for k in 1:num_gates
        sp_mat, is_d, sign0 = build_direct_sparse_tau(gates[k], N, basis, sortOrder, spec_masks[k]; antihermitian=antihermitian)
        is_diag[k] = is_d
        sign0_val[k] = sign0
        sp_mat_typed = datatype <: Real ? SparseMatrixCSC{datatype, Int32}(real.(sp_mat)) : SparseMatrixCSC{datatype, Int32}(sp_mat)
        if stream_tau
            tau_dev[k] = sp_mat_typed # Option B: store on CPU Host RAM to save VRAM
        else
            tau_dev[k] = CUDA_mod.CUSPARSE.CuSparseMatrixCSC(sp_mat_typed) # store on GPU VRAM
        end
    end
    d = length(basis)
    w1 = CUDA_mod.CuArray(zeros(datatype, d))
    w2 = CUDA_mod.CuArray(zeros(datatype, d))
    return GpuGateOps(tau_dev, is_diag, sign0_val, w1, w2)
end

function get_gpu_gate_ops(gates, N::Int, basis::AbstractVector{<:Integer}; antihermitian::Bool=false, datatype::Type{<:Number}=ComplexF64, stream_tau::Bool=true)
    key = hash((objectid(gates), objectid(basis), N, antihermitian, datatype, stream_tau))
    if haskey(_GPU_GATE_OPS_CACHE, key)
        return _GPU_GATE_OPS_CACHE[key]
    end
    gpu_ops = prepare_gpu_gate_ops(gates, N, basis; antihermitian=antihermitian, datatype=datatype, stream_tau=stream_tau)
    _GPU_GATE_OPS_CACHE[key] = gpu_ops
    return gpu_ops
end

function gpu_apply_gate_exp!(v_out, v_in, gpu_ops::GpuGateOps, k::Int, a::Float64; antihermitian::Bool=false, inverse::Bool=false)
    is_d = gpu_ops.is_diag[k]
    sign0 = gpu_ops.sign0_val[k]
    w1 = gpu_ops.w1
    w2 = gpu_ops.w2
    T = eltype(v_in)

    if is_d
        if antihermitian
            copyto!(v_out, v_in)
        else
            tau_mat = _get_gpu_tau_mat(gpu_ops, k)
            a_val = inverse ? -a : a
            phase_val = T(exp(2im * a_val * sign0))
            mul!(w1, tau_mat, v_in)
            v_out .= v_in .+ ((phase_val - T(1.0)) / T(2.0 * sign0)) .* w1
        end
    else
        tau_mat = _get_gpu_tau_mat(gpu_ops, k)
        a_val = inverse ? -a : a
        ca = cos(a_val)
        sa = sin(a_val)
        coeff_sin = T(antihermitian ? sa : 1im * sa)
        coeff_cos_m1 = T(antihermitian ? (1.0 - ca) : (ca - 1.0))

        mul!(w1, tau_mat, v_in)
        mul!(w2, tau_mat, w1)
        v_out .= v_in .+ coeff_cos_m1 .* w2 .+ coeff_sin .* w1
    end
    return v_out
end

"""
    to_device_vector(v, use_gpu::Bool)

Convert vector `v` to a GPU `CuVector` if `use_gpu` is true and CUDA is loaded/functional,
otherwise return `v`.
"""
function to_device_vector(v::AbstractVector, use_gpu::Bool, datatype::Type{<:Number}=eltype(v))
    v_typed = (eltype(v) == datatype) ? v : datatype.(v)
    if use_gpu && _has_cuda()
        CUDA_mod = _get_cuda()
        return v_typed isa CUDA_mod.CuArray ? v_typed : CUDA_mod.CuArray(v_typed)
    end
    return v_typed
end

"""
    to_device_ops(ops, use_gpu::Bool)

Convert a collection of operators (e.g. `LinearMap`s) to GPU sparse matrices `CuSparseMatrixCSC`
if `use_gpu` is true and CUDA is loaded/functional, otherwise return `ops` unchanged.
"""
function to_device_ops(ops::AbstractVector, use_gpu::Bool, datatype::Type{<:Number}=ComplexF64)
    if use_gpu && _has_cuda()
        CUDA_mod = _get_cuda()
        return [CUDA_mod.CUSPARSE.CuSparseMatrixCSC(SparseMatrixCSC{datatype, Int32}(sparse(op))) for op in ops]
    end
    return ops
end

"""
    apply_unitary_checkpoints(A, gates, ref, basis, N, num_exponentials) -> phis

Evolves the state `ref` forward through all parameters `A` and returns a list
of intermediate state checkpoints `phis`, where `phis[1]` is `ref` and `phis[end]`
is the fully evolved state. Supports GPU execution via `use_gpu=true`.
"""
function apply_unitary_checkpoints(A::AbstractArray, gates, ref::AbstractArray, basis, N::Int, num_exponentials::Int; antihermitian::Bool=false, use_gpu::Bool=false, datatype::Type{<:Number}=ComplexF64)
    P = num_exponentials
    num_gates = length(gates)
    M = P * num_gates

    if use_gpu && _has_cuda()
        CUDA_mod = _get_cuda()
        gpu_ops = get_gpu_gate_ops(gates, N, basis; antihermitian=antihermitian, datatype=datatype)
        ref_dev = to_device_vector(ref, use_gpu, datatype)
        phis = Vector{typeof(ref_dev)}(undef, M + 1)
        phis[1] = copy(ref_dev)
        curr = 1
        for l in 1:P
            for param_idx in 1:num_gates
                a = Float64(A[curr])
                phis[curr+1] = similar(ref_dev)
                gpu_apply_gate_exp!(phis[curr+1], phis[curr], gpu_ops, param_idx, a; antihermitian=antihermitian, inverse=false)
                curr += 1
            end
        end
        return phis
    else
        ref_dev = (eltype(ref) == datatype) ? ref : datatype.(ref)
        phis = Vector{typeof(ref_dev)}(undef, M + 1)
        phis[1] = ref_dev
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
end

"""
    backward_adjoint_propagation(A, gates, tau_terms, phis, init_adjoint_state, basis, N, num_exponentials) -> grad_A

Propagates the `adjoint_state` backward starting from `init_adjoint_state`,
computing the gradient of the parameters at each step using the forward state checkpoints `phis`.
Supports GPU execution via `use_gpu=true`.
"""
function backward_adjoint_propagation(A::AbstractArray, gates, tau_terms, phis::Vector, init_adjoint_state::AbstractVector, basis, N::Int, num_exponentials::Int; antihermitian::Bool=false, use_gpu::Bool=false, datatype::Type{<:Number}=ComplexF64)
    P = num_exponentials
    num_gates = length(gates)
    M = P * num_gates
    grad_A = Vector{Float64}(undef, M)

    if use_gpu && _has_cuda()
        CUDA_mod = _get_cuda()
        gpu_ops = get_gpu_gate_ops(gates, N, basis; antihermitian=antihermitian, datatype=datatype)
        adj_curr = copy(init_adjoint_state)
        ref_sample = phis[1]
        adj_next = similar(ref_sample)

        curr = M
        for l in P:-1:1
            for param_idx in num_gates:-1:1
                a = Float64(A[curr])
                tau_mat = _get_gpu_tau_mat(gpu_ops, param_idx)
                mul!(gpu_ops.w1, tau_mat, phis[curr+1])
                dot_val = dot(adj_curr, gpu_ops.w1)
                grad_A[curr] = antihermitian ? -real(dot_val) : imag(dot_val)

                gpu_apply_gate_exp!(adj_next, adj_curr, gpu_ops, param_idx, a; antihermitian=antihermitian, inverse=true)
                adj_curr, adj_next = adj_next, adj_curr
                curr -= 1
            end
        end
        CUDA_mod.synchronize()
        return grad_A
    else
        adjoint_state = copy(init_adjoint_state)
        curr = M
        for l in P:-1:1
            coefs = A[((l-1)*num_gates+1):(l*num_gates)]
            ops_inv = TamFermion.fgateToExpSector(gates, -coefs, N, basis; antihermitian=antihermitian)

            for param_idx in num_gates:-1:1
                op_inv = ops_inv[param_idx]
                tau_term = tau_terms[param_idx]

                dot_val = dot(adjoint_state, tau_term * phis[curr+1])
                if antihermitian
                    grad_A[curr] = -real(dot_val)
                else
                    grad_A[curr] = imag(dot_val)
                end

                adjoint_state = op_inv * adjoint_state
                curr -= 1
            end
        end
        return grad_A
    end
end

function apply_unitary(A::AbstractArray, gates, ref::AbstractArray, basis, N::Int, num_exponentials::Int; antihermitian::Bool=false, use_gpu::Bool=false, datatype::Type{<:Number}=ComplexF64)
    P = num_exponentials
    num_gates = length(gates)
    M = P * num_gates

    if use_gpu && _has_cuda()
        gpu_ops = get_gpu_gate_ops(gates, N, basis; antihermitian=antihermitian, datatype=datatype)
        ref_dev = to_device_vector(ref, use_gpu, datatype)
        v_curr = copy(ref_dev)
        v_next = similar(ref_dev)
        curr = 1
        for l in 1:P
            for param_idx in 1:num_gates
                a = Float64(A[curr])
                gpu_apply_gate_exp!(v_next, v_curr, gpu_ops, param_idx, a; antihermitian=antihermitian, inverse=false)
                v_curr, v_next = v_next, v_curr
                curr += 1
            end
        end
        return v_curr
    else
        ref_dev = (eltype(ref) == datatype) ? ref : datatype.(ref)
        v_curr = copy(ref_dev)
        for l in 1:P
            coefs = A[((l-1)*num_gates+1):(l*num_gates)]
            ops = TamFermion.fgateToExpSector(gates, coefs, N, basis; antihermitian=antihermitian)
            for op in ops
                v_curr = op * v_curr
            end
        end
        return v_curr
    end
end

# ═══════════════════════════════════════════════════════════════════════
# OVERLAP LOSS
# ═══════════════════════════════════════════════════════════════════════

"""
A : coefficients stored
gates : output of get_fermionic_gates
tau_terms : (not used unless gradient is computed)
ref :
target :
basis :
N : number of sites
"""
function adjoint_loss(A::AbstractArray, gates, tau_terms, ref::AbstractArray, target::AbstractArray, basis, N::Int; num_exponentials::Int=1, antihermitian::Bool=false, use_gpu::Bool=false, datatype::Type{<:Number}=ComplexF64)
    ref_evolved = apply_unitary(A, gates, ref, basis, N, num_exponentials; antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype)
    target_dev = to_device_vector(target, use_gpu, datatype)
    return 1 - abs2(dot(target_dev, ref_evolved))
end

function ChainRulesCore.rrule(::typeof(adjoint_loss), A::AbstractArray, gates, tau_terms, ref::AbstractArray, target::AbstractArray, basis, N::Int; num_exponentials::Int=1, antihermitian::Bool=false, use_gpu::Bool=false, datatype::Type{<:Number}=ComplexF64)
    t = @elapsed begin
        target_dev = to_device_vector(target, use_gpu, datatype)
        phis = apply_unitary_checkpoints(A, gates, ref, basis, N, num_exponentials; antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype)
        evolved_ref = phis[end]
        overlap = dot(target_dev, evolved_ref)
        loss = 1 - abs2(overlap)
        println("loss: $loss")
    end
    println("Forward time: $t")

    function adjoint_loss_pullback(y)
        t = @elapsed begin
            init_adjoint_state = (2 * overlap * conj(y)) * target_dev
            grad_A = backward_adjoint_propagation(A, gates, tau_terms, phis, init_adjoint_state, basis, N, num_exponentials; antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype)
        end
        println("Gradient time: $t")
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
    find_multi_start_initialization(f, optf, M::Int; kwargs...) -> (A_init, local_multistart_losses, local_best_start_idx, multistart_run)

Perform multi-start initialization by sampling `initialization_samples` random configurations of size `M`.
Evaluate the gradient of each configuration with function `f`, select the top `multi_start_samples` candidates based on gradient norm,
run quick optimization sweeps of at most `multi_start_iters` with `optf`, and return the best overall parameter configuration along with candidate loss trajectories.
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
    grow_coefficients(old_coeffs, old_num_exponentials, new_num_exponentials, num_gates) -> Vector{Float64}

Extend a Trotter coefficient vector that was optimized for `old_num_exponentials` layers into
one usable for a larger `new_num_exponentials`, so the newly-added (later) layers can be
optimized starting from an already-converged shorter ansatz instead of from scratch.

Coefficients are stored as `new_num_exponentials * num_gates` contiguous per-layer blocks
(layer `l`'s parameters live at `A[(l-1)*num_gates+1 : l*num_gates]`, matching
[`apply_unitary`](@ref) / [`apply_unitary_checkpoints`](@ref)), and layer 1 is applied first
(closest to the reference state). Growing therefore keeps `old_coeffs` unchanged as the first
`old_num_exponentials` (earlier) layers, and appends zeros for the new (later) layers, ready to
be optimized.

`new_num_exponentials` must be `>= old_num_exponentials`, and `length(old_coeffs)` must equal
`old_num_exponentials * num_gates`.
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

    f = (A, p=nothing) -> begin
        if loss_type == :overlap
            return adjoint_loss(A, gates, tau_terms, ref_prep, target_prep, basis, N; num_exponentials=num_exponentials, antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype)
        elseif loss_type == :energy
            return energy_loss(A, gates, tau_terms, target_prep, ref_prep, basis, N; num_exponentials=num_exponentials, antihermitian=antihermitian)
        else
            error("Unknown loss_type: $loss_type")
        end
    end

    optf = Optimization.OptimizationFunction(f, Optimization.AutoZygote())
    M = num_exponentials * length(gates)

    state2_vec = target isa AbstractVector ? target : state2
    H_mat = target isa AbstractMatrix ? target : H

    initial_loss = if loss_type == :overlap
        1.0 - abs2(dot(state2_vec, ref))
    elseif loss_type == :energy
        real(dot(ref, H_mat * ref))
    else
        error("Unknown loss_type: $loss_type")
    end

    metrics = Dict{String,Vector{Any}}()
    if !isnothing(loaded_metrics) && haskey(loaded_metrics, "loss") && !isempty(loaded_metrics["loss"])
        metrics["loss"] = copy(loaded_metrics["loss"])
    else
        metrics["loss"] = Float64[initial_loss]
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
        metrics["energy"] = Float64[!isnothing(H_mat) ? real(dot(ref, H_mat * ref)) : NaN]
    end
    if !isnothing(loaded_metrics) && haskey(loaded_metrics, "overlap") && !isempty(loaded_metrics["overlap"])
        metrics["overlap"] = copy(loaded_metrics["overlap"])
    elseif loss_type == :energy
        metrics["overlap"] = Float64[!isnothing(state2_vec) ? (1.0 - abs2(dot(state2_vec, ref))) : NaN]
    end
    for k in keys(metric_functions)
        metrics[k] = Any[]
    end

    println("Initial loss ($loss_type): $initial_loss")

    if loss_type == :overlap && initial_loss < 1e-15
        println("States are already equal")
        A_zero = zeros(Float64, M)
        return A_zero, initial_loss, metrics
    end

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
        final_overlap = !isnothing(target_prep) ? (1.0 - abs2(dot(target_prep, ref_evolved_cpu))) : NaN
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
    grow_mode::Symbol=:chain
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

        A_opt, final_loss, metrics = optimize_unitary(
            gates, tau_terms, state1, opt_target, basis, N;
            loss_type=loss_type,
            H=H,
            state2=state2,
            num_exponentials=num_exponentials,
            maxiters=maxiters,
            optimizer=optimizer,
            perturb_optimization=perturb_optimization,
            initialization_samples=initialization_samples,
            multi_start_samples=multi_start_samples,
            multi_start_iters=multi_start_iters,
            initial_coefficients=current_coeffs,
            initial_history=init_history,
            loaded_metrics=loaded_m,
            antihermitian=antihermitian,
            use_gpu=use_gpu,
            datatype=datatype,
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
        state1_prep, _ = (datatype <: Real) ? strip_global_phase(state1) : (state1, 1.0)
        state2_prep, _ = (datatype <: Real) ? strip_global_phase(state2) : (state2, 1.0)
        ref_evolved = apply_unitary(A_opt, gates, state1_prep, basis, N, num_exponentials; antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype)
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

end # module TrotterOptimization