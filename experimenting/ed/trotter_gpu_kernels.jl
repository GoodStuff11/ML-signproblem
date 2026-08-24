#=
trotter_gpu_kernels.jl

GPU gate operations and low-level matrix-vector acceleration for Trotter optimization.
=#

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
    _get_cuda()

Retrieve the active CUDA module safely across parent modules or Main.
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

"""
    _has_cuda() -> Bool

Check whether CUDA is loaded and a functional GPU is available.
"""
function _has_cuda()
    c = _get_cuda()
    return c !== nothing && c.has_cuda_gpu()
end

"""
    GpuGateOps

Precomputed GPU data structures for ultra-fast gate exponentials and matrix-vector operations.
"""
struct GpuGateOps
    tau_cpu::Vector{Any}
    tau_dev::Vector{Any}
    is_diag::Vector{Bool}
    sign0_val::Vector{Float64}
    w1::Any
    w2::Any
    pre_cached_all::Bool
end

const _GPU_GATE_OPS_CACHE = Dict{UInt64, GpuGateOps}()

"""
    build_direct_sparse_tau(g, N, basis, sortOrder, spec_mask; antihermitian=false)

Construct direct sparse representation of a single gate term for GPU acceleration.
"""
function build_direct_sparse_tau(g::TamFermion.FGate, N::Int, basis::AbstractVector{<:Integer}, sortOrder, spec_mask; antihermitian::Bool=false)
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
        if j <= d && sorted_basis[j] == t
            itgtI[k] = sortOrder[j]
        else
            throw(ArgumentError("The given gate maps states out of the provided basis."))
        end
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

"""
    get_gpu_gate_ops(gates, N, basis; antihermitian=false, datatype=ComplexF64) -> GpuGateOps

Retrieve or build precomputed GPU gate operators for rapid matrix-vector gate applications.
"""
function get_gpu_gate_ops(gates, N::Int, basis::AbstractVector{<:Integer}; antihermitian::Bool=false, datatype::Type{<:Number}=ComplexF64)
    CUDA_mod = _get_cuda()
    if CUDA_mod === nothing
        error("CUDA not loaded or available")
    end

    cache_key = hash((length(gates), N, length(basis), antihermitian, datatype))
    if haskey(_GPU_GATE_OPS_CACHE, cache_key)
        return _GPU_GATE_OPS_CACHE[cache_key]
    end

    num_gates = length(gates)
    d = length(basis)
    tau_cpu = Vector{Any}(undef, num_gates)
    tau_dev = Vector{Any}(undef, num_gates)
    is_diag = Vector{Bool}(undef, num_gates)
    sign0_val = Vector{Float64}(undef, num_gates)

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

    # Pre-cache all tau matrices on GPU only if total colptr memory is small (< 4 GB)
    colptr_bytes_total = num_gates * (d + 1) * 4
    pre_cache_all = colptr_bytes_total < 4 * 1024^3

    for (k, g) in enumerate(gates)
        sp_mat, diag_flag, s0 = build_direct_sparse_tau(g, N, basis, sortOrder, spec_masks[k]; antihermitian=antihermitian)
        is_diag[k] = diag_flag
        sign0_val[k] = Float64(s0)

        if diag_flag && antihermitian
            tau_cpu[k] = nothing
            tau_dev[k] = nothing
        else
            sp_typed = SparseMatrixCSC{datatype, Int32}(sp_mat)
            tau_cpu[k] = sp_typed
            if pre_cache_all
                tau_dev[k] = CUDA_mod.CUSPARSE.CuSparseMatrixCSC(sp_typed)
            else
                tau_dev[k] = nothing
            end
        end
    end

    w1 = CUDA_mod.zeros(datatype, d)
    w2 = CUDA_mod.zeros(datatype, d)

    ops = GpuGateOps(tau_cpu, tau_dev, is_diag, sign0_val, w1, w2, pre_cache_all)
    _GPU_GATE_OPS_CACHE[cache_key] = ops
    return ops
end

"""
    _get_gpu_tau_mat(gpu_ops, k)

Retrieve the sparse tau operator for gate `k` from GPU cache or stream from host.
"""
function _get_gpu_tau_mat(gpu_ops::GpuGateOps, k::Int)
    if gpu_ops.tau_dev[k] !== nothing
        return gpu_ops.tau_dev[k]
    end
    if gpu_ops.tau_cpu[k] === nothing
        return nothing
    end
    CUDA_mod = _get_cuda()
    return CUDA_mod.CUSPARSE.CuSparseMatrixCSC(gpu_ops.tau_cpu[k])
end

"""
    gpu_apply_gate_exp!(v_out, v_in, gpu_ops, k, a; antihermitian=false, inverse=false)

Apply the exponential of gate `k` with parameter `a` to device vector `v_in`, storing result in `v_out`.
"""
function gpu_apply_gate_exp!(v_out::AbstractVector, v_in::AbstractVector, gpu_ops::GpuGateOps, k::Int, a::Float64; antihermitian::Bool=false, inverse::Bool=false)
    CUDA_mod = _get_cuda()
    a_val = inverse ? -a : a

    if gpu_ops.is_diag[k]
        if antihermitian
            copyto!(v_out, v_in)
        else
            sign0 = gpu_ops.sign0_val[k]
            phase_val = exp(2im * a_val * sign0)
            tau = gpu_ops.tau_dev[k]
            CUDA_mod.CUSPARSE.mv!('N', eltype(v_out)(1.0), tau, v_in, eltype(v_out)(0.0), gpu_ops.w1, 'O')
            coeff = eltype(v_out)((phase_val - 1.0) / (2.0 * sign0))
            v_out .= v_in .+ coeff .* gpu_ops.w1
        end
        return v_out
    end

    tau = gpu_ops.tau_dev[k]
    ca = cos(a_val)
    sa = sin(a_val)

    # w1 = tau * v_in
    CUDA_mod.CUSPARSE.mv!('N', eltype(v_out)(1.0), tau, v_in, eltype(v_out)(0.0), gpu_ops.w1, 'O')
    # w2 = tau * w1 = tau^2 * v_in
    CUDA_mod.CUSPARSE.mv!('N', eltype(v_out)(1.0), tau, gpu_ops.w1, eltype(v_out)(0.0), gpu_ops.w2, 'O')

    if antihermitian
        coeff_tau2 = eltype(v_out)(1.0 - ca)
        coeff_sin = eltype(v_out)(sa)
    else
        coeff_tau2 = eltype(v_out)(ca - 1.0)
        coeff_sin = eltype(v_out)(im * sa)
    end

    v_out .= v_in .+ coeff_tau2 .* gpu_ops.w2 .+ coeff_sin .* gpu_ops.w1
    return v_out
end

"""
    to_device_vector(v, use_gpu, datatype)

Transfer vector `v` to GPU device memory if `use_gpu=true`, converted to `datatype`.
"""
function to_device_vector(v::AbstractVector, use_gpu::Bool, datatype::Type{<:Number})
    v_typed = if datatype <: Real && eltype(v) <: Complex
        datatype.(real.(strip_global_phase(v)[1]))
    else
        datatype.(v)
    end
    if use_gpu && _has_cuda()
        CUDA_mod = _get_cuda()
        if v isa CUDA_mod.CuArray && eltype(v) == datatype
            return v
        else
            return CUDA_mod.CuArray(v_typed)
        end
    else
        return (eltype(v) == datatype) ? v : v_typed
    end
end

"""
    to_device_ops(ops, use_gpu, datatype)

Transfer sparse operators to GPU CuSparseMatrixCSC if `use_gpu=true`.
"""
function to_device_ops(ops::Vector{<:AbstractMatrix}, use_gpu::Bool, datatype::Type{<:Number})
    if use_gpu && _has_cuda()
        CUDA_mod = _get_cuda()
        return [CUDA_mod.CUSPARSE.CuSparseMatrixCSC(SparseMatrixCSC{datatype, Int32}(sparse(op))) for op in ops]
    end
    return ops
end
