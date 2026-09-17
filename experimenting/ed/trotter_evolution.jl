#=
trotter_evolution.jl

State evolution and strided rematerialization checkpointing routines for Trotter circuits.
=#

"""
    StridedCheckpoints{DevVec}

Container for strided state checkpoints stored on GPU VRAM for rematerialization.
- `checkpoints`: Array of GPU device vectors spaced by `stride` steps (`checkpoints[1]` is step 0, `checkpoints[end]` is step M).
- `stride`: Step stride K between checkpoints.
- `total_gates`: Total number of gates M.
"""
struct StridedCheckpoints{DevVec}
    checkpoints::Vector{DevVec}
    stride::Int
    total_gates::Int
end

Base.length(sc::StridedCheckpoints) = length(sc.checkpoints)
last_checkpoint(sc::StridedCheckpoints) = sc.checkpoints[end]
last_checkpoint(v::AbstractVector) = v[end]

"""
    determine_checkpoint_stride(M, basis_len, datatype) -> K

Automatically determine the optimal checkpoint stride K based on available GPU VRAM.
Queries `CUDA.memory_info()` to select K=1 (all checkpoints on VRAM) if memory allows,
or computes the optimal rematerialization stride K to prevent OOM without any PCIe bus transfers.
"""
function determine_checkpoint_stride(M::Int, basis_len::Int, datatype::Type{<:Number})
    if !_has_cuda()
        return 1
    end

    CUDA_mod = _get_cuda()
    free_bytes, total_bytes = CUDA_mod.memory_info()
    bytes_per_vec = basis_len * sizeof(datatype)

    # Reserve 4 GiB safety buffer for CUSPARSE, local scratch, and Zygote runtime
    safety_buffer = 4.0 * (1024^3)
    usable_bytes = max(0.0, free_bytes - safety_buffer) * 0.75

    max_checkpoints = floor(Int, usable_bytes / bytes_per_vec)

    if max_checkpoints >= (M + 1)
        # Everything fits safely in VRAM at stride 1
        return 1
    elseif max_checkpoints <= 4
        # Highly constrained VRAM: use square root heuristic
        return max(1, ceil(Int, sqrt(M)))
    else
        # Choose minimum K such that ceil(M / K) + 4 <= max_checkpoints
        k_candidate = max(2, ceil(Int, M / (max_checkpoints - 4)))
        return k_candidate
    end
end

"""
    apply_unitary_checkpoints(A, gates, ref, basis, N, num_exponentials; ...) -> phis

Evolves the state `ref` forward through all parameters `A` and returns strided checkpoints on GPU.
Automatically calculates optimal stride K to prevent GPU OOM.
"""
function apply_unitary_checkpoints(A::AbstractArray, gates, ref::AbstractArray, basis, N::Int, num_exponentials::Int;
    antihermitian::Bool=false, use_gpu::Bool=false, datatype::Type{<:Number}=ComplexF64,
    stream_tau::Union{Nothing,Bool}=nothing)
    
    P = num_exponentials
    num_gates = length(gates)
    M = P * num_gates

    if use_gpu && _has_cuda()
        CUDA_mod = _get_cuda()
        gpu_ops = get_gpu_gate_ops(gates, N, basis; antihermitian=antihermitian, datatype=datatype, stream_tau=stream_tau)
        ref_dev = to_device_vector(ref, use_gpu, datatype)
        d = length(basis)

        K = determine_checkpoint_stride(M, d, datatype)

        num_ckpts = cld(M, K) + 1
        DevVec = typeof(ref_dev)
        ckpts = Vector{DevVec}(undef, num_ckpts)
        ckpts[1] = copy(ref_dev)

        v_curr = copy(ref_dev)
        v_next = similar(ref_dev)

        ckpt_idx = 2
        curr = 1
        for l in 1:P
            for param_idx in 1:num_gates
                a = Float64(A[curr])
                gpu_apply_gate_exp!(v_next, v_curr, gpu_ops, param_idx, a; antihermitian=antihermitian, inverse=false)
                v_curr, v_next = v_next, v_curr

                if (curr % K == 0) || (curr == M)
                    ckpts[ckpt_idx] = copy(v_curr)
                    ckpt_idx += 1
                end
                curr += 1
            end
        end

        return StridedCheckpoints{DevVec}(ckpts, K, M)
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
    backward_adjoint_propagation(A, gates, tau_terms, phis, init_adjoint_state, basis, N, num_exponentials; ...) -> grad_A

Propagates the adjoint state backward using local GPU rematerialization between strided checkpoints.
"""
function backward_adjoint_propagation(A::AbstractArray, gates, tau_terms, phis::Union{Vector, StridedCheckpoints}, init_adjoint_state::AbstractVector, basis, N::Int, num_exponentials::Int;
    antihermitian::Bool=false, use_gpu::Bool=false, datatype::Type{<:Number}=ComplexF64,
    stream_tau::Union{Nothing,Bool}=nothing)
    
    P = num_exponentials
    num_gates = length(gates)
    M = P * num_gates
    grad_A = Vector{Float64}(undef, M)

    if use_gpu && _has_cuda()
        CUDA_mod = _get_cuda()
        gpu_ops = get_gpu_gate_ops(gates, N, basis; antihermitian=antihermitian, datatype=datatype, stream_tau=stream_tau)
        adj_curr = to_device_vector(init_adjoint_state, use_gpu, datatype)
        adj_curr = (adj_curr === init_adjoint_state) ? copy(adj_curr) : adj_curr
        adj_next = similar(adj_curr)

        if phis isa StridedCheckpoints
            K = phis.stride
            num_blocks = cld(M, K)

            local_phis = [similar(adj_curr) for _ in 1:(K + 1)]

            for block in num_blocks:-1:1
                start_step = (block - 1) * K
                end_step = min(block * K, M)
                block_len = end_step - start_step

                # 1. Rematerialize forward states inside this block directly in fast GPU VRAM
                local_phis[1] .= phis.checkpoints[block]
                for s in 1:block_len
                    curr_step = start_step + s
                    layer = cld(curr_step, num_gates)
                    param_idx = (curr_step - 1) % num_gates + 1
                    a = Float64(A[curr_step])
                    gpu_apply_gate_exp!(local_phis[s+1], local_phis[s], gpu_ops, param_idx, a; antihermitian=antihermitian, inverse=false)
                end

                # 2. Backpropagate adjoint state within this block
                for s in block_len:-1:1
                    curr_step = start_step + s
                    layer = cld(curr_step, num_gates)
                    param_idx = (curr_step - 1) % num_gates + 1
                    a = Float64(A[curr_step])

                    phi_curr = local_phis[s+1]

                    if gpu_ops.is_diag[param_idx] && antihermitian
                        grad_A[curr_step] = 0.0
                        tau_mat = nothing
                    else
                        tau_mat = _get_gpu_tau_mat(gpu_ops, param_idx)
                        CUDA_mod.CUSPARSE.mv!('N', eltype(gpu_ops.w1)(1.0), tau_mat, phi_curr, eltype(gpu_ops.w1)(0.0), gpu_ops.w1, 'O')
                        dot_val = dot(adj_curr, gpu_ops.w1)
                        if antihermitian
                            grad_A[curr_step] = -real(dot_val)
                        else
                            grad_A[curr_step] = imag(dot_val)
                        end
                    end

                    gpu_apply_gate_exp!(adj_next, adj_curr, gpu_ops, param_idx, a; antihermitian=antihermitian, inverse=true, tau=tau_mat)
                    adj_curr, adj_next = adj_next, adj_curr
                end
            end
        else
            for curr in M:-1:1
                layer = cld(curr, num_gates)
                param_idx = (curr - 1) % num_gates + 1
                a = Float64(A[curr])

                phi_curr = phis[curr+1]

                if gpu_ops.is_diag[param_idx] && antihermitian
                    grad_A[curr] = 0.0
                    tau_mat = nothing
                else
                    tau_mat = _get_gpu_tau_mat(gpu_ops, param_idx)
                    CUDA_mod.CUSPARSE.mv!('N', eltype(gpu_ops.w1)(1.0), tau_mat, phi_curr, eltype(gpu_ops.w1)(0.0), gpu_ops.w1, 'O')
                    dot_val = dot(adj_curr, gpu_ops.w1)
                    if antihermitian
                        grad_A[curr] = -real(dot_val)
                    else
                        grad_A[curr] = imag(dot_val)
                    end
                end

                gpu_apply_gate_exp!(adj_next, adj_curr, gpu_ops, param_idx, a; antihermitian=antihermitian, inverse=true, tau=tau_mat)
                adj_curr, adj_next = adj_next, adj_curr
            end
        end

        return grad_A
    else
        adjoint_state = (eltype(init_adjoint_state) == datatype) ? copy(init_adjoint_state) : datatype.(init_adjoint_state)
        curr = M
        for l in P:-1:1
            coefs = A[((l-1)*num_gates+1):(l*num_gates)]
            ops_inv = TamFermion.fgateToExpSector(gates, -coefs, N, basis; antihermitian=antihermitian)

            for param_idx in num_gates:-1:1
                op_inv = ops_inv[param_idx]
                tau_term = tau_terms[param_idx]

                phi_vec = (phis isa StridedCheckpoints) ? phis.checkpoints[curr+1] : phis[curr+1]
                dot_val = dot(adjoint_state, tau_term * phi_vec)
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

"""
    apply_unitary(A, gates, ref, basis, N, num_exponentials; ...) -> evolved_state

Evolve the reference state `ref` through all parameter layers without storing intermediate checkpoints.
"""
function apply_unitary(A::AbstractArray, gates, ref::AbstractArray, basis, N::Int, num_exponentials::Int;
    antihermitian::Bool=false, use_gpu::Bool=false, datatype::Type{<:Number}=ComplexF64,
    param_map::Union{Nothing,AbstractVector{Int}}=nothing,
    stream_tau::Union{Nothing,Bool}=nothing)
    A = expand_shared_coefficients(A, param_map, length(gates), num_exponentials)
    P = num_exponentials
    num_gates = length(gates)
    M = P * num_gates

    if use_gpu && _has_cuda()
        gpu_ops = get_gpu_gate_ops(gates, N, basis; antihermitian=antihermitian, datatype=datatype, stream_tau=stream_tau)
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
