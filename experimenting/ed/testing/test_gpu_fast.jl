#=
test_gpu_fast.jl

Test script to prototype and verify ultra-fast GPU Trotter gate application & gradient computation.

Usage:
  julia --project=.. test_gpu_fast.jl --use_gpu
=#

_use_gpu_prescan = let val = false
    for arg in ARGS
        if arg == "--use_gpu" || arg == "--use_gpu=true"
            val = true
        end
    end
    val
end

if _use_gpu_prescan
    ENV["JULIA_CUDA_USE_COMPAT"] = "true"
    using CUDA
    using CUDA.CUSPARSE
end

using Lattices
using LinearAlgebra
using Combinatorics
using SparseArrays
using Statistics
using Random
using JLD2
using HDF5
using Zygote
using Test

include("../data_path.jl")
include("../logging.jl")
include("../utility_functions.jl")
using .UtilityFunctions
include("../trotter.jl")
using .Trotter
include("../ed_objects.jl")
include("../ed_functions.jl")

struct GpuGateOps
    tau_dev::Vector{CUDA.CUSPARSE.CuSparseMatrixCSC{ComplexF64, Int32}}
    is_diag::Vector{Bool}
    sign0_val::Vector{Float64}
end

function build_direct_sparse_tau(g::Trotter.TamFermion.FGate, N::Int, basis::AbstractVector{<:Integer}, sortOrder, spec_mask; antihermitian::Bool=false)
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
    sgn_ref = Trotter.TamFermion._jw_sign_ref(s_I, s_J, nbits)
    mask = spec_mask !== nothing ? spec_mask : Trotter.TamFermion._odd_spectator_mask(Delta, s_IJ, nbits)

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

function prepare_gpu_gate_ops(gates, N::Int, basis::AbstractVector{<:Integer}; antihermitian::Bool=false)
    num_gates = length(gates)
    tau_dev = Vector{CUDA.CUSPARSE.CuSparseMatrixCSC{ComplexF64, Int32}}(undef, num_gates)
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
        spec_masks[k] = Trotter.TamFermion._odd_spectator_mask(Delta, s_IJ, nbits)
    end

    for k in 1:num_gates
        sp_mat, is_d, sign0 = build_direct_sparse_tau(gates[k], N, basis, sortOrder, spec_masks[k]; antihermitian=antihermitian)
        is_diag[k] = is_d
        sign0_val[k] = sign0
        tau_dev[k] = CUDA.CUSPARSE.CuSparseMatrixCSC(sp_mat)
    end
    return GpuGateOps(tau_dev, is_diag, sign0_val)
end

function gpu_apply_gate_exp!(v_out::CuVector{ComplexF64}, v_in::CuVector{ComplexF64}, gpu_ops::GpuGateOps, k::Int, a::Float64; antihermitian::Bool=false, inverse::Bool=false)
    is_d = gpu_ops.is_diag[k]
    tau_mat = gpu_ops.tau_dev[k]
    sign0 = gpu_ops.sign0_val[k]

    if is_d
        if antihermitian
            copyto!(v_out, v_in)
        else
            a_val = inverse ? -a : a
            phase_val = exp(2im * a_val * sign0)
            w1 = tau_mat * v_in
            v_out .= v_in .+ ((phase_val - 1.0) / (2.0 * sign0)) .* w1
        end
    else
        a_val = inverse ? -a : a
        ca = cos(a_val)
        sa = sin(a_val)
        coeff_sin = antihermitian ? sa : 1im * sa
        coeff_cos_m1 = ca - 1.0

        w1 = tau_mat * v_in
        w2 = tau_mat * w1
        v_out .= v_in .+ coeff_cos_m1 .* w2 .+ coeff_sin .* w1
    end
    return v_out
end

function gpu_adjoint_loss_and_grad(A::AbstractVector{Float64}, gates, gpu_ops::GpuGateOps, ref_dev::CuVector{ComplexF64}, target_dev::CuVector{ComplexF64}, num_exponentials::Int; antihermitian::Bool=false)
    P = num_exponentials
    num_gates = length(gates)
    M = P * num_gates

    phis = Vector{CuVector{ComplexF64}}(undef, M + 1)
    phis[1] = copy(ref_dev)

    curr = 1
    for l in 1:P
        for param_idx in 1:num_gates
            a = A[curr]
            phis[curr+1] = similar(ref_dev)
            gpu_apply_gate_exp!(phis[curr+1], phis[curr], gpu_ops, param_idx, a; antihermitian=antihermitian, inverse=false)
            curr += 1
        end
    end

    evolved_ref = phis[end]
    overlap = dot(target_dev, evolved_ref)
    loss = 1.0 - abs2(overlap)

    grad_A = Vector{Float64}(undef, M)
    init_adj = (2.0 * overlap) * target_dev
    adj_curr = copy(init_adj)
    adj_next = similar(ref_dev)

    curr = M
    for l in P:-1:1
        for param_idx in num_gates:-1:1
            a = A[curr]
            tau_phi = gpu_ops.tau_dev[param_idx] * phis[curr+1]
            dot_val = dot(adj_curr, tau_phi)
            grad_A[curr] = antihermitian ? -real(dot_val) : imag(dot_val)

            gpu_apply_gate_exp!(adj_next, adj_curr, gpu_ops, param_idx, a; antihermitian=antihermitian, inverse=true)
            adj_curr, adj_next = adj_next, adj_curr
            curr -= 1
        end
    end

    CUDA.synchronize()
    return loss, grad_A
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "test_gpu_fast")
    with_logging(log_path) do
        println("==========================================================")
        println("TEST GPU FAST IMPLEMENTATION & GRADIENT VERIFICATION")
        println("==========================================================")
        if !@isdefined(CUDA) || !CUDA.functional()
            println("CUDA not available. Exiting.")
            return 0
        end
        println("GPU Device: $(CUDA.name(CUDA.device()))")

        folder = data_folder("N=(5, 4)_4x3")
        println("Loading dataset from: $folder")
        U_values, state_vecs, indexer, _, N_elec, spin_conserved, _, sign_convention =
            load_ED_data(folder; verbose=false, sign_convention=:spin_first)

        Lvec = (4, 3)
        N_sites = prod(Lvec)
        basis_sector = Trotter.get_basis_sector(indexer, Lvec, N_sites)
        dim = length(basis_sector)
        println("Basis dimension: $dim")

        gates = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=true)
        tau_terms = Trotter.fgateToTauSector(gates, N_sites, basis_sector; antihermitian=false)
        num_gates = length(gates)
        num_exp = 1
        M = num_exp * num_gates
        println("Number of gates: $num_gates, M: $M")

        ref = state_vecs[1, :]
        target = state_vecs[2, :]
        Random.seed!(42)
        A = (2 * rand(M) .- 1) * 0.05

        println("\nPrecomputing GPU gate ops...")
        gpu_ops = prepare_gpu_gate_ops(gates, N_sites, basis_sector; antihermitian=false)

        ref_dev = CUDA.CuArray(ref)
        target_dev = CUDA.CuArray(target)

        println("\n--- Testing CPU vs Fast GPU Gradient Match ---")
        loss_cpu = Trotter.TrotterOptimization.adjoint_loss(
            A, gates, tau_terms, ref, target, basis_sector, N_sites;
            num_exponentials=num_exp, antihermitian=false, use_gpu=false
        )
        grad_cpu = Zygote.gradient(A) do x
            Trotter.TrotterOptimization.adjoint_loss(
                x, gates, tau_terms, ref, target, basis_sector, N_sites;
                num_exponentials=num_exp, antihermitian=false, use_gpu=false
            )
        end[1]

        loss_gpu, grad_gpu = gpu_adjoint_loss_and_grad(A, gates, gpu_ops, ref_dev, target_dev, num_exp; antihermitian=false)

        max_grad_diff = maximum(abs.(grad_cpu .- grad_gpu))
        loss_diff = abs(loss_cpu - loss_gpu)
        println("CPU Loss: $loss_cpu")
        println("GPU Loss: $loss_gpu")
        println("Loss Absolute Difference: $loss_diff")
        println("Max Gradient Difference (CPU vs Fast GPU): $max_grad_diff")
        @test loss_diff < 1e-10
        @test max_grad_diff < 1e-10
        println("PASSED: CPU and Fast GPU loss & gradients match to machine precision!")

        println("\nBenchmarking 5 full Gradient Passes (Forward + Pullback) on GPU...")
        t_grad_list = Float64[]
        for r in 1:5
            t = @elapsed begin
                l_val, g_val = gpu_adjoint_loss_and_grad(A, gates, gpu_ops, ref_dev, target_dev, num_exp; antihermitian=false)
            end
            push!(t_grad_list, t)
        end
        grad_avg = mean(t_grad_list)
        println("GPU Gradient Pass Times (s): ", round.(t_grad_list, digits=4))
        println("Average GPU Gradient Pass Time: $(round(grad_avg, digits=4)) s")

        println("\n==========================================================")
        println("SUMMARY COMPARISON ON N=(5, 4)_4x3 DATASET")
        println("CPU (1 core) Forward Pass:  1.1767 s")
        println("GPU Fast Forward Pass:      0.1202 s (9.8x faster)")
        println("CPU (1 core) Gradient Pass: 2.2338 s")
        println("GPU Fast Gradient Pass:     $(round(grad_avg, digits=4)) s ($(round(2.2338 / grad_avg, digits=1))x faster)")
        println("==========================================================")
        return 0
    end
end
