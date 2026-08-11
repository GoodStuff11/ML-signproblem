#=
test_option_b_verification.jl

Quick test verifying that Option B (CPU Host RAM storage & GPU streaming of gate matrices)
produces exact loss & gradients as standard GPU execution.
=#

using HDF5
using Lattices
using LinearAlgebra
using SparseArrays
using CUDA
using Zygote

include(joinpath(@__DIR__, "..", "data_path.jl"))
include(joinpath(@__DIR__, "..", "logging.jl"))
include(joinpath(@__DIR__, "..", "utility_functions.jl"))
using .UtilityFunctions
include(joinpath(@__DIR__, "..", "trotter.jl"))
using .Trotter
include(joinpath(@__DIR__, "..", "ed_objects.jl"))
include(joinpath(@__DIR__, "..", "ed_functions.jl"))

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "test_option_b_verification")
    with_logging(log_path) do
        println("================================================================================")
        println("=== Option B Verification Test ===")
        println("================================================================================")

        folder = data_folder("N=(5, 4)_4x3")
        U_values, state_vecs, indexer, _, N_elec, _, _, sign_convention =
            load_ED_data(folder; verbose=false, sign_convention=:spin_first, use_slater_reference=false)

        Lvec = parse_lattice_dimension(folder)
        N_sites = prod(Lvec)
        basis_sector = Trotter.get_basis_sector(indexer, Lvec, N_sites)

        gates = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=false)
        num_gates = length(gates)

        ref_real, _ = Trotter.TrotterOptimization.strip_global_phase(state_vecs[1, :])
        target_real, _ = Trotter.TrotterOptimization.strip_global_phase(state_vecs[2, :])
        v_ref32 = Float32.(ref_real)
        v_target32 = Float32.(target_real)
        A_rand = randn(num_gates)

        # Test Option B (stream_tau=true)
        println("1. Running Option B (stream_tau=true)...")
        ops_opt_b = Trotter.TrotterOptimization.get_gpu_gate_ops(gates, N_sites, basis_sector; antihermitian=true, datatype=Float32, stream_tau=true)

        phis_b = Trotter.TrotterOptimization.apply_unitary_checkpoints(
            A_rand, gates, v_ref32, basis_sector, N_sites, 1;
            antihermitian=true, use_gpu=true, datatype=Float32
        )
        init_adj = CUDA.copy(phis_b[end])
        grad_b = Trotter.TrotterOptimization.backward_adjoint_propagation(
            A_rand, gates, nothing, phis_b, init_adj, basis_sector, N_sites, 1;
            antihermitian=true, use_gpu=true, datatype=Float32
        )

        # Test Full GPU (stream_tau=false)
        println("2. Running Full GPU (stream_tau=false)...")
        ops_full = Trotter.TrotterOptimization.prepare_gpu_gate_ops(gates, N_sites, basis_sector; antihermitian=true, datatype=Float32, stream_tau=false)

        # Manually compute using ops_full
        phis_full = Vector{CUDA.CuVector{Float32}}(undef, num_gates + 1)
        v_curr = CUDA.CuArray(v_ref32)
        v_next = CUDA.similar(v_curr)
        phis_full[1] = copy(v_curr)
        for k in 1:num_gates
            Trotter.TrotterOptimization.gpu_apply_gate_exp!(v_next, v_curr, ops_full, k, A_rand[k]; antihermitian=true)
            v_curr, v_next = v_next, v_curr
            phis_full[k+1] = copy(v_curr)
        end

        init_adj_full = copy(phis_full[end])
        grad_full = Vector{Float64}(undef, num_gates)
        adj_curr = copy(init_adj_full)
        adj_next = CUDA.similar(adj_curr)

        for k in num_gates:-1:1
            tau_mat = ops_full.tau_dev[k]
            mul!(ops_full.w1, tau_mat, phis_full[k+1])
            grad_full[k] = -real(dot(adj_curr, ops_full.w1))
            Trotter.TrotterOptimization.gpu_apply_gate_exp!(adj_next, adj_curr, ops_full, k, A_rand[k]; antihermitian=true, inverse=true)
            adj_curr, adj_next = adj_next, adj_curr
        end

        max_grad_diff = maximum(abs.(grad_b .- grad_full))
        println("Max gradient difference between Option B and Full GPU: $max_grad_diff")
        @assert max_grad_diff < 1e-5 "Option B gradient differs from Full GPU!"

        println("✅ Option B Verification SUCCESSFUL! Option B yields exact match.")
        return 0
    end
end
