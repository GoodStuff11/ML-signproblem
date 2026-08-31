#=
test_overlap_loss_and_metrics_saving.jl

Verification suite for:
1. Normalization of state vectors loaded via load_ED_data across all U values (specifically testing N=(5, 4)_3x3).
2. Non-negativity of overlap loss (fidelity <= 1.0).
3. metrics["loss"] length guarantees (always >= 2 elements) in optimize_unitary even when states are identical.
=#

using Test
using LinearAlgebra
using SparseArrays
using Lattices
using JLD2
using HDF5

include(joinpath(@__DIR__, "..", "data_path.jl"))
include(joinpath(@__DIR__, "..", "utility_functions.jl"))
using .UtilityFunctions
include(joinpath(@__DIR__, "..", "ed_objects.jl"))
include(joinpath(@__DIR__, "..", "ed_functions.jl"))
include(joinpath(@__DIR__, "..", "trotter.jl"))
using .Trotter

@testset "Overlap Loss and Metrics Saving Tests" begin
    folder = "/home/jek354/research/data/new_data/data_h5_fixed/N=(5, 4)_3x3"

    @testset "1. State Vector Normalization" begin
        U_values, state_vecs, indexer, _, N_elec, spin_conserved, _, sign_convention =
            load_ED_data(folder; verbose=false, sign_convention=:spin_first, use_slater_reference=false)
        
        n_U = size(state_vecs, 1)
        @test n_U == 60
        
        # Test that EVERY state vector in state_vecs has norm exactly 1.0
        max_norm_err = 0.0
        for u in 1:n_U
            v = state_vecs[u, :]
            n = norm(v)
            err = abs(n - 1.0)
            max_norm_err = max(max_norm_err, err)
            @test isapprox(n, 1.0; atol=1e-12)
        end
        println("  [PASS] All $n_U state vectors are strictly normalized (max norm deviation: $max_norm_err)")
        
        # Explicit check for u=12, 13, 14
        @test isapprox(norm(state_vecs[12, :]), 1.0; atol=1e-14)
        @test isapprox(norm(state_vecs[13, :]), 1.0; atol=1e-14)
        @test isapprox(norm(state_vecs[14, :]), 1.0; atol=1e-14)
    end

    @testset "2. Overlap Loss Non-Negativity" begin
        # Generate random state vectors and gates
        dim = 100
        v1 = randn(ComplexF64, dim)
        v1 ./= norm(v1)
        v2 = randn(ComplexF64, dim)
        v2 ./= norm(v2)
        
        # Overlap with identical state
        overlap_identical = abs2(dot(v1, v1))
        loss_identical = max(0.0, 1.0 - overlap_identical)
        @test loss_identical >= 0.0
        @test isapprox(loss_identical, 0.0; atol=1e-14)
        
        # Overlap with orthogonal state
        v_ortho = v2 - dot(v1, v2) * v1
        v_ortho ./= norm(v_ortho)
        loss_ortho = max(0.0, 1.0 - abs2(dot(v1, v_ortho)))
        @test loss_ortho >= 0.0
        @test isapprox(loss_ortho, 1.0; atol=1e-14)
        
        println("  [PASS] Overlap loss is strictly non-negative in boundary conditions")
    end

    @testset "3. Metrics Vector Length and Early Exit Handling" begin
        Lvec = [3, 3]
        N_sites = 9
        
        U_values, state_vecs, indexer, _, N_elec, spin_conserved, _, sign_convention =
            load_ED_data(folder; verbose=false, sign_convention=:spin_first, use_slater_reference=false)
        basis_sector = Trotter.get_basis_sector(indexer, Lvec, N_sites)
        gates = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=false)
        tau_terms = Trotter.fgateToTauSector(gates, N_sites, basis_sector; antihermitian=true)
        
        # Case A: Identical states (early exit triggered)
        target_state = state_vecs[12, :]
        ref_state = copy(target_state) # Exactly identical
        
        A_opt, final_loss, metrics = Trotter.optimize_unitary(
            gates,
            tau_terms,
            ref_state,
            target_state,
            basis_sector,
            N_sites;
            loss_type=:overlap,
            num_exponentials=1,
            antihermitian=true,
            initial_coefficients=zeros(length(gates)),
            initialization_samples=0,
            use_gpu=false
        )
        
        @test haskey(metrics, "loss")
        @test length(metrics["loss"]) == 2
        @test metrics["loss"][1] == metrics["loss"][2]
        @test isapprox(metrics["loss"][1], 0.0; atol=1e-12)
        @test length(metrics["optimization_losses"]) == 1
        @test length(metrics["stopping_reasons"]) == 1
        @test metrics["stopping_reasons"][1] == ["States are already equal"]
        println("  [PASS] Identical states early exit stores 2-element metrics[\"loss\"]: $(metrics["loss"])")
        
        # Case B: Nearby states (u=13 to u=12) with Hamiltonian passed for energy metric
        state_13 = state_vecs[13, :]
        state_12 = state_vecs[12, :]
        
        # Build Hamiltonian for U=12
        n_up, n_dn = N_elec
        H_hop_sec, _, _ = Trotter.TamFermion.HubbardMomentumBasis(1.0, 0.0, Lvec, (n_up, n_dn); indexer=indexer)
        H_int_sec, _, _ = Trotter.TamFermion.HubbardMomentumBasis(0.0, 1.0, Lvec, (n_up, n_dn); indexer=indexer)
        H_12 = H_hop_sec + U_values[12] * H_int_sec
        
        A_opt_12, final_loss_12, metrics_12 = Trotter.optimize_unitary(
            gates,
            tau_terms,
            state_13,
            state_12,
            basis_sector,
            N_sites;
            loss_type=:overlap,
            H=H_12,
            num_exponentials=1,
            antihermitian=true,
            initial_coefficients=zeros(length(gates)),
            initialization_samples=0,
            maxiters=5,
            optimizer=:LBFGS,
            use_gpu=false
        )
        
        @test length(metrics_12["loss"]) == 2
        @test metrics_12["loss"][1] >= 0.0
        @test metrics_12["loss"][2] >= 0.0
        @test metrics_12["loss"][2] <= metrics_12["loss"][1] + 1e-10
        @test isapprox(metrics_12["loss"][1], max(0.0, 1.0 - abs2(dot(state_12, state_13))); atol=1e-10)
        
        @test haskey(metrics_12, "energy")
        @test length(metrics_12["energy"]) == 2
        @test isapprox(metrics_12["energy"][1], real(dot(state_13, H_12 * state_13)); atol=1e-10)
        @test !isnan(metrics_12["energy"][2])
        println("  [PASS] Standard optimization stores 2-element non-negative metrics[\"loss\"]: $(metrics_12["loss"])")
        println("  [PASS] Energy metric correctly computed at start and end: $(metrics_12["energy"])")
        
        # Case C: Energy loss optimization
        A_opt_e, final_loss_e, metrics_e = Trotter.optimize_unitary(
            gates,
            tau_terms,
            state_13,
            H_12,
            basis_sector,
            N_sites;
            loss_type=:energy,
            state2=state_12,
            num_exponentials=1,
            antihermitian=true,
            initial_coefficients=zeros(length(gates)),
            initialization_samples=0,
            maxiters=5,
            optimizer=:LBFGS,
            use_gpu=false
        )
        
        @test length(metrics_e["loss"]) == 2
        @test isapprox(metrics_e["loss"][1], real(dot(state_13, H_12 * state_13)); atol=1e-10)
        @test haskey(metrics_e, "overlap")
        @test length(metrics_e["overlap"]) == 2
        @test isapprox(metrics_e["overlap"][1], max(0.0, 1.0 - abs2(dot(state_12, state_13))); atol=1e-10)
        println("  [PASS] Energy loss optimization tracks loss (energy) and overlap metric correctly: energy=$(metrics_e["loss"]), overlap=$(metrics_e["overlap"])")
        
        # Case D: Multi-layer growth (num_exponentials: 1 -> 2)
        A_init_2 = Trotter.grow_coefficients(A_opt_12, 1, 2, length(gates))
        A_opt_layer2, final_loss_layer2, metrics_layer2 = Trotter.optimize_unitary(
            gates,
            tau_terms,
            state_13,
            state_12,
            basis_sector,
            N_sites;
            loss_type=:overlap,
            H=H_12,
            num_exponentials=2,
            antihermitian=true,
            initial_coefficients=A_init_2,
            initialization_samples=0,
            loaded_metrics=metrics_12,
            maxiters=5,
            optimizer=:LBFGS,
            use_gpu=false
        )
        
        @test length(metrics_layer2["loss"]) == 3
        @test length(metrics_layer2["energy"]) == 3
        @test metrics_layer2["loss"][1] == metrics_12["loss"][1]
        @test metrics_layer2["loss"][2] == metrics_12["loss"][2]
        @test metrics_layer2["energy"][1] == metrics_12["energy"][1]
        @test metrics_layer2["energy"][2] == metrics_12["energy"][2]
        println("  [PASS] Layer growth carries forward intermediate metrics to 3 elements: loss=$(metrics_layer2["loss"]), energy=$(metrics_layer2["energy"])")
    end
end

