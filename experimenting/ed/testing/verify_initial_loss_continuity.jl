#=
verify_initial_loss_continuity.jl

Comprehensive verification script for:
1. Resuming optimization from an already optimized state: verifying that initial_loss == previous final_loss.
2. Increasing num_exponentials with grow_mode=:chain: verifying that initial_loss == previous final_loss.
3. Datatype override: verifying that Hermitian mode overrides Real datatypes to ComplexF64 without error,
   and to_device_vector cleanly converts complex inputs to real datatypes for antihermitian mode without InexactError.

Usage:
  julia --project=.. testing/verify_initial_loss_continuity.jl
=#

ENV["JULIA_CUDA_USE_COMPAT"] = "true"
using CUDA

using Lattices
using LinearAlgebra
using SparseArrays
using JLD2
using HDF5

include(joinpath(@__DIR__, "..", "data_path.jl"))
include(joinpath(@__DIR__, "..", "logging.jl"))
include(joinpath(@__DIR__, "..", "utility_functions.jl"))
using .UtilityFunctions
include(joinpath(@__DIR__, "..", "trotter.jl"))
using .Trotter
include(joinpath(@__DIR__, "..", "ed_objects.jl"))
include(joinpath(@__DIR__, "..", "ed_functions.jl"))

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "verify_initial_loss_continuity")
    with_logging(log_path) do
        println("================================================================================")
        println("STARTING VERIFICATION OF INITIAL LOSS CONTINUITY & DATATYPE HANDLING")
        println("================================================================================")

        folder = data_folder("N=(3, 3)_3x3")
        U_values, state_vecs, indexer, _, N_elec, _, _, sign_convention =
            load_ED_data(folder; verbose=false, sign_convention=:spin_first, use_slater_reference=true)

        n_up, n_dn = N_elec
        Lvec = parse_lattice_dimension(folder)
        N_sites = prod(Lvec)
        u_idx = 33
        target_u = U_values[u_idx]

        basis_sector = Trotter.get_basis_sector(indexer, Lvec, N_sites)
        gates_anti = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=false)
        gates_herm = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=true)
        tau_terms_anti = Trotter.fgateToTauSector(gates_anti, N_sites, basis_sector; antihermitian=true)
        tau_terms_herm = Trotter.fgateToTauSector(gates_herm, N_sites, basis_sector; antihermitian=false)

        state1 = state_vecs[1, :] # Slater reference
        state2 = state_vecs[u_idx, :] # Target state at U_idx=33

        # -------------------------------------------------------------------------
        # TEST 1: Initial optimization (Stage 0) -> gives A_opt_1, final_loss_1
        # -------------------------------------------------------------------------
        println("\n--- TEST 1: Initial Optimization (num_exponentials=1, maxiters=10) ---")
        A_opt_1, final_loss_1, metrics_1 = optimize_unitary(
            gates_anti, tau_terms_anti, state1, state2, basis_sector, N_sites;
            loss_type=:overlap,
            num_exponentials=1,
            maxiters=10,
            optimizer=:LBFGS,
            initialization_samples=5,
            antihermitian=true,
            use_gpu=true,
            datatype=Float32
        )
        println("Stage 0 Final Loss: $final_loss_1")

        # -------------------------------------------------------------------------
        # TEST 2: Resume optimization (same num_exponentials=1) starting from A_opt_1
        # -------------------------------------------------------------------------
        println("\n--- TEST 2: Resume Optimization (num_exponentials=1 from A_opt_1) ---")
        A_opt_resume, final_loss_resume, metrics_resume = optimize_unitary(
            gates_anti, tau_terms_anti, state1, state2, basis_sector, N_sites;
            loss_type=:overlap,
            num_exponentials=1,
            maxiters=5,
            optimizer=:LBFGS,
            initial_coefficients=A_opt_1,
            loaded_metrics=metrics_1,
            antihermitian=true,
            use_gpu=true,
            datatype=Float32
        )
        resumed_initial_loss = metrics_resume["optimization_losses"][1][1]
        println("Resumed Initial Loss (first step of callback): $resumed_initial_loss")
        println("Previous Final Loss:                          $final_loss_1")
        loss_diff_resume = abs(resumed_initial_loss - final_loss_1)
        println("Difference: $loss_diff_resume")
        @assert loss_diff_resume < 1e-6 "TEST 2 FAILED: Resumed initial loss does not match previous final loss!"
        println("✓ TEST 2 PASSED: Resumed initial loss matches previous final loss!")

        # -------------------------------------------------------------------------
        # TEST 3: Grow mode chain (num_exponentials=1 -> 2)
        # -------------------------------------------------------------------------
        println("\n--- TEST 3: Grow Mode Chain (num_exponentials=1 -> 2 from A_opt_1) ---")
        A_grown = grow_coefficients(A_opt_1, 1, 2, length(gates_anti))
        println("A_grown length: $(length(A_grown)) (expected $(2 * length(gates_anti)))")
        @assert norm(A_grown[(length(gates_anti)+1):end]) == 0.0 "New layers are not zero-initialized!"

        A_opt_grow, final_loss_grow, metrics_grow = optimize_unitary(
            gates_anti, tau_terms_anti, state1, state2, basis_sector, N_sites;
            loss_type=:overlap,
            num_exponentials=2,
            maxiters=5,
            optimizer=:LBFGS,
            initial_coefficients=A_grown,
            antihermitian=true,
            use_gpu=true,
            datatype=Float32
        )
        grown_initial_loss = metrics_grow["optimization_losses"][1][1]
        println("Grown Initial Loss (first step of callback): $grown_initial_loss")
        println("Previous Final Loss (from 1 exponential):   $final_loss_1")
        loss_diff_grow = abs(grown_initial_loss - final_loss_1)
        println("Difference: $loss_diff_grow")
        @assert loss_diff_grow < 1e-6 "TEST 3 FAILED: Grown initial loss does not match previous final loss!"
        println("✓ TEST 3 PASSED: Grown initial loss matches previous final loss!")

        # -------------------------------------------------------------------------
        # TEST 4: Datatype Override & Safety Check
        # -------------------------------------------------------------------------
        println("\n--- TEST 4: Datatype Override & Conversion Safety ---")
        
        # 4a. Hermitian mode with Real datatype: must override to ComplexF64 and run without error
        scan_instr = Dict{String,Any}(
            "num_exponentials" => 1,
            "antihermitian" => false,
            "u_range" => u_idx:u_idx,
            "U_values" => U_values
        )
        # interaction_scan_map_to_state with datatype=Float32 and antihermitian=false:
        println("Testing Hermitian mode with datatype=Float32 (expecting override to ComplexF64)...")
        res_herm = Trotter.interaction_scan_map_to_state(
            state_vecs, scan_instr, gates_herm, tau_terms_herm, basis_sector, N_sites;
            maxiters=3,
            optimizer=:LBFGS,
            initialization_samples=2,
            save_folder=nothing,
            loss_type=:overlap,
            U_values=U_values,
            antihermitian=false,
            use_gpu=true,
            datatype=Float32
        )
        println("Hermitian run completed successfully with final loss: ", res_herm["loss_metrics"][1])
        @assert !isnan(res_herm["loss_metrics"][1]) "TEST 4a FAILED: Hermitian loss is NaN!"
        println("✓ TEST 4a PASSED: Hermitian mode cleanly handled datatype override!")

        # 4b. to_device_vector with complex input and Float32 datatype: must not throw InexactError
        complex_vec = state_vecs[1, :] # Vector{ComplexF64}
        real_dev = Trotter.to_device_vector(complex_vec, true, Float32)
        println("Converted vector type: $(typeof(real_dev)), eltype: $(eltype(real_dev)), norm: $(norm(real_dev))")
        @assert eltype(real_dev) == Float32 "TEST 4b FAILED: Expected Float32 eltype!"
        @assert abs(norm(real_dev) - 1.0f0) < 1e-5 "TEST 4b FAILED: Norm deviated from 1!"
        println("✓ TEST 4b PASSED: to_device_vector successfully converts Complex to Float32 without InexactError!")

        println("\n================================================================================")
        println("ALL VERIFICATIONS COMPLETED SUCCESSFULLY AND PASSED!")
        println("================================================================================")
        return 0
    end
end
