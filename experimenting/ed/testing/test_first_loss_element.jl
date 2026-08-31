#=
test_first_loss_element.jl

Verifies that the first element of the loss history (metrics["loss"]) is always
the loss corresponding to a unitary where all coefficients are zero.
For overlap loss, this is the overlap between target and reference states: 1 - |<target|ref>|^2.
For energy loss, this is the energy of the reference state: <ref|H|ref>.
=#

using Test
using LinearAlgebra
using SparseArrays
using Lattices
using JLD2
using HDF5
using Zygote

include(joinpath(@__DIR__, "..", "data_path.jl"))
include(joinpath(@__DIR__, "..", "utility_functions.jl"))
using .UtilityFunctions
include(joinpath(@__DIR__, "..", "ed_objects.jl"))
include(joinpath(@__DIR__, "..", "ed_functions.jl"))
include(joinpath(@__DIR__, "..", "ed_optimization.jl"))
include(joinpath(@__DIR__, "..", "trotter.jl"))
using .Trotter

@testset "First Loss Element Metrics Verification" begin
    folder = "/home/jek354/research/data/new_data/data_h5_fixed/N=(5, 4)_3x3"
    Lvec = [3, 3]
    N_sites = 9

    U_values, state_vecs, indexer, _, N_elec, spin_conserved, _, sign_convention =
        load_ED_data(folder; verbose=false, sign_convention=:spin_first, use_slater_reference=false)
    basis_sector = Trotter.get_basis_sector(indexer, Lvec, N_sites)
    
    # Excitations/gates setup
    gates = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=false)
    tau_terms = Trotter.fgateToTauSector(gates, N_sites, basis_sector; antihermitian=true)

    # Pick two states at different U indices
    state_ref = state_vecs[1, :]
    state_target = state_vecs[10, :]

    # 1. Calculate the exact expected zero-coefficients overlap loss
    expected_overlap_loss = max(0.0, 1.0 - abs2(dot(state_target, state_ref)))
    println("Expected zero-coefficients overlap loss: $expected_overlap_loss")

    # 2. Reconstruct H at U_idx=10 for energy loss test
    subspace = reconstruct_subspace(indexer, spin_conserved)
    H_hopping, H_interaction = create_hubbard_matrices(subspace; indexer=indexer, get_indexer=false,
        sign_convention=:spin_first, lattice_ordering=ColSnake()
    )
    H_10 = H_hopping + U_values[10] * H_interaction
    expected_energy_loss = real(dot(state_ref, H_10 * state_ref))
    println("Expected zero-coefficients energy loss: $expected_energy_loss")

    @testset "Trotter Optimization - Overlap Loss (with warm-start/non-zero init)" begin
        # Non-zero initial coefficients
        init_coeffs = fill(0.05, length(gates))
        
        _, _, metrics = Trotter.optimize_unitary(
            gates,
            tau_terms,
            state_ref,
            state_target,
            basis_sector,
            N_sites;
            loss_type=:overlap,
            num_exponentials=1,
            antihermitian=true,
            initial_coefficients=init_coeffs,
            initialization_samples=0,
            maxiters=3,
            optimizer=:LBFGS,
            use_gpu=false
        )
        
        @test haskey(metrics, "loss")
        @test !isempty(metrics["loss"])
        # First element of the loss must be the zero-coefficients loss, not the initial warm-started loss
        @test isapprox(metrics["loss"][1], expected_overlap_loss; atol=1e-12)
        println("  [PASS] Trotter overlap metrics[\"loss\"][1] matches zero-coefficients loss!")
    end

    @testset "Trotter Optimization - Energy Loss (with warm-start/non-zero init)" begin
        init_coeffs = fill(0.05, length(gates))
        
        _, _, metrics = Trotter.optimize_unitary(
            gates,
            tau_terms,
            state_ref,
            state_target,
            basis_sector,
            N_sites;
            loss_type=:energy,
            H=H_10,
            num_exponentials=1,
            antihermitian=true,
            initial_coefficients=init_coeffs,
            initialization_samples=0,
            maxiters=3,
            optimizer=:LBFGS,
            use_gpu=false
        )
        
        @test haskey(metrics, "loss")
        @test !isempty(metrics["loss"])
        @test isapprox(metrics["loss"][1], expected_energy_loss; atol=1e-12)
        println("  [PASS] Trotter energy metrics[\"loss\"][1] matches zero-coefficients loss!")
    end

    @testset "ED Optimization - Overlap Loss (with warm-start/non-zero init)" begin
        init_coeffs = [fill(0.05, length(gates))] # vector of vectors (for each order)
        
        # Call the ED optimize_unitary directly
        _, _, _, _, _, metrics, _ = optimize_unitary(
            state_ref,
            state_target,
            indexer;
            spin_conserved=spin_conserved,
            maxiters=3,
            optimization_scheme=[2],
            gradient=:adjoint_gradient,
            antihermitian=true,
            optimizer=:LBFGS,
            initial_coefficients=init_coeffs,
            initialization_samples=0,
            loss_type=:overlap,
            H=H_10,
            use_gpu=false,
            num_exponentials=1
        )
        
        @test haskey(metrics, "loss")
        @test !isempty(metrics["loss"])
        @test isapprox(metrics["loss"][1], expected_overlap_loss; atol=1e-12)
        println("  [PASS] ED overlap metrics[\"loss\"][1] matches zero-coefficients loss!")
    end

    @testset "ED Optimization - Energy Loss (with warm-start/non-zero init)" begin
        init_coeffs = [fill(0.05, length(gates))]
        
        _, _, _, _, _, metrics, _ = optimize_unitary(
            state_ref,
            state_target,
            indexer;
            spin_conserved=spin_conserved,
            maxiters=3,
            optimization_scheme=[2],
            gradient=:adjoint_gradient,
            antihermitian=true,
            optimizer=:LBFGS,
            initial_coefficients=init_coeffs,
            initialization_samples=0,
            loss_type=:energy,
            H=H_10,
            use_gpu=false,
            num_exponentials=1
        )
        
        @test haskey(metrics, "loss")
        @test !isempty(metrics["loss"])
        @test isapprox(metrics["loss"][1], expected_energy_loss; atol=1e-12)
        println("  [PASS] ED energy metrics[\"loss\"][1] matches zero-coefficients loss!")
    end
end
