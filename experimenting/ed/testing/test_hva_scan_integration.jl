#=
test_hva_scan_integration.jl

Tests for the pieces that let `run_trotter_scan_optimization.jl --ansatz=hva`
work: the DOF-matching layer count, the momentum-sector -> real-space lift, the
spin-exchange diagnostic, and param_map threading through the scan driver.

Run from the `ed/` directory:

    julia --project=.. testing/test_hva_scan_integration.jl
=#

using Test
using LinearAlgebra
using SparseArrays
using Random
using JLD2

const ED_DIR = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ED_DIR, "trotter.jl"))

using .Trotter
using .Trotter.TamFermion
using .Trotter.TrotterOptimization

@testset "hva_layers_matching_dof" begin
    Lvec = (3, 2)
    gates_hva, pmap_full = TamFermion.enumerate_ferm_excitations_HVA(Lvec)
    _, pmap_none = TamFermion.enumerate_ferm_excitations_HVA(Lvec; tie=:none)
    n_hva = length(gates_hva)

    # tie=:full on (3,2) OBC has 4 groups (see the HVA gate-set tests).
    @test num_shared_params(pmap_full, n_hva) == 4

    info = hva_layers_matching_dof(pmap_full, n_hva, 180, 1)
    @test info.params_per_layer == 4
    @test info.reference_dof == 180
    @test info.layers == 45
    @test info.hva_dof == 180
    @test info.exact

    # Scaling in the reference layer count is linear.
    @test hva_layers_matching_dof(pmap_full, n_hva, 180, 3).layers == 135

    # Rounding modes, on a deliberately inexact ratio.
    @test hva_layers_matching_dof(pmap_full, n_hva, 11, 1; mode=:ceil).layers == 3
    @test hva_layers_matching_dof(pmap_full, n_hva, 11, 1; mode=:floor).layers == 2
    @test hva_layers_matching_dof(pmap_full, n_hva, 11, 1; mode=:round).layers == 3
    @test !hva_layers_matching_dof(pmap_full, n_hva, 11, 1).exact

    # Never fewer than one layer, even when the HVA already over-parameterizes.
    @test hva_layers_matching_dof(pmap_none, n_hva, 1, 1; mode=:floor).layers == 1

    # param_map=nothing means one parameter per gate.
    @test hva_layers_matching_dof(nothing, n_hva, n_hva, 7).layers == 7

    # The gate-vector convenience form agrees with the count form.
    gates_std = TamFermion.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true)
    @test hva_layers_matching_dof(pmap_full, n_hva, gates_std, 2) ==
          hva_layers_matching_dof(pmap_full, n_hva, length(gates_std), 2)

    @test_throws ArgumentError hva_layers_matching_dof(pmap_full, n_hva, 180, 1; mode=:nonsense)
    @test_throws ArgumentError hva_layers_matching_dof(pmap_full, n_hva, 0, 1)
    @test_throws ArgumentError hva_layers_matching_dof(pmap_full, n_hva, 180, 0)
end

@testset "realspace_basis ordering" begin
    Lvec, nvec = (3, 2), (2, 1)
    N = prod(Lvec)
    basis, bu, bd = realspace_basis(Lvec, nvec)

    @test length(basis) == length(bu) * length(bd)
    @test length(bu) == binomial(N, nvec[1])
    @test length(bd) == binomial(N, nvec[2])

    # Up index slow, down index fast -- the ordering HubbardRealSpace's matrix uses.
    d_dn = length(bd)
    for iu in eachindex(bu), id in eachindex(bd)
        @test basis[(iu-1)*d_dn+id] == TamFermion.combineSpinInts(bu[iu], bd[id], N)
    end

    H = TamFermion.HubbardRealSpace(1.0, 4.0, Lvec, nvec; use_pbc=true, returnBasis=false)
    @test size(H, 1) == length(basis)
end

@testset "momentum_sector_to_realspace round trip" begin
    Random.seed!(20260917)
    Lvec, nvec = (2, 2), (1, 1)
    N = prod(Lvec)
    n_up, n_dn = nvec

    basis, bu, bd = realspace_basis(Lvec, nvec)
    d_up, d_dn = length(bu), length(bd)

    F_up, _ = TamFermion.SlaterCOB_RtoK_nparticle(Lvec, n_up)
    F_dn, _ = TamFermion.SlaterCOB_RtoK_nparticle(Lvec, n_dn)
    @test F_up' * F_up ≈ I
    @test F_dn' * F_dn ≈ I

    # Forward-transform known real-space vectors, then invert with the helper.
    # `basis` doubles as the "momentum sector" basis here (the full space, listed
    # in the same ordering), which is exactly the ordering assumption under test.
    v_real = [normalize!(randn(ComplexF64, d_up * d_dn)) for _ in 1:3]
    v_mom = [vec(F_dn * reshape(v, d_dn, d_up) * transpose(F_up)) for v in v_real]

    got = momentum_sector_to_realspace(reduce(vcat, transpose.(v_mom)), basis, Lvec, nvec)
    @test size(got) == (3, d_up * d_dn)
    for i in 1:3
        @test got[i, :] ≈ v_real[i]
    end

    # A vector-of-vectors input is accepted too.
    got2 = momentum_sector_to_realspace(v_mom, basis, Lvec, nvec)
    @test got2 ≈ got

    # A genuinely restricted sector: drop half the basis and check the embedding
    # puts amplitudes in the right slots (the dropped ones stay zero).
    sub = basis[1:2:end]
    v_sub = normalize!(randn(ComplexF64, length(sub)))
    out = momentum_sector_to_realspace(reshape(v_sub, 1, :), sub, Lvec, nvec)
    @test size(out) == (1, d_up * d_dn)
    # Undo the transform by hand and confirm only the sector slots are populated.
    back = vec(F_dn * reshape(out[1, :], d_dn, d_up) * transpose(F_up))
    idx = Dict(UInt(basis[i]) => i for i in eachindex(basis))
    for i in eachindex(basis)
        expected = i in [idx[UInt(s)] for s in sub] ? nothing : 0.0
        isnothing(expected) || @test abs(back[i]) < 1e-10
    end

    @test_throws DimensionMismatch momentum_sector_to_realspace(
        reshape(randn(ComplexF64, 3), 1, 3), basis, Lvec, nvec)
end

@testset "momentum <-> real space preserves the Hubbard energy" begin
    # The physics check the runner performs at startup, on a state that is an
    # honest eigenvector: diagonalize in real space, push to momentum space, and
    # confirm the helper brings it back with the same energy.
    Random.seed!(20260917)
    Lvec, nvec = (2, 2), (1, 1)
    N = prod(Lvec)
    basis, bu, bd = realspace_basis(Lvec, nvec)
    d_up, d_dn = length(bu), length(bd)

    H_hop = TamFermion.HubbardRealSpace(1.0, 0.0, Lvec, nvec; use_pbc=true, returnBasis=false)
    H_int = TamFermion.HubbardRealSpace(0.0, 1.0, Lvec, nvec; use_pbc=true, returnBasis=false)
    u = 3.0
    H_real = Matrix(H_hop + u * H_int)
    vals, vecs = eigen(Hermitian(H_real))
    psi_real = vecs[:, 1]

    F_up, _ = TamFermion.SlaterCOB_RtoK_nparticle(Lvec, nvec[1])
    F_dn, _ = TamFermion.SlaterCOB_RtoK_nparticle(Lvec, nvec[2])
    psi_mom = vec(F_dn * reshape(ComplexF64.(psi_real), d_dn, d_up) * transpose(F_up))

    back = momentum_sector_to_realspace(reshape(psi_mom, 1, :), basis, Lvec, nvec)[1, :]
    @test norm(back) ≈ 1.0
    @test real(dot(back, H_real * back)) ≈ vals[1]

    # check_realspace_transform accepts a faithful transform ...
    K = kron(F_up, F_dn)
    H_mom = K * H_real * K'
    chk = check_realspace_transform(psi_mom, back, H_mom, H_real)
    @test chk.energy_mom ≈ chk.energy_real
    @test chk.norm_mom ≈ chk.norm_real

    # ... and rejects one that quietly scrambles the state.
    scrambled = circshift(back, 1)
    @test_throws ErrorException check_realspace_transform(psi_mom, scrambled, H_mom, H_real)
end

@testset "spin exchange diagnostic" begin
    Lvec, nvec = (2, 2), (2, 2)
    N = prod(Lvec)
    basis, bu, bd = realspace_basis(Lvec, nvec)
    d = length(basis)

    # P is an involution on this sector.
    Random.seed!(1)
    v = normalize!(randn(ComplexF64, d))
    @test apply_spin_exchange(apply_spin_exchange(v, basis, N, nvec), basis, N, nvec) ≈ v

    # A state built on the SAME orbitals for both spins is a P eigenstate ...
    idx = Dict(UInt(basis[i]) => i for i in eachindex(basis))
    sym = zeros(ComplexF64, d)
    sym[idx[UInt(TamFermion.combineSpinInts(bu[1], bd[1], N))]] = 1.0
    bnd_sym = spin_tied_fidelity_bound(sym, sym, basis, N, nvec)
    @test abs(bnd_sym.ref_parity) ≈ 1.0
    @test bnd_sym.target_is_eigenstate
    @test bnd_sym.bound ≈ 1.0

    # ... while one on DIFFERENT orbitals per spin has zero overlap with its own
    # mirror, which is exactly the 0.5 cap the runner warns about.
    asym = zeros(ComplexF64, d)
    asym[idx[UInt(TamFermion.combineSpinInts(bu[1], bd[2], N))]] = 1.0
    target = (sym + apply_spin_exchange(sym, basis, N, nvec))
    normalize!(target)
    bnd = spin_tied_fidelity_bound(asym, target, basis, N, nvec)
    @test bnd.ref_parity ≈ 0.0 atol = 1e-12
    @test bnd.target_is_eigenstate
    @test bnd.bound ≈ 0.5

    # Undefined outside an n_up == n_dn sector.
    @test_throws ArgumentError apply_spin_exchange(v, basis, N, (2, 1))
end

@testset "interaction_scan_map_to_state threads param_map" begin
    Random.seed!(20260917)
    Lvec, nvec = (2, 2), (1, 1)
    N = prod(Lvec)
    basis, _, _ = realspace_basis(Lvec, nvec)
    d = length(basis)

    gates, pmap = TamFermion.enumerate_ferm_excitations_HVA(Lvec)
    n_params = num_shared_params(pmap, length(gates))
    tau_terms = TamFermion.fgateToTauSector(gates, N, basis; antihermitian=false)

    P = 2
    U_values = [1.0, 2.0]
    states = Matrix{ComplexF64}(undef, 2, d)
    for i in 1:2
        states[i, :] = normalize!(randn(ComplexF64, d))
    end

    save_folder = mktempdir()
    instructions = Dict{String,Any}(
        "u_range" => 1:1, "starting level" => 1, "ending level" => 1,
        "num_exponentials" => P, "antihermitian" => false)

    data = interaction_scan_map_to_state(
        states, instructions, gates, tau_terms, basis, N;
        maxiters=5, optimizer=:LBFGS, initialization_samples=0,
        U_values=U_values, antihermitian=false,
        save_folder=save_folder, save_name="pmap_test",
        param_map=pmap)

    # The saved/returned coefficients live in the REDUCED space.
    @test length(data["coefficients"][1]) == P * n_params
    saved = JLD2.load(joinpath(save_folder, "pmap_test_u_1.jld2"))["dict"]
    @test length(saved["coefficients"]) == P * n_params
    shared = JLD2.load(joinpath(save_folder, "pmap_test_shared.jld2"))["dict"]
    @test shared["param_map"] == pmap

    # Without param_map the vector keeps its full per-gate length -- the
    # regression guarantee for every existing caller.
    data2 = interaction_scan_map_to_state(
        states, instructions, gates, tau_terms, basis, N;
        maxiters=5, optimizer=:LBFGS, initialization_samples=0,
        U_values=U_values, antihermitian=false)
    @test length(data2["coefficients"][1]) == P * length(gates)
end
