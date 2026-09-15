#=
test_hva_gates_and_param_map.jl

Tests for enumerate_ferm_excitations_HVA and for shared Trotter coefficients.
Run from the `ed/` directory:

    julia --project=.. testing/test_hva_gates_and_param_map.jl
=#

using Test
using LinearAlgebra
using Random

const ED_DIR = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ED_DIR, "trotter.jl"))
include(joinpath(ED_DIR, "logging.jl"))

using .Trotter
using .Trotter.TamFermion
using .Trotter.TrotterOptimization

# ───────────────────────────────────────────────────────────────────────
# Shared helpers
# ───────────────────────────────────────────────────────────────────────

"""Build the combined 2N-bit sector basis in the SAME ordering that
`HubbardRealSpace` uses for its matrix: up index slow, down index fast."""
function build_sector_basis(Lvec, nvec)
    N = prod(Lvec)
    bu = TamFermion.getReducedHilSpace(N, nvec[1]; returnOcc=false)
    bd = TamFermion.getReducedHilSpace(N, nvec[2]; returnOcc=false)
    return [TamFermion.combineSpinInts(bu[iu], bd[id], N)
            for iu in eachindex(bu) for id in eachindex(bd)]
end

dense_taus(gates, N, basis) =
    [Matrix(t) for t in TamFermion.fgateToTauSector(gates, N, basis; antihermitian=false)]

# ───────────────────────────────────────────────────────────────────────
# All testsets run inside this single (@main) / with_logging wrapper so
# every run is logged under testing/logs/. Later tasks should add their
# @testset blocks inside this same with_logging do-block rather than
# introducing a second wrapper.
# ───────────────────────────────────────────────────────────────────────
function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "test_hva_gates_and_param_map")
    with_logging(log_path) do

@testset "HVA gate set" begin
    @testset "structure on $(Lvec)" for Lvec in [(2, 2), (3, 2)]
        N = prod(Lvec)
        gates, pmap = TamFermion.enumerate_ferm_excitations_HVA(Lvec)

        # param_map is a contiguous surjection onto 1:n_params
        @test length(pmap) == length(gates)
        @test sort(unique(pmap)) == collect(1:maximum(pmap))

        # The first N gates are the on-site interactions, one per site.
        @test all(is_diagonal_gate, gates[1:N])
        @test all(!is_diagonal_gate, gates[N+1:end])
        # ...and they all share one coefficient under the default tie=:full
        @test all(pmap[1:N] .== pmap[1])
        @test !any(pmap[N+1:end] .== pmap[1])

        # Every gate respects the h.c.-deduplication convention.
        @test all(g -> (g.cre_up, g.cre_dn) <= (g.ann_up, g.ann_dn), gates)

        # Hopping gates come in up/down pairs, one spin channel each.
        for g in gates[N+1:end]
            up_only = (g.cre_dn == 0 && g.ann_dn == 0)
            dn_only = (g.cre_up == 0 && g.ann_up == 0)
            @test up_only ⊻ dn_only
        end
    end

    @testset "gate counts and grouping on (2,2) OBC" begin
        gates, pmap = TamFermion.enumerate_ferm_excitations_HVA((2, 2))
        # 4 on-site; horizontal bonds (1,3),(2,4) parity 0 and none parity 1;
        # vertical bonds (1,2),(3,4) parity 0 and none parity 1.
        # 2 spin channels per bond => 4 + 4 + 4 = 12 gates, 3 parameter groups.
        @test length(gates) == 12
        @test maximum(pmap) == 3
        @test count(==(1), pmap) == 4
        @test count(==(2), pmap) == 4
        @test count(==(3), pmap) == 4
    end

    @testset "generators are Hermitian and commute within a group" begin
        Lvec, nvec = (3, 2), (2, 1)
        N = prod(Lvec)
        basis = build_sector_basis(Lvec, nvec)
        gates, pmap = TamFermion.enumerate_ferm_excitations_HVA(Lvec)
        taus = dense_taus(gates, N, basis)

        for t in taus
            @test t ≈ t'
        end
        for k in 1:N
            @test isdiag(taus[k])
        end
        for k in eachindex(taus), l in eachindex(taus)
            if k < l && pmap[k] == pmap[l]
                @test norm(taus[k] * taus[l] - taus[l] * taus[k]) < 1e-10
            end
        end
    end

    @testset "tie modes" begin
        Lvec = (3, 2)
        N = prod(Lvec)
        gates, pmap_full = TamFermion.enumerate_ferm_excitations_HVA(Lvec; tie=:full)
        _, pmap_none = TamFermion.enumerate_ferm_excitations_HVA(Lvec; tie=:none)
        _, pmap_spin = TamFermion.enumerate_ferm_excitations_HVA(Lvec; tie=:spin)

        @test pmap_none == collect(1:length(gates))

        # (3,2) OBC: horizontal parity-0 bonds (1,3),(2,4); parity-1 (3,5),(4,6);
        # vertical parity-0 (1,2),(3,4),(5,6); vertical parity-1 is EMPTY and
        # must be dropped, so 4 groups rather than 5.
        @test maximum(pmap_full) == 4

        # :spin gives one parameter per site plus one per bond.
        n_bonds = (length(gates) - N) ÷ 2
        @test maximum(pmap_spin) == N + n_bonds
        @test pmap_spin[1:N] == collect(1:N)
        for b in 1:n_bonds
            @test pmap_spin[N+2b-1] == pmap_spin[N+2b]
        end
    end

    @testset "odd periodic axis is rejected" begin
        @test_throws ArgumentError TamFermion.enumerate_ferm_excitations_HVA((3, 2); use_pbc=true)
        @test_throws ArgumentError TamFermion.enumerate_ferm_excitations_HVA((3, 2); tie=:nonsense)
    end
end

        nothing
    end
end
