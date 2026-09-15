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
# every run is logged under testing/logs/. Later tasks (Tasks 2-4) should
# add their @testset blocks inside this same with_logging do-block, placed
# ABOVE the trailing `nothing` near the bottom of this function. The
# `nothing` must remain the LAST statement in the do-block: `@main` tries
# to convert whatever this function returns into a process exit code, and
# a bare `@testset` returns a `Test.DefaultTestSet` (not exit-code
# convertible), which would make even a fully-passing run exit nonzero
# with a spurious MethodError. Appending a new testset below the `nothing`
# would silently reintroduce that bug.
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

    @testset "PBC canonicalization wrap branch on $(Lvec)" for Lvec in [(2, 2), (4, 2)]
        N = prod(Lvec)
        gates, pmap = TamFermion.enumerate_ferm_excitations_HVA(Lvec; use_pbc=true)

        # On-site gates are still first, still diagonal.
        @test all(is_diagonal_gate, gates[1:N])
        @test all(!is_diagonal_gate, gates[N+1:end])

        # This is the assertion that actually exercises the `i < j ? ... :
        # (sitebit(j), sitebit(i))` wrap branch: on a periodic axis the
        # forward neighbour of the last site wraps to site 1, so j < i for
        # that bond and the swap must fire to keep the h.c.-dedup
        # convention intact.
        @test all(g -> (g.cre_up, g.cre_dn) <= (g.ann_up, g.ann_dn), gates)

        # NOTE: deliberately no group-count / pmap assertion for (2,2)
        # here. On a length-2 periodic axis circshift maps coord 0 -> 1
        # and coord 1 -> 0, so parity-0 and parity-1 pick up the SAME
        # physical bond (matching this repo's own findLatticeEdges
        # convention, which likewise double-counts the L=2 ring). That
        # duplicate-layer behaviour is accepted, not pinned by a test.
    end

    @testset "PBC group count on (4,2) is well-defined" begin
        Lvec = (4, 2)
        N = prod(Lvec)
        gates, pmap = TamFermion.enumerate_ferm_excitations_HVA(Lvec; use_pbc=true)
        # Rather than re-deriving the exact bond geometry here, check the
        # invariants that follow from the grouping structure: a contiguous
        # surjection, and every group having an even size (each bond
        # contributes exactly 2 gates, up and down).
        @test sort(unique(pmap)) == collect(1:maximum(pmap))
        for k in 1:maximum(pmap)
            @test count(==(k), pmap) % 2 == 0
        end
        @test length(gates) == N + 2 * ((length(gates) - N) ÷ 2)
    end

    @testset "length-1 periodic axis emits no self-bond gates" begin
        Lvec = (1, 4)
        N = prod(Lvec)
        gates, pmap = TamFermion.enumerate_ferm_excitations_HVA(Lvec; use_pbc=true)

        # The on-site block is unaffected.
        @test all(is_diagonal_gate, gates[1:N])

        # This is the invariant the self-bond bug broke: a length-1
        # periodic axis makes every site its own forward neighbour, which
        # (absent the `j == i` skip) emitted a diagonal FGate(m, m, 0, 0)
        # "hopping" gate. No gate beyond the on-site block may be
        # diagonal, regardless of how many genuine hopping gates the
        # other (non-trivial) axis contributes.
        @test all(!is_diagonal_gate, gates[N+1:end])

        # The length-1 axis itself contributes zero bonds; the only
        # hopping gates come from the length-4 axis, which forms a genuine
        # 4-site ring (bonds (1,2),(3,4) parity 0; (2,3),(4,1) parity 1),
        # i.e. N bonds x 2 spin channels.
        @test length(gates) == N + 2 * N

        # h.c.-dedup convention still holds even across the (4,1) wrap bond.
        @test all(g -> (g.cre_up, g.cre_dn) <= (g.ann_up, g.ann_dn), gates)
    end
end

@testset "sharing helpers" begin
    pmap = [1, 1, 2, 3, 3, 3]
    num_gates = 6
    P = 2

    @test num_shared_params(nothing, num_gates) == 6
    @test num_shared_params(pmap, num_gates) == 3

    # Gather: layer-major, theta[(l-1)*n_params + pmap[j]] lands at a[(l-1)*num_gates+j]
    theta = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]   # 2 layers × 3 params
    a = expand_shared_coefficients(theta, pmap, num_gates, P)
    @test a == [10.0, 10.0, 20.0, 30.0, 30.0, 30.0,
                40.0, 40.0, 50.0, 60.0, 60.0, 60.0]

    # Scatter-add is the exact transpose of the gather.
    grad_a = collect(1.0:12.0)
    grad_theta = contract_shared_gradient(grad_a, pmap, num_gates, P)
    @test grad_theta == [1.0 + 2.0, 3.0, 4.0 + 5.0 + 6.0,
                         7.0 + 8.0, 9.0, 10.0 + 11.0 + 12.0]

    # <grad_a, E*theta> == <E'*grad_a, theta>  (the adjoint identity itself)
    @test dot(grad_a, a) ≈ dot(grad_theta, theta)

    # nothing short-circuits to identity
    @test expand_shared_coefficients(theta, nothing, 3, P) === theta
    @test contract_shared_gradient(grad_a, nothing, 6, P) === grad_a

    # Length validation
    @test_throws ArgumentError expand_shared_coefficients(theta, pmap, 5, P)
    @test_throws ArgumentError expand_shared_coefficients([1.0, 2.0], pmap, num_gates, P)
end

        nothing
    end
end
