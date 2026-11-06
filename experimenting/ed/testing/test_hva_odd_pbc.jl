#=
test_hva_odd_pbc.jl

Checks the periodic HVA on lattices with an odd axis (length > 2), where the bonds of a periodic
ring cannot be split into two commuting groups. enumerate_ferm_excitations_HVA(...; use_pbc=true)
now places the wrap bond (L-1, 0) of such an axis in a third hopping group with its own coefficient
under tie=:full (Option 1; "per edge color" in Bosse & Montanaro, PRB 106, 165130 (2022)).

Checks:
  1. The periodic HVA's hopping bonds are exactly the edges of the periodic lattice
     (findLatticeEdges(...; use_pbc=true)), compared with multiplicity per spin (a length-2 periodic
     axis counts each bond twice, in both), on 3x2, 2x3, 3x3, 4x3, 3x4, 5x3.
  2. tie=:full gives 1 + sum over axes of (3 on an odd periodic axis, else 2) coefficients, and the wrap
     bonds of each odd axis sit alone in their own group.
  3. Within every hopping group the bonds are site-disjoint, and the generator matrices commute
     (max ||[tau_a, tau_b]|| over pairs in a group, real-space sector).
  4. The sum of all hopping generators is proportional to the periodic hopping Hamiltonian
     HubbardRealSpace(1, 0, ...; use_pbc=true) and the sum of on-site generators to
     HubbardRealSpace(0, 1, ...), with zero residual (least-squares scale printed).
  5. Lattices whose periodic axes are all even (or length <= 2), and every open-boundary lattice,
     produce exactly the same gates and param_map as the enumerator on branch main (commit faf9b1e,
     loaded from git), so their earlier results are reproducible.
  6. run_hva_parameter_sweep.jl on "N=(3, 3)_3x2", U=15 (u_idx=60, --ref=eigenstate, periodic target,
     no --hva_pbc) now picks the periodic HVA (log line), reports n_params_per_layer = 6, and reaches an
     infidelity below the floor of the old open-bond HVA on the same problem (best 5.78e-3 over
     P = 2..65 in benchmarks/hva_parameter_sweep_N_3_3_3x2_u60_eigref). Settings: P=10, tie=full,
     overlap loss, 1 run, 10 initialization samples, maxiters 2000 (those of the original sweep).

Usage (from experimenting/ed):
  julia --project=.. testing/test_hva_odd_pbc.jl
No command-line arguments. Prints PASS/FAIL per check and exits with status 1 if any check fails.
Output is also written to logs/<date>/test_hva_odd_pbc_<timestamp>_<pid>.log.
=#

using LinearAlgebra
using SparseArrays

include(joinpath(@__DIR__, "..", "logging.jl"))
include(joinpath(@__DIR__, "..", "trotter.jl"))
using .Trotter
using .Trotter.TamFermion: HubbardRealSpace, findLatticeEdges, enumerate_ferm_excitations_HVA

const ED_DIR = normpath(joinpath(@__DIR__, ".."))
const PROJECT = normpath(joinpath(@__DIR__, "..", ".."))
const MAIN_COMMIT = "faf9b1e"
const OLD_FLOOR = 5.78e-3
const N_FAILED = Ref(0)

function check(name, ok::Bool, detail="")
    println(ok ? "PASS" : "FAIL", "  ", name, isempty(detail) ? "" : "  ($detail)")
    ok || (N_FAILED[] += 1)
    return ok
end

# The enumerator as it was on main, loaded from git into its own module tree.
module OldCode
const _src = read(`git -C $(@__DIR__) show $(Main.MAIN_COMMIT):experimenting/ed/TamFermion.jl`, String)
include(joinpath(@__DIR__, "..", "utility_functions.jl"))
include(joinpath(@__DIR__, "..", "TamLib.jl"))
const _path = tempname() * "_TamFermion_main.jl"
write(_path, _src)
include(_path)
end

hop_bond(g) = minmax(trailing_zeros(g.cre_up | g.cre_dn) + 1, trailing_zeros(g.ann_up | g.ann_dn) + 1)
is_onsite(g) = g.cre_up == g.ann_up == g.cre_dn == g.ann_dn
odd_ring(L) = L > 2 && isodd(L)

function test_bonds_and_groups()
    println("\n== 1-3. Periodic HVA bonds, groups, commutation ==")
    for Lvec in ((3, 2), (2, 3), (3, 3), (4, 3), (3, 4), (5, 3))
        N = prod(Lvec)
        gates, pmap = enumerate_ferm_excitations_HVA(Lvec; use_pbc=true, tie=:full)
        hop = [(k, g) for (k, g) in enumerate(gates) if !is_onsite(g)]
        up = [hop_bond(g) for (_, g) in hop if g.cre_dn == 0]
        dn = [hop_bond(g) for (_, g) in hop if g.cre_up == 0]
        # Compared as multisets: on a length-2 periodic axis findLatticeEdges (and so the periodic
        # HubbardRealSpace Hamiltonian) counts each bond twice, and the HVA emits it twice too.
        edges = sort([minmax(e...) for e in findLatticeEdges(collect(Lvec); use_pbc=true) if e[1] != e[2]])
        check("$(Lvec[1])x$(Lvec[2]): HVA bonds = periodic lattice edges (with multiplicity), per spin",
            sort(up) == edges && sort(dn) == edges && count(is_onsite, gates) == N,
            "$(length(up)) up bonds, $(length(dn)) down bonds, $(length(edges)) periodic edges " *
            "($(length(unique(edges))) distinct)")

        expected = 1 + sum(odd_ring(L) ? 3 : 2 for L in Lvec)
        check("$(Lvec[1])x$(Lvec[2]): tie=:full coefficient count", maximum(pmap) == expected,
            "$(maximum(pmap)) coefficients, expected $expected")

        # wrap-bond groups: every bond (L-1 -> 0) along an odd axis, and nothing else
        groups = Dict{Int,Vector{Tuple{Int,Int}}}()
        for (k, g) in hop
            push!(get!(groups, pmap[k], Tuple{Int,Int}[]), hop_bond(g))
        end
        disjoint = all(begin
            bs = unique(v); sites = reduce(vcat, [[b...] for b in bs]); length(sites) == length(unique(sites))
        end for v in values(groups))
        check("$(Lvec[1])x$(Lvec[2]): bonds within every hopping group are site-disjoint", disjoint,
            "group sizes " * join([length(unique(groups[p])) for p in sort(collect(keys(groups)))], ","))

        # commutators of generator matrices within each group (small real-space sector)
        nvec = (2, 1)
        basis, _, _ = realspace_basis(Lvec, nvec)
        taus = Trotter.fgateToTauSector(gates, N, basis; antihermitian=false)
        mats = [Matrix(t) for t in taus]
        worst = 0.0
        for p in unique(pmap[N+1:end])
            idx = findall(==(p), pmap)
            for a in idx, b in idx
                a < b && (worst = max(worst, opnorm(mats[a] * mats[b] - mats[b] * mats[a])))
            end
        end
        check("$(Lvec[1])x$(Lvec[2]): generators commute within every hopping group", worst < 1e-12,
            "max ||[tau_a, tau_b]|| = $(worst), sector $nvec dim $(length(basis))")
    end
end

function test_generator_sums()
    println("\n== 4. Sum of HVA generators = periodic Hubbard terms ==")
    for (Lvec, nvec) in (((3, 2), (3, 3)), ((3, 3), (2, 2)), ((4, 3), (2, 1)), ((5, 3), (1, 1)))
        N = prod(Lvec)
        gates, _ = enumerate_ferm_excitations_HVA(Lvec; use_pbc=true, tie=:full)
        basis, _, _ = realspace_basis(Lvec, nvec)
        taus = Trotter.fgateToTauSector(gates, N, basis; antihermitian=false)
        S_hop = sum(Matrix(taus[k]) for k in eachindex(gates) if !is_onsite(gates[k]))
        S_int = sum(Matrix(taus[k]) for k in eachindex(gates) if is_onsite(gates[k]))
        H_hop = Matrix(HubbardRealSpace(1.0, 0.0, Lvec, nvec; use_pbc=true, returnBasis=false))
        H_int = Matrix(HubbardRealSpace(0.0, 1.0, Lvec, nvec; use_pbc=true, returnBasis=false))
        H_hop_open = Matrix(HubbardRealSpace(1.0, 0.0, Lvec, nvec; use_pbc=false, returnBasis=false))
        for (name, S, H) in (("hopping", S_hop, H_hop), ("on-site", S_int, H_int))
            c = real(dot(H, S)) / real(dot(S, S))
            res = norm(H - c * S) / norm(H)
            check("$(Lvec[1])x$(Lvec[2]) $nvec: sum of $name generators ∝ periodic $name term",
                res < 1e-12, "scale $(round(c, digits=12)), relative residual $(res)")
        end
        c = real(dot(H_hop_open, S_hop)) / real(dot(S_hop, S_hop))
        check("$(Lvec[1])x$(Lvec[2]) $nvec: ...and is NOT proportional to the open-boundary hopping term",
            norm(H_hop_open - c * S_hop) / norm(H_hop_open) > 1e-3,
            "relative residual $(norm(H_hop_open - c * S_hop) / norm(H_hop_open))")
    end
end

function test_unchanged_vs_main()
    println("\n== 5. Unchanged output vs main ($MAIN_COMMIT) where no axis is an odd ring ==")
    cases = [((2, 2), true), ((4, 2), true), ((4, 4), true), ((2, 4), true), ((6, 2), true),
             ((3, 2), false), ((3, 3), false), ((4, 3), false), ((5, 4), false), ((4, 2), false)]
    for (Lvec, pbc) in cases, tie in (:full, :spin, :none)
        g_new, p_new = enumerate_ferm_excitations_HVA(Lvec; use_pbc=pbc, tie=tie)
        g_old, p_old = OldCode.TamFermion.enumerate_ferm_excitations_HVA(Lvec; use_pbc=pbc, tie=tie)
        same = length(g_new) == length(g_old) && p_new == p_old &&
               all((a.cre_up, a.ann_up, a.cre_dn, a.ann_dn) == (b.cre_up, b.ann_up, b.cre_dn, b.ann_dn)
                   for (a, b) in zip(g_new, g_old))
        check("$(Lvec[1])x$(Lvec[2]) use_pbc=$pbc tie=$tie: same gates and param_map as main", same,
            "$(length(g_new)) gates, $(maximum(p_new)) coefficients")
    end
    threw = try
        OldCode.TamFermion.enumerate_ferm_excitations_HVA((3, 3); use_pbc=true); false
    catch e
        e isa ArgumentError
    end
    check("main's enumerator rejected 3x3 use_pbc=true (the reason odd lattices fell back to open bonds)", threw)
end

function test_sweep_3x2()
    println("\n== 6. run_hva_parameter_sweep.jl, N=(3, 3)_3x2, U=15, periodic target ==")
    jl = Base.julia_cmd()
    out = joinpath(mktempdir(), "sweep.csv")
    t0 = time()
    log = read(ignorestatus(`$jl -t 4 --project=$PROJECT $(joinpath(ED_DIR, "run_hva_parameter_sweep.jl")) "N=(3, 3)_3x2" 60
        --ref=eigenstate --tie=full --layers=10 --loss=overlap --runs=1 --initialization_samples=10 --maxiters=2000 --out=$out`), String)
    println("    sweep wall time $(round(time() - t0, digits=1)) s")
    check("the sweep chose the periodic HVA without --hva_pbc",
        occursin("--hva_pbc not given; using true", log),
        something(match(r"--hva_pbc not given[^\n]*", log), (match="(line missing)",)).match)
    if !isfile(out)
        check("sweep wrote its CSV", false, "log tail: " * join(last(split(log, '\n'), 10), " | "))
        return
    end
    lines = readlines(out)
    hdr = split(lines[1], ',')
    rows = [Dict(zip(hdr, split(l, ','))) for l in lines[2:end] if !isempty(l)]
    println("    CSV rows:")
    foreach(l -> println("      ", l), lines)
    row = only(rows)
    check("n_params_per_layer = 6 (on-site + 3 groups on the length-3 ring + 2 on the length-2 axis)",
        row["n_params_per_layer"] == "6", "n_params_per_layer = $(row["n_params_per_layer"]), n_params = $(row["n_params"])")
    inf = parse(Float64, row["infidelity"])
    check("P=10 infidelity is below the old open-bond HVA's floor ($OLD_FLOOR, best over P=2..65)",
        inf < OLD_FLOOR, "infidelity = $inf")
end

function (@main)(ARGS)
    log_path = make_log_path(ED_DIR, "test_hva_odd_pbc")
    with_logging(log_path) do
        test_bonds_and_groups()
        test_generator_sums()
        test_unchanged_vs_main()
        test_sweep_3x2()
        println("\n", N_FAILED[] == 0 ? "ALL CHECKS PASSED" : "$(N_FAILED[]) CHECK(S) FAILED")
    end
    return N_FAILED[] == 0 ? 0 : 1
end
