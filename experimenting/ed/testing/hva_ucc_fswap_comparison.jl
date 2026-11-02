#=
hva_ucc_fswap_comparison.jl

The qubit table of the HVA benchmark with fermionic swaps (paper appendix "Comparison with the Hamiltonian
variational ansatz", tab:hva_fswap): the pruned momentum-space UCC layer against the HVA, both on a line of
Jordan–Wigner qubits whose strings are removed with fermionic swaps (fSWAPs).

  UCC: the pruned layer of hva_ucc_fermionic_comparison.jl, from its `ucc_setup` and the same bisection (keep the
       K largest |A_g|, smallest K with fidelity >= F*, raw truncation, no re-optimization), so K equals that
       script's (and tab:hva's) angle count. The kept double excitations are routed string-free with
       `route_string_free` (ucc_fswap_routing.jl, strategy "route", lookahead 16) and costed with `routed_cost`:
       2 CNOTs per fSWAP, the Yordanov et al. circuit (2w + 5 CNOTs, w = 4 once string-free) per excitation, and
       as-soon-as-possible depth in CNOT layers. This is how the fSWAP column of the encoding table (tab:encoding)
       costs the full layer. The greedy router breaks ties by input order, so the kept excitations are fed in the
       enumeration order that table used (enumerate_ferm_excitations(2, Lvec; conserve_mom, conserve_sz)); in the
       coefficient files' order the counts move by up to ~10%.
  HVA: Cade et al.'s fermionic-swap network (`cade_swap_network_cost`, open bonds), read from the jw_fswap,
       jw_fswap_hop (fSWAPs fused with a hopping term; both counted as fSWAPs), jw_cnot and jw_depth columns of data-summary/hva_ucc_fermionic_comparison.csv (eigenstate reference), so that CSV must
       be current (run hva_ucc_fermionic_comparison.jl first).
The periodic 4x2 case of testing/hva_ucc_4x2_periodic_comparison.jl is not included: cade_swap_network_cost builds
open bonds only.

Check (PASS/FAIL line): the full 4x3 (5,4) layer routed this way reproduces the encoding table's 418 fSWAPs and
15,032 CNOTs (data-summary/ucc_fswap_routing.csv, strategy route).

Usage (from experimenting/ed):
  julia --project=.. testing/hva_ucc_fswap_comparison.jl [--targets=0.95,0.99]
Arguments:
  --targets=<list> (optional): comma-separated target fidelities F*. Default: 0.95,0.99. Each must be a value
                  present in hva_ucc_fermionic_comparison.csv's F_target column (the HVA side is read from it).
Prints the PASS/FAIL check and one table row per (system, U, F*): UCC angles kept, fSWAPs, CNOTs, CNOT depth;
HVA layers, fSWAPs, CNOTs, CNOT depth. Log in testing/logs/<date>/.
=#

module Cmp
include(joinpath(@__DIR__, "..", "hva_ucc_fermionic_comparison.jl"))
include(joinpath(@__DIR__, "..", "ucc_fswap_routing.jl"))
end
using .Cmp: ucc_setup, CASES, circuit_excitations, route_string_free, routed_cost, make_log_path, with_logging
using .Cmp.CSV, .Cmp.DataFrames, Printf

const ED_DIR = normpath(joinpath(@__DIR__, ".."))
const LATTICE_OF_N = Dict(6 => (3, 2), 8 => (4, 2), 9 => (3, 3), 12 => (4, 3))

"""
    parse_cli_args(args) -> Vector{Float64}

`--targets=` (comma-separated F*), default [0.95, 0.99].
"""
function parse_cli_args(args)
    targets = [0.95, 0.99]
    for a in args
        if startswith(a, "--targets=")
            targets = parse.(Float64, split(split(a, "=", limit=2)[2], ","))
        else
            error("Unknown command line argument: $a")
        end
    end
    return targets
end

mkey(e) = (Tuple(sort(collect(e.creators))), Tuple(sort(collect(e.annihilators))))

"""Rank of every double excitation in the enumeration order of ucc_fswap_routing.jl's `lattice_excitations`."""
function enumeration_rank(N)
    g = Cmp.Trotter.TamFermion.enumerate_ferm_excitations(2, LATTICE_OF_N[N]; conserve_mom=true, conserve_sz=true,
        include_diagonal=false)
    return Dict(mkey(e) => i for (i, e) in enumerate(circuit_excitations(g, N; antihermitian=true)))
end

"""fSWAP-routed cost of the excitations with nonzero coefficient in `A`, fed in enumeration order."""
function routed(gates, N, A, rank)
    exc = filter(e -> e.kind === :double, circuit_excitations(gates, N; antihermitian=true, coefficients=A, drop_tol=0.0))
    all(e -> haskey(rank, mkey(e)), exc) || error("an excitation is missing from the enumeration order")
    ops, _ = route_string_free(sort(exc; by=e -> rank[mkey(e)]), 2N)
    return merge(routed_cost(ops, 2N), (n_exc=length(exc),))
end

function run(targets)
    hva = CSV.read(joinpath(ED_DIR, "data-summary", "hva_ucc_fermionic_comparison.csv"), DataFrame)
    hva = hva[(hva.series .== "HVA") .& (hva.reference .== "eigenstate"), :]
    ranks = Dict{Int,Dict}()
    rows = []
    for (label, folder, Lvec, U, _) in CASES
        S = ucc_setup(folder, Lvec, U)
        rank = get!(() -> enumeration_rank(S.N), ranks, S.N)
        if label == "4x3 (5,4)" && U == 2.0
            r = routed(S.gates, S.N, S.A, rank)
            ok = r.n_exc == 1092 && r.n_fswap == 418 && r.n_two_qubit == 15032
            println("CHECK full 4x3 layer routed: $(r.n_exc) excitations, $(r.n_fswap) fSWAPs, $(r.n_two_qubit) CNOTs: ",
                ok ? "PASS (encoding table: 1092 / 418 / 15,032)" : "FAIL (encoding table: 1092 / 418 / 15,032)")
        end
        order = sortperm(abs.(S.A); rev=true)
        keepK(K) = (Ak = zeros(length(S.A)); Ak[order[1:K]] = S.A[order[1:K]]; Ak)
        for Fstar in targets
            h = hva[(hva.system .== label) .& (hva.U .== U) .& (hva.F_target .== Fstar), :]
            nrow(h) == 1 || error("$label U=$U F*=$Fstar: expected one HVA row in the CSV, found $(nrow(h))")
            # fSWAPs of the swap network: plain fSWAPs plus fSWAPs fused with a hopping term
            hv = (layers=h.layers_costed[1], fswap=h.jw_fswap[1] + h.jw_fswap_hop[1], cnot=h.jw_cnot[1], depth=h.jw_depth[1])
            if S.l0 > 1 - Fstar
                push!(rows, (label, U, Fstar, nothing, hv))
                continue
            end
            lo, hi = 0, length(S.A)
            while hi - lo > 1
                mid = (lo + hi) ÷ 2
                S.infid(keepK(mid), S.row) <= 1 - Fstar ? (hi = mid) : (lo = mid)
            end
            r = routed(S.gates, S.N, keepK(hi), rank)
            r.n_exc == hi || error("$label U=$U F*=$Fstar: routed $(r.n_exc) excitations, kept $hi")
            push!(rows, (label, U, Fstar, (K=hi, P=length(S.A), fswap=r.n_fswap, cnot=r.n_two_qubit, depth=r.depth), hv))
        end
    end
    println("\nsystem | U/t | F* | UCC angles | UCC fSWAPs | UCC CNOTs | UCC depth | HVA layers | HVA fSWAPs | HVA CNOTs | HVA depth")
    for (label, U, Fstar, u, hv) in rows
        ucc = isnothing(u) ? "— | — | — | —" : "$(u.K)/$(u.P) | $(u.fswap) | $(u.cnot) | $(u.depth)"
        println("$label | $(Int(U)) | $Fstar | $ucc | $(hv.layers) | $(hv.fswap) | $(hv.cnot) | $(hv.depth)")
    end
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "hva_ucc_fswap_comparison")
    with_logging(log_path) do
        cd(ED_DIR)
        run(parse_cli_args(ARGS))
    end
end
