#=
test_hva_pbc_depth.jl

Checks the hardware cost of the periodic HVA (enumerate_ferm_excitations_HVA(...; use_pbc=true, tie=:full),
including the third wrap-bond group on an odd axis) as computed by ucc_hardware_encoding.jl.

1. Fermionic tweezer register (fermionic_encoding_cost, González-Cuadra et al. 2023; no connectivity
   constraint, so a wrap bond costs the same as any bond). Hand count per HVA layer:
     two-mode gates = N interaction gates + 2 x (distinct bonds) tunneling gates
       (a length-2 periodic axis emits each bond twice, in consecutive groups; the two copies act on the
        same modes with nothing in between, so they fuse exactly into one gate),
     depth          = 1 (on-site) + number of hopping groups after that fusion
       (each group is site-disjoint, spin-up and spin-down copies act on disjoint modes).
   Checked for P = 1, 2, 5 layers on 3x2, 3x3, 4x3 (odd axis) and 4x2 (control, matches the 4x2 log:
   2 layers = 64 two-mode gates, depth 8), periodic and open. The reported depth must equal the hand count
   (the printed critical path counts HVA gates before fusion, so it can exceed the depth on a length-2 axis).
2. Exactness of the scheduled/fused circuit on 3x2 (3,3) periodic, P = 2: the native gate list,
   rebuilt as a product of exponentials with random angles per HVA gate in the compiled order, gives the
   same state as the HVA circuit in program order (fusion merges equal generators; reordering only
   swaps gates on disjoint modes).
3. Jordan–Wigner qubits: only --hva_qubit_circuit=per_term accepts periodic bonds (cade_swap_network_cost
   is open-boundary only). Prints per_term CNOT count and depth for periodic vs open, and the open
   swap-network numbers, to show what a periodic JW depth would be measured with.

Usage (from experimenting/ed):
  julia --project=.. testing/test_hva_pbc_depth.jl
No command-line arguments. Prints PASS/FAIL per check, exits 1 if any fails.
Output is also written to logs/<date>/test_hva_pbc_depth_<timestamp>_<pid>.log.
=#

using LinearAlgebra
using Random

include(joinpath(@__DIR__, "..", "logging.jl"))
include(joinpath(@__DIR__, "..", "ucc_hardware_encoding.jl"))
using .Trotter
using .Trotter.TamFermion: enumerate_ferm_excitations_HVA, findLatticeEdges

const ED_DIR = normpath(joinpath(@__DIR__, ".."))
const N_FAILED = Ref(0)
function check(name, ok::Bool, detail="")
    println(ok ? "PASS" : "FAIL", "  ", name, isempty(detail) ? "" : "  ($detail)")
    ok || (N_FAILED[] += 1)
    return ok
end

function hand_count(Lvec, pbc)
    N = prod(Lvec)
    distinct = length(unique(minmax(e...) for e in findLatticeEdges(collect(Lvec); use_pbc=pbc) if e[1] != e[2]))
    groups = sum(L -> (L == 1 ? 0 : (!pbc ? (L == 2 ? 1 : 2) : (L == 2 ? 1 : (isodd(L) ? 3 : 2)))), Lvec)
    return (two_mode=N + 2distinct, depth=1 + groups)
end

function test_fermionic_register()
    println("\n== 1. Fermionic register: periodic HVA gates and depth vs hand count ==")
    for Lvec in ((3, 2), (3, 3), (4, 3), (4, 2)), pbc in (true, false)
        N = prod(Lvec)
        gates, pm = enumerate_ferm_excitations_HVA(Lvec; use_pbc=pbc, tie=:full)
        h = hand_count(Lvec, pbc)
        for P in (1, 2, 5)
            r = fermionic_encoding_cost(gates, N; antihermitian=false, param_map=pm, num_exponentials=P)
            check("$(Lvec[1])x$(Lvec[2]) $(pbc ? "periodic" : "open    ") P=$P: two-mode gates and depth = hand count",
                r.n_two_qubit == P * h.two_mode && r.depth == P * h.depth,
                "tool $(r.n_two_qubit) gates / depth $(r.depth); hand $(P*h.two_mode) / $(P*h.depth); " *
                "$(r.n_tunneling) tunneling, $(r.n_interaction) interaction, $(r.n_fused) fused; " *
                "critical path (excitations) $(r.bounds.critical_path); $(r.order_strategy)")
        end
    end
end

# Apply exp(-i θ τ) for a list of (generator modes) in the 2N-mode Fock sector, via the repo's own
# fgateToTauSector on HVA FGates.
function test_exactness()
    println("\n== 2. Scheduled + fused native circuit implements the HVA circuit (3x2 (3,3) periodic, P=2) ==")
    Lvec, nvec, P = (3, 2), (3, 3), 2
    N = prod(Lvec)
    gates, pm = enumerate_ferm_excitations_HVA(Lvec; use_pbc=true, tie=:full)
    basis, _, _ = realspace_basis(Lvec, nvec)
    taus = [Matrix(t) for t in Trotter.fgateToTauSector(gates, N, basis; antihermitian=false)]
    rng = MersenneTwister(7)
    theta = randn(rng, P * maximum(pm))
    angle(l, g) = theta[(l-1)*maximum(pm)+pm[g]]
    psi0 = normalize(randn(rng, ComplexF64, length(basis)))
    # program order
    psi = copy(psi0)
    for l in 1:P, g in eachindex(gates)
        psi = exp(-im * angle(l, g) * taus[g]) * psi
    end
    # compiled order: the cost function's execution order over excitations (one per HVA gate per layer)
    r = fermionic_encoding_cost(gates, N; antihermitian=false, param_map=pm, num_exponentials=P)
    ex = r.excitations
    phi = copy(psi0)
    for b in r.execution_order
        e = ex[b]
        phi = exp(-im * angle(e.layer, e.gate_index) * taus[e.gate_index]) * phi
    end
    check("compiled execution order gives the same state as program order", norm(psi - phi) < 1e-10,
        "||psi_program - psi_compiled|| = $(norm(psi - phi)), order strategy: $(r.order_strategy)")
    # fusion: consecutive equal generators (length-2 axis duplicates) — exp(aτ)exp(bτ) = exp((a+b)τ)
    dup = [(g1, g2) for g1 in eachindex(gates), g2 in eachindex(gates) if g1 < g2 &&
           (gates[g1].cre_up, gates[g1].ann_up, gates[g1].cre_dn, gates[g1].ann_dn) ==
           (gates[g2].cre_up, gates[g2].ann_up, gates[g2].cre_dn, gates[g2].ann_dn)]
    worst = maximum(norm(exp(-im * 0.3 * taus[a]) * exp(-im * 0.5 * taus[b]) - exp(-im * 0.8 * taus[a])) for (a, b) in dup)
    check("fused duplicate gates (length-2 axis) are equal generators, so merging is exact", worst < 1e-12,
        "$(length(dup)) duplicate pairs, max ||e^{-i0.3τ}e^{-i0.5τ} - e^{-i0.8τ}|| = $worst")
end

function test_jordan_wigner()
    println("\n== 3. Jordan–Wigner qubits: what a periodic HVA depth is measured with ==")
    for Lvec in ((3, 2), (3, 3), (4, 3))
        N = prod(Lvec)
        for pbc in (true, false)
            gates, pm = enumerate_ferm_excitations_HVA(Lvec; use_pbc=pbc, tie=:full)
            r = yordanov_encoding_cost(gates, N; antihermitian=false, param_map=pm, num_exponentials=1)
            println("    $(Lvec[1])x$(Lvec[2]) $(pbc ? "periodic" : "open    ") per_term, 1 layer: $(r.n_two_qubit) CNOTs, CNOT depth $(r.depth)")
        end
        _, pm_o = enumerate_ferm_excitations_HVA(Lvec; use_pbc=false, tie=:full)
        s = cade_swap_network_cost(Lvec; num_exponentials=1, param_map=pm_o)
        println("    $(Lvec[1])x$(Lvec[2]) open     swap network, 1 layer: $(s.n_two_qubit) CNOTs, CNOT depth $(s.depth)")
    end
    check("periodic HVA is costed on JW qubits only via per_term (no periodic swap network exists in the code)", true)
end

function (@main)(ARGS)
    log_path = make_log_path(ED_DIR, "test_hva_pbc_depth")
    with_logging(log_path) do
        test_fermionic_register()
        test_exactness()
        test_jordan_wigner()
        println("\n", N_FAILED[] == 0 ? "ALL CHECKS PASSED" : "$(N_FAILED[]) CHECK(S) FAILED")
    end
    return N_FAILED[] == 0 ? 0 : 1
end
