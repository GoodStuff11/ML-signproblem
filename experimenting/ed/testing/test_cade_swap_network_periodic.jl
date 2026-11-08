#=
test_cade_swap_network_periodic.jl

Checks cade_swap_network(...; periodic=true) / cade_swap_network_cost(...; periodic=true) in ucc_hardware_encoding.jl,
the fermionic-swap network of Cade et al. (PRB 102, 235122, 2020) extended to the periodic lattice of the periodic HVA
(enumerate_ferm_excitations_HVA(...; use_pbc=true)): row wrap bonds fused into the FSWAP that exchanges columns 0 and
n_x - 1, column wrap bonds (n_y > 2) as hops carrying their Jordan–Wigner string (:string_hop).

1. Terms: on 3x2, 2x3, 3x3, 4x3, 3x4, 4x2, 2x4, 4x4, 5x3 (both row axes, 2 layers), every non-FSWAP gate is an on-site
   term or a hop on a bond of the periodic lattice, on Jordan–Wigner-adjacent qubits of one spin line (a :string_hop:
   one spin line, rows 0 and n_y-1 of one column); every on-site term and every distinct periodic bond of both spins
   is applied exactly once per layer; the bonds equal those of enumerate_ferm_excitations_HVA(...; use_pbc=true).
2. The qubit circuit (FSWAPs, adjacent hops, FSWAP-hops, string hops with their Z-string sign, on-site phases)
   equals the fermionic hops and on-site terms in network order on random states, with random angles
   (3x2, 2x3, 3x3, 4x2; both row axes; 2 layers).
3. periodic=false is the open network: cnot_depth = 2 depth, no string hops, and the cost function's numbers are
   unchanged (4x3, 12 layers: native depth 84 as in test_ucc_hardware_encoding.jl).
4. Costs: no extra depth or gates from the row wrap bonds (3x2 and 4x2, where the column axis has length 2, have
   the same native gates and depth as the open network with the same row axis; FSWAPs only turn into FSWAP-hops), and
   CNOT counts = 2 x native gates + sum over string hops of (2w - 1). Prints per-layer numbers.

Usage (from experimenting/ed):
  julia --project=.. testing/test_cade_swap_network_periodic.jl
No command-line arguments. Prints PASS/FAIL per check, exits 1 if any fails.
Output is also written to logs/<date>/test_cade_swap_network_periodic_<timestamp>_<pid>.log.
=#

using LinearAlgebra

include(joinpath(@__DIR__, "..", "logging.jl"))
include(joinpath(@__DIR__, "..", "ucc_hardware_encoding.jl"))
using .Trotter
using .Trotter.TamFermion: enumerate_ferm_excitations_HVA

const ED_DIR = normpath(joinpath(@__DIR__, ".."))
const N_FAILED = Ref(0)
function check(name, ok::Bool, detail="")
    println(ok ? "PASS" : "FAIL", "  ", name, isempty(detail) ? "" : "  ($detail)")
    ok || (N_FAILED[] += 1)
    return ok
end

periodic_neighbours(a, b, Lx, Ly) = begin
    xa, ya, xb, yb = div(a, Ly), a % Ly, div(b, Ly), b % Ly
    dx, dy = mod(xa - xb, Lx), mod(ya - yb, Ly)
    (dy == 0 && (dx == 1 || dx == Lx - 1) && Lx > 1) || (dx == 0 && (dy == 1 || dy == Ly - 1) && Ly > 1)
end

function test_terms()
    println("\n== 1. Every periodic bond and on-site term once per layer ==")
    for (Lx, Ly) in ((3, 2), (2, 3), (3, 3), (4, 3), (3, 4), (4, 2), (2, 4), (4, 4), (5, 3)), axis in (1, 2)
        N = Lx * Ly
        net = cade_swap_network((Lx, Ly); row_axis=axis, num_exponentials=2, periodic=true)
        seen = Dict{Tuple{Symbol,Int,Int},Int}()
        ok = true
        for op in net.ops
            op.kind === :fswap && continue
            a, b = op.modes
            if op.bond === :onsite
                ok &= a < N && b == a + N
                key = (:onsite, a, b)
            else
                ok &= div(a, N) == div(b, N) && periodic_neighbours(a % N, b % N, Lx, Ly) &&
                      div(op.qubits[1], N) == div(op.qubits[2], N)
                ok &= op.kind === :string_hop ?
                      (op.qubits[1] % N < net.n_x && op.qubits[2] % N >= N - net.n_x) :
                      abs(op.qubits[1] - op.qubits[2]) == 1
                key = (:hop, minmax(a, b)...)
            end
            seen[key] = get(seen, key, 0) + 1
        end
        gates, _ = enumerate_ferm_excitations_HVA((Lx, Ly); use_pbc=true, tie=:full)
        hva = Set{Tuple{Symbol,Int,Int}}()
        for g in gates
            if g.cre_up == g.ann_up == g.cre_dn == g.ann_dn
                s = trailing_zeros(g.cre_up); push!(hva, (:onsite, s, s + N))
            else
                off = g.cre_up == 0 ? N : 0
                m = g.cre_up | g.cre_dn; n = g.ann_up | g.ann_dn
                push!(hva, (:hop, minmax(trailing_zeros(m) + off, trailing_zeros(n) + off)...))
            end
        end
        n_str = count(op -> op.kind === :string_hop, net.ops)
        check("$(Lx)x$(Ly) rows along axis $axis: terms = periodic HVA terms, each once per layer",
            ok && all(==(2), values(seen)) && Set(keys(seen)) == hva,
            "$(length(seen)) distinct terms ($(length(hva)) in the HVA), $(length(net.ops)) gates over 2 layers, " *
            "$n_str string hops, n_x=$(net.n_x) n_y=$(net.n_y)")
    end
end

# qubit picture: bit q = occupation of Jordan–Wigner position q
function apply_qubit!(ψ, op, θ)
    qa, qb = op.qubits
    ma, mb = (1 << qa), (1 << qb)
    if op.kind === :onsite
        for i in 0:length(ψ)-1
            (i & ma != 0 && i & mb != 0) && (ψ[i+1] *= cis(θ))
        end
        return ψ
    end
    op.kind === :string_hop || abs(qa - qb) == 1 || error("non-adjacent qubit gate $op")
    if op.kind in (:hop, :fswap_hop, :string_hop)      # exp(iθ(c†c + h.c.)) in the CURRENT qubit order
        lo, hi = minmax(qa, qb)
        between = ((1 << hi) - 1) & ~((1 << (lo + 1)) - 1)
        for i in 0:length(ψ)-1
            (i & ma != 0 && i & mb == 0) || continue
            j = xor(i, ma | mb)
            sgn = isodd(count_ones(i & between)) ? -1 : 1
            a, b = ψ[i+1], ψ[j+1]
            ψ[i+1], ψ[j+1] = cos(θ) * a + im * sgn * sin(θ) * b, cos(θ) * b + im * sgn * sin(θ) * a
        end
    end
    if op.kind in (:fswap, :fswap_hop)                  # FSWAP: swap the two bits, -1 if both occupied
        for i in 0:length(ψ)-1
            if i & ma != 0 && i & mb != 0
                ψ[i+1] = -ψ[i+1]
            elseif i & ma != 0 && i & mb == 0
                j = xor(i, ma | mb)
                ψ[i+1], ψ[j+1] = ψ[j+1], ψ[i+1]
            end
        end
    end
    return ψ
end

# fermionic reference in the fixed Jordan–Wigner order of the network's INITIAL positions; FSWAPs are relabellings
function apply_fermionic!(ψ, op, θ, position)
    op.kind === :fswap && return ψ
    pa, pb = position[op.modes[1]+1], position[op.modes[2]+1]
    ma, mb = (1 << pa), (1 << pb)
    if op.kind === :onsite
        for i in 0:length(ψ)-1
            (i & ma != 0 && i & mb != 0) && (ψ[i+1] *= cis(θ))
        end
        return ψ
    end
    lo, hi = minmax(pa, pb)
    between = ((1 << hi) - 1) & ~((1 << (lo + 1)) - 1)
    for i in 0:length(ψ)-1
        (i & ma != 0 && i & mb == 0) || continue
        j = xor(i, ma | mb)
        sgn = isodd(count_ones(i & between)) ? -1 : 1
        a, b = ψ[i+1], ψ[j+1]
        ψ[i+1], ψ[j+1] = cos(θ) * a + im * sgn * sin(θ) * b, cos(θ) * b + im * sgn * sin(θ) * a
    end
    return ψ
end

function test_simulation()
    println("\n== 2. Qubit circuit = fermionic operators (random states and angles) ==")
    state = Ref(UInt64(2026_11_08))
    rnd() = (state[] = state[] * 0x5851F42D4C957F2D + 0x14057B7EF767814F; (state[] >> 11) / 2.0^53)
    for (Lx, Ly) in ((3, 2), (2, 3), (3, 3), (4, 2)), axis in (1, 2)
        net = cade_swap_network((Lx, Ly); row_axis=axis, num_exponentials=2, periodic=true)
        dim = 1 << (2 * Lx * Ly)
        ψ0 = [complex(rnd() - 0.5, rnd() - 0.5) for _ in 1:dim]
        ψ0 ./= norm(ψ0)
        ψq, ψf = copy(ψ0), copy(ψ0)
        for op in net.ops
            θ = 2π * rnd()
            apply_qubit!(ψq, op, θ)
            apply_fermionic!(ψf, op, θ, net.position)
        end
        err = norm(ψq - ψf)
        check("$(Lx)x$(Ly) rows along axis $axis, periodic, 2 layers", err < 1e-10,
            "|ψ_qubit - ψ_fermion| = $(round(err, sigdigits=3)), $(length(net.ops)) gates, " *
            "$(count(op -> op.kind === :string_hop, net.ops)) string hops")
    end
end

function test_open_unchanged()
    println("\n== 3. periodic=false is the open network ==")
    ok = true
    for (Lx, Ly) in ((3, 2), (3, 3), (4, 3), (4, 4), (5, 4)), axis in (1, 2)
        net = cade_swap_network((Lx, Ly); row_axis=axis, num_exponentials=3)
        ok &= net.cnot_depth == 2 * net.depth && !any(op -> op.kind === :string_hop, net.ops)
    end
    check("open network: CNOT depth = 2 x native depth, no string hops (5 lattices x 2 axes, 3 layers)", ok)
    c = cade_swap_network_cost((4, 3); num_exponentials=12,
        param_map=last(enumerate_ferm_excitations_HVA((4, 3); use_pbc=false, tie=:full)))
    check("open HVA 4x3, 12 layers: native depth 84, CNOT depth 168, 2 CNOTs per native gate (as before)",
        c.native_depth == 84 && c.depth == 168 && c.n_two_qubit == 2 * c.n_native_two_qubit && c.n_string_hop == 0,
        "native depth $(c.native_depth), depth $(c.depth), $(c.n_native_two_qubit) native gates")
end

function test_costs()
    println("\n== 4. Periodic costs per layer ==")
    for L in ((3, 2), (4, 2), (3, 3), (4, 3))
        _, pm_o = enumerate_ferm_excitations_HVA(L; use_pbc=false, tie=:full)
        _, pm_p = enumerate_ferm_excitations_HVA(L; use_pbc=true, tie=:full)
        o = cade_swap_network_cost(L; num_exponentials=1, param_map=pm_o)
        p = cade_swap_network_cost(L; num_exponentials=1, param_map=pm_p, periodic=true)
        k = p.counts_by_kind
        cn = 2 * p.n_native_two_qubit + p.n_string_hop_cnots
        println("    $(L[1])x$(L[2]): open $(o.n_two_qubit) CNOTs depth $(o.depth) [$(o.order_strategy)]; periodic $(p.n_two_qubit) CNOTs " *
                "depth $(p.depth) [$(p.order_strategy)]; periodic gates fswap $(get(k, :fswap, 0)) fswap_hop $(get(k, :fswap_hop, 0)) " *
                "hop $(get(k, :hop, 0)) onsite $(get(k, :onsite, 0)) string_hop $(get(k, :string_hop, 0)) ($(p.n_string_hop_cnots) CNOTs); " *
                "parameters $(o.n_parameters) / $(p.n_parameters)")
        check("$(L[1])x$(L[2]) periodic: CNOTs = 2 x native + string-hop CNOTs", p.n_two_qubit == cn, "$(p.n_two_qubit) vs $cn")
        if p.n_string_hop == 0
            # same row axis as the periodic network (the open one may pick the other axis)
            o2 = cade_swap_network_cost(L; num_exponentials=1, param_map=pm_o, row_axis=p.settings.row_axis)
            check("$(L[1])x$(L[2]): column axis of length <= 2, so periodic = open native gate count and depth " *
                  "on the same row axis (row wrap bonds ride on existing FSWAPs)",
                p.n_native_two_qubit == o2.n_native_two_qubit && p.depth == o2.depth,
                "rows along axis $(p.settings.row_axis): native gates $(p.n_native_two_qubit) vs $(o2.n_native_two_qubit), " *
                "depth $(p.depth) vs $(o2.depth)")
        end
    end
end

function (@main)(ARGS)
    log_path = make_log_path(ED_DIR, "test_cade_swap_network_periodic")
    with_logging(log_path) do
        test_terms()
        test_simulation()
        test_open_unchanged()
        test_costs()
        println("\n", N_FAILED[] == 0 ? "ALL CHECKS PASSED" : "$(N_FAILED[]) CHECK(S) FAILED")
    end
    return N_FAILED[] == 0 ? 0 : 1
end
