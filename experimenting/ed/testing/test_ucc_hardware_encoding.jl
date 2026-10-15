#=
test_ucc_hardware_encoding.jl

Tests for ../ucc_hardware_encoding.jl (yordanov_encoding_cost, fermionic_encoding_cost). Every
number the cost functions rely on is checked against its source, and every optimization is checked
to leave the circuit unchanged:

  1. Yordanov per-excitation two-qubit counts and depths against the formulas stated in
     Yordanov et al., PRA 102, 062612 (2020), Secs. IV A-B (staircase), and the balanced-tree depths.
  2. Yordanov single-qubit counts (21 / 16 fused, 7 / 6 fused), counted gate by gate from the
     paper's own circuit figures in its arXiv TeX source (downloaded to a temporary directory).
  3. González-Cuadra et al., PNAS 120, e2304294120 (2023), Fig. 3(a): the 5-layer, 10-gate
     decomposition reproduces the pair-tunneling gate on the 4-mode Fock space; the swapped
     tunneling pairing is the same gate with θ₁ -> -θ₁; two tunneling gates on one pair merge into one.
  4. Exactness of the optimized circuits: applying the factors in the execution order each encoding
     chose gives the same state as program order (3x2 lattice, N↑ = N↓ = 2, random coefficients);
     `gate_order=:best` is shown to give a different state, as documented.
  5. The unoptimized Yordanov two-qubit count of the momentum-space UCC gate set agrees with the
     independent Python enumeration in 68f6bef0eaaf5c0928a922c6/gate_count_scripts (Σ(2w+5)).
  6. Cost tables for the lattices used in the paper, and for a pruned (--target_fidelity) run.
  11. The Cade et al. (PRB 102, 235122, 2020) fermionic-swap network for the HVA: every gate is a lattice
     term on Jordan–Wigner-adjacent qubits and each term is applied once per layer; the qubit circuit
     equals the fermionic operators on random states; depths against Cade Table I.

Usage (from experimenting/ed):
  julia --project=.. testing/test_ucc_hardware_encoding.jl
No command-line arguments. Prints PASS/FAIL per check and exits with status 1 if any check fails.
Output is also written to logs/<date>/test_ucc_hardware_encoding_<timestamp>_<pid>.log.
=#

using LinearAlgebra
using JLD2

include(joinpath(@__DIR__, "..", "logging.jl"))
include(joinpath(@__DIR__, "..", "data_path.jl"))
include(joinpath(@__DIR__, "..", "ucc_hardware_encoding.jl"))
using .Trotter.TamFermion: FGate, getReducedHilSpace, fgateToExpSector
import .Trotter: load_saved_dict

const REFERENCE_SCRIPTS = normpath(joinpath(@__DIR__, "..", "..", "gate_count_scripts"))   # moved from 68f6bef0eaaf5c0928a922c6/gate_count_scripts
const N_FAILED = Ref(0)

function check(name, ok::Bool, detail="")
    println(ok ? "PASS" : "FAIL", "  ", name, isempty(detail) ? "" : "  ($detail)")
    ok || (N_FAILED[] += 1)
    return ok
end

mask(bits) = UInt32(sum((UInt32(1) << b for b in bits); init=UInt32(0)))

# ─── 1. Yordanov formulas ───────────────────────────────────────────────

function test_yordanov_formulas()
    println("\n== 1. Yordanov per-excitation counts vs. the paper's formulas ==")
    N = 16
    println(" kind     w | 2q (paper)  depth staircase (paper) | depth tree")
    all_ok = true
    for w in 4:14
        # modes 0 < 1 < 2 < 2+b sorted, string pairs (0,1) and (2,2+b): w = 2 + 1 + b
        b = w - 3
        g = FGate(mask((0, 2)), mask((1, 2 + b)), 0, 0)
        base = yordanov_encoding_cost([g], N; optimize=false)
        tree = yordanov_encoding_cost([g], N; optimize=true, jw_ordering=:blocked)
        paper_2q = 2w + 5
        paper_depth = w == 4 ? 11 : max(13, 2w - 1)
        m = w - 4
        tree_depth = m == 0 ? 11 : max(13, 2 * (m <= 1 ? 0 : ceil(Int, log2(m))) + 9)
        @assert base.mean_support == w
        println(" double  $(lpad(w,2)) | $(lpad(base.n_two_qubit,3)) ($(lpad(paper_2q,3)))   $(lpad(base.depth,3)) ($(lpad(paper_depth,3)))              | $(lpad(tree.depth,3))")
        all_ok &= base.n_two_qubit == paper_2q && base.depth == paper_depth && tree.depth == tree_depth &&
                  tree.n_two_qubit == paper_2q && base.n_single_qubit == 21 && tree.n_single_qubit == 16
    end
    for w in 2:12
        g = FGate(mask((0,)), mask((w - 1,)), 0, 0)
        base = yordanov_encoding_cost([g], N; optimize=false)
        tree = yordanov_encoding_cost([g], N; optimize=true, jw_ordering=:blocked)
        paper_2q = 2w - 1
        paper_depth = w == 2 ? 3 : max(5, 2w - 3)
        m = w - 2
        tree_depth = m == 0 ? 3 : max(5, 2 * (m <= 1 ? 0 : ceil(Int, log2(m))) + 3)
        println(" single  $(lpad(w,2)) | $(lpad(base.n_two_qubit,3)) ($(lpad(paper_2q,3)))   $(lpad(base.depth,3)) ($(lpad(paper_depth,3)))              | $(lpad(tree.depth,3))")
        all_ok &= base.n_two_qubit == paper_2q && base.depth == paper_depth && tree.depth == tree_depth &&
                  base.n_single_qubit == 7 && tree.n_single_qubit == 6
    end
    check("Yordanov two-qubit counts 2w+5 / 2w-1, depths max(13,2w-1) / max(5,2w-3), tree depths, 1q 21/16 and 7/6", all_ok)
    # hermitian convention: +2 single-qubit gates (S, S† conjugation), same two-qubit cost
    g = FGate(mask((0, 2)), mask((1, 5)), 0, 0)
    a = yordanov_encoding_cost([g], N; optimize=false, antihermitian=true)
    h = yordanov_encoding_cost([g], N; optimize=false, antihermitian=false)
    check("non-antihermitian factor adds exactly 2 single-qubit gates", h.n_single_qubit == a.n_single_qubit + 2 && h.n_two_qubit == a.n_two_qubit)
end

# ─── 2. Yordanov single-qubit counts from the paper's TeX ───────────────

function qcircuit_rows(tex::String, label::String)
    i = findfirst("\\label{$label}", tex)
    j = findprev("\\Qcircuit", tex, first(i))
    block = tex[first(j):first(i)-1]
    rows = Vector{String}[]
    for r in split(block, "\\\\")
        cells = strip.(split(r, '&'))
        any(c -> startswith(c, "\\gate") || startswith(c, "\\ctrl") || startswith(c, "\\targ"), cells) || continue
        push!(rows, [c for c in cells if startswith(c, "\\gate") || startswith(c, "\\ctrl") || startswith(c, "\\targ")])
    end
    return rows
end

function count_single_qubit(rows)
    total, runs = 0, 0
    for r in rows
        one = [startswith(c, "\\gate") for c in r]
        total += count(one)
        runs += count(k -> one[k] && (k == 1 || !one[k-1]), eachindex(one))
    end
    return total, runs
end

function test_yordanov_single_qubit_from_paper()
    println("\n== 2. Yordanov single-qubit gates, counted from the paper's figures (arXiv:2005.14475 TeX) ==")
    dir = mktempdir()
    tarball = joinpath(dir, "src.tar.gz")
    ok = success(pipeline(`curl -sL --max-time 60 -o $tarball https://arxiv.org/e-print/2005.14475`)) &&
         success(`tar xzf $tarball -C $dir`)
    texs = ok ? filter(f -> endswith(f, ".tex"), readdir(dir; join=true)) : String[]
    if isempty(texs)
        println("SKIP  could not download the arXiv source; checks 2 not run")
        N_FAILED[] += 1
        return
    end
    tex = read(first(texs), String)
    for (label, want_total, want_runs, what) in (
        ("fig:d_q_exc_full", 21, 16, "double qubit excitation, Fig. 6"),
        ("fig:s_exchange", 7, 6, "single qubit excitation, Fig. 4b"))
        rows = qcircuit_rows(tex, label)
        total, runs = count_single_qubit(rows)
        for r in rows
            println("    ", join(startswith(c, "\\gate") ? "1" : "2" for c in r), "   (1 = single-qubit gate, 2 = end of a two-qubit gate)")
        end
        check("$what: $total single-qubit gates, $runs after merging adjacent ones (code uses $want_total / $want_runs)",
            total == want_total && runs == want_runs)
    end
end

# ─── 3. González-Cuadra et al. Fig. 3(a) ────────────────────────────────

const σm = ComplexF64[0 1; 0 0]
const Zm = ComplexF64[1 0; 0 -1]
const I2 = Matrix{ComplexF64}(I, 2, 2)
annihilator(j, n) = reduce(kron, [k < j ? Zm : (k == j ? σm : I2) for k in 1:n])
number(j, n) = annihilator(j, n)' * annihilator(j, n)
U_t(i, j, n, θ1, θ2, θ3) = exp(-im * (θ1 / 2 * (exp(-im * θ2) * annihilator(i, n)' * annihilator(j, n) +
                                               exp(im * θ2) * annihilator(j, n)' * annihilator(i, n)) +
                                      θ3 / 2 * (number(i, n) - number(j, n))))
U_int(i, j, n, θ) = exp(-im * θ * number(i, n) * number(j, n))
function U_pt(i, j, k, l, n, θ1, θ2)
    A = annihilator(i, n)' * annihilator(j, n)' * annihilator(k, n) * annihilator(l, n)
    return exp(-im * θ1 * (exp(-im * θ2) * A + exp(im * θ2) * A'))
end
phase_fidelity(U, V) = abs(tr(U' * V)) / size(U, 1)

function test_gonzalez_cuadra_decomposition()
    println("\n== 3. González-Cuadra et al. Fig. 3(a) pair-tunneling decomposition ==")
    n = 4
    i, j, k, l = 1, 2, 3, 4
    worst = 1.0
    for (θ1, θ2) in ((0.37, 1.1), (-1.3, 0.2), (2.2, -2.9), (0.9, 2.4), (π / 2, π / 2))
        layer_t(a) = U_t(i, k, n, a...) * U_t(j, l, n, a...)
        layer_i(φ) = U_int(i, j, n, φ) * U_int(k, l, n, φ)
        V = layer_t((π / 2, θ2 / 2 + π, 0.0)) * layer_i(θ1) * layer_t((π / 2, θ2 / 2 + π / 2, 0.0)) *
            layer_i(-θ1) * layer_t((sqrt(8) * π / sqrt(27), θ2 / 2 - π / 4, 2π / sqrt(27)))
        f = phase_fidelity(U_pt(i, j, k, l, n, θ1, θ2), V)
        println("    θ₁=$(round(θ1, digits=3)) θ₂=$(round(θ2, digits=3)):  |tr(U_pt† V)|/16 = $f")
        worst = min(worst, f)
    end
    check("6 tunneling + 4 interaction gates in 5 layers reproduce U^(pt)(θ₁,θ₂) up to a global phase", worst > 1 - 1e-12, "worst fidelity $(worst)")

    # the code's own native-gate list for one double excitation
    e = HardwareExcitation(:double, [0, 1], [2, 3], 1, 1)
    layers = fermionic_native_layers(e, ((0, 2), (1, 3)))
    check("fermionic_native_layers: 5 layers, 6 tunneling + 4 interaction, matching the figure's pairs",
        length(layers) == 5 && count(op -> op[1] === :tunnel, reduce(vcat, layers)) == 6 &&
        count(op -> op[1] === :interact, reduce(vcat, layers)) == 4 &&
        layers[1] == [(:tunnel, 0, 2), (:tunnel, 1, 3)] && layers[2] == [(:interact, 0, 1), (:interact, 2, 3)])

    # the other pairing: c†_i c†_j c_l c_k = -c†_i c†_j c_k c_l, so U^pt_{ijlk}(θ) = U^pt_{ijkl}(-θ)
    θ1, θ2 = 0.81, -0.4
    check("swapped tunneling pairing is U^(pt) with θ₁ -> -θ₁ (exact)",
        phase_fidelity(U_pt(i, j, l, k, n, θ1, θ2), U_pt(i, j, k, l, n, -θ1, θ2)) > 1 - 1e-12)

    # merging: two tunneling gates on one pair are again a single tunneling gate
    m = 2
    X = annihilator(1, m)' * annihilator(2, m) + annihilator(2, m)' * annihilator(1, m)
    Y = im * (annihilator(1, m)' * annihilator(2, m) - annihilator(2, m)' * annihilator(1, m))
    Z = number(1, m) - number(2, m)
    worst = 1.0
    for (a, b) in (((0.7, 0.3, -0.5), (1.9, -1.2, 0.8)), ((2.5, 2.0, 1.0), (0.4, -0.9, -2.2)))
        P = U_t(1, 2, m, b...) * U_t(1, 2, m, a...)
        H = im * log(P)                                   # P = exp(-iH)
        cx, cy, cz = (real(tr(G' * H)) / real(tr(G' * G)) for G in (X, Y, Z))
        # e^{-iθ2} c†_1 c_2 + h.c. = cosθ2 X - sinθ2 Y, so U_t(θ1, θ2, θ3) = exp(-i[θ1/2 (cosθ2 X - sinθ2 Y) + θ3/2 Z])
        θ1m, θ2m, θ3m = 2 * hypot(cx, cy), atan(-cy, cx), 2cz
        worst = min(worst, phase_fidelity(P, U_t(1, 2, m, θ1m, θ2m, θ3m)))
    end
    check("product of two tunneling gates on the same pair equals one tunneling gate U^(t)(θ⃗')", worst > 1 - 1e-12, "worst fidelity $(worst)")
end

# ─── 4. Exactness of the optimized execution orders ─────────────────────

function apply_in_order(gates, coefs, order, N, basis, ψ)
    ops = fgateToExpSector(gates[order], coefs[order], N, basis; antihermitian=true)
    for op in ops
        ψ = op * ψ
    end
    return ψ
end

function test_reordering_is_exact()
    println("\n== 4. Optimized execution orders implement the same circuit (3x2, N↑ = N↓ = 2) ==")
    Lvec, N = (3, 2), 6
    gates = enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=false)
    ups = getReducedHilSpace(N, 2, false)
    basis = vec([UInt64(u) | (UInt64(d) << N) for u in ups, d in ups])
    # deterministic pseudo-random coefficients and state
    coefs = [0.6 * sin(1.7k + 0.3) for k in 1:length(gates)]
    ψ0 = normalize(ComplexF64[cos(0.37k) + im * sin(1.1k) for k in 1:length(basis)])
    ref = apply_in_order(gates, coefs, collect(1:length(gates)), N, basis, ψ0)
    for (name, r) in (
        ("Yordanov, optimized", yordanov_encoding_cost(gates, N; optimize=true)),
        ("fermionic, optimized", fermionic_encoding_cost(gates, N; optimize=true)),
        ("fermionic, objective=:two_qubit", fermionic_encoding_cost(gates, N; optimize=true, objective=:two_qubit)))
        order = [r.excitations[b].gate_index for b in r.execution_order]
        moved = count(k -> order[k] != k, eachindex(order))
        d = norm(apply_in_order(gates, coefs, order, N, basis, ψ0) - ref)
        check("$name: reported execution order ($moved of $(length(order)) factors moved) gives the same state", d < 1e-10, "‖ψ - ψ_ref‖ = $d")
    end
    # The cost functions keep program order when the scheduler does not beat it, so also apply the
    # commutation-aware scheduler's own orders directly (qubit resources and mode resources).
    ex = circuit_excitations(gates, N)
    modes = [excitation_modes(e) for e in ex]
    supports = [jw_support(e, collect(0:2N-1)) for e in ex]
    for (name, res, dur) in (
        ("schedule_blocks on JW qubits (Yordanov blocks)", supports,
            [yordanov_excitation_cost(e.kind, length(s)).depth for (e, s) in zip(ex, supports)]),
        ("schedule_blocks on modes (fermionic blocks)", modes, [e.kind === :double ? 5 : 1 for e in ex]))
        order, start, _ = schedule_blocks(res, dur, modes, 2N, 2N; reorder=true)
        order = order[sortperm(start[order]; alg=Base.Sort.DEFAULT_STABLE)]
        moved = count(k -> order[k] != k, eachindex(order))
        d = norm(apply_in_order(gates, coefs, [ex[b].gate_index for b in order], N, basis, ψ0) - ref)
        check("$name: reordered sequence ($moved of $(length(order)) factors moved) gives the same state",
            d < 1e-10 && moved > 0, "‖ψ - ψ_ref‖ = $d")
    end
    r = yordanov_encoding_cost(gates, N; optimize=true, gate_order=:best)
    order = [r.excitations[b].gate_index for b in r.execution_order]
    d = norm(apply_in_order(gates, coefs, order, N, basis, ψ0) - ref)
    check("gate_order=:best is a different circuit, as documented", d > 1e-3, "‖ψ - ψ_ref‖ = $d")
end

# ─── 5. Cross-check against the independent Python enumeration ──────────

function python_reference(lattices)
    dir = mktempdir()
    for f in ("enumerate.py", "gates2.py")
        cp(joinpath(REFERENCE_SCRIPTS, f), joinpath(dir, f))
    end
    code = """
import io, contextlib, sys
sys.path.insert(0, '.')
with contextlib.redirect_stdout(io.StringIO()):
    import gates2
for L in [$(join(("($(a),$(b))" for (a, b) in lattices), ","))]:
    blocks, N, M = gates2.build(*L)
    print(L[0], L[1], len(blocks), sum(2*b[1]+5 for b in blocks))
"""
    out = cd(() -> read(`python3 -I -c $code`, String), dir)   # -I: isolated, cwd added via sys.path
    rows = Dict{Tuple{Int,Int},Tuple{Int,Int}}()
    for line in split(strip(out), '\n')
        a, b, P, c = parse.(Int, split(line))
        rows[(a, b)] = (P, c)
    end
    return rows
end

function test_against_python_reference()
    println("\n== 5. Unoptimized Yordanov two-qubit count vs. gate_count_scripts (Python, Σ(2w+5)) ==")
    lattices = [(3, 2), (4, 2), (3, 3), (4, 3), (4, 4)]
    ref = python_reference(lattices)
    all_ok = true
    for Lvec in lattices
        N = prod(Lvec)
        gates = enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=false)
        r = yordanov_encoding_cost(gates, N; optimize=false)
        P_py, c_py = ref[Lvec]
        println("    $(Lvec[1])x$(Lvec[2]): factors $(r.n_excitations) (Python $P_py), two-qubit $(r.n_two_qubit) (Python $c_py)")
        all_ok &= r.n_excitations == P_py && r.n_two_qubit == c_py
    end
    check("factor count and Σ(2w+5) match the independent Python enumeration for all lattices", all_ok)
end

# ─── 6. Cost tables ─────────────────────────────────────────────────────

fmt(x) = lpad(string(x), 8)

function print_tables()
    println("\n== 6. Cost of the momentum-space UCC circuit (num_exponentials = 1, antihermitian) ==")
    println("  lattice  encoding                          |   2-qubit   1-qubit     depth")
    for Lvec in [(3, 2), (4, 2), (3, 3), (4, 3), (4, 4)]
        N = prod(Lvec)
        gates = enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=false)
        rows = [
            ("Yordanov, unoptimized", yordanov_encoding_cost(gates, N; optimize=false)),
            ("Yordanov, optimized", yordanov_encoding_cost(gates, N; optimize=true)),
            ("Yordanov, free gate order*", yordanov_encoding_cost(gates, N; optimize=true, gate_order=:best)),
            ("fermionic, unoptimized", fermionic_encoding_cost(gates, N; optimize=false)),
            ("fermionic, optimized", fermionic_encoding_cost(gates, N; optimize=true)),
            ("fermionic, free gate order*", fermionic_encoding_cost(gates, N; optimize=true, gate_order=:best)),
        ]
        for (name, r) in rows
            println("  $(Lvec[1])x$(Lvec[2])      $(rpad(name, 33)) | $(fmt(r.n_two_qubit))  $(fmt(r.n_single_qubit))  $(fmt(r.depth))")
        end
        y = rows[2][2]
        println("           (Yordanov: $(y.n_excitations) factors, mean JW support $(round(y.mean_support, digits=3)), Σ block depths $(y.depth_sequential); fermionic: $(rows[5][2].n_fused) gates merged)")
    end
    println("  * a different circuit (non-commuting factors reordered); coefficients would need re-optimizing")

    println("\n  Pruned run: N=(2, 2)_3x2, slater reference, target_fidelity=0.9998, U index 30")
    folder = data_folder("N=(2, 2)_3x2")
    d = load_saved_dict(joinpath(folder, "trotter_N=6_ref_slater_antihermitian_target_fidelity=0.9998_u_30.jld2"))
    shared = load_saved_dict(joinpath(folder, "trotter_N=6_ref_slater_antihermitian_target_fidelity=0.9998_shared.jld2"))
    gates = [g isa FGate ? g : FGate(g.cre_up, g.ann_up, g.cre_dn, g.ann_dn) for g in shared["gates"]]
    A = d["coefficients"]
    for (name, f) in (("Yordanov", yordanov_encoding_cost), ("fermionic", fermionic_encoding_cost))
        full = f(gates, 6; optimize=true)
        pruned = f(gates, 6; optimize=true, coefficients=A)
        println("    $(rpad(name, 10)) all $(full.n_excitations) factors: 2q $(full.n_two_qubit), 1q $(full.n_single_qubit), depth $(full.depth)" *
                "  |  $(pruned.n_excitations) nonzero: 2q $(pruned.n_two_qubit), 1q $(pruned.n_single_qubit), depth $(pruned.depth)")
    end
    pr = yordanov_encoding_cost(gates, 6; coefficients=A)
    check("pruned coefficients: only the nonzero factors are compiled, and only they count as parameters",
        pr.n_excitations == pr.n_parameters == count(!iszero, A) == length(d["active_indices"]))
end

# ─── 7. Jordan–Wigner orderings and why the depth is large ──────────────

function test_orderings_and_depth()
    println("\n== 7. Jordan–Wigner ordering: blocked (all ↑ then all ↓) vs interleaved (0↑ 0↓ 1↑ 1↓ …) ==")
    N = 4
    pos = jw_ordering_positions(:interleaved, N)
    line = fill("", 2N)
    for m in 0:2N-1
        line[pos[m+1]+1] = "$(m % N)$(m < N ? "↑" : "↓")"
    end
    check("interleaved qubit line is 0↑ 0↓ 1↑ 1↓ 2↑ 2↓ 3↑ 3↓", join(line, " ") == "0↑ 0↓ 1↑ 1↓ 2↑ 2↓ 3↑ 3↓", join(line, " "))

    println("  lattice  ordering      mean w   2-qubit   1-qubit     depth")
    ok_opt = true
    for Lvec in [(3, 2), (4, 2), (3, 3), (4, 3), (4, 4)]
        N = prod(Lvec)
        gates = enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=false)
        rs = Dict(o => yordanov_encoding_cost(gates, N; optimize=true, jw_ordering=o) for o in (:blocked, :interleaved, :optimized))
        for o in (:blocked, :interleaved, :optimized)
            r = rs[o]
            println("  $(Lvec[1])x$(Lvec[2])      $(rpad(o, 12)) $(lpad(round(r.mean_support, digits=3), 7)) $(fmt(r.n_two_qubit))  $(fmt(r.n_single_qubit))  $(fmt(r.depth))")
        end
        ok_opt &= rs[:optimized].n_two_qubit <= min(rs[:blocked].n_two_qubit, rs[:interleaved].n_two_qubit)
    end
    check(":optimized ordering is never worse than :blocked or :interleaved", ok_opt)

    println("\n== 8. Why the depth is large: bounds on the depth (optimized settings) ==")
    println("  depth units: Yordanov = two-qubit-gate layers; fermionic = native-gate layers (a double excitation is 5)")
    println("  lattice  encoding    factors | sequential  crit.path |  depth (given order) | best order: longest-first only / best of all rules  [winning rule] | lower bound  max at once")
    ok_bounds = true
    for Lvec in [(3, 2), (4, 2), (3, 3), (4, 3), (4, 4)]
        N = prod(Lvec)
        gates = enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=false)
        for (name, f) in (("Yordanov ", yordanov_encoding_cost), ("fermionic", fermionic_encoding_cost))
            r = f(gates, N; optimize=true)
            free = f(gates, N; optimize=true, gate_order=:best)
            b = r.bounds
            # the previous behaviour: one greedy pass, longest block first, no order constraints
            ex = free.excitations
            res = name == "fermionic" ? [excitation_modes(e) for e in ex] : [jw_support(e, invperm(free.jw_order .+ 1) .- 1) for e in ex]
            durs = name == "fermionic" ? [e.kind === :double ? 5 : 1 for e in ex] :
                   [yordanov_excitation_cost(e.kind, length(q); parity_network=:tree).depth for (e, q) in zip(ex, res)]
            _, _, single = schedule_blocks(res, durs, [Int[] for _ in ex], 2N, 2N; reorder=true)
            println("  $(Lvec[1])x$(Lvec[2])      $name  $(lpad(r.n_excitations, 6)) | $(fmt(b.sequential))   $(fmt(b.critical_path)) |  $(fmt(r.depth))            | $(fmt(single)) / $(fmt(free.depth))  [$(free.order_strategy)] | $(fmt(b.resource_bound))  $(lpad(b.max_concurrent, 6))")
            # merged native gates can pull the fermionic depth slightly below the block-level bound
            ok_bounds &= free.depth + (name == "fermionic" ? free.n_fused : 0) >= b.resource_bound &&
                         (name == "fermionic" || free.depth <= single)
            # fermionic depth counts merged native layers, so it may sit slightly below the block bounds
            slack = name == "fermionic" ? r.n_fused : 0
            ok_bounds &= r.depth + slack >= b.critical_path && r.depth <= b.sequential
        end
    end
    check("given-order depth lies between the critical path and the sequential sum; best-order depth never beats the lower bound and never loses to the single greedy pass", ok_bounds)
end

# ─── 9. Command line ────────────────────────────────────────────────────

function test_command_line()
    println("\n== 9. Command-line entry point (julia --project=.. ucc_hardware_encoding.jl ...) ==")
    script = joinpath(@__DIR__, "..", "ucc_hardware_encoding.jl")
    project = joinpath(@__DIR__, "..", "..")
    jl = Base.julia_cmd()
    out = read(ignorestatus(`$jl --project=$project $script --lattice=4x3 --encoding=yordanov --optimize=true --jw_ordering=interleaved`), String)
    gates = enumerate_ferm_excitations(2, (4, 3); conserve_mom=true, conserve_sz=true, include_diagonal=false)
    r = yordanov_encoding_cost(gates, 12; optimize=true, jw_ordering=:interleaved)
    println(join(("    " * l for l in split(out, '\n') if occursin(":", l) && !occursin("settings", l)), "\n"))
    check("CLI --jw_ordering=interleaved prints the library's two-qubit count, single-qubit count and depth",
        occursin("two-qubit gates     : $(r.n_two_qubit)\n", out) && occursin("single-qubit gates  : $(r.n_single_qubit)\n", out) &&
        occursin("depth               : $(r.depth) ", out) && occursin("jw_ordering = :interleaved", out))
    out = read(ignorestatus(`$jl --project=$project $script --lattice=3x2 --encoding=fermionic --optimize=both --gate_order=best`), String)
    f0 = fermionic_encoding_cost(enumerate_ferm_excitations(2, (3, 2); conserve_mom=true, conserve_sz=true, include_diagonal=false), 6; optimize=false, gate_order=:best)
    f1 = fermionic_encoding_cost(enumerate_ferm_excitations(2, (3, 2); conserve_mom=true, conserve_sz=true, include_diagonal=false), 6; optimize=true, gate_order=:best)
    check("CLI --optimize=both --gate_order=best prints both fermionic results",
        occursin("optimize=false", out) && occursin("optimize=true", out) &&
        occursin("depth               : $(f0.depth) ", out) && occursin("depth               : $(f1.depth) ", out))
    p = run(pipeline(ignorestatus(`$jl --project=$project $script --lattice=4x3 --jw_ordering=zigzag`); stdout=devnull, stderr=devnull))
    check("CLI rejects an invalid option value with a non-zero exit code", p.exitcode != 0, "exit code $(p.exitcode)")
end

# ─── 10. Free-parameter count vs. gate count ────────────────────────────

function test_parameter_counts()
    println("\n== 10. Free parameters vs. gates (factors) ==")
    ok = true
    for (tie, want) in ((:full, 5), (:spin, 29), (:none, 46))
        g, pm = enumerate_ferm_excitations_HVA((4, 3); use_pbc=false, tie=tie)
        y = yordanov_encoding_cost(g, 12; antihermitian=false, param_map=pm)
        f = fermionic_encoding_cost(g, 12; antihermitian=false, param_map=pm)
        println("    HVA 4x3 open, tie=$tie: $(y.n_excitations) gates, $(y.n_parameters) free parameters per layer (expected $want)")
        ok &= y.n_parameters == want == f.n_parameters && y.n_excitations == 46
    end
    check("HVA 4x3 (open): 46 gates per layer carrying 5 / 29 / 46 parameters for tie = full / spin / none", ok)
    g, pm = enumerate_ferm_excitations_HVA((4, 3); use_pbc=false, tie=:full)
    y = yordanov_encoding_cost(g, 12; antihermitian=false, param_map=pm, num_exponentials=12)
    check("HVA tie=full with 12 layers: 60 parameters (= the EHV DOF in gate_count_analysis.tex) on 552 gates",
        y.n_parameters == 60 && y.n_excitations == 552, "$(y.n_parameters) parameters, $(y.n_excitations) gates")
    gates = enumerate_ferm_excitations(2, (4, 3); conserve_mom=true, conserve_sz=true, include_diagonal=false)
    check("momentum-space UCC 4x3: one parameter per gate, 1092", yordanov_encoding_cost(gates, 12).n_parameters == 1092)
    script = joinpath(@__DIR__, "..", "ucc_hardware_encoding.jl")
    out = read(ignorestatus(`$(Base.julia_cmd()) --project=$(joinpath(@__DIR__, "..", "..")) $script --lattice=4x3 --ansatz=hva --hva_pbc=false --antihermitian=false --num_exponentials=12 --encoding=yordanov --optimize=true`), String)
    println(join(("    " * l for l in split(out, '\n') if occursin("free parameters", l) || occursin("gates (factors)", l)), "\n"))
    check("CLI prints the free-parameter count separately from the gate count",
        occursin("free parameters     : 60\n", out) && occursin("gates (factors)     : 552 ", out))
end

# ─── 11. Cade et al. swap network for the HVA ──────────────────────────

# Qubit-level simulation of the network: bit q of the index is the occupation of Jordan–Wigner position q.
function _apply_qubit_op!(ψ, op, θ)
    qa, qb = op.qubits
    ma, mb = (1 << qa), (1 << qb)
    if op.kind === :onsite
        for i in 0:length(ψ)-1
            (i & ma != 0 && i & mb != 0) && (ψ[i+1] *= cis(θ))
        end
        return ψ
    end
    abs(qa - qb) == 1 || error("non-adjacent qubit gate $op")
    if op.kind in (:hop, :fswap_hop)               # exp(iθ(XX+YY)/2) on adjacent qubits: no string
        for i in 0:length(ψ)-1
            (i & ma != 0 && i & mb == 0) || continue
            j = xor(i, ma | mb)
            a, b = ψ[i+1], ψ[j+1]
            ψ[i+1], ψ[j+1] = cos(θ) * a + im * sin(θ) * b, cos(θ) * b + im * sin(θ) * a
        end
    end
    if op.kind in (:fswap, :fswap_hop)             # FSWAP: swap the two bits, -1 if both occupied
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

# Fermionic reference: exp(iθ(c†_a c_b + h.c.)) and exp(iθ n_a n_b) on the Fock space with the fixed
# Jordan–Wigner order of the network's initial positions (strings included); FSWAPs are relabellings.
function _apply_fermionic_op!(ψ, op, θ, position)
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

function test_cade_swap_network()
    println("\n== 11. Cade et al. (PRB 102, 235122, 2020) swap network for the HVA ==")
    # (a) every gate is a lattice term on Jordan–Wigner-adjacent qubits of one spin line, each term once per layer
    ok_terms = true
    for (Lx, Ly) in ((2, 2), (3, 2), (2, 3), (3, 3), (4, 3), (3, 4), (4, 4), (5, 4), (1, 4), (4, 1)), axis in (1, 2)
        N = Lx * Ly
        net = cade_swap_network((Lx, Ly); row_axis=axis, num_exponentials=2)
        xy(s) = (div(s, Ly), s % Ly)
        seen = Dict{Tuple{Symbol,Int,Int},Int}()
        for op in net.ops
            op.kind === :fswap && continue
            a, b = op.modes
            if op.bond === :onsite
                ok_terms &= a < N && b == a + N
                key = (:onsite, a, b)
            else
                (xa, ya), (xb, yb) = xy(a % N), xy(b % N)
                ok_terms &= div(a, N) == div(b, N) && abs(xa - xb) + abs(ya - yb) == 1 &&
                            div(op.qubits[1], N) == div(op.qubits[2], N) && abs(op.qubits[1] - op.qubits[2]) == 1
                key = (:hop, minmax(a, b)...)
            end
            seen[key] = get(seen, key, 0) + 1
        end
        n_bonds = 2 * ((Lx - 1) * Ly + Lx * (Ly - 1))
        ok_terms &= length(seen) == N + n_bonds && all(==(2), values(seen))
    end
    check("every gate acts on Jordan–Wigner-adjacent qubits (or is an on-site phase), and every on-site term and " *
          "every open-boundary bond of both spins is applied exactly once per layer (10 lattices x 2 row axes, 2 layers)", ok_terms)

    # (b) the network implements the fermionic operators it claims (FSWAP bookkeeping and signs)
    ok_state = true
    rng_state = Ref(UInt64(12345))
    rnd() = (rng_state[] = rng_state[] * 0x5851F42D4C957F2D + 0x14057B7EF767814F; (rng_state[] >> 11) / 2.0^53)
    for (Lx, Ly) in ((3, 2), (2, 3), (4, 2), (3, 3)), axis in (1, 2)
        net = cade_swap_network((Lx, Ly); row_axis=axis, num_exponentials=2)
        dim = 1 << (2 * Lx * Ly)
        ψ0 = [complex(rnd() - 0.5, rnd() - 0.5) for _ in 1:dim]
        ψ0 ./= norm(ψ0)
        θs = [2π * rnd() for _ in net.ops]
        ψq, ψf = copy(ψ0), copy(ψ0)
        for (op, θ) in zip(net.ops, θs)
            _apply_qubit_op!(ψq, op, θ)
            _apply_fermionic_op!(ψf, op, θ, net.position)
        end
        err = norm(ψq - ψf)
        println("    $(Lx)x$(Ly), rows along axis $axis: |ψ_qubit - ψ_fermion| = $(round(err, sigdigits=3)) ($(length(net.ops)) gates, depth $(net.depth))")
        ok_state &= err < 1e-10
    end
    check("qubit circuit (FSWAP, adjacent hops, combined FSWAP-hops, on-site phases) = the fermionic hops and on-site " *
          "terms in network order, on random states (3x2, 2x3, 4x2, 3x3; both row axes; 2 layers)", ok_state)

    # (c) depth per layer against Cade Table I and Sec. I C
    ok_even, ok_odd, rows = true, true, String[]
    for nx in 2:7, ny in 2:5
        d = cade_swap_network((nx, ny); row_axis=1).depth
        cade = iseven(nx) ? 2nx + 1 : 2nx + 2
        iseven(nx) ? (ok_even &= d == cade) : (ok_odd &= d <= cade)
        ny == 2 && push!(rows, "n_x=$nx: $d (Cade $cade)")
    end
    println("    native depth per layer: ", join(rows, ", "))
    check("even n_x: native depth per layer = 2n_x + 1, as in Cade Table I (n_x = 2..7, n_y = 2..5)", ok_even)
    check("odd n_x: native depth per layer <= Cade's 2n_x + 2 (the explicit network reaches 2n_x + 1)", ok_odd)
    t = [(4, 4) => 9, (4, 5) => 9, (6, 6) => 13]
    check("Table I examples (fully connected): 4x4 -> 9, 4x5 -> 9, 6x6 -> 13",
        all(cade_swap_network(L; row_axis=1).depth == d for (L, d) in t))
    check("3x3: 57 native two-qubit gates per layer (Cade Sec. I C: 9 + 2x3x(6+2) = 57)",
        length(cade_swap_network((3, 3); row_axis=1).ops) == 57)
    c = cade_swap_network_cost((4, 3); num_exponentials=12,
        param_map=last(enumerate_ferm_excitations_HVA((4, 3); use_pbc=false, tie=:full)))
    println("    HVA 4x3, 12 layers: $(c.n_native_two_qubit) native gates, native depth $(c.native_depth); " *
            "CNOT level $(c.n_two_qubit) / depth $(c.depth); $(c.n_parameters) parameters; $(c.order_strategy)")
    check("HVA 4x3, 12 layers: rows along the length-3 axis, native depth 12 x 7 = 84, 60 parameters",
        c.native_depth == 84 && c.n_parameters == 60 && c.settings.row_axis == 2)

    # (d) command line
    script = joinpath(@__DIR__, "..", "ucc_hardware_encoding.jl")
    jl, project = Base.julia_cmd(), joinpath(@__DIR__, "..", "..")
    out = read(ignorestatus(`$jl --project=$project $script --lattice=4x3 --ansatz=hva --hva_pbc=false --antihermitian=false --num_exponentials=12 --encoding=yordanov --optimize=true`), String)
    println(join(("    " * l for l in split(out, '\n') if occursin("native", l) || occursin("depth  ", l)), "\n"))
    check("CLI --ansatz=hva --encoding=yordanov uses the swap network by default and prints its native and CNOT-level depth",
        occursin("Cade et al. 2020", out) && occursin("native depth        : 84", out) && occursin("depth               : 168 ", out))
    p = run(pipeline(ignorestatus(`$jl --project=$project $script --lattice=4x2 --ansatz=hva --antihermitian=false --encoding=yordanov`); stdout=devnull, stderr=devnull))
    check("CLI rejects --hva_pbc=true with the swap network (Cade's construction is for open boundaries)", p.exitcode != 0)
end

function (@main)(ARGS)
    log_path = make_log_path(joinpath(@__DIR__, ".."), "test_ucc_hardware_encoding")
    with_logging(log_path) do
        test_yordanov_formulas()
        test_yordanov_single_qubit_from_paper()
        test_gonzalez_cuadra_decomposition()
        test_reordering_is_exact()
        test_against_python_reference()
        print_tables()
        test_orderings_and_depth()
        test_command_line()
        test_parameter_counts()
        test_cade_swap_network()
        println("\n", N_FAILED[] == 0 ? "ALL CHECKS PASSED" : "$(N_FAILED[]) CHECK(S) FAILED")
    end
    return N_FAILED[] == 0 ? 0 : 1
end
