#=
hva_ucc_fermionic_comparison.jl

Fair hardware-cost comparison of the momentum-space UCC layer and the Hamiltonian-variational ansatz (HVA) on
a fermionic tweezer register (González-Cuadra et al. 2023 native gates, `fermionic_encoding_cost` in
ucc_hardware_encoding.jl), at a matched target fidelity F*:

  UCC: the pruned layer, computed directly from the saved coefficients trotter_N=<N>_ref_slater_antihermitian_u_<i>.jld2
       (i = U / 0.25, the convention of fig2.csv): keep the K largest |A_g|, zero the rest (raw truncation, no
       re-optimization, as in Fig. 3), and find by bisection the smallest K with fidelity >= F*, evaluating the
       fidelity against the ED target with Trotter.apply_unitary (CPU). The target row of load_ED_data is the one that
       reproduces the file's saved loss with all coefficients kept (checked to 1e-6). The stored pruning analyses are
       not used: for 3x3 they predate the current coefficient files (2026-08-11 against 2026-08-25).
  HVA: open-boundary (position-space) targets and an open HVA, i.e. Cade et al.'s setting (run_hva_parameter_sweep.jl
       --target_bc=open; benchmarks/hva_parameter_sweep_*_obc), tie=full, best run per layer count. Layers to F*:
       (raw) read off the sweep, log-log interpolation between the sampled layer counts at the first crossing;
       (fit) the power law infidelity = a L^b fitted in log space to all sampled layer counts (hva_analysis.ipynb
       cell 2), solved for 1 - F*. The cost is computed at ceil(layers).

Both are costed with optimize=true, in the trained gate order (gate_order=:given) and in the best order found
(gate_order=:best: a different circuit, whose coefficients would need re-optimizing). For each order the CSV
has the two-mode gates (tunneling + interaction), single-mode gates (phases) and native depth.
The same circuits are also costed on Jordan–Wigner qubits (best gate order): UCC with the Yordanov et al. 2020
excitation circuits (`yordanov_encoding_cost`: CNOTs, parametrized and fixed single-qubit gates, depth in CNOT
layers), the HVA with Cade et al.'s fermionic-swap network (`cade_swap_network_cost`: native FSWAP, FSWAP+hop,
hop and on-site gates, 2 CNOTs each; single-qubit gates not modelled). Besides the CSV, the script writes the
two tables (fermionic register, Jordan–Wigner qubits) to <out>.md.
Caveat: the UCC target is the periodic (torus) ground state in a momentum sector; the HVA target here is the
open-boundary ground state, a different state. The periodic-target HVA plateaus on odd axes (open bonds), so
the open-boundary data are the HVA's natural, favourable setting.

Usage (from experimenting/ed):
  julia --project=.. hva_ucc_fermionic_comparison.jl [--targets=0.95,0.99] [--out=data-summary/hva_ucc_fermionic_comparison.csv]
Arguments:
  --targets=<list> (optional): comma-separated target fidelities F*. Default: 0.95,0.99.
  --out=<path> (optional): output CSV, relative to experimenting/ed. Default: data-summary/hva_ucc_fermionic_comparison.csv
                  The Markdown tables go to the same path with extension .md.
Writes one CSV row per (system, U, F*, series) and prints a summary table.
=#

using LinearAlgebra, Printf, CSV, DataFrames, JLD2, LsqFit, Lattices, SparseArrays, HDF5
include(joinpath(@__DIR__, "data_path.jl"))
include(joinpath(@__DIR__, "logging.jl"))
include(joinpath(@__DIR__, "utility_functions.jl"))
using .UtilityFunctions
include(joinpath(@__DIR__, "trotter.jl"))
using .Trotter
include(joinpath(@__DIR__, "ucc_hardware_encoding.jl"))
include(joinpath(@__DIR__, "ed_objects.jl"))
include(joinpath(@__DIR__, "ed_functions.jl"))

const DATA = "/home/jek354/research/data/new_data/data_h5_fixed"
const BENCH = joinpath(@__DIR__, "benchmarks")

# (label, data folder, lattice, U, HVA open-boundary series: (reference, sweep folder))
const CASES = [
    ("3x2 (3,3)", "N=(3, 3)_3x2", (3, 2), 15.0, [("eigenstate", "hva_parameter_sweep_N_3_3_3x2_u60_eigref_obc")]),
    ("3x3 (3,3)", "N=(3, 3)_3x3", (3, 3), 2.0,  [("eigenstate", "hva_parameter_sweep_N_3_3_3x3_u8_eigref_obc")]),
    ("3x3 (3,3)", "N=(3, 3)_3x3", (3, 3), 15.0, [("eigenstate", "hva_parameter_sweep_N_3_3_3x3_u60_eigref_obc")]),
    ("4x3 (5,4)", "N=(5, 4)_4x3", (4, 3), 2.0,  [("eigenstate", "hva_parameter_sweep_N_5_4_4x3_u8_eigref_obc"),
                                                 ("slater", "hva_parameter_sweep_N_5_4_4x3_u9_slaterref_obc")]),
    ("4x3 (5,4)", "N=(5, 4)_4x3", (4, 3), 15.0, [("eigenstate", "hva_parameter_sweep_N_5_4_4x3_u60_eigref_obc"),
                                                 ("slater", "hva_parameter_sweep_N_5_4_4x3_u61_slaterref_obc")]),
]

function parse_arguments(args)
    targets = [0.95, 0.99]
    out = joinpath("data-summary", "hva_ucc_fermionic_comparison.csv")
    for a in args
        if startswith(a, "--targets=")
            targets = parse.(Float64, split(split(a, "=", limit=2)[2], ","))
        elseif startswith(a, "--out=")
            out = split(a, "=", limit=2)[2]
        else
            error("unknown argument $a")
        end
    end
    return targets, out
end

# two-mode (tunneling + interaction) and single-mode (phase) gate counts and native depth, in the trained gate
# order (given) and the best order found (best); fusion can make the counts differ between the two
counts(c) = (two_mode_given=c.g.n_two_qubit, single_mode_given=c.g.n_single_qubit, depth_given=c.g.depth,
             two_mode_best=c.b.n_two_qubit, single_mode_best=c.b.n_single_qubit, depth_best=c.b.depth,
             tunneling_best=c.b.n_tunneling, interaction_best=c.b.n_interaction)
# Jordan–Wigner qubits, best gate order. UCC: Yordanov circuits; HVA: Cade swap network (2 CNOTs per native gate,
# single-qubit gates not modelled -> missing)
jw_ucc(gates, N; kw...) = (r = yordanov_encoding_cost(gates, N; optimize=true, gate_order=:best, kw...);
    (jw_cnot=r.n_two_qubit, jw_rot_1q=r.n_rotations, jw_fixed_1q=r.n_single_qubit - r.n_rotations, jw_depth=r.depth,
     jw_fswap=missing, jw_fswap_hop=missing, jw_hop=missing, jw_onsite=missing))
jw_hva(Lvec, L, pm) = (r = cade_swap_network_cost(Lvec; num_exponentials=L, param_map=pm); k = r.counts_by_kind;
    (jw_cnot=r.n_two_qubit, jw_rot_1q=missing, jw_fixed_1q=missing, jw_depth=r.depth,
     jw_fswap=get(k, :fswap, 0), jw_fswap_hop=get(k, :fswap_hop, 0), jw_hop=get(k, :hop, 0), jw_onsite=get(k, :onsite, 0)))
cost(gates, N; kw...) = (g = fermionic_encoding_cost(gates, N; optimize=true, gate_order=:given, kw...),
                         b = fermionic_encoding_cost(gates, N; optimize=true, gate_order=:best, kw...))

function ucc_setup(folder, Lvec, U)
    N = prod(Lvec); dir = joinpath(DATA, folder); col = round(Int, U / 0.25)
    f = joinpath(dir, "trotter_N=$(N)_ref_slater_antihermitian_u_$(col).jld2")
    shared = Trotter.load_saved_dict(replace(f, r"_u_\d+\.jld2$" => "_shared.jld2"))
    FG = Trotter.TamFermion.FGate
    gates = [g isa FG ? g : FG(g.cre_up, g.ann_up, g.cre_dn, g.ann_dn) for g in shared["gates"]]
    d = Trotter.load_saved_dict(f)
    haskey(d, "gate_keys") && (gates = gates[gate_permutation(d["gate_keys"], gate_keys(gates))])
    A = Float64.(d["coefficients"])
    saved_loss = Float64(d["metrics"]["loss"][end])
    _, sv, idx, _ = load_ED_data(dir; verbose=false, sign_convention=:spin_first, use_slater_reference=true)
    basis = Trotter.get_basis_sector(idx, collect(Lvec), N)
    ref = ComplexF64.(sv[1, :])
    infid(Akeep, row) = (psi = Trotter.apply_unitary(Akeep, gates, ref, basis, N, 1; antihermitian=true, use_gpu=false);
                         t = sv[row, :]; 1 - abs2(dot(t, psi)) / (norm(t)^2 * norm(psi)^2))
    losses = [(row, infid(A, row)) for row in (col, col + 1) if row <= size(sv, 1)]
    row, l0 = losses[argmin([abs(l - saved_loss) for (_, l) in losses])]
    abs(l0 - saved_loss) < 1e-6 || error("$folder U=$U: no target row reproduces the saved loss $saved_loss (got $losses)")
    return (; N, gates, A, row, infid, l0)
end

function ucc_row(label, folder, Lvec, U, Fstar, S)
    base = (series="UCC pruned", reference="slater", target="periodic (momentum sector)")
    S.l0 > 1 - Fstar && return merge(base, (status="full layer below target (F=$(round(1 - S.l0, digits=4)))",))
    order = sortperm(abs.(S.A); rev=true)
    keepK(K) = (Ak = zeros(length(S.A)); Ak[order[1:K]] = S.A[order[1:K]]; Ak)
    lo, hi = 0, length(S.A)            # invariant: F(keep hi) >= F*, F(keep lo) < F* (K = 0 is the bare reference)
    while hi - lo > 1
        mid = (lo + hi) ÷ 2
        S.infid(keepK(mid), S.row) <= 1 - Fstar ? (hi = mid) : (lo = mid)
    end
    K = hi; Ak = keepK(K); infK = S.infid(Ak, S.row)
    infK <= 1 - Fstar || error("$label U=$U: bisection result K=$K misses the target")
    c = cost(S.gates, S.N; coefficients=Ak, drop_tol=0.0)
    c.g.n_excitations == K || @warn "$label U=$U: costed $(c.g.n_excitations) gates, expected $K"
    return merge(base, (status="ok", layers_raw=missing, layers_fit=missing, layers_costed=1,
        F_achieved=1 - infK, threshold=abs(S.A[order[K]]), params=K, params_total=length(S.A),
        counts(c)..., jw_ucc(S.gates, S.N; coefficients=Ak)...))
end

function hva_rows(label, Lvec, U, series, Fstar)
    rows = []
    N = prod(Lvec)
    gates, pm = enumerate_ferm_excitations_HVA(Lvec; use_pbc=false, tie=:full)
    for (ref, sweep) in series
        dir = joinpath(BENCH, sweep)
        df = reduce(vcat, [CSV.read(joinpath(dir, f), DataFrame) for f in readdir(dir) if endswith(f, ".csv")]; cols=:union)
        df = sort!(combine(groupby(df, :n_layers), :infidelity => minimum => :infidelity, :n_params => first => :n_params), :n_layers)
        eps = 1 - Fstar
        k = findfirst(<=(eps), df.infidelity)
        L_raw = isnothing(k) ? missing : k == 1 ? Float64(df.n_layers[1]) :
            exp(log(df.n_layers[k-1]) + (log(eps) - log(df.infidelity[k-1])) *
                (log(df.n_layers[k]) - log(df.n_layers[k-1])) / (log(df.infidelity[k]) - log(df.infidelity[k-1])))
        q = curve_fit((x, p) -> p[1] .+ p[2] .* x, log.(Float64.(df.n_layers)), log.(df.infidelity), [0.0, -1.0]).param
        L_fit = q[2] < 0 ? exp((log(eps) - q[1]) / q[2]) : missing
        L = ceil(Int, coalesce(L_raw, L_fit, NaN))
        base = (series="HVA", reference=ref, target="open boundary (position space)")
        c = cost(gates, N; num_exponentials=L, antihermitian=false, param_map=pm)
        push!(rows, merge(base, (status=ismissing(L_raw) ? "from fit (not reached in sweep)" : "ok",
            layers_raw=L_raw, layers_fit=L_fit, layers_costed=L, F_achieved=missing, threshold=missing,
            params=L * (df.n_params[1] ÷ df.n_layers[1]), params_total=missing,
            counts(c)..., jw_hva(Lvec, L, pm)...,
            sweep_layers="$(minimum(df.n_layers))-$(maximum(df.n_layers))", fit_exponent=q[2])))
    end
    return rows
end

fmtn(x) = ismissing(x) ? "—" : string(x)
size_label(r) = r.series == "HVA" ? "$(r.layers_costed) layer$(r.layers_costed == 1 ? "" : "s") ($(r.params) angles)" : "1 layer, $(r.params) of $(r.params_total) angles"
ansatz_label(r) = r.series == "HVA" ? "HVA ($(r.reference) ref.)" : "UCC, pruned"

function markdown_tables(rows)
    io = IOBuffer()
    ok = [r for r in eachrow(rows) if r.status != "" && !startswith(r.status, "full layer below")]
    unreachable = [r for r in eachrow(rows) if startswith(r.status, "full layer below")]
    println(io, "# Hardware cost at matched target fidelity: pruned UCC layer vs HVA\n")
    println(io, "Generated by `experimenting/ed/hva_ucc_fermionic_comparison.jl`; data in the CSV of the same name. UCC: momentum-space layer, ",
        "periodic target, Slater reference, raw truncation to the fewest angles with F ≥ F*. HVA: open-boundary target and open HVA ",
        "(Cade et al.'s setting), layers from the sweep or its power-law fit. Best gate order in both tables.\n")
    println(io, "## Fermionic tweezer register (González-Cuadra et al. 2023)\n")
    println(io, "| System | U/t | F* | Ansatz | Circuit size | Tunneling gates (two-mode) | Interaction gates (two-mode) | Phase gates (single-mode) | Depth (native layers) |")
    println(io, "|---|---|---|---|---|---|---|---|---|")
    for r in ok
        println(io, "| $(r.system) | $(Int(r.U)) | $(r.F_target) | $(ansatz_label(r)) | $(size_label(r)) | $(r.tunneling_best) | $(r.interaction_best) | $(r.single_mode_best) | $(r.depth_best) |")
    end
    println(io, "\n## Jordan–Wigner qubits\n")
    println(io, "UCC: Yordanov et al. 2020 excitation circuits. HVA: Cade et al. 2020 fermionic-swap network, 2 CNOTs per native gate; its single-qubit gates are not modelled (—).\n")
    println(io, "| System | U/t | F* | Ansatz | Circuit size | CNOT gates | Single-qubit rotations (parametrized) | Single-qubit gates (fixed) | HVA native gates: FSWAP / FSWAP+hop / hop / on-site | Depth (CNOT layers) |")
    println(io, "|---|---|---|---|---|---|---|---|---|---|")
    for r in ok
        native = r.series == "HVA" ? "$(r.jw_fswap) / $(r.jw_fswap_hop) / $(r.jw_hop) / $(r.jw_onsite)" : "—"
        println(io, "| $(r.system) | $(Int(r.U)) | $(r.F_target) | $(ansatz_label(r)) | $(size_label(r)) | $(r.jw_cnot) | $(fmtn(r.jw_rot_1q)) | $(fmtn(r.jw_fixed_1q)) | $native | $(r.jw_depth) |")
    end
    isempty(unreachable) || println(io, "\nNot in the tables: ", join(["$(r.system), U/t = $(Int(r.U)), F* = $(r.F_target): the unpruned UCC layer does not reach F*" for r in unreachable], "; "), ".")
    return String(take!(io))
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "hva_ucc_fermionic_comparison")
    with_logging(log_path) do
        targets, out = parse_arguments(ARGS)
        rows = DataFrame()
        for (label, folder, Lvec, U, series) in CASES
            S = ucc_setup(folder, Lvec, U)
            println("$label U=$U: target row $(S.row), full-layer infidelity $(S.l0)")
            for Fstar in targets
            meta = (system=label, U=U, F_target=Fstar)
            push!(rows, merge(meta, ucc_row(label, folder, Lvec, U, Fstar, S)); cols=:union)
            for r in hva_rows(label, Lvec, U, series, Fstar)
                push!(rows, merge(meta, r); cols=:union)
            end
            end
        end
        path = joinpath(@__DIR__, out)
        CSV.write(path, rows)
        md = replace(path, r"\.csv$" => ".md")
        write(md, markdown_tables(rows))
        println("wrote $path\nwrote $md\n")
        print(read(md, String))
        show(stdout, MIME("text/plain"), select(rows, intersect(["system", "U", "F_target", "series", "reference", "status",
            "layers_costed", "params", "two_mode_best", "single_mode_best", "depth_best", "two_mode_given", "single_mode_given", "depth_given"], names(rows))); allrows=true, allcols=true)
        println()
    end
end
