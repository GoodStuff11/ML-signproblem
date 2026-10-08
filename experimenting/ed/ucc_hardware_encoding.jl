#=
ucc_hardware_encoding.jl

Hardware cost of the Trotterized UCC circuit built in `trotter.jl` -- the product
∏_l ∏_g exp(a_{l,g} τ_g) over the `Vector{FGate}` returned by `enumerate_ferm_excitations` and the
`num_exponentials` layers -- for two encodings:

1. `yordanov_encoding_cost`: a qubit processor. The 2N spin-orbitals are Jordan–Wigner encoded and every
   excitation is compiled with the CNOT-efficient fermionic-excitation circuits of
     Y. S. Yordanov, D. R. M. Arvidsson-Shukur and C. H. W. Barnes, "Efficient quantum circuits for
     quantum computational chemistry", Phys. Rev. A 102, 062612 (2020), arXiv:2005.14475.
2. `fermionic_encoding_cost`: a fermionic processor whose register holds the 2N spin-orbitals directly
   (one mode per tweezer), compiled into the native tunneling and interaction gates of
     D. González-Cuadra, D. Bluvstein, M. Kalinowski, R. Kaubruegger, N. Maskara, P. Naldesi,
     T. V. Zache, A. M. Kaufman, M. D. Lukin, H. Pichler, B. Vermersch, J. Ye and P. Zoller,
     "Fermionic quantum processing with programmable neutral atom arrays", PNAS 120, e2304294120
     (2023), arXiv:2303.06985.

Both return the number of two-qubit (two-mode) gates, single-qubit (single-mode) gates and the circuit
depth, after the optimizations listed in each docstring; pass `optimize=false` for the unoptimized
baseline. `ucc_hardware_cost_table` runs both on the same circuit.

Mode convention (from `TamFermion.excitation_operator_sector`): spin-orbitals are 0-based and
spin-blocked, spin-up momentum k is mode k and spin-down momentum k is mode N + k, where N is the
number of lattice sites. This is also the default Jordan–Wigner qubit order.

The file works both as a library (`include` it; its `(@main)` is only defined when the file itself is
the program being run, so including scripts can define their own) and as a command-line script.
Tests: `testing/test_ucc_hardware_encoding.jl`.

Usage:
  julia --project=.. ucc_hardware_encoding.jl --lattice=<Lx>x<Ly> [options]

The circuit is either the full gate set of a lattice (default), or a saved optimization run when
--coefficients_file is given. Every option below maps to a keyword of `yordanov_encoding_cost` /
`fermionic_encoding_cost`; options marked "default: follows --optimize" take the value of --optimize
(true = the optimized choice listed first under "Valid options") unless set explicitly.

Circuit options:
  --lattice=<Lx>x<Ly> (required): lattice dimensions, e.g. "4x3" or "3x3". N = Lx*Ly sites, 2N modes.
  --ansatz=<ansatz> (optional): which gate set to enumerate. Default: "standard".
                        Valid options:
                        - "standard": enumerate_ferm_excitations(2, (Lx, Ly); conserve_mom=true,
                          conserve_sz=true, include_diagonal=!antihermitian) -- the momentum-space UCC
                          gate set, sorted, in the order the scan optimizes it.
                        - "hva": enumerate_ferm_excitations_HVA((Lx, Ly); use_pbc, tie) -- real-space
                          Hamiltonian-variational gates; requires --antihermitian=false.
  --hva_tie=<tie> (optional, --ansatz=hva only): coefficient sharing passed as `tie`. Default: "full".
                        Valid options: "full", "spin", "none" (see enumerate_ferm_excitations_HVA).
                        Sharing changes the reported free-parameter count, not the gate count or depth.
  --hva_pbc=<true|false> (optional, --ansatz=hva only): periodic bonds (`use_pbc`). Default: true.
                        enumerate_ferm_excitations_HVA rejects true for an odd axis length > 2
                        (e.g. 3x2); use false there.
  --num_exponentials=<n> (optional): number of Trotter layers (the gate list is repeated n times).
                        Default: 1, or inferred from the length of --coefficients_file's coefficients.
  --antihermitian=<true|false> (optional): generator convention. true: factors exp(a(τ - τ†)),
                        diagonal gates dropped; false: factors exp(i a(τ + τ†)), diagonal gates kept
                        (+2 single-qubit gates per excitation under Yordanov). Default: true.
  --coefficients_file=<path> (optional): a per-U coefficient file of a saved scan
                        (<prefix>_u_<idx>.jld2, e.g. a pruned --target_fidelity run). Its gates are
                        read from the sibling <prefix>_shared.jld2 (reordered by the file's "gate_keys"
                        when present), and factors with |coefficient| <= --drop_tol are dropped.
                        --lattice must match the run; --ansatz/--hva_* are then ignored. Default: none.
  --drop_tol=<x> (optional): drop factors with |coefficient| <= x (only with --coefficients_file).
                        Default: 0.0, i.e. exactly the pruned (zeroed) parameters.

Encoding options:
  --encoding=<encoding> (optional): which hardware to cost. Default: "both".
                        Valid options:
                        - "yordanov": Jordan–Wigner qubits, Yordanov et al. (2020) excitation circuits.
                        - "fermionic": fermionic tweezer register, González-Cuadra et al. (2023) gates.
                        - "both": both of the above.
  --optimize=<choice> (optional): Default: "both".
                        Valid options:
                        - "true": every optimization on (unless overridden below).
                        - "false": the unoptimized baseline (unless overridden below).
                        - "both": print the unoptimized and the optimized result.
  --gate_order=<order> (optional): Default: "given".
                        Valid options:
                        - "given": the circuit as trained; factors sharing a spin-orbital keep their order.
                        - "best": factors may go in any order; the shallowest schedule found by
                          packing_priorities is used. A different circuit (coefficients would need
                          re-optimizing); a heuristic whose true optimum lies between the reported
                          depth and the reported lower bound.
  --order_trials=<n> (optional, --gate_order=best): random tie-break orders tried on top of the four
                        fixed packing rules. Default: 32.
  --reorder=<true|false> (optional): commutation-aware scheduling of the given order, kept only if it
                        is shallower than program order. Default: follows --optimize.

Yordanov options (ignored by --encoding=fermionic):
  --parity_network=<network> (optional): Default: follows --optimize ("tree" if true).
                        Valid options:
                        - "tree": balanced parity tree, string depth ⌈log₂ m⌉.
                        - "staircase": the paper's CNOT staircase, string depth m - 1.
  --jw_ordering=<ordering> (optional): Default: follows --optimize ("optimized" if true, else "blocked").
                        Valid options:
                        - "optimized": hill-climbing search for the order with the shortest strings.
                        - "blocked": all spin-up orbitals, then all spin-down (0↑ 1↑ … 0↓ 1↓ …).
                        - "interleaved": both spins of each orbital together (0↑ 0↓ 1↑ 1↓ …).
  --fuse_single_qubit=<true|false> (optional): merge adjacent single-qubit gates inside each
                        excitation circuit (21 -> 16, 7 -> 6). Default: follows --optimize.

Fermionic options (ignored by --encoding=yordanov):
  --fuse=<true|false> (optional): merge consecutive native gates on the same modes. Default: follows --optimize.
  --choose_pairing=<true|false> (optional): pick each double excitation's tunneling pairing to
                        maximize merges. Default: follows --optimize.
  --spin_conserving=<true|false> (optional): only tunneling gates between modes of the same spin.
                        Default: true (not tied to --optimize).
  --objective=<objective> (optional): what "better" means when comparing schedules. Default: "depth".
                        Valid options:
                        - "depth": depth first, then two-mode gate count.
                        - "two_qubit": two-mode gate count first, then depth.

Boolean options accept "true" or "false"; a bare flag (e.g. --fuse) means true.

Output: for every (encoding, optimize) combination, the two-qubit, single-qubit and depth totals, the
number of free variational parameters and of gates (factors),
the depth bounds (sequential sum, critical path of the given order, lower bound, gates at once),
the schedule that produced the depth, and the settings used. Also written to
logs/<date>/ucc_hardware_encoding_<timestamp>_<pid>.log.

Examples:
  julia --project=.. ucc_hardware_encoding.jl --lattice=4x3
  julia --project=.. ucc_hardware_encoding.jl --lattice=4x3 --encoding=yordanov --jw_ordering=interleaved --optimize=true
  julia --project=.. ucc_hardware_encoding.jl --lattice=4x4 --gate_order=best --order_trials=64
  julia --project=.. ucc_hardware_encoding.jl --lattice=3x2 --num_exponentials=2 --encoding=fermionic --fuse=false
  julia --project=.. ucc_hardware_encoding.jl --lattice=4x2 --ansatz=hva --antihermitian=false
  julia --project=.. ucc_hardware_encoding.jl --lattice=3x2 \
      --coefficients_file="<data root>/N=(2, 2)_3x2/trotter_N=6_ref_slater_antihermitian_target_fidelity=0.9998_u_30.jld2"
=#

if !isdefined(Main, :Trotter)
    include("trotter.jl")
end
using .Trotter

# ═══════════════════════════════════════════════════════════════════════
# EXCITATIONS OF THE CIRCUIT
# ═══════════════════════════════════════════════════════════════════════

"""
    HardwareExcitation

One factor exp(a τ_g) of the UCC circuit, in circuit order.

- `kind`: `:double` (c†_p c†_q c_r c_s - h.c.), `:single` (c†_p c_q - h.c.), `:number_pair`
  (n_p n_q, only for a non-antihermitian circuit) or `:number` (n_p, likewise).
- `creators`, `annihilators`: 0-based spin-orbital indices (see the file header). For the diagonal
  kinds both hold the modes of the number operators.
- `gate_index`: index of the `FGate` in the input gate vector; `layer`: 1-based Trotter layer.
"""
struct HardwareExcitation
    kind::Symbol
    creators::Vector{Int}
    annihilators::Vector{Int}
    gate_index::Int
    layer::Int
end

excitation_modes(e::HardwareExcitation) = sort!(unique(vcat(e.creators, e.annihilators)))

_mask_bits(mask::UInt64) = [b for b in 0:63 if (mask >> b) & 1 == 1]

"""
    classify_excitation(g, N) -> (kind, creators, annihilators)

Read an `FGate` (fields `cre_up`, `ann_up`, `cre_dn`, `ann_dn`, site bit-masks) on an `N`-site lattice
as one of the kinds of [`HardwareExcitation`](@ref). Throws for operators neither encoding below
covers: number-controlled excitations (a mode both created and annihilated, e.g. c†_p n_q c_r, which
`enumerate_ferm_excitations` only produces with `allow_overlap=true`), excitations of rank > 2, and
diagonal products of more than two number operators.
"""
function classify_excitation(g, N::Integer)
    s_I = UInt64(g.cre_up) | (UInt64(g.cre_dn) << N)
    s_J = UInt64(g.ann_up) | (UInt64(g.ann_dn) << N)
    if s_I == s_J
        m = _mask_bits(s_I)
        length(m) == 1 && return (:number, m, copy(m))
        length(m) == 2 && return (:number_pair, m, copy(m))
        throw(ArgumentError("diagonal gate on $(length(m)) modes (a product of more than two number operators) is not supported"))
    end
    (s_I & s_J) != 0 && throw(ArgumentError("number-controlled excitation (modes $(_mask_bits(s_I & s_J)) both created and annihilated) is not supported"))
    n_I, n_J = count_ones(s_I), count_ones(s_J)
    n_I == n_J || throw(ArgumentError("gate does not conserve particle number ($n_I creators, $n_J annihilators)"))
    n_I == 1 && return (:single, _mask_bits(s_I), _mask_bits(s_J))
    n_I == 2 && return (:double, _mask_bits(s_I), _mask_bits(s_J))
    throw(ArgumentError("rank-$n_I excitation is not supported (only single and double excitations)"))
end

"""
    circuit_excitations(gates, N; coefficients=nothing, num_exponentials=nothing,
                        antihermitian=true, param_map=nothing, drop_tol=0.0) -> Vector{HardwareExcitation}

Expand the gate list into the circuit's factors, in the order `apply_unitary` applies them (layer by
layer, gates in input order within a layer).

- `coefficients`: the optimized coefficient vector, laid out as in `apply_unitary`
  (`A[(l-1)*length(gates)+g]`), or the reduced vector when `param_map` is given. Factors with
  `|a| <= drop_tol` are the identity and are dropped -- with the default `drop_tol=0.0`, exactly the
  parameters a pruned (`--target_fidelity`) run zeroed. `nothing` keeps every factor.
- `num_exponentials`: number of Trotter layers. Defaults to `length(coefficients) ÷ length(gates)`
  (or ÷ the number of shared parameters with `param_map`), or 1 without coefficients.
- `antihermitian`: the circuit's generator convention. Diagonal gates have a vanishing generator in
  the antihermitian convention and are dropped.
- `param_map`: the HVA coefficient-sharing map (`enumerate_ferm_excitations_HVA`), expanded with
  `expand_shared_coefficients`.
"""
function circuit_excitations(gates::AbstractVector, N::Integer;
    coefficients=nothing, num_exponentials=nothing, antihermitian::Bool=true,
    param_map=nothing, drop_tol::Real=0.0)
    n_gates = length(gates)
    n_per_layer = isnothing(param_map) ? n_gates : maximum(param_map)
    n_exp = !isnothing(num_exponentials) ? num_exponentials :
            isnothing(coefficients) ? 1 : length(coefficients) ÷ n_per_layer
    A = nothing
    if !isnothing(coefficients)
        length(coefficients) == n_exp * n_per_layer ||
            throw(ArgumentError("length(coefficients)=$(length(coefficients)) is not num_exponentials=$n_exp × $n_per_layer"))
        A = isnothing(param_map) ? coefficients : expand_shared_coefficients(coefficients, param_map, n_gates, n_exp)
    end
    kinds = [classify_excitation(g, N) for g in gates]
    excitations = HardwareExcitation[]
    for l in 1:n_exp, g in 1:n_gates
        kind, cre, ann = kinds[g]
        antihermitian && kind in (:number, :number_pair) && continue
        !isnothing(A) && abs(A[(l-1)*n_gates+g]) <= drop_tol && continue
        push!(excitations, HardwareExcitation(kind, cre, ann, g, l))
    end
    return excitations
end

"""
    free_parameter_count(excitations; param_map=nothing) -> Int

Number of independent variational parameters in the circuit: without `param_map` every factor has
its own parameter; with the HVA tie map, gates sharing a `param_map` entry share one parameter per
Trotter layer, counted once if at least one of its gates survives (pruning, dropped diagonal gates).
For the gate count and depth it makes no difference how parameters are shared.
"""
free_parameter_count(excitations::Vector{HardwareExcitation}; param_map=nothing) =
    isnothing(param_map) ? length(excitations) :
    length(unique((e.layer, param_map[e.gate_index]) for e in excitations))

# ═══════════════════════════════════════════════════════════════════════
# SCHEDULING (shared by both encodings)
# ═══════════════════════════════════════════════════════════════════════

# Minimal binary min-heap of tuples, so this file needs no package beyond what `trotter.jl` loads.
function _heap_push!(h::Vector{T}, x::T) where {T}
    push!(h, x)
    i = length(h)
    while i > 1
        p = i ÷ 2
        h[p] <= h[i] && break
        h[p], h[i] = h[i], h[p]
        i = p
    end
    return h
end

function _heap_pop!(h::Vector)
    top = h[1]
    last = pop!(h)
    if !isempty(h)
        h[1] = last
        i, n = 1, length(h)
        while true
            l, r, s = 2i, 2i + 1, i
            l <= n && h[l] < h[s] && (s = l)
            r <= n && h[r] < h[s] && (s = r)
            s == i && break
            h[i], h[s] = h[s], h[i]
            i = s
        end
    end
    return top
end

"""
    schedule_blocks(resources, durations, mode_sets, n_resources, n_modes; reorder=true)
        -> (order, start, makespan)

Place each block `b` (an excitation, occupying the 0-based resources `resources[b]` -- qubits or modes
-- for `durations[b]` layers) on a timeline where blocks sharing a resource cannot overlap.

- `reorder=false`: as-soon-as-possible in program order.
- `reorder=true`: commutation-aware list scheduling. Two factors acting on disjoint sets of
  spin-orbitals are even fermionic operators on disjoint modes and commute exactly, so only blocks
  that share a mode in `mode_sets` keep their relative program order; among the blocks whose
  mode-predecessors are all placed, the one that can start earliest goes next (ties: smallest
  `priority`, default longer block first, then program order). The circuit implemented is unchanged.

Returns the execution `order` (block indices), the `start` layer of every block and the `makespan`.
"""
function schedule_blocks(resources::Vector{Vector{Int}}, durations::Vector{Int},
    mode_sets::Vector{Vector{Int}}, n_resources::Integer, n_modes::Integer; reorder::Bool=true,
    priority::Union{Nothing,Vector{Int}}=nothing)
    P = length(durations)
    rank = isnothing(priority) ? -durations : priority
    free = zeros(Int, n_resources)
    start = zeros(Int, P)
    earliest(b) = isempty(resources[b]) ? 0 : maximum(free[r+1] for r in resources[b])
    function place!(b)
        start[b] = earliest(b)
        for r in resources[b]
            free[r+1] = start[b] + durations[b]
        end
    end
    if !reorder
        foreach(place!, 1:P)
        return collect(1:P), start, maximum(start .+ durations; init=0)
    end

    queue = [Int[] for _ in 1:n_modes]
    for b in 1:P, m in mode_sets[b]
        push!(queue[m+1], b)
    end
    head = ones(Int, n_modes)
    is_ready(b) = all(queue[m+1][head[m+1]] == b for m in mode_sets[b])
    pushed = falses(P)
    heap = Tuple{Int,Int,Int}[]
    for b in 1:P
        if is_ready(b)
            _heap_push!(heap, (earliest(b), rank[b], b))
            pushed[b] = true
        end
    end
    order = Int[]
    while !isempty(heap)
        key, negdur, b = _heap_pop!(heap)
        # free times only grow, so a stale key is a lower bound: re-queue it with the true start
        s = earliest(b)
        if s > key
            _heap_push!(heap, (s, negdur, b))
            continue
        end
        place!(b)
        push!(order, b)
        for m in mode_sets[b]
            head[m+1] += 1
            head[m+1] > length(queue[m+1]) && continue
            c = queue[m+1][head[m+1]]
            if !pushed[c] && is_ready(c)
                _heap_push!(heap, (earliest(c), rank[c], c))
                pushed[c] = true
            end
        end
    end
    length(order) == P || error("schedule_blocks: dependency cycle (placed $(length(order)) of $P blocks)")
    return order, start, maximum(start .+ durations; init=0)
end

"""
    depth_bounds(resources, durations, mode_sets, n_resources) -> NamedTuple

Why a circuit's depth is what it is. For blocks as in [`schedule_blocks`](@ref):
- `sequential`: Σ durations, the depth with no parallelism at all;
- `critical_path`: longest chain of blocks where each shares a spin-orbital with the next, in
  program order. Such blocks are treated as non-commuting (they act on common modes), so no exact
  schedule of this gate order can beat it;
- `resource_bound`: max over qubits/modes of the total duration of the blocks using it -- no
  schedule of *any* gate order can beat it;
- `max_concurrent`: the most blocks that could ever run at once (`n_resources ÷` the smallest block
  size).
"""
function depth_bounds(resources::Vector{Vector{Int}}, durations::Vector{Int},
    mode_sets::Vector{Vector{Int}}, n_resources::Integer)
    finish = zeros(Int, length(durations))
    last_on_mode = Dict{Int,Int}()
    for b in eachindex(durations)
        s = maximum((finish[last_on_mode[m]] for m in mode_sets[b] if haskey(last_on_mode, m)); init=0)
        finish[b] = s + durations[b]
        for m in mode_sets[b]
            last_on_mode[m] = b
        end
    end
    load = zeros(Int, n_resources)
    for (r, d) in zip(resources, durations), q in r
        load[q+1] += d
    end
    smallest = max(1, minimum((length(r) for r in resources if !isempty(r)); init=n_resources))
    return (sequential=sum(durations; init=0), critical_path=maximum(finish; init=0),
        resource_bound=maximum(load; init=0), max_concurrent=n_resources ÷ smallest)
end

"""
    packing_priorities(resources, durations, n_resources; n_random=8) -> Vector{Pair{String,Vector{Int}}}

Tie-break priorities (smaller goes first) for [`schedule_blocks`](@ref) when the gate order is free
(`gate_order=:best`). Finding the shallowest order is a resource-constrained scheduling problem
(NP-hard), so the cost functions run the greedy scheduler once per rule below and keep the best:
- `"longest first"`: longest block first;
- `"widest first"`: block on the most qubits/modes first, then longest;
- `"busiest resource first"`: block touching the most heavily loaded qubit/mode first (the one that
  sets the lower bound `resource_bound` of [`depth_bounds`](@ref)), then longest;
- `"program order"`: input order;
- `"random k"`, k = 1..`n_random`: fixed pseudo-random orders (deterministic LCG, no `Random`).
"""
function packing_priorities(resources::Vector{Vector{Int}}, durations::Vector{Int}, n_resources::Integer;
    n_random::Integer=8)
    P = length(durations)
    big = maximum(durations; init=0) + 1
    load = zeros(Int, n_resources)
    for (r, d) in zip(resources, durations), q in r
        load[q+1] += d
    end
    busiest = [maximum((load[q+1] for q in r); init=0) for r in resources]
    rules = Pair{String,Vector{Int}}[
        "longest first" => -durations,
        "widest first" => [-(length(r) * big + d) for (r, d) in zip(resources, durations)],
        "busiest resource first" => [-(b * big + d) for (b, d) in zip(busiest, durations)],
        "program order" => collect(1:P),
    ]
    state = UInt64(0x9E3779B97F4A7C15)
    for k in 1:n_random
        keys = Vector{UInt64}(undef, P)
        for b in 1:P
            state = state * 0x5851F42D4C957F2D + 0x14057B7EF767814F
            keys[b] = state
        end
        rank = zeros(Int, P)
        rank[sortperm(keys)] = 1:P
        push!(rules, "random $k" => rank)
    end
    return rules
end

# ═══════════════════════════════════════════════════════════════════════
# YORDANOV 2020 (Jordan–Wigner qubit encoding)
# ═══════════════════════════════════════════════════════════════════════

"""
    jw_support(e, position) -> Vector{Int}

0-based qubits a factor touches under a Jordan–Wigner encoding where mode `m` sits on qubit
`position[m+1]`. With the excitation's 2 (single) or 4 (double) qubits sorted, q₁ < q₂ (< q₃ < q₄),
the parity string covers the qubits strictly between q₁ and q₂ (and between q₃ and q₄), so the
support is q₁:q₂ (∪ q₃:q₄) whichever of the modes are creators -- the number of qubits Yordanov
et al. call n^(sf) = k - i + 1 and n^(df) = j - i + l - k + 2. Number operators carry no string.
"""
function jw_support(e::HardwareExcitation, position::AbstractVector{<:Integer})
    p = sort!([position[m+1] for m in excitation_modes(e)])
    e.kind in (:number, :number_pair) && return p
    q = Int[]
    for k in 1:2:length(p)
        append!(q, p[k]:p[k+1])
    end
    return q
end

_parity_depth(m::Integer, network::Symbol) =
    m <= 1 ? 0 : (network === :tree ? ceil(Int, log2(m)) : m - 1)

"""
    yordanov_excitation_cost(kind, w; parity_network=:staircase, fuse_single_qubit=false,
                             antihermitian=true) -> (two_qubit, single_qubit, depth, rotations)

Gate counts of one factor on `w` Jordan–Wigner qubits (`length(jw_support(...))`). `depth` is the
two-qubit-gate depth, the quantity Yordanov et al. report as "CNOT depth".

- `:double` -- Yordanov et al., Sec. IV B, Fig. 8 (built on the double qubit excitation of
  Fig. 6): 2w + 5 two-qubit gates (CNOTs plus the two CZs that apply the parity sign); depth 11 for
  w = 4 and max(13, 2w - 1) for w ≥ 5. Single-qubit gates: the 21 of Fig. 6 (8 parametrized R_y);
  the parity network adds only CNOTs and CZs.
- `:single` -- Yordanov et al., Sec. IV A, Fig. 7 (built on Fig. 4b): 2w - 1 two-qubit gates;
  depth 3 for w = 2 and max(5, 2w - 3) for w ≥ 3. Single-qubit gates: the 7 of Fig. 4b (2 R_y).
- `:number_pair` -- not in Yordanov et al.: exp(iθ n_p n_q) is a controlled phase, compiled as
  2 CNOTs and 3 R_z (depth 2). `:number` -- one R_z.

Options (all exact):
- `parity_network`: `:staircase` is the paper's CNOT staircase, which computes the parity of the
  m = w - 4 (double) or w - 2 (single) string qubits in depth m - 1. `:tree` is the balanced-tree
  rearrangement the paper suggests (Sec. IV B, citing Cowtan et al.): the same m - 1 CNOTs in depth
  ⌈log₂ m⌉, so the depths above become max(13, 2⌈log₂ m⌉ + 9) and max(5, 2⌈log₂ m⌉ + 3).
- `fuse_single_qubit`: merge single-qubit gates that are adjacent on the same qubit into one
  (any product of single-qubit gates is a single-qubit gate): 21 → 16 for Fig. 6 and 7 → 6 for
  Fig. 4b, counted gate by gate in the figures (`testing/test_ucc_hardware_encoding.jl`).
- `antihermitian=false`: the factor is exp(i a (τ + τ†)) instead of exp(a (τ - τ†)). Conjugating
  one excitation qubit by S = R_z(π/2) maps one to the other, adding 2 single-qubit gates.
"""
function yordanov_excitation_cost(kind::Symbol, w::Integer; parity_network::Symbol=:staircase,
    fuse_single_qubit::Bool=false, antihermitian::Bool=true)
    parity_network in (:staircase, :tree) || throw(ArgumentError("parity_network must be :staircase or :tree, got :$parity_network"))
    conj = antihermitian ? 0 : 2
    if kind === :double
        w >= 4 || throw(ArgumentError("a double excitation needs w ≥ 4, got $w"))
        m = w - 4
        depth = m == 0 ? 11 : max(13, 2 * _parity_depth(m, parity_network) + 9)
        return (two_qubit=2w + 5, single_qubit=(fuse_single_qubit ? 16 : 21) + conj, depth=depth, rotations=8)
    elseif kind === :single
        w >= 2 || throw(ArgumentError("a single excitation needs w ≥ 2, got $w"))
        m = w - 2
        depth = m == 0 ? 3 : max(5, 2 * _parity_depth(m, parity_network) + 3)
        return (two_qubit=2w - 1, single_qubit=(fuse_single_qubit ? 6 : 7) + conj, depth=depth, rotations=2)
    elseif kind === :number_pair
        return (two_qubit=2, single_qubit=3, depth=2, rotations=3)
    elseif kind === :number
        return (two_qubit=0, single_qubit=1, depth=0, rotations=1)
    end
    throw(ArgumentError("unknown excitation kind :$kind"))
end

"""
    total_jw_support(excitations, position) -> Int

Σ over the non-diagonal factors of `length(jw_support(e, position))`. Both Yordanov two-qubit counts
are 2w + const, so minimizing this minimizes the total two-qubit-gate count.
"""
total_jw_support(excitations, position) =
    sum((length(jw_support(e, position)) for e in excitations if !(e.kind in (:number, :number_pair))); init=0)

"""
    jw_ordering_positions(ordering, N) -> Vector{Int}

Qubit position of every mode (mode m on qubit `position[m+1]`) for a fixed Jordan–Wigner order:
- `:blocked` -- all spin-up orbitals, then all spin-down: k↑ on qubit k, k↓ on qubit N + k (the
  order `TamFermion` builds its operators in);
- `:interleaved` -- orbital by orbital, both spins together: k↑ on qubit 2k, k↓ on qubit 2k + 1,
  i.e. 0↑ 0↓ 1↑ 1↓ … For the momentum-space ansatz the "orbitals" are the momenta k in
  `TamFermion`'s momentum index order; for a real-space gate set (HVA) they are the lattice sites.
"""
function jw_ordering_positions(ordering::Symbol, N::Integer)
    ordering === :blocked && return collect(0:2N-1)
    ordering === :interleaved && return [m < N ? 2m : 2(m - N) + 1 for m in 0:2N-1]
    throw(ArgumentError("fixed Jordan–Wigner ordering must be :blocked or :interleaved, got :$ordering"))
end

"""
    optimize_jw_ordering(excitations, n_modes, N; max_passes=200) -> (position, total_support)

Choose the Jordan–Wigner order of the spin-orbitals (any order is a valid encoding) to minimize
[`total_jw_support`](@ref). Starts from the spin-blocked order (mode m on qubit m) and the
interleaved order (k↑, k↓ adjacent), then runs pairwise-swap hill climbing from each: every swap of
two qubits' modes is tried and kept if it lowers the total; only factors touching the two swapped
modes are re-evaluated. Deterministic. Returns `position` (mode m on qubit `position[m+1]`).
"""
function optimize_jw_ordering(excitations::Vector{HardwareExcitation}, n_modes::Integer, N::Integer;
    max_passes::Integer=200)
    strung = [e for e in excitations if !(e.kind in (:number, :number_pair))]
    by_mode = [Int[] for _ in 1:n_modes]
    for (i, e) in enumerate(strung), m in excitation_modes(e)
        push!(by_mode[m+1], i)
    end
    support(i, pos) = length(jw_support(strung[i], pos))
    blocked = collect(0:n_modes-1)
    starts = n_modes == 2N ? [blocked, jw_ordering_positions(:interleaved, N)] : [blocked]

    best_pos, best_cost = blocked, typemax(Int)
    stamp = zeros(Int, length(strung))
    tick = 0
    for pos0 in starts
        pos = copy(pos0)
        at = zeros(Int, n_modes)              # at[q+1] = mode on qubit q
        for m in 0:n_modes-1
            at[pos[m+1]+1] = m
        end
        cost = total_jw_support(strung, pos)
        for _ in 1:max_passes
            improved = false
            for qa in 0:n_modes-2, qb in qa+1:n_modes-1
                x, y = at[qa+1], at[qb+1]
                tick += 1
                touched = Int[]
                for i in Iterators.flatten((by_mode[x+1], by_mode[y+1]))
                    stamp[i] == tick && continue
                    stamp[i] = tick
                    push!(touched, i)
                end
                isempty(touched) && continue
                before = sum(support(i, pos) for i in touched)
                pos[x+1], pos[y+1] = qb, qa
                delta = sum(support(i, pos) for i in touched) - before
                if delta < 0
                    at[qa+1], at[qb+1] = y, x
                    cost += delta
                    improved = true
                else
                    pos[x+1], pos[y+1] = qa, qb
                end
            end
            improved || break
        end
        if cost < best_cost
            best_pos, best_cost = pos, cost
        end
    end
    return best_pos, best_cost
end

"""
    yordanov_encoding_cost(gates, N; coefficients=nothing, num_exponentials=nothing,
                           antihermitian=true, param_map=nothing, drop_tol=0.0,
                           optimize=true, parity_network=nothing, jw_ordering=nothing,
                           reorder=nothing, fuse_single_qubit=nothing) -> NamedTuple

Cost of the UCC circuit on a qubit processor: Jordan–Wigner encoding, each factor compiled with the
fermionic-excitation circuits of Yordanov, Arvidsson-Shukur and Barnes, Phys. Rev. A 102, 062612
(2020) (see [`yordanov_excitation_cost`](@ref) for the per-factor counts and their figure numbers).

Inputs are the objects `trotter.jl` builds: `gates` from `enumerate_ferm_excitations` (or
`enumerate_ferm_excitations_HVA`, with its `param_map`), `N` the number of lattice sites, and
optionally the coefficient vector of a saved run; see [`circuit_excitations`](@ref) for
`coefficients`, `num_exponentials`, `antihermitian`, `param_map` and `drop_tol`.

`optimize=true` turns on every optimization below; each can be overridden individually (`nothing`
means "follow `optimize`"). All of them leave the implemented unitary unchanged.
- `parity_network` (`:tree` / `:staircase`): balanced-tree parity computation, log-depth strings.
- `jw_ordering`: the Jordan–Wigner order of the spin-orbitals on the qubit line.
  `:blocked` -- all spin-up then all spin-down (`TamFermion`'s order; default for `optimize=false`);
  `:interleaved` -- 0↑ 0↓ 1↑ 1↓ …, both spins of each orbital adjacent
  ([`jw_ordering_positions`](@ref)); `:optimized` -- chosen by [`optimize_jw_ordering`](@ref) to
  minimize the total string length, hence the two-qubit count (default for `optimize=true`; it
  starts from both fixed orders, so it is never worse than either).
- `reorder` (`true` / `false`): commutation-aware scheduling ([`schedule_blocks`](@ref)); the depth
  reported is the better of this and program order.
- `fuse_single_qubit` (`true` / `false`): merge adjacent single-qubit gates inside each factor.

`gate_order` (not tied to `optimize`):
- `:given` (default): the circuit as trained. Factors that act on a common spin-orbital keep their
  program order (they are treated as non-commuting); only factors on disjoint spin-orbitals move.
  With the sorted order of `enumerate_ferm_excitations`, almost every pair of consecutive factors
  shares a spin-orbital, so the depth is close to `depth_sequential` whatever the scheduler does.
- `:best`: the factors may go in any order, and the shallowest order found is used for the count:
  the greedy scheduler is run with every rule of [`packing_priorities`](@ref) (and the given order)
  and the best result is kept (`order_strategy` names the winner); `order_trials` (default 32) sets
  how many random tie-break orders are tried on top of the four fixed rules. This is a *different* circuit
  (non-commuting factors are reordered), so its coefficients would have to be re-optimized in that
  order. It is a heuristic, not a proven optimum: the true optimum lies between `depth` and
  `bounds.resource_bound`.

Depth model: each factor is a block occupying all qubits of its Jordan–Wigner support for its
two-qubit-gate depth; blocks sharing a qubit do not overlap (as in the `gate_count_scripts` of the
UCC paper). `depth` is therefore the two-qubit-gate depth of the whole circuit, with
interleaving of neighbouring blocks' gates not exploited.

Returns a `NamedTuple` with
- `n_two_qubit`, `n_single_qubit`, `depth` -- the totals asked for;
- `n_rotations` (parametrized single-qubit rotations), `depth_sequential` (Σ block depths, no
  parallelism), `n_qubits`, `n_excitations` (factors = gates), `n_parameters` (free variational
  parameters, [`free_parameter_count`](@ref)), `counts_by_kind` (`Dict` kind => number of factors),
  `mean_support`, `bounds` ([`depth_bounds`](@ref): why the depth is what it is);
- `jw_order` (`jw_order[q+1]` = mode on qubit q), `execution_order` (indices into
  `excitations`), `order_strategy` (which schedule produced `depth`), `excitations`, and
  `settings` (the options actually used).
"""
function yordanov_encoding_cost(gates::AbstractVector, N::Integer;
    coefficients=nothing, num_exponentials=nothing, antihermitian::Bool=true, param_map=nothing,
    drop_tol::Real=0.0, optimize::Bool=true, parity_network::Union{Nothing,Symbol}=nothing,
    jw_ordering::Union{Nothing,Symbol}=nothing, reorder::Union{Nothing,Bool}=nothing,
    fuse_single_qubit::Union{Nothing,Bool}=nothing, gate_order::Symbol=:given, order_trials::Integer=32)
    gate_order in (:given, :best) || throw(ArgumentError("gate_order must be :given or :best, got :$gate_order"))
    network = something(parity_network, optimize ? :tree : :staircase)
    ordering = something(jw_ordering, optimize ? :optimized : :blocked)
    ordering in (:blocked, :interleaved, :optimized) ||
        throw(ArgumentError("jw_ordering must be :blocked, :interleaved or :optimized, got :$ordering"))
    do_reorder = something(reorder, optimize)
    do_fuse = something(fuse_single_qubit, optimize)
    n_modes = 2N

    excitations = circuit_excitations(gates, N; coefficients=coefficients, num_exponentials=num_exponentials,
        antihermitian=antihermitian, param_map=param_map, drop_tol=drop_tol)
    position = ordering === :optimized ? first(optimize_jw_ordering(excitations, n_modes, N)) :
               jw_ordering_positions(ordering, N)

    supports = [jw_support(e, position) for e in excitations]
    costs = [yordanov_excitation_cost(e.kind, length(s); parity_network=network,
        fuse_single_qubit=do_fuse, antihermitian=antihermitian) for (e, s) in zip(excitations, supports)]
    durations = [c.depth for c in costs]
    mode_sets = [excitation_modes(e) for e in excitations]

    order, _, depth = schedule_blocks(supports, durations, mode_sets, n_modes, n_modes; reorder=false)
    strategy = "given order, as soon as possible"
    if do_reorder || gate_order === :best
        order_r, _, depth_r = schedule_blocks(supports, durations, mode_sets, n_modes, n_modes; reorder=true)
        depth_r < depth && ((order, depth, strategy) = (order_r, depth_r, "given order, commutation-aware"))
    end
    if gate_order === :best
        free = [Int[] for _ in excitations]
        for (name, prio) in packing_priorities(supports, durations, n_modes; n_random=order_trials)
            order_f, _, depth_f = schedule_blocks(supports, durations, free, n_modes, n_modes; reorder=true, priority=prio)
            depth_f < depth && ((order, depth, strategy) = (order_f, depth_f, "free order: " * name))
        end
    end

    jw_order = zeros(Int, n_modes)
    for m in 0:n_modes-1
        jw_order[position[m+1]+1] = m
    end
    counts_by_kind = Dict{Symbol,Int}()
    for e in excitations
        counts_by_kind[e.kind] = get(counts_by_kind, e.kind, 0) + 1
    end
    strung = [length(s) for (e, s) in zip(excitations, supports) if !(e.kind in (:number, :number_pair))]
    return (
        encoding="Yordanov et al., PRA 102, 062612 (2020): Jordan–Wigner qubits",
        n_two_qubit=sum((c.two_qubit for c in costs); init=0),
        n_single_qubit=sum((c.single_qubit for c in costs); init=0),
        depth=depth,
        n_rotations=sum((c.rotations for c in costs); init=0),
        depth_sequential=sum(durations; init=0),
        n_qubits=n_modes,
        n_excitations=length(excitations),
        n_parameters=free_parameter_count(excitations; param_map=param_map),
        counts_by_kind=counts_by_kind,
        mean_support=isempty(strung) ? 0.0 : sum(strung) / length(strung),
        bounds=depth_bounds(supports, durations, mode_sets, n_modes),
        order_strategy=strategy,
        jw_order=jw_order,
        execution_order=order,
        excitations=excitations,
        settings=(parity_network=network, jw_ordering=ordering, reorder=do_reorder,
            fuse_single_qubit=do_fuse, antihermitian=antihermitian, gate_order=gate_order),
    )
end

# ═══════════════════════════════════════════════════════════════════════
# GONZÁLEZ-CUADRA ET AL. 2023 (fermionic register, native fermionic gates)
# ═══════════════════════════════════════════════════════════════════════

"""
    pair_tunneling_pairings(e, N; spin_conserving=true) -> Vector{NTuple{2,Tuple{Int,Int}}}

Ways to pair the creators (c₁, c₂) of a double excitation with its annihilators (a₁, a₂) for the
tunneling layers of the pair-tunneling decomposition: ((c₁,a₁),(c₂,a₂)) as drawn in
González-Cuadra et al. Fig. 3(a), and ((c₁,a₂),(c₂,a₁)), which is the same gate with θ₁ → -θ₁ since
c†c†c_{a₁}c_{a₂} = -c†c†c_{a₂}c_{a₁}. With `spin_conserving=true` only pairings whose tunneling
gates connect two modes of the same spin are kept (exactly one for an opposite-spin excitation,
both for a same-spin one); if none is spin conserving, both are returned.
"""
function pair_tunneling_pairings(e::HardwareExcitation, N::Integer; spin_conserving::Bool=true)
    c1, c2 = e.creators
    a1, a2 = e.annihilators
    options = [((c1, a1), (c2, a2)), ((c1, a2), (c2, a1))]
    spin(m) = m >= N
    if spin_conserving
        kept = [o for o in options if all(spin(p[1]) == spin(p[2]) for p in o)]
        isempty(kept) || return kept
    end
    return options
end

"""
    fermionic_native_layers(e, pairing) -> Vector{Vector{Tuple{Symbol,Int,Int}}}

Native-gate layers of one factor, in time order. Each gate is `(kind, p, q)` with `kind` one of
`:tunnel` (U^(t)_{p,q}, a two-mode tunneling gate), `:interact` (U^(int)_{p,q} = exp(-iθ n_p n_q))
or `:phase` (single-mode exp(-iθ n_p), `q = -1`); two-mode gates have `p < q`.

- `:double` -- the pair-tunneling gate U^(pt)_{c₁,c₂,a₁,a₂}(θ₁, θ₂), decomposed exactly in
  González-Cuadra et al. Fig. 3(a) into five layers: tunneling U^(t)(√8π/√27, θ₂/2 - π/4, 2π/√27)
  on the two `pairing` pairs; interaction U^(int)(-θ₁) on (c₁,c₂) and (a₁,a₂); tunneling
  U^(t)(π/2, θ₂/2 + π/2, 0); interaction U^(int)(+θ₁); tunneling U^(t)(π/2, θ₂/2 + π, 0) --
  6 tunneling + 4 interaction gates, depth 5, no single-mode gates (checked to machine precision on
  the 4-mode Fock space in `testing/test_ucc_hardware_encoding.jl`). Their disentangled-UCC ansatz
  uses U^(pt)(θ, π/2) for exactly these factors.
- `:single` -- one tunneling gate U^(t)(θ, π/2, 0) (their UCC ansatz, same equation).
- `:number_pair` -- one interaction gate. `:number` -- one single-mode phase.
"""
function fermionic_native_layers(e::HardwareExcitation, pairing=nothing)
    srt(p, q) = p < q ? (p, q) : (q, p)
    if e.kind === :double
        (p1, p2) = pairing
        T = [(:tunnel, srt(p1...)...), (:tunnel, srt(p2...)...)]
        I = [(:interact, srt(e.creators...)...), (:interact, srt(e.annihilators...)...)]
        return [T, I, copy(T), copy(I), copy(T)]
    elseif e.kind === :single
        return [[(:tunnel, srt(e.creators[1], e.annihilators[1])...)]]
    elseif e.kind === :number_pair
        return [[(:interact, srt(e.creators...)...)]]
    elseif e.kind === :number
        return [[(:phase, e.creators[1], -1)]]
    end
    throw(ArgumentError("unknown excitation kind :$(e.kind)"))
end

"""
    NativeFermionicCircuit(n_modes; fuse=true)

Native-gate list being assembled, with a peephole pass applied as gates are appended
([`append_native!`](@ref)).
"""
mutable struct NativeFermionicCircuit
    ops::Vector{Tuple{Symbol,Int,Int}}
    last::Vector{Int}       # last[m+1] = index in `ops` of the latest gate on mode m (0 if none)
    fuse::Bool
    n_fused::Int
end
NativeFermionicCircuit(n_modes::Integer; fuse::Bool=true) =
    NativeFermionicCircuit(Tuple{Symbol,Int,Int}[], zeros(Int, n_modes), fuse, 0)

"""
    can_fuse(circ, op) -> Bool

`true` when the latest gate on every mode of `op` is one and the same gate of the same kind on the
same modes. Then nothing acted on those modes in between, and the two merge exactly into one native
gate: two tunneling gates on a pair are two SU(2) rotations of the pair's single-particle space
(the generators c†_p c_q + h.c., i(c†_p c_q - h.c.) and n_p - n_q of U^(t)), whose product is again a
U^(t); interaction gates and single-mode phases on the same modes add their angles.
"""
function can_fuse(circ::NativeFermionicCircuit, op::Tuple{Symbol,Int,Int})
    kind, p, q = op
    j = circ.last[p+1]
    j == 0 && return false
    circ.ops[j] == op || return false
    return q < 0 || circ.last[q+1] == j
end

"""
    append_native!(circ, op)

Append a native gate, merging it into the previous gate on its modes when [`can_fuse`](@ref) holds
and `circ.fuse` is on.
"""
function append_native!(circ::NativeFermionicCircuit, op::Tuple{Symbol,Int,Int})
    if circ.fuse && can_fuse(circ, op)
        circ.n_fused += 1
        return circ
    end
    push!(circ.ops, op)
    j = length(circ.ops)
    circ.last[op[2]+1] = j
    op[3] >= 0 && (circ.last[op[3]+1] = j)
    return circ
end

"""
    native_depth(ops, n_modes) -> Int

As-soon-as-possible layering of a native-gate list: each gate goes one layer after the latest gate
on any of its modes. Gates on disjoint modes run in parallel (tweezer arrays reconfigure to bring any
pair of modes together, González-Cuadra et al., so there is no connectivity constraint).
"""
function native_depth(ops::Vector{Tuple{Symbol,Int,Int}}, n_modes::Integer)
    level = zeros(Int, n_modes)
    depth = 0
    for (_, p, q) in ops
        l = 1 + (q < 0 ? level[p+1] : max(level[p+1], level[q+1]))
        level[p+1] = l
        q >= 0 && (level[q+1] = l)
        depth = max(depth, l)
    end
    return depth
end

"""
    compile_fermionic(excitations, order, n_modes, N; fuse=true, choose_pairing=true,
                      spin_conserving=true) -> NativeFermionicCircuit

Emit the native gates of `excitations[order]`. With `choose_pairing`, each double excitation uses
the allowed tunneling pairing ([`pair_tunneling_pairings`](@ref)) whose first tunneling layer merges
with the most gates already in the circuit; otherwise the first allowed pairing.
"""
function compile_fermionic(excitations::Vector{HardwareExcitation}, order::AbstractVector{Int},
    n_modes::Integer, N::Integer; fuse::Bool=true, choose_pairing::Bool=true, spin_conserving::Bool=true)
    circ = NativeFermionicCircuit(n_modes; fuse=fuse)
    for b in order
        e = excitations[b]
        pairing = nothing
        if e.kind === :double
            options = pair_tunneling_pairings(e, N; spin_conserving=spin_conserving)
            pairing = first(options)
            if choose_pairing && fuse && length(options) > 1
                n_merge(o) = count(op -> can_fuse(circ, op), first(fermionic_native_layers(e, o)))
                pairing = options[argmax(map(n_merge, options))]
            end
        end
        for layer in fermionic_native_layers(e, pairing), op in layer
            append_native!(circ, op)
        end
    end
    return circ
end

"""
    fermionic_encoding_cost(gates, N; coefficients=nothing, num_exponentials=nothing,
                            antihermitian=true, param_map=nothing, drop_tol=0.0,
                            optimize=true, fuse=nothing, choose_pairing=nothing, reorder=nothing,
                            spin_conserving=true, objective=:depth) -> NamedTuple

Cost of the UCC circuit on the fermionic processor of González-Cuadra et al., PNAS 120,
e2304294120 (2023): one spin-orbital per register mode, no Jordan–Wigner strings, and the native
gate set G = {U^(t)_{p,q}(θ⃗), U^(int)_{p,q}(θ)} (their Eqs. 2–3) plus single-mode phases. Each
factor is compiled as in [`fermionic_native_layers`](@ref); the double excitations of the
momentum-space UCC ansatz are their pair-tunneling gates, at 10 two-mode gates and depth 5 each.

Inputs as in [`yordanov_encoding_cost`](@ref) (the generator convention does not change the native
cost: it only sets the phase θ₂ of the tunneling gates).

`optimize=true` turns on every optimization below; `nothing` means "follow `optimize`". All are
exact.
- `fuse` (`true` / `false`): merge consecutive native gates on the same modes ([`can_fuse`](@ref)).
  In the sorted gate order of `enumerate_ferm_excitations`, consecutive excitations often share a
  tunneling pair, so the last tunneling layer of one and the first of the next merge.
- `choose_pairing` (`true` / `false`): pick each double excitation's tunneling pairing to maximize
  merges ([`compile_fermionic`](@ref)).
- `reorder` (`true` / `false`): also compile the commutation-aware schedule of
  [`schedule_blocks`](@ref) (resources = modes) and keep whichever of it and program order is better
  under `objective` (`:depth`: depth, then two-mode count; `:two_qubit`: the reverse).
- `spin_conserving` (default `true`, not tied to `optimize`): only use tunneling gates between two
  modes of the same spin, i.e. no spin-flipping tunneling gates are assumed available.
- `gate_order` (`:given` default / `:best`, not tied to `optimize`): as in
  [`yordanov_encoding_cost`](@ref). With `:best` every [`packing_priorities`](@ref) rule is
  compiled (with merging and pairing choice) and the best under `objective` is kept -- a different
  circuit whose coefficients would need re-optimizing in that order.

Returns a `NamedTuple` with
- `n_two_qubit` (= two-mode gates, `n_tunneling` + `n_interaction`), `n_single_qubit` (= single-mode
  phases), `depth` (native-gate layers) -- the totals asked for;
- `n_tunneling`, `n_interaction`, `n_fused` (gates removed by merging), `bounds`
  ([`depth_bounds`](@ref), in pair-tunneling layers), `n_modes`, `n_excitations` (factors = gates),
  `n_parameters` ([`free_parameter_count`](@ref)),
  `counts_by_kind`, `execution_order`, `order_strategy`, `excitations`, `ops` (the native gate
  list), `settings`.
"""
function fermionic_encoding_cost(gates::AbstractVector, N::Integer;
    coefficients=nothing, num_exponentials=nothing, antihermitian::Bool=true, param_map=nothing,
    drop_tol::Real=0.0, optimize::Bool=true, fuse::Union{Nothing,Bool}=nothing,
    choose_pairing::Union{Nothing,Bool}=nothing, reorder::Union{Nothing,Bool}=nothing,
    spin_conserving::Bool=true, objective::Symbol=:depth, gate_order::Symbol=:given,
    order_trials::Integer=32)
    gate_order in (:given, :best) || throw(ArgumentError("gate_order must be :given or :best, got :$gate_order"))
    objective in (:depth, :two_qubit) || throw(ArgumentError("objective must be :depth or :two_qubit, got :$objective"))
    do_fuse = something(fuse, optimize)
    do_pair = something(choose_pairing, optimize)
    do_reorder = something(reorder, optimize)
    n_modes = 2N

    excitations = circuit_excitations(gates, N; coefficients=coefficients, num_exponentials=num_exponentials,
        antihermitian=antihermitian, param_map=param_map, drop_tol=drop_tol)
    mode_sets = [excitation_modes(e) for e in excitations]
    compile(order) = compile_fermionic(excitations, order, n_modes, N; fuse=do_fuse,
        choose_pairing=do_pair, spin_conserving=spin_conserving)
    n2(c) = count(op -> op[1] !== :phase, c.ops)
    score(c) = objective === :depth ? (native_depth(c.ops, n_modes), n2(c)) : (n2(c), native_depth(c.ops, n_modes))

    order = collect(1:length(excitations))
    circ = compile(order)
    durations = [length(fermionic_native_layers(e, e.kind === :double ?
        first(pair_tunneling_pairings(e, N; spin_conserving=spin_conserving)) : nothing)) for e in excitations]
    strategy = "given order"
    # emit in start-time order (stable in the scheduler's order): blocks sharing a mode never
    # overlap in time, so this keeps the scheduled order on every mode
    function try_schedule!(deps, prio, name)
        sched_order, start, _ = schedule_blocks(mode_sets, durations, deps, n_modes, n_modes; reorder=true, priority=prio)
        sched_order = sched_order[sortperm(start[sched_order]; alg=Base.Sort.DEFAULT_STABLE)]
        circ_r = compile(sched_order)
        if score(circ_r) < score(circ)
            order, circ, strategy = sched_order, circ_r, name
        end
    end
    if do_reorder || gate_order === :best
        try_schedule!(mode_sets, nothing, "given order, commutation-aware")
    end
    if gate_order === :best
        free = [Int[] for _ in excitations]
        for (name, prio) in packing_priorities(mode_sets, durations, n_modes; n_random=order_trials)
            try_schedule!(free, prio, "free order: " * name)
        end
    end

    counts_by_kind = Dict{Symbol,Int}()
    for e in excitations
        counts_by_kind[e.kind] = get(counts_by_kind, e.kind, 0) + 1
    end
    n_tunnel = count(op -> op[1] === :tunnel, circ.ops)
    n_interact = count(op -> op[1] === :interact, circ.ops)
    return (
        encoding="González-Cuadra et al., PNAS 120, e2304294120 (2023): fermionic tweezer register",
        n_two_qubit=n_tunnel + n_interact,
        n_single_qubit=count(op -> op[1] === :phase, circ.ops),
        depth=native_depth(circ.ops, n_modes),
        n_tunneling=n_tunnel,
        n_interaction=n_interact,
        n_fused=circ.n_fused,
        order_strategy=strategy,
        bounds=depth_bounds(mode_sets, durations, mode_sets, n_modes),
        n_modes=n_modes,
        n_excitations=length(excitations),
        n_parameters=free_parameter_count(excitations; param_map=param_map),
        counts_by_kind=counts_by_kind,
        execution_order=order,
        excitations=excitations,
        ops=circ.ops,
        settings=(fuse=do_fuse, choose_pairing=do_pair, reorder=do_reorder,
            spin_conserving=spin_conserving, objective=objective, gate_order=gate_order),
    )
end

# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════

"""
    ucc_hardware_cost_table(gates, N; kwargs...) -> Vector{NamedTuple}

Run both encodings, unoptimized and optimized, on the same circuit and return one row per run with
`encoding`, `optimized`, `n_two_qubit`, `n_single_qubit`, `depth`. Keyword arguments are passed to
both cost functions (`coefficients`, `num_exponentials`, `antihermitian`, `param_map`, `drop_tol`).
"""
function ucc_hardware_cost_table(gates::AbstractVector, N::Integer; kwargs...)
    rows = NamedTuple[]
    for (name, f) in (("Yordanov (JW qubits)", yordanov_encoding_cost), ("fermionic processor", fermionic_encoding_cost)),
        opt in (false, true)
        r = f(gates, N; optimize=opt, kwargs...)
        push!(rows, (encoding=name, optimized=opt, n_two_qubit=r.n_two_qubit,
            n_single_qubit=r.n_single_qubit, depth=r.depth, n_excitations=r.n_excitations, n_parameters=r.n_parameters))
    end
    return rows
end

# ═══════════════════════════════════════════════════════════════════════
# COMMAND LINE (see the file header for every option)
# ═══════════════════════════════════════════════════════════════════════

function _parse_bool(name::AbstractString, val::AbstractString)
    val == "true" && return true
    val == "false" && return false
    throw(ArgumentError("--$name expects true or false, got \"$val\""))
end

function _parse_choice(name::AbstractString, val::AbstractString, valid)
    s = Symbol(val)
    s in valid || throw(ArgumentError("--$name must be one of $(join(string.(valid), ", ")), got \"$val\""))
    return s
end

"""
    parse_ucc_hardware_arguments(args::Vector{String}) -> NamedTuple

Parse the command-line options documented in the file header into a `NamedTuple` of settings.
Options left unset are `nothing` where the cost functions' own default applies.
"""
function parse_ucc_hardware_arguments(args::Vector{String})
    o = Dict{Symbol,Any}(
        :lattice => nothing, :ansatz => :standard, :hva_tie => :full, :hva_pbc => true,
        :num_exponentials => nothing, :antihermitian => true, :coefficients_file => nothing,
        :drop_tol => 0.0, :encoding => :both, :optimize => "both", :gate_order => :given,
        :order_trials => 32, :reorder => nothing, :parity_network => nothing, :jw_ordering => nothing,
        :fuse_single_qubit => nothing, :fuse => nothing, :choose_pairing => nothing,
        :spin_conserving => true, :objective => :depth)
    bools = (:hva_pbc, :antihermitian, :reorder, :fuse_single_qubit, :fuse, :choose_pairing, :spin_conserving)
    for arg in args
        startswith(arg, "--") || throw(ArgumentError("unexpected argument \"$arg\" (all options are --name=value)"))
        name, val = occursin('=', arg) ? split(arg[3:end], '='; limit=2) : (arg[3:end], nothing)
        key = Symbol(name)
        haskey(o, key) || throw(ArgumentError("unknown option --$name (see the header of ucc_hardware_encoding.jl)"))
        if key in bools
            o[key] = isnothing(val) ? true : _parse_bool(name, val)
            continue
        end
        isnothing(val) && throw(ArgumentError("--$name needs a value (--$name=<value>)"))
        if key === :lattice
            m = match(r"^(\d+)x(\d+)$", val)
            isnothing(m) && throw(ArgumentError("--lattice must look like 4x3, got \"$val\""))
            o[key] = (parse(Int, m[1]), parse(Int, m[2]))
        elseif key in (:num_exponentials, :order_trials)
            o[key] = parse(Int, val)
            o[key] >= 1 || throw(ArgumentError("--$name must be ≥ 1"))
        elseif key === :drop_tol
            o[key] = parse(Float64, val)
        elseif key === :coefficients_file
            o[key] = String(val)
        elseif key === :ansatz
            o[key] = _parse_choice(name, val, (:standard, :hva))
        elseif key === :hva_tie
            o[key] = _parse_choice(name, val, (:full, :spin, :none))
        elseif key === :encoding
            o[key] = _parse_choice(name, val, (:yordanov, :fermionic, :both))
        elseif key === :optimize
            # stored as a String: in Julia `:true` is the Bool `true`, not a Symbol
            val in ("true", "false", "both") || throw(ArgumentError("--optimize must be one of true, false, both, got \"$val\""))
            o[key] = String(val)
        elseif key === :gate_order
            o[key] = _parse_choice(name, val, (:given, :best))
        elseif key === :parity_network
            o[key] = _parse_choice(name, val, (:tree, :staircase))
        elseif key === :jw_ordering
            o[key] = _parse_choice(name, val, (:optimized, :blocked, :interleaved))
        elseif key === :objective
            o[key] = _parse_choice(name, val, (:depth, :two_qubit))
        end
    end
    isnothing(o[:lattice]) && throw(ArgumentError("--lattice=<Lx>x<Ly> is required"))
    if o[:ansatz] === :hva && o[:antihermitian] && isnothing(o[:coefficients_file])
        throw(ArgumentError("--ansatz=hva requires --antihermitian=false (its on-site gates are diagonal)"))
    end
    return NamedTuple(o)
end

"""
    load_cli_circuit(opts) -> (gates, N, coefficients, param_map, num_exponentials, description)

Build the gate list (and coefficients, for a saved run) the command line asked for.
"""
function load_cli_circuit(opts)
    Lvec = opts.lattice
    N = prod(Lvec)
    if !isnothing(opts.coefficients_file)
        f = opts.coefficients_file
        isfile(f) || throw(ArgumentError("--coefficients_file not found: $f"))
        occursin(r"_u_-?\d+\.jld2$", f) || throw(ArgumentError("--coefficients_file must be a per-U file <prefix>_u_<idx>.jld2"))
        shared_path = replace(f, r"_u_-?\d+\.jld2$" => "_shared.jld2")
        isfile(shared_path) || throw(ArgumentError("shared gate file not found next to the coefficients: $shared_path"))
        shared = Trotter.load_saved_dict(shared_path)
        FG = Trotter.TamFermion.FGate
        gates = [g isa FG ? g : FG(g.cre_up, g.ann_up, g.cre_dn, g.ann_dn) for g in shared["gates"]]
        d = Trotter.load_saved_dict(f)
        if haskey(d, "gate_keys")
            perm = gate_permutation(d["gate_keys"], gate_keys(gates))
            isnothing(perm) && throw(ArgumentError("the file's gate_keys are not the shared gate set"))
            gates = gates[perm]
        end
        maximum(g -> max(g.cre_up, g.ann_up, g.cre_dn, g.ann_dn), gates) < (UInt64(1) << N) ||
            throw(ArgumentError("the run's gates use more than N = $N sites; check --lattice"))
        A = d["coefficients"]
        param_map = get(shared, "param_map", nothing)
        desc = "saved run $(basename(f)) ($(count(!iszero, A)) of $(length(A)) coefficients nonzero)"
        return gates, N, A, param_map, opts.num_exponentials, desc
    end
    if opts.ansatz === :hva
        gates, param_map = enumerate_ferm_excitations_HVA(Lvec; use_pbc=opts.hva_pbc, tie=opts.hva_tie)
        return gates, N, nothing, param_map, something(opts.num_exponentials, 1), "HVA gate set, tie=$(opts.hva_tie), pbc=$(opts.hva_pbc)"
    end
    gates = enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=!opts.antihermitian)
    return gates, N, nothing, nothing, something(opts.num_exponentials, 1), "momentum-space UCC gate set (enumerate_ferm_excitations)"
end

function _print_result(name, opt, r)
    println("\n--- $name, optimize=$opt ---")
    println("  two-qubit gates     : $(r.n_two_qubit)")
    println("  single-qubit gates  : $(r.n_single_qubit)")
    println("  depth               : $(r.depth)   [schedule: $(r.order_strategy)]")
    b = r.bounds
    println("  depth bounds        : sequential $(b.sequential), critical path of the given order $(b.critical_path), " *
            "lower bound for any order $(b.resource_bound), at most $(b.max_concurrent) gates at once" *
            (get(r, :n_fused, 0) > 0 ? " (bounds are before merging, so the merged depth can fall below them)" : ""))
    println("  free parameters     : $(r.n_parameters)")
    println("  gates (factors)     : $(r.n_excitations)  $(join(("$k=$v" for (k, v) in sort(collect(r.counts_by_kind))), ", "))")
    if haskey(r, :n_tunneling)
        println("  native gates        : $(r.n_tunneling) tunneling + $(r.n_interaction) interaction ($(r.n_fused) merged away)")
    else
        println("  qubits / mean JW w  : $(r.n_qubits) / $(round(r.mean_support, digits=3)),  parametrized rotations $(r.n_rotations)")
    end
    println("  settings            : $(r.settings)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    include(joinpath(@__DIR__, "logging.jl"))

    function (@main)(ARGS)
        log_path = make_log_path(@__DIR__, "ucc_hardware_encoding")
        with_logging(log_path) do
            opts = parse_ucc_hardware_arguments(ARGS)
            gates, N, A, param_map, n_exp, desc = load_cli_circuit(opts)
            println("Lattice $(opts.lattice[1])x$(opts.lattice[2]) (N = $N sites, $(2N) modes); circuit: $desc")
            println("$(length(gates)) gates per layer, num_exponentials = $(something(n_exp, "inferred from coefficients")), antihermitian = $(opts.antihermitian)")
            common = (coefficients=A, num_exponentials=n_exp, antihermitian=opts.antihermitian, param_map=param_map,
                drop_tol=opts.drop_tol, reorder=opts.reorder, gate_order=opts.gate_order, order_trials=opts.order_trials)
            encodings = opts.encoding === :both ? (:yordanov, :fermionic) : (opts.encoding,)
            optimizes = opts.optimize == "both" ? (false, true) : (opts.optimize == "true",)
            for enc in encodings, opt in optimizes
                if enc === :yordanov
                    r = yordanov_encoding_cost(gates, N; optimize=opt, common..., parity_network=opts.parity_network,
                        jw_ordering=opts.jw_ordering, fuse_single_qubit=opts.fuse_single_qubit)
                    _print_result("Yordanov 2020 (Jordan–Wigner qubits)", opt, r)
                else
                    r = fermionic_encoding_cost(gates, N; optimize=opt, common..., fuse=opts.fuse,
                        choose_pairing=opts.choose_pairing, spin_conserving=opts.spin_conserving, objective=opts.objective)
                    _print_result("González-Cuadra 2023 (fermionic tweezer register)", opt, r)
                end
            end
        end
        return 0
    end
end
