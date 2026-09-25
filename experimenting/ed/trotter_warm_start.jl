#=
trotter_warm_start.jl

Safe reuse of saved coefficient vectors (resume, adjacent-U warm start, layer growth).

A coefficient vector only means something together with the exact gate ORDER it was
optimized with: the circuit is a product of non-commuting exponentials, so the same set of
(gate, coefficient) pairs applied in a different order is a different unitary. That order
has changed at least once (commit 9efb09f, 2026-08-31, started sorting gates with
`sortGatesByIJ`), and reference/target states can change too, so a naively reloaded vector
can silently land on an essentially random point of the landscape (loss ≈ 1 for overlap).
The helpers below:
  1. record every gate's identity (`gate_keys`) next to the coefficients when saving;
  2. find the gate order a loaded vector was optimized in (recorded `gate_keys`, or for
     older files a list of candidate legacy orders), as a permutation of the current gates;
  3. verify the warm start by re-evaluating its loss IN THAT GATE ORDER, discarding it when
     no candidate order reproduces the saved loss (or beats the zero-coefficient loss).
The caller then keeps optimizing with `gates[perm]` / `tau_terms[perm]`, so the resumed
circuit is exactly the saved one. (Remapping the coefficients onto the current order
instead would change the circuit, which is why that was wrong.)
=#

"""
    gate_key(g) -> NTuple{4,UInt64}

Identity of a gate, independent of its position in the gate list.
"""
gate_key(g) = (UInt64(g.cre_up), UInt64(g.ann_up), UInt64(g.cre_dn), UInt64(g.ann_dn))

"""
    gate_keys(gates) -> Vector{NTuple{4,UInt64}}
"""
gate_keys(gates) = [gate_key(g) for g in gates]

"""
    gate_permutation(from_keys, to_keys) -> Vector{Int} or nothing

Permutation `perm` with `to_keys[perm] == from_keys`, i.e. `gates[perm]` lists the current
`gates` (with keys `to_keys`) in the order `from_keys` was saved in. Returns `nothing` if the
two lists are not the same set of distinct gates.
"""
function gate_permutation(from_keys::AbstractVector, to_keys::AbstractVector)
    G = length(to_keys)
    (length(from_keys) == G && G > 0) || return nothing
    pos = Dict{eltype(to_keys),Int}()
    for (i, k) in enumerate(to_keys)
        pos[k] = i
    end
    length(pos) == G || return nothing
    perm = Vector{Int}(undef, G)
    for (i, k) in enumerate(from_keys)
        j = get(pos, k, 0)
        j == 0 && return nothing
        perm[i] = j
    end
    allunique(perm) || return nothing
    return perm
end

"""
    align_warm_start(coeffs, saved_dict, gates, eval_loss; kwargs...) -> NamedTuple or nothing

Find the gate order `coeffs` (loaded from `saved_dict`, possibly already grown with zero
layers) was optimized in, and verify it with `eval_loss(A, perm) -> loss`, which must
evaluate the loss of coefficients `A` driving the gates `gates[perm]` (with the matching
`tau_terms[perm]`). Returns `(coefficients=coeffs, perm=perm, name=...)` -- the coefficients
are NOT reordered; the caller must keep using `gates[perm]` for them -- or `nothing` when the
warm start should be discarded. `perm == 1:length(gates)` means the current order.

Candidate gate orders, in order:
- `saved_dict["gate_keys"]` present: that recorded order (trusted, checked once);
- otherwise: the current order, then each `legacy_gate_orders` entry (name => keys), then
  the gate list stored in `shared_file` (if it exists).

With `expected_loss` (the saved loss for the SAME target and loss type), the first candidate
reproducing it within `atol + rtol*|expected_loss|` wins, and if none does the warm start is
discarded (the file does not describe this problem any more). Without `expected_loss` (e.g. a
warm start from a neighbouring U), the lowest-loss candidate is used when it beats the
zero-coefficient loss by more than `atol`; otherwise the warm start is discarded. With
`param_map !== nothing` coefficients live in the reduced shared-parameter space and the gate
order is fixed by the ansatz, so only the current order is tried (verification only).
"""
function align_warm_start(coeffs::AbstractVector, saved_dict, gates, eval_loss;
    expected_loss::Union{Nothing,Real}=nothing,
    legacy_gate_orders=Pair{String,Vector{NTuple{4,UInt64}}}[],
    shared_file::Union{Nothing,String}=nothing,
    param_map::Union{Nothing,AbstractVector{Int}}=nothing,
    label::String="warm start",
    atol::Float64=1e-3,
    rtol::Float64=1e-3)

    current = gate_keys(gates)
    G = length(gates)
    identity_perm = collect(1:G)
    candidates = Pair{String,Vector{Int}}[]
    add!(name, p) = (isnothing(p) || any(c -> c.second == p, candidates)) || push!(candidates, name => p)

    if !isnothing(param_map)
        add!("current gate order (shared parameters)", identity_perm)
    elseif haskey(saved_dict, "gate_keys")
        perm = gate_permutation(saved_dict["gate_keys"], current)
        if isnothing(perm)
            @warn "$label: the saved gate set differs from the current gate set; discarding the warm start."
            return nothing
        end
        add!(perm == identity_perm ? "recorded gate order (= current order)" : "recorded gate order", perm)
    else
        add!("current gate order", identity_perm)
        for (name, keys) in legacy_gate_orders
            add!(name, gate_permutation(keys, current))
        end
        if !isnothing(shared_file) && isfile(shared_file)
            shared = try
                JLD2.load(shared_file)["dict"]
            catch
                nothing
            end
            if !isnothing(shared) && haskey(shared, "gates")
                add!("gate order in $(basename(shared_file))", gate_permutation(gate_keys(shared["gates"]), current))
            end
        end
    end

    println("  Verifying $label ($(length(candidates)) candidate gate order(s)" *
            (isnothing(expected_loss) ? "" : ", saved loss $expected_loss") * ")")
    c = Float64.(coeffs)
    losses = Float64[]
    for (name, perm) in candidates
        l = Float64(eval_loss(c, perm))
        push!(losses, l)
        println("    $name: loss = $l")
        if !isnothing(expected_loss) && abs(l - expected_loss) <= atol + rtol * abs(expected_loss)
            println("  -> using $name")
            return (coefficients=c, perm=perm, name=name)
        end
    end

    if !isnothing(expected_loss)
        @warn "$label: no candidate gate order reproduces the saved loss $expected_loss " *
              "(got $(losses)); the gate order, reference state, target or conventions have changed " *
              "since it was saved. Discarding the warm start."
        return nothing
    end
    zero_loss = Float64(eval_loss(zeros(Float64, length(c)), identity_perm))
    best = argmin(losses)
    if losses[best] < zero_loss - atol
        println("  -> using $(candidates[best].first) (loss $(losses[best]) vs zero-coefficient loss $zero_loss)")
        return (coefficients=c, perm=candidates[best].second, name=candidates[best].first)
    end
    @warn "$label: best candidate loss $(losses[best]) does not beat the zero-coefficient loss $zero_loss; discarding the warm start."
    return nothing
end
