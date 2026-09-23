#=
trotter_warm_start.jl

Safe reuse of saved coefficient vectors (resume, adjacent-U warm start, layer growth).

A coefficient vector only means something together with the exact gate ORDER it was
optimized with. That order has changed at least once (commit 9efb09f, 2026-08-31, started
sorting gates with `sortGatesByIJ`), and reference/target states can change too, so a
naively reloaded vector can silently land on an essentially random point of the landscape
(loss ≈ 1 for overlap). The helpers below:
  1. record every gate's identity (`gate_keys`) next to the coefficients when saving;
  2. remap a loaded vector onto the current gate order by gate identity; and
  3. verify the warm start by re-evaluating its loss, discarding it when no candidate
     alignment reproduces the saved loss (or beats the zero-coefficient loss).
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
    remap_coefficients(coeffs, from_keys, to_keys) -> Vector or nothing

Reorder a layer-major coefficient vector saved for gates `from_keys` so that entry `i` of
every layer drives gate `to_keys[i]`. Returns `nothing` if the two gate lists are not the
same set of distinct gates, or if `length(coeffs)` is not a whole number of layers.
"""
function remap_coefficients(coeffs::AbstractVector, from_keys::AbstractVector, to_keys::AbstractVector)
    G = length(to_keys)
    (length(from_keys) == G && G > 0 && length(coeffs) % G == 0) || return nothing
    pos = Dict{eltype(from_keys),Int}()
    for (i, k) in enumerate(from_keys)
        pos[k] = i
    end
    length(pos) == G || return nothing
    perm = Vector{Int}(undef, G)
    for (i, k) in enumerate(to_keys)
        j = get(pos, k, 0)
        j == 0 && return nothing
        perm[i] = j
    end
    P = length(coeffs) ÷ G
    out = similar(coeffs)
    for l in 0:(P-1), i in 1:G
        out[l*G+i] = coeffs[l*G+perm[i]]
    end
    return out
end

"""
    align_warm_start(coeffs, saved_dict, gates, eval_loss; kwargs...) -> Union{Vector,Nothing}

Map `coeffs` (loaded from `saved_dict`, possibly already grown with zero layers) onto the
current `gates` and verify it with `eval_loss(A_full) -> loss`. Returns the aligned vector,
or `nothing` when the warm start should be discarded.

Candidate alignments, in order:
- `saved_dict["gate_keys"]` present: that recorded order, remapped (trusted, checked once);
- otherwise: the vector as saved, then each `legacy_gate_orders` entry (name => keys), then
  the gate list stored in `shared_file` (if it exists).

With `expected_loss` (the saved loss for the SAME target and loss type), the first candidate
reproducing it within `atol + rtol*|expected_loss|` wins, and if none does the warm start is
discarded (the file does not describe this problem any more). Without `expected_loss` (e.g. a
warm start from a neighbouring U), the lowest-loss candidate is used when it beats the
zero-coefficient loss by more than `atol`; otherwise the warm start is discarded. With `param_map !== nothing` coefficients live
in the reduced shared-parameter space, so no remapping is attempted (verification only).
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
    candidates = Pair{String,Vector{Float64}}[]
    add!(name, c) = (isnothing(c) || any(p -> p.second == c, candidates)) || push!(candidates, name => Float64.(c))

    if !isnothing(param_map)
        add!("as saved (shared parameters, not remapped)", coeffs)
    elseif haskey(saved_dict, "gate_keys")
        remapped = remap_coefficients(coeffs, saved_dict["gate_keys"], current)
        if isnothing(remapped)
            @warn "$label: the saved gate set differs from the current gate set; discarding the warm start."
            return nothing
        end
        add!("recorded gate order", remapped)
    else
        add!("as saved (current gate order)", coeffs)
        for (name, keys) in legacy_gate_orders
            add!(name, remap_coefficients(coeffs, keys, current))
        end
        if !isnothing(shared_file) && isfile(shared_file)
            shared = try
                JLD2.load(shared_file)["dict"]
            catch
                nothing
            end
            if !isnothing(shared) && haskey(shared, "gates")
                add!("gate order in $(basename(shared_file))", remap_coefficients(coeffs, gate_keys(shared["gates"]), current))
            end
        end
    end

    println("  Verifying $label ($(length(candidates)) candidate alignment(s)" *
            (isnothing(expected_loss) ? "" : ", saved loss $expected_loss") * ")")
    losses = Float64[]
    for (name, c) in candidates
        l = Float64(eval_loss(c))
        push!(losses, l)
        println("    $name: loss = $l")
        if !isnothing(expected_loss) && abs(l - expected_loss) <= atol + rtol * abs(expected_loss)
            println("  -> using $name")
            return c
        end
    end

    if !isnothing(expected_loss)
        @warn "$label: no candidate alignment reproduces the saved loss $expected_loss " *
              "(got $(losses)); the gate order, reference state, target or conventions have changed " *
              "since it was saved. Discarding the warm start."
        return nothing
    end
    zero_loss = Float64(eval_loss(zeros(Float64, length(coeffs))))
    best = argmin(losses)
    if losses[best] < zero_loss - atol
        println("  -> using $(candidates[best].first) (loss $(losses[best]) vs zero-coefficient loss $zero_loss)")
        return candidates[best].second
    end
    @warn "$label: best candidate loss $(losses[best]) does not beat the zero-coefficient loss $zero_loss; discarding the warm start."
    return nothing
end
