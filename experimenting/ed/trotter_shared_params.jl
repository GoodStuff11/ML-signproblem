#=
trotter_shared_params.jl

Shared Trotter coefficients.

Letting several gates be driven by one variational parameter is a *linear
reparameterisation* of the circuit:

    a = E * theta                       (a gather)
    d loss / d theta = E' * d loss / d a   (a scatter-ADD)

where `a` is the per-step coefficient vector the circuit already consumes
(length `P * num_gates`, layer-major) and `theta` is the reduced parameter
vector (length `P * n_params`). Concretely

    a[(l-1)*num_gates + j] == theta[(l-1)*n_params + param_map[j]]

for layer `l in 1:P` and gate `j in 1:num_gates`.

Because the reparameterisation sits strictly *above* the circuit, none of the
checkpointing, rematerialisation or GPU adjoint code needs to know about it:
those routines keep receiving a full length-`P*num_gates` vector. In particular
the local name `param_idx` inside those loops remains a *gate* index used to
look up `gpu_ops`, exactly as before.

The chain rule gives the scatter-add whether or not the gates sharing a
parameter commute. Commutation matters for the ansatz being a faithful HVA
layer, not for gradient correctness.

`param_map === nothing` means "no sharing" and short-circuits every helper here
to identity, which is what guarantees byte-for-byte unchanged behaviour for all
existing callers.
=#

"""
    check_antihermitian_diagonal_gates(gates, antihermitian::Bool)

Guard shared by every loss/optimization entry point: under the antihermitian
convention, `tau_g_operator_sector` maps a diagonal gate to the ZERO operator
(TamFermion.jl), so any coefficient driving a diagonal gate would be silently
unoptimisable. Throws `ArgumentError` when `antihermitian=true` and `gates`
contains a diagonal gate; short-circuits to a no-op when `antihermitian=false`
so the ordinary (non-antihermitian) path pays nothing extra.
"""
function check_antihermitian_diagonal_gates(gates, antihermitian::Bool)
    if antihermitian && any(TamFermion.is_diagonal_gate, gates)
        throw(ArgumentError(
            "antihermitian=true with a gate set containing diagonal gates: " *
            "tau_g_operator_sector maps those to the zero operator, so their " *
            "coefficients would be unoptimisable. The HVA gate set from " *
            "enumerate_ferm_excitations_HVA is Hermitian by construction; use " *
            "antihermitian=false."))
    end
    return nothing
end

"""
    num_shared_params(param_map, num_gates) → Int

Number of distinct variational parameters **per Trotter layer**.
`nothing` means one parameter per gate.
"""
num_shared_params(param_map::Nothing, num_gates::Int) = num_gates
num_shared_params(param_map::AbstractVector{Int}, num_gates::Int) = maximum(param_map)

function _check_shared(param_map::AbstractVector{Int}, theta_len::Int, num_gates::Int, P::Int)
    if length(param_map) != num_gates
        throw(ArgumentError(
            "length(param_map) = $(length(param_map)) does not match num_gates = $num_gates"))
    end
    n_params = maximum(param_map)
    if sort(unique(param_map)) != collect(1:n_params)
        throw(ArgumentError(
            "param_map = $param_map is not a contiguous surjection onto 1:$n_params " *
            "(it has a gap); every param_map produced by this branch's helpers is " *
            "contiguous by construction (_renumber_contiguous), so a gappy map would " *
            "leave a dead, never-driven parameter slot with a zero gradient"))
    end
    if theta_len != P * n_params
        throw(ArgumentError(
            "coefficient vector has length $theta_len but num_exponentials * n_params = " *
            "$P * $n_params = $(P * n_params)"))
    end
    return n_params
end

"""
    expand_shared_coefficients(theta, param_map, num_gates, P) → a

Gather the reduced parameter vector `theta` into the full per-step coefficient
vector `a` of length `P * num_gates` that the circuit consumes.
"""
expand_shared_coefficients(theta::AbstractVector, ::Nothing, num_gates::Int, P::Int) = theta

function expand_shared_coefficients(theta::AbstractVector{T}, param_map::AbstractVector{Int},
    num_gates::Int, P::Int) where {T}
    n_params = _check_shared(param_map, length(theta), num_gates, P)
    a = Vector{T}(undef, P * num_gates)
    for l in 1:P
        off_a = (l - 1) * num_gates
        off_t = (l - 1) * n_params
        for j in 1:num_gates
            a[off_a+j] = theta[off_t+param_map[j]]
        end
    end
    return a
end

"""
    contract_shared_gradient(grad_a, param_map, num_gates, P) → grad_theta

Scatter-**add** a full per-step gradient back onto the reduced parameters. This
is the exact transpose of `expand_shared_coefficients`, so a shared parameter's
gradient is the sum of the gradients of every step it drives.
"""
contract_shared_gradient(grad_a::AbstractVector, ::Nothing, num_gates::Int, P::Int) = grad_a

function contract_shared_gradient(grad_a::AbstractVector{T}, param_map::AbstractVector{Int},
    num_gates::Int, P::Int) where {T}
    if length(param_map) != num_gates
        throw(ArgumentError(
            "length(param_map) = $(length(param_map)) does not match num_gates = $num_gates"))
    end
    n_params = maximum(param_map)
    if length(grad_a) != P * num_gates
        throw(ArgumentError(
            "grad_a has length $(length(grad_a)) but num_exponentials * num_gates = " *
            "$(P * num_gates)"))
    end
    grad_theta = zeros(T, P * n_params)   # MUST be zeros: entries are accumulated
    for l in 1:P
        off_a = (l - 1) * num_gates
        off_t = (l - 1) * n_params
        for j in 1:num_gates
            grad_theta[off_t+param_map[j]] += grad_a[off_a+j]
        end
    end
    return grad_theta
end

"""
    hva_layers_matching_dof(param_map, num_gates, reference_num_gates,
                            reference_num_exponentials=1; mode=:ceil) → NamedTuple

How many HVA (shared-coefficient) layers the ansatz needs for its variational
degrees of freedom to match the *standard* (one-parameter-per-gate) ansatz built
from `reference_num_gates` gates over `reference_num_exponentials` Trotter layers.

The standard ansatz carries `reference_num_gates * reference_num_exponentials`
free parameters. A shared-coefficient ansatz carries
`num_shared_params(param_map, num_gates)` free parameters *per layer*, so the
answer is that ratio, rounded per `mode` (`:ceil` — the default, so the HVA never
ends up under-parameterized; `:floor`; `:round`) and clamped to at least 1.

`reference_num_gates` may instead be the reference gate vector itself, and
`param_map === nothing` means one parameter per gate.

Returns `(; layers, params_per_layer, hva_dof, reference_dof, exact)`, where
`exact` reports whether the two parameter counts coincide exactly rather than
being rounded apart.

# Example
```julia
gates_std = enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true)
gates_hva, pmap = enumerate_ferm_excitations_HVA(Lvec)
info = hva_layers_matching_dof(pmap, length(gates_hva), gates_std, 3)
info.layers    # HVA layer count whose DOF count matches 3 standard layers
```
"""
function hva_layers_matching_dof(param_map::Union{Nothing,AbstractVector{Int}},
    num_gates::Int,
    reference_num_gates::Int,
    reference_num_exponentials::Int=1;
    mode::Symbol=:ceil)

    if !(mode in (:ceil, :floor, :round))
        throw(ArgumentError("mode must be one of :ceil, :floor, :round; got :$mode"))
    end
    if num_gates <= 0
        throw(ArgumentError("num_gates must be positive; got $num_gates"))
    end
    if reference_num_gates <= 0
        throw(ArgumentError("reference_num_gates must be positive; got $reference_num_gates"))
    end
    if reference_num_exponentials <= 0
        throw(ArgumentError(
            "reference_num_exponentials must be positive; got $reference_num_exponentials"))
    end

    n_params = num_shared_params(param_map, num_gates)
    if n_params <= 0
        throw(ArgumentError("param_map yields $n_params parameters per layer"))
    end

    reference_dof = reference_num_gates * reference_num_exponentials
    ratio = reference_dof / n_params
    layers = if mode === :ceil
        ceil(Int, ratio)
    elseif mode === :floor
        floor(Int, ratio)
    else
        round(Int, ratio)
    end
    layers = max(layers, 1)

    return (layers=layers,
        params_per_layer=n_params,
        hva_dof=layers * n_params,
        reference_dof=reference_dof,
        exact=(layers * n_params == reference_dof))
end

hva_layers_matching_dof(param_map::Union{Nothing,AbstractVector{Int}}, num_gates::Int,
    reference_gates::AbstractVector, reference_num_exponentials::Int=1;
    mode::Symbol=:ceil) =
    hva_layers_matching_dof(param_map, num_gates, length(reference_gates),
        reference_num_exponentials; mode=mode)
