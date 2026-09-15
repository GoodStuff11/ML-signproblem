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
