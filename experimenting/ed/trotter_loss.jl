#=
trotter_loss.jl

Overlap and energy loss functions with custom ChainRules rrules for Trotter optimization.

Coefficient sharing (`param_map`) is handled entirely here: the reduced parameter
vector is expanded to the full per-step vector on the way in, and the full per-step
gradient is contracted back on the way out. See trotter_shared_params.jl. The
circuit routines below this layer never see a `param_map`.
=#

# ═══════════════════════════════════════════════════════════════════════
# OVERLAP LOSS
# ═══════════════════════════════════════════════════════════════════════

"""
    adjoint_loss(A, gates, tau_terms, ref, target, basis, N; ...) -> loss

Compute the infidelity loss: 1 - |<target|U(A)|ref>|^2.

With `param_map !== nothing`, `A` is the reduced parameter vector of length
`num_exponentials * maximum(param_map)` and gates sharing an entry of `param_map`
are driven by the same coefficient.

Throws `ArgumentError` for `antihermitian=true` with a gate set containing
diagonal gates (see `check_antihermitian_diagonal_gates`): `tau_g_operator_sector`
maps those to the zero operator, so the numbers would be silently plausible but
wrong. The guard is applied on both the primal and the `rrule`, since Zygote
calls the `rrule` directly and would otherwise bypass it.
"""
function adjoint_loss(A::AbstractArray, gates, tau_terms, ref::AbstractArray, target::AbstractArray, basis, N::Int;
    num_exponentials::Int=1, antihermitian::Bool=false, use_gpu::Bool=false, datatype::Type{<:Number}=ComplexF64,
    param_map::Union{Nothing,AbstractVector{Int}}=nothing)
    check_antihermitian_diagonal_gates(gates, antihermitian)
    A_steps = expand_shared_coefficients(A, param_map, length(gates), num_exponentials)
    ref_evolved = apply_unitary(A_steps, gates, ref, basis, N, num_exponentials; antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype)
    target_dev = to_device_vector(target, use_gpu, datatype)
    return 1.0 - abs2(dot(target_dev, ref_evolved))
end

function ChainRulesCore.rrule(::typeof(adjoint_loss), A::AbstractArray, gates, tau_terms, ref::AbstractArray, target::AbstractArray, basis, N::Int;
    num_exponentials::Int=1, antihermitian::Bool=false, use_gpu::Bool=false, datatype::Type{<:Number}=ComplexF64,
    param_map::Union{Nothing,AbstractVector{Int}}=nothing)
    check_antihermitian_diagonal_gates(gates, antihermitian)
    num_gates = length(gates)
    A_steps = expand_shared_coefficients(A, param_map, num_gates, num_exponentials)
    t = @elapsed begin
        target_dev = to_device_vector(target, use_gpu, datatype)
        phis = apply_unitary_checkpoints(A_steps, gates, ref, basis, N, num_exponentials;
            antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype)
        evolved_ref = last_checkpoint(phis)
        overlap = dot(target_dev, evolved_ref)
        loss = 1.0 - abs2(overlap)
        println("loss: $loss")
    end
    println("Forward time: $t")

    function adjoint_loss_pullback(y)
        t = @elapsed begin
            init_adjoint_state = (2 * overlap * conj(y)) * target_dev
            grad_steps = backward_adjoint_propagation(A_steps, gates, tau_terms, phis, init_adjoint_state, basis, N, num_exponentials;
                antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype)
            grad_A = contract_shared_gradient(grad_steps, param_map, num_gates, num_exponentials)
        end
        println("Gradient time: $t")
        return NoTangent(), grad_A, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return loss, adjoint_loss_pullback
end

# ═══════════════════════════════════════════════════════════════════════
# ENERGY LOSS
# ═══════════════════════════════════════════════════════════════════════

"""
    energy_loss(A, gates, tau_terms, H, ref, basis, N; ...) -> loss

Compute the variational energy loss: <ref|U(A)^† H U(A)|ref>.

With `param_map !== nothing`, `A` is the reduced parameter vector of length
`num_exponentials * maximum(param_map)`.

Throws `ArgumentError` for `antihermitian=true` with a gate set containing
diagonal gates, on both the primal and the `rrule` (see
`check_antihermitian_diagonal_gates`).
"""
function energy_loss(A::AbstractArray, gates, tau_terms, H, ref::AbstractArray, basis, N::Int;
    num_exponentials::Int=1, antihermitian::Bool=false, use_gpu::Bool=false, datatype::Type{<:Number}=ComplexF64,
    param_map::Union{Nothing,AbstractVector{Int}}=nothing)
    check_antihermitian_diagonal_gates(gates, antihermitian)
    A_steps = expand_shared_coefficients(A, param_map, length(gates), num_exponentials)
    ref_evolved = apply_unitary(A_steps, gates, ref, basis, N, num_exponentials; antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype)
    return real(dot(ref_evolved, H * ref_evolved))
end

function ChainRulesCore.rrule(::typeof(energy_loss), A::AbstractArray, gates, tau_terms, H, ref::AbstractArray, basis, N::Int;
    num_exponentials::Int=1, antihermitian::Bool=false, use_gpu::Bool=false, datatype::Type{<:Number}=ComplexF64,
    param_map::Union{Nothing,AbstractVector{Int}}=nothing)
    check_antihermitian_diagonal_gates(gates, antihermitian)
    num_gates = length(gates)
    A_steps = expand_shared_coefficients(A, param_map, num_gates, num_exponentials)
    phis = apply_unitary_checkpoints(A_steps, gates, ref, basis, N, num_exponentials;
        antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype)
    evolved_ref = last_checkpoint(phis)
    loss = real(dot(evolved_ref, H * evolved_ref))
    println("loss: $loss")
    function energy_loss_pullback(y)
        init_adjoint_state = (-2 * conj(y)) * (H * evolved_ref)
        grad_steps = backward_adjoint_propagation(A_steps, gates, tau_terms, phis, init_adjoint_state, basis, N, num_exponentials;
            antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype)
        grad_A = contract_shared_gradient(grad_steps, param_map, num_gates, num_exponentials)
        return NoTangent(), grad_A, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return loss, energy_loss_pullback
end
