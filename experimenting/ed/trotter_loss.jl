#=
trotter_loss.jl

Overlap and energy loss functions with custom ChainRules rrules for Trotter optimization.
=#

# ═══════════════════════════════════════════════════════════════════════
# OVERLAP LOSS
# ═══════════════════════════════════════════════════════════════════════

"""
    adjoint_loss(A, gates, tau_terms, ref, target, basis, N; ...) -> loss

Compute the infidelity loss: 1 - |<target|U(A)|ref>|^2.
"""
function adjoint_loss(A::AbstractArray, gates, tau_terms, ref::AbstractArray, target::AbstractArray, basis, N::Int;
    num_exponentials::Int=1, antihermitian::Bool=false, use_gpu::Bool=false, datatype::Type{<:Number}=ComplexF64)
    ref_evolved = apply_unitary(A, gates, ref, basis, N, num_exponentials; antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype)
    target_dev = to_device_vector(target, use_gpu, datatype)
    return max(0.0, 1.0 - abs2(dot(target_dev, ref_evolved)))
end

function ChainRulesCore.rrule(::typeof(adjoint_loss), A::AbstractArray, gates, tau_terms, ref::AbstractArray, target::AbstractArray, basis, N::Int;
    num_exponentials::Int=1, antihermitian::Bool=false, use_gpu::Bool=false, datatype::Type{<:Number}=ComplexF64)
    t = @elapsed begin
        target_dev = to_device_vector(target, use_gpu, datatype)
        phis = apply_unitary_checkpoints(A, gates, ref, basis, N, num_exponentials;
            antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype)
        evolved_ref = last_checkpoint(phis)
        overlap = dot(target_dev, evolved_ref)
        loss = max(0.0, 1.0 - abs2(overlap))
        println("loss: $loss")
    end
    println("Forward time: $t")

    function adjoint_loss_pullback(y)
        t = @elapsed begin
            init_adjoint_state = (2 * overlap * conj(y)) * target_dev
            grad_A = backward_adjoint_propagation(A, gates, tau_terms, phis, init_adjoint_state, basis, N, num_exponentials;
                antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype)
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
"""
function energy_loss(A::AbstractArray, gates, tau_terms, H, ref::AbstractArray, basis, N::Int;
    num_exponentials::Int=1, antihermitian::Bool=false, use_gpu::Bool=false, datatype::Type{<:Number}=ComplexF64)
    ref_evolved = apply_unitary(A, gates, ref, basis, N, num_exponentials; antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype)
    return real(dot(ref_evolved, H * ref_evolved))
end

function ChainRulesCore.rrule(::typeof(energy_loss), A::AbstractArray, gates, tau_terms, H, ref::AbstractArray, basis, N::Int;
    num_exponentials::Int=1, antihermitian::Bool=false, use_gpu::Bool=false, datatype::Type{<:Number}=ComplexF64)
    phis = apply_unitary_checkpoints(A, gates, ref, basis, N, num_exponentials;
        antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype)
    evolved_ref = last_checkpoint(phis)
    loss = real(dot(evolved_ref, H * evolved_ref))
    println("loss: $loss")
    function energy_loss_pullback(y)
        init_adjoint_state = (-2 * conj(y)) * (H * evolved_ref)
        grad_A = backward_adjoint_propagation(A, gates, tau_terms, phis, init_adjoint_state, basis, N, num_exponentials;
            antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype)
        return NoTangent(), grad_A, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return loss, energy_loss_pullback
end
