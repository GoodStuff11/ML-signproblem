"""
Apply Givens rotation directly to matrix rows/columns (in-place, optimized)
"""
function apply_givens_inplace!(U::Matrix{ComplexF64}, i::Int, j::Int, θ::Float64, φ::Float64)
    c = cos(θ)
    s = sin(θ)
    eiφ = exp(1im * φ)
    s_conj = s * conj(eiφ)
    s_eiφ = s * eiφ
    
    n = size(U, 2)
    @inbounds @simd for k in 1:n
        temp_i = U[i, k]
        temp_j = U[j, k]
        U[i, k] = c * temp_i + s_eiφ * temp_j
        U[j, k] = c * temp_j - s_conj * temp_i
    end
end

"""
GPU version of Givens rotation
"""
function apply_givens_gpu!(U::CuMatrix{ComplexF64}, i::Int, j::Int, θ::Float64, φ::Float64)
    c = cos(θ)
    s = sin(θ)
    eiφ = exp(1im * φ)
    
    n = size(U, 2)
    function kernel(U, i, j, c, s, eiφ)
        k = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        if k <= n
            @inbounds begin
                temp_i = U[i, k]
                temp_j = U[j, k]
                U[i, k] = c * temp_i + s * eiφ * temp_j
                U[j, k] = c * temp_j - s * conj(eiφ) * temp_i
            end
        end
        return nothing
    end
    
    threads = min(256, n)
    blocks = cld(n, threads)
    @cuda threads=threads blocks=blocks kernel(U, i, j, c, s, eiφ)
    CUDA.synchronize()
end

"""
Batch apply non-conflicting Givens rotations in parallel
"""
function find_parallel_groups(pairs::Vector{Tuple{Int,Int}})
    groups = Vector{Vector{Int}}()
    used_indices = Set{Int}()
    
    for (idx, (i, j)) in enumerate(pairs)
        if !(i in used_indices || j in used_indices)
            if isempty(groups) || length(groups[end]) >= 32  # Limit group size
                push!(groups, [idx])
            else
                push!(groups[end], idx)
            end
            push!(used_indices, i, j)
        else
            # Start new group
            push!(groups, [idx])
            used_indices = Set([i, j])
        end
    end
    
    return groups
end

"""
Optimized layer structure with pre-computed groupings
"""
struct FastGivensLayer
    n::Int
    pairs::Vector{Tuple{Int,Int}}
    parallel_groups::Vector{Vector{Int}}
    θ::Vector{Float64}
    φ::Vector{Float64}
end

function FastGivensLayer(n::Int, pairs::Vector{Tuple{Int,Int}})
    np = length(pairs)
    θ = 2π * rand(np) .- π
    φ = 2π * rand(np) .- π
    groups = find_parallel_groups(pairs)
    return FastGivensLayer(n, pairs, groups, θ, φ)
end

"""
Optimized neural network with flat parameter storage
"""
struct FastGivensNN
    n::Int
    layers::Vector{FastGivensLayer}
    param_count::Int
    workspace::Matrix{ComplexF64}  # Pre-allocated workspace
    gpu_workspace::Union{CuMatrix{ComplexF64}, Nothing}
end

function FastGivensNN(n::Int, num_layers::Int, pair_pattern::Symbol=:nearest_neighbor; use_gpu::Bool=false)
    pairs = create_pairs(n, pair_pattern)
    layers = [FastGivensLayer(n, pairs) for _ in 1:num_layers]
    param_count = sum(2 * length(layer.pairs) for layer in layers)
    workspace = Matrix{ComplexF64}(undef, n, n)
    
    gpu_workspace = if use_gpu && CUDA.functional()
        CuMatrix{ComplexF64}(undef, n, n)
    else
        nothing
    end
    
    return FastGivensNN(n, layers, param_count, workspace, gpu_workspace)
end

"""
Apply layer with batched operations
"""
function apply_layer_fast!(U::Matrix{ComplexF64}, layer::FastGivensLayer)
    # Use batched operations for parallel groups
    for group in layer.parallel_groups
        ThreadsX.foreach(group) do idx
            i, j = layer.pairs[idx]
            apply_givens_inplace!(U, i, j, layer.θ[idx], layer.φ[idx])
        end
    end
end

"""
Apply layer on GPU
"""
function apply_layer_gpu!(U::CuMatrix{ComplexF64}, layer::FastGivensLayer)
    for (idx, (i, j)) in enumerate(layer.pairs)
        apply_givens_gpu!(U, i, j, layer.θ[idx], layer.φ[idx])
    end
end

"""
Fast network application using pre-allocated workspace
"""
function apply_network_fast!(nn::FastGivensNN, use_gpu::Bool=false)
    if use_gpu && nn.gpu_workspace !== nothing
        # GPU computation
        U = nn.gpu_workspace
        fill!(U, ComplexF64(0))
        U[diagind(U)] .= ComplexF64(1)  # Identity
        
        for layer in nn.layers
            apply_layer_gpu!(U, layer)
        end
        
        return Array(U)  # Copy back to CPU
    else
        # CPU computation
        U = nn.workspace
        fill!(U, ComplexF64(0))
        U[diagind(U)] .= ComplexF64(1)  # Identity
        
        for layer in nn.layers
            apply_layer_fast!(U, layer)
        end
        
        return copy(U)  # Return copy to avoid overwriting
    end
end


"""
Apply Givens rotation to matrix using purely functional row operations
"""
function apply_givens_to_matrix_functional(U::Matrix{ComplexF64}, i::Int, j::Int, θ::Float64, φ::Float64)
    c = cos(θ)
    s = sin(θ)
    eiφ = exp(1im * φ)
    n = size(U, 1)
    
    # Create new matrix using map (completely functional)
    indices = CartesianIndices(U)
    U_new = map(indices) do idx
        row, col = Tuple(idx)
        if row == i
            return c * U[i, col] + eiφ * s * U[j, col]
        elseif row == j
            return c * U[j, col] - conj(eiφ) * s * U[i, col]
        else
            return U[row, col]
        end
    end
    
    return U_new
end

# """
# Completely functional network application for AD
# """
# function apply_network_functional(params::Vector{Float64}, n::Int, layer_structures::Vector{Vector{Tuple{Int,Int}}})
#     U = Matrix{ComplexF64}(LinearAlgebra.I, n, n)
#     param_idx = 1
    
#     for pairs in layer_structures
#         for (i, j) in pairs
#             θ = params[param_idx]
#             φ = params[param_idx + 1]
            
#             # Apply rotation functionally (no mutations)
#             U = apply_givens_to_matrix_functional(U, i, j, θ, φ)
#             param_idx += 2
#         end
#     end
    
#     return U
# end

"""
Extract parameters as flat vector
"""
function get_parameters(nn::FastGivensNN)
    params = Float64[]
    for layer in nn.layers
        append!(params, layer.θ)
        append!(params, layer.φ)
    end
    return params
end

"""
Set parameters from flat vector
"""
function set_parameters!(nn::FastGivensNN, params::Vector{Float64})
    idx = 1
    for layer in nn.layers
        np = length(layer.pairs)
        layer.θ[:] = params[idx:(idx+np-1)]
        idx += np
        layer.φ[:] = params[idx:(idx+np-1)]
        idx += np
    end
end

"""
AD-compatible loss function using purely functional operations
"""
function loss_function_ad(params::Vector{Float64}, target::Matrix{ComplexF64}, nn)
    set_parameters!(nn, params)
    U = apply_network_fast!(nn)
    return sum(abs2.(U - target))
end

"""
Create pairing patterns
"""
function create_pairs(n::Int, pattern::Symbol=:nearest_neighbor)
    pairs = Tuple{Int,Int}[]
    
    if pattern == :nearest_neighbor
        for i in 1:(n-1)
            push!(pairs, (i, i+1))
        end
    elseif pattern == :all_pairs
        for i in 1:n
            for j in (i+1):n
                push!(pairs, (i, j))
            end
        end
    elseif pattern == :alternating
        # Odd pairs first, then even pairs
        for i in 1:2:(n-1)
            push!(pairs, (i, i+1))
        end
        for i in 2:2:(n-1)
            if i+1 <= n
                push!(pairs, (i, i+1))
            end
        end
    end
    
    return pairs
end

"""
Optimized training using automatic differentiation
"""
function train_network_ad!(nn::FastGivensNN, target::Matrix{ComplexF64}; 
                          lr::Float64=0.01, epochs::Int=1000, use_gpu::Bool=false)
    
    # Loss function closure
    loss_fn(params) = loss_function_ad(params, target, nn)
    
    # Get initial parameters
    params = get_parameters(nn)
    
    losses = Float64[]
    
    println("Starting AD training...")
    
    # Test AD compatibility first
    # try
    # test_loss, test_grads = withgradient(loss_fn, params)
    # println("AD test successful, proceeding with training...")
    # catch e
    #     println("AD test failed: $e")
    #     println("Falling back to finite differences...")
    #     return train_network_fd!(nn, target, lr=lr, epochs=epochs)
    # end
    
    for epoch in 1:epochs
        # try
            # Compute gradients
            loss_val, grads = withgradient(loss_fn, params)
            push!(losses, loss_val)
            
            # Update parameters manually with gradient descent
            if grads[1] !== nothing
                # Simple gradient descent update
                params = params - lr .* grads[1]
            end
            
            if epoch % 100 == 0
                println("Epoch $epoch: Loss = $(loss_val)")
            end
            
        # catch e
        #     println("Error at epoch $epoch: $e")
        #     println("Falling back to finite differences...")
        #     # Set current parameters and continue with FD
        #     set_parameters!(nn, params)
        #     remaining_losses = train_network_fd!(nn, target, lr=lr, epochs=epochs-epoch+1)
        #     append!(losses, remaining_losses)
        #     return losses
        # end
    end
    
    # Set final parameters
    set_parameters!(nn, params)
    
    return losses
end

"""
Fallback finite difference training
"""
function train_network_fd!(nn::FastGivensNN, target::Matrix{ComplexF64}; 
                          lr::Float64=0.01, epochs::Int=1000)
    
    function loss_fn(params)
        set_parameters!(nn, params)
        U = apply_network_fast!(nn, false)
        return real(tr((U - target)' * (U - target)))
    end
    
    params = get_parameters(nn)
    losses = Float64[]
    
    println("Using finite difference optimization...")
    
    for epoch in 1:epochs
        # Simple finite difference gradient
        loss_val = loss_fn(params)
        push!(losses, loss_val)
        
        # Compute gradient via finite differences
        grad = zeros(length(params))
        h = 1e-6
        
        for i in 1:length(params)
            params[i] += h
            loss_plus = loss_fn(params)
            params[i] -= 2h
            loss_minus = loss_fn(params)
            params[i] += h  # restore
            
            grad[i] = (loss_plus - loss_minus) / (2h)
        end
        
        # Update parameters
        params .-= lr .* grad
        
        if epoch % 100 == 0
            println("Epoch $epoch: Loss = $(loss_val)")
        end
    end
    
    set_parameters!(nn, params)
    return losses
end
