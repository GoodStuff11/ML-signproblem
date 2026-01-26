using LinearAlgebra
using SparseArrays
using Random
using ExponentialUtilities
using Zygote
using Lattices
using Combinatorics

# Include the actual optimization file
include("ed_objects.jl") # Dependency
include("ed_functions.jl") # Dependency
include("ed_optimization.jl")

Random.seed!(42)
dim = 10
state1 = normalize(randn(ComplexF64, dim))
state2 = normalize(randn(ComplexF64, dim))

H0 = randn(ComplexF64, dim, dim)
H0 = H0 + H0' # Hermitian
Hk = randn(ComplexF64, dim, dim)
Hk = Hk + Hk'

t_val = 0.5

# Convert to sparse for consistency with benchmark usage
H0s = sparse(H0)
Hks = sparse(Hk)

println("--- Testing ed_optimization.jl AuxiliaryMatrix ---")

# 1. Zygote (Reference)
function f(t)
    H = t * Hk + H0
    U = exp(1im * H)
    overlap = state2' * U * state1
    return 1 - abs2(overlap)
end

grad_zygote = Zygote.gradient(f, t_val)[1]
println("Zygote Grad: $grad_zygote")

# 2. Auxiliary Matrix (Using ed_optimization.jl struct)
H_eff = t_val * Hks + H0s
M = AuxiliaryMatrix(H_eff, Hks, dim)
v_in = zeros(ComplexF64, 2 * dim)
v_in[1:dim] = state1

println("M type: ", typeof(M))
println("Size M: ", size(M))

v_out = expv(1im, M, v_in)
psi_u = expv(1im, H_eff, state1)
d_psi = v_out[dim+1:end]

overlap_val = state2' * psi_u
grad_aux = -2 * real(conj(overlap_val) * (state2' * d_psi))
println("Aux Grad:    $grad_aux")

diff = abs(grad_aux - grad_zygote)
println("Error:       $diff")

if diff < 1e-9
    println("✅ Match!")
else
    println("❌ Mismatch!")
end
