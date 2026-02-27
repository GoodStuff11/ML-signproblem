using SparseArrays
using LinearAlgebra
using ExponentialUtilities
using KrylovKit
N = 1000
A = sprand(N, N, 0.05)
A = A + A' # Symmetric
H = Hermitian(A)
v = randn(ComplexF64, N)
v /= norm(v)

println("Norm of H: ", norm(H, Inf))

# Try exponentiate
println("Running KrylovKit.exponentiate...")
@time psi_k, info = exponentiate(H, 1.0im, v, ishermitian=true)
println("Info: ", info)

# Try expv
println("Running ExponentialUtilities.expv...")
@time psi_e = expv(1.0im, H, v)
