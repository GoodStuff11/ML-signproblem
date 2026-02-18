
using CUDA
using LinearAlgebra
using SparseArrays
using ExponentialUtilities

println("Testing CUDA operations")

try
    println("Test 1: Dense Matrix Exponential")
    A = CUDA.rand(ComplexF64, 10, 10)
    E = exp(A)
    println("exp(A) successful")
catch e
    println("exp(A) failed: $e")
end

try
    println("Test 2: Sparse Matrix - Vector Multiplication (Mixed Types)")
    A_cpu = sparse([1, 2], [1, 2], [1.0, 2.0], 10, 10)
    A = CuSparseMatrixCSC(A_cpu)
    v = CUDA.rand(ComplexF64, 10)
    w = CUDA.zeros(ComplexF64, 10)
    mul!(w, A, v)
    println("mul!(w, A, v) successful")
    
    mul!(w, A, v, 1.0+0im, 0.0+0im)
    println("5-arg mul! successful")
catch e
    println("Sparse mul! failed: $e")
end

try
    println("Test 3: ExponentialUtilities expv with Sparse Matrix")
    A_cpu = sparse([1, 2], [1, 2], [1.0, 2.0], 10, 10)
    A = CuSparseMatrixCSC(A_cpu)
    v = CUDA.rand(ComplexF64, 10)
    phi = expv(1.0im, A, v)
    println("expv(A, v) successful")
catch e
    println("expv(A, v) failed: $e")
end
