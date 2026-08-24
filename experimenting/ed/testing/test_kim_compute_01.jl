#=
test_kim_compute_01.jl

Diagnostic test script to verify whether kim-compute-01 still exhibits
CUDA error: unknown error (code 999, ERROR_UNKNOWN) or primary context retain issues.
=#

ENV["JULIA_CUDA_USE_COMPAT"] = "true"

using LinearAlgebra
using SparseArrays
using Test

println("==================================================")
println("KIM-COMPUTE-01 CUDA DIAGNOSTIC TEST")
println("==================================================")
println("Hostname: ", gethostname())

println("\n--- Step 1: Loading CUDA.jl ---")
try
    using CUDA
    using CUDA.CUSPARSE
    println("CUDA module loaded successfully.")
catch e
    println("FAILED at loading CUDA:")
    showerror(stdout, e, catch_backtrace())
    println()
    exit(1)
end

println("\n--- Step 2: Querying Device & Functional Status ---")
try
    is_func = CUDA.functional()
    println("CUDA.functional(): $is_func")
    if !is_func
        println("CUDA is NOT functional on this node.")
        exit(1)
    end
    dev = CUDA.device()
    println("CUDA Device: $(CUDA.name(dev))")
    free_mem, tot_mem = CUDA.memory_info()
    println("VRAM: $(round(free_mem / 1024^3, digits=2)) GiB free / $(round(tot_mem / 1024^3, digits=2)) GiB total")
catch e
    println("FAILED at querying device status:")
    showerror(stdout, e, catch_backtrace())
    println()
    exit(1)
end

println("\n--- Step 3: Primary Context Retain & CuArray Allocation ---")
try
    # This was the exact point of failure for ERROR_UNKNOWN (code 999)
    println("Allocating CuArray{ComplexF64}(undef, 1_000_000)...")
    arr = CuArray{ComplexF64}(undef, 1_000_000)
    fill!(arr, ComplexF64(1.0, 2.0))
    CUDA.synchronize()
    println("Successfully allocated and initialized CuArray. First element: $(Array(arr[1:1])[1])")
catch e
    println("FAILED at CuArray allocation (reproducing code 999 error):")
    showerror(stdout, e, catch_backtrace())
    println()
    exit(1)
end

println("\n--- Step 4: Broadcast Kernel Execution & Math ---")
try
    x = CUDA.rand(Float32, 100_000)
    y = 2.5f0 .* x .+ 1.0f0
    CUDA.synchronize()
    println("Broadcast kernel executed successfully. Sample: x[1]=$(Array(x[1:1])[1]), y[1]=$(Array(y[1:1])[1])")
catch e
    println("FAILED at broadcast kernel execution:")
    showerror(stdout, e, catch_backtrace())
    println()
    exit(1)
end

println("\n--- Step 5: CUSPARSE Matrix-Vector Multiplication ---")
try
    I = [1, 2, 3, 4]
    J = [2, 1, 4, 3]
    V = ComplexF32[1.0f0, 1.0f0, -1.0f0, 1.0f0]
    sp = sparse(I, J, V, 4, 4)
    sp_dev = CUDA.CUSPARSE.CuSparseMatrixCSC(sp)
    v_in = CuArray(ComplexF32[1.0, 2.0, 3.0, 4.0])
    v_out = CUDA.zeros(ComplexF32, 4)
    CUDA.CUSPARSE.mv!('N', 1.0f0 + 0.0f0im, sp_dev, v_in, 0.0f0 + 0.0f0im, v_out, 'O')
    CUDA.synchronize()
    println("CUSPARSE mv! executed successfully. Result: $(Array(v_out))")
catch e
    println("FAILED at CUSPARSE operations:")
    showerror(stdout, e, catch_backtrace())
    println()
    exit(1)
end

println("\n==================================================")
println("ALL KIM-COMPUTE-01 CUDA TESTS PASSED!")
println("==================================================")
