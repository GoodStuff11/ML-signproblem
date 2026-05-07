using SparseArrays
using LinearAlgebra
using BenchmarkTools
using CUDA
# CUDA.set_runtime_version!("12.8")
using ExponentialUtilities

# Check if CUDA is available
println("CUDA available: ", CUDA.has_cuda_gpu())

dim = 5000
num_params = 300
N_steps = 50
density = 0.05

println("Creating mock data...")
# CPU mock ops
ops = [sprand(ComplexF64, dim, dim, density) for _ in 1:num_params]
ops = [op + op' for op in ops] # make hermitian

# GPU mock ops
println("Transferring ops to GPU...")
@time ops_gpu = CUDA.CUSPARSE.CuSparseMatrixCSC.(ops)

# Mock checkpoint vectors
println("Creating checkpoint vectors...")
phis = [rand(ComplexF64, dim) for _ in 1:(N_steps+1)]
chis = [rand(ComplexF64, dim) for _ in 1:(N_steps+1)]

println("Transferring checkpoint vectors to GPU...")
@time phis_gpu = CuArray.(phis)
@time chis_gpu = CuArray.(chis)

weights = rand(N_steps+1)
@time weights_gpu = CuArray(weights)

println("Benchmarking CPU gradient accumulation loop...")
function cpu_accumulate(ops, phis, chis, weights, grad_t)
    Threads.@threads for i in eachindex(grad_t)
        M = ops[i]
        val = 0.0 + 0.0im
        for k in 1:length(weights)
            term = dot(chis[k], M, phis[k])
            val += term * weights[k]
        end
        grad_t[i] = real(val)
    end
end

grad_t_cpu = zeros(num_params)
cpu_accumulate(ops, phis, chis, weights, grad_t_cpu) # compile
b_cpu = @benchmark cpu_accumulate($ops, $phis, $chis, $weights, $grad_t_cpu) samples=10
display(b_cpu)

println("\nBenchmarking GPU gradient accumulation loop...")
function gpu_accumulate_naive(ops_gpu, phis_gpu, chis_gpu, weights, grad_t_gpu)
    for i in eachindex(grad_t_gpu)
        M = ops_gpu[i]
        val = 0.0 + 0.0im
        for k in 1:length(weights)
            # using M * phis_gpu might allocate. 
            # Better to use dot(chis, M, phis) if supported. 
            # We'll see if it works and benchmark it.
            # term = dot(chis_gpu[k], M, phis_gpu[k])
            term = dot(chis_gpu[k], M * phis_gpu[k])
            val += term * weights[k]
        end
        grad_t_gpu[i] = real(val)
    end
end

grad_t_gpu = zeros(num_params)
gpu_accumulate_naive(ops_gpu, phis_gpu, chis_gpu, weights, grad_t_gpu) # compile
b_gpu = @benchmark CUDA.@sync(gpu_accumulate_naive($ops_gpu, $phis_gpu, $chis_gpu, $weights, $grad_t_gpu)) samples=10
display(b_gpu)

println("\nBenchmarking CPU expv...")
A = ops[1]
v = phis[1]
b_expv_cpu = @benchmark expv(1.0im, $A, $v) samples=5
display(b_expv_cpu)

println("\nBenchmarking GPU expv...")
A_gpu = ops_gpu[1]
v_gpu = phis_gpu[1]
b_expv_gpu = @benchmark CUDA.@sync(expv(1.0im, $A_gpu, $v_gpu)) samples=5
display(b_expv_gpu)
