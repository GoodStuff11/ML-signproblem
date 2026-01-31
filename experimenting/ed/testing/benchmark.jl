using Random
using Zygote
using LinearAlgebra
using SparseArrays

println("Loading testing.jl...")
include("testing.jl")

d_analytic = d_objective_dense([1, 1])

# Original slow implementation for comparison
function d_objective_original(a)
    X = sum(i -> a[i] * M[i], 1:DIM)
    1 / N * map(j -> sum(i -> v_2' * (I + X / N)^(N - i) * M[j] * (I + X / N)^i * v_1, 1:N), 1:DIM)
end

println("Analytic (Optimized): ", d_analytic)
try
    d_orig = d_objective_original([1, 1])
    println("Original: ", d_orig)
    println("Diff vs Original: ", norm(d_analytic - d_orig))
catch e
    println("Original implementation failed: ", e)
end
println("Original matches Finite Diff? ", isapprox(d_analytic, finite_difference_check, atol=1e-3))


function run_large_benchmark()
    println("\n--- Starting Large Benchmark ---")

    # Set up large problem
    # "Hs_dim will be very large"
    # "DIM will also be sufficiently large"

    global Hs_dim = 2000
    global DIM = 100
    global N = 1000 # Keep N reasonable for benchmark time, or 1000 as per file

    println("Parameters: Hs_dim=$Hs_dim, DIM=$DIM, N=$N")

    global v_1 = rand(Hs_dim)
    global v_2 = rand(Hs_dim)

    # Generate disjoint sparse matrices
    # Split entries among DIM matrices
    indices = [(i, j) for i in 1:Hs_dim for j in 1:Hs_dim]
    shuffle!(indices)

    global M = SparseMatrixCSC{Float64,Int}[]
    chunk_size = div(length(indices), DIM)

    start_idx = 1
    for k in 1:DIM
        end_idx = min(start_idx + chunk_size, length(indices) + 1)
        my_ind = indices[start_idx:end_idx-1]

        Is = [x[1] for x in my_ind]
        Js = [x[2] for x in my_ind]
        Vs = rand(length(my_ind)) .* 0.01

        push!(M, sparse(Is, Js, Vs, Hs_dim, Hs_dim))

        start_idx = end_idx
    end

    a_test = rand(DIM)

    println("Running d_objective_dense (warmup)...")
    d_objective_dense(a_test)

    println("Running d_objective_dense (measured)...")
    @time res = d_objective_dense(a_test)

    println("Running d_objective_sparse (warmup)...")
    d_objective_sparse(a_test)

    println("Running d_objective_sparse (measured)...")
    @time res = d_objective_sparse(a_test)

    # Redefine objective for the large case to ensure it uses the correct global M
    large_objective = (a) -> v_1' * exp(Matrix(sum(i -> a[i] * M[i], 1:DIM))) * v_2

    # println("\n--- Finite Difference Check ---")
    # println("Running Finite Difference (DIM=$DIM)...")
    # fd_grad = zeros(Float64, DIM)
    # epsilon = 1e-5
    # # Calculate FD for each dimension.
    # # Base value
    # val_base = large_objective(a_test)

    # @time for i in 1:DIM
    #     a_perturbed = copy(a_test)
    #     a_perturbed[i] += epsilon
    #     val_perturbed = large_objective(a_perturbed)
    #     fd_grad[i] = (val_perturbed - val_base) / epsilon
    # end

    # println("FD Grad norm: ", norm(fd_grad))
    # println("Optimized Grad norm: ", norm(res))
    # println("Diff vs FD: ", norm(res - fd_grad))
    # println("Matches within tolerance? ", isapprox(res, fd_grad, rtol=1e-2))

    println("\n--- Zygote Check ---")
    println("Running Zygote gradient...")
    @time zygote_grad_tuple = gradient(large_objective, a_test)
    zygote_grad = zygote_grad_tuple[1]

    println("Zygote Grad norm: ", norm(zygote_grad))
    println("Diff vs Zygote: ", norm(res - zygote_grad))
    println("Matches Zygote? ", isapprox(res, zygote_grad, atol=1e-5))

    println("Result size: ", size(res))
    println("Done.")
end

run_large_benchmark()
