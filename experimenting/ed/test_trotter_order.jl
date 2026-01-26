using Lattices
using LinearAlgebra
using Combinatorics
using SparseArrays
using Random
using ExponentialUtilities
using JSON
using JLD2

# Use a minimal set of includes to avoid conflicts
include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("utility_functions.jl")

# --- Setup ---
println("Loading N=3 Data for Indexer...")
dic = load_saved_dict("data/N=3_2x3/meta_data_and_E.jld2")
indexer = dic["indexer"]
dim = length(indexer.inv_comb_dict)

Random.seed!(422)
state1 = normalize(randn(ComplexF64, dim))
state2 = normalize(randn(ComplexF64, dim))

println("Dimension: $dim")

# Create 3 distinct operators
println("Creating operators...")
h1_dict = create_randomized_nth_order_operator(1, indexer; magnitude=1.0)
h2_dict = create_randomized_nth_order_operator(1, indexer; magnitude=1.0)
h3_dict = create_randomized_nth_order_operator(1, indexer; magnitude=1.0)

rows1, cols1, signs1, _ = build_n_body_structure(h1_dict, indexer)
H1 = make_hermitian(sparse(rows1, cols1, signs1, dim, dim))

rows2, cols2, signs2, _ = build_n_body_structure(h2_dict, indexer)
H2 = make_hermitian(sparse(rows2, cols2, signs2, dim, dim))

rows3, cols3, signs3, _ = build_n_body_structure(h3_dict, indexer)
H3 = make_hermitian(sparse(rows3, cols3, signs3, dim, dim))

println("Norms: H1=$(norm(H1)), H2=$(norm(H2)), H3=$(norm(H3))")
println("Norm H1-H2: ", norm(H1 - H2))

ops = [H1, H2, H3]
t_vals = [0.5, -0.3, 0.2]

# --- Test Evolution Accuracy ---
println("\nAccuracy Test (Overlap with Exact):")
H_exact = t_vals[1] * H1 + t_vals[2] * H2 + t_vals[3] * H3
psi_exact = expv(1im, H_exact, state1)
overlap_exact = state2' * psi_exact

for order in [1, 2, 4]
    println("Order $order:")
    for steps in [1, 5, 20]
        println("  Steps $steps...")
        psi_t = trotter_evolve(state1, ops, t_vals, order, steps)
        overlap_t = state2' * psi_t
        err = abs(overlap_exact - overlap_t)
        println("    Error = $err")
    end
end

# --- Test Gradient Consistency ---
println("\nGradient Consistency Test (Adjoint Trotter vs Numerical FD):")
steps = 5
order = 2
grad_adj = trotter_gradient_adjoint(state1, state2, ops, t_vals, order, steps)
println("Adjoint Grad: ", grad_adj)

eps = 1e-7
grad_fd = zeros(3)
for i in 1:3
    t_plus = copy(t_vals)
    t_plus[i] += eps
    t_minus = copy(t_vals)
    t_minus[i] -= eps

    val_plus = 1 - abs2(state2' * trotter_evolve(state1, ops, t_plus, order, steps))
    val_minus = 1 - abs2(state2' * trotter_evolve(state1, ops, t_minus, order, steps))
    grad_fd[i] = (val_plus - val_minus) / (2 * eps)
end
println("Numerical Grad: ", grad_fd)
println("Norm Diff: ", norm(grad_adj - grad_fd))
