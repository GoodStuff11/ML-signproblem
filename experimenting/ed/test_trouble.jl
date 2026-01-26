using Lattices
using LinearAlgebra
using Combinatorics
using SparseArrays
using Random
using ExponentialUtilities
using JSON
using JLD2

include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("utility_functions.jl")

println("Loading N=3 Data for Indexer...")
dic = load_saved_dict("data/N=3_2x3/meta_data_and_E.jld2")
indexer = dic["indexer"]
dim = length(indexer.inv_comb_dict)

Random.seed!(422)
state1 = normalize(randn(ComplexF64, dim))

# Create 1 simple operator
println("Creating operator...")
h1_dict = create_randomized_nth_order_operator(1, indexer; magnitude=1.0)
rows1, cols1, signs1, _ = build_n_body_structure(h1_dict, indexer)
H1 = make_hermitian(sparse(rows1, cols1, signs1, dim, dim))

println("Norm H1: ", norm(H1))
if norm(H1) == 0
    println("WARNING: H1 is zero!")
end

println("Testing expv(1im, H1, state1)...")
try
    res = expv(1im, H1, state1)
    println("Success! Norm result: ", norm(res))
catch e
    println("Caught error: ", e)
    rethrow(e)
end
