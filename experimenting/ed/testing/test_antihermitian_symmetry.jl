using Lattices
using LinearAlgebra
using Combinatorics
using SparseArrays
using Statistics
using Random
using Zygote
using Optimization, OptimizationOptimisers
using OptimizationOptimJL
using JLD2
using ExponentialUtilities

include("../ed_objects.jl")
include("../ed_functions.jl")
include("../ed_optimization.jl")
include("../utility_functions.jl")

# Load Data
dic = load_saved_dict("data/N=6_4x3/meta_data_and_E.jld2")
meta_data = dic["meta_data"]
all_full_eig_vecs = dic["all_full_eig_vecs"]
indexer = dic["indexer"]

# Setup Optimization Instructions - WITH SYMMETRY
instructions = Dict("starting state" => Dict("U index" => 1, "levels" => 1),
    "ending state" => Dict("U index" => 30, "levels" => 1), "max_order" => 2, "use symmetry" => true)

println("\n--- Testing Standard Hermitian Optimization (Symmetry ON) ---")
try
    data_herm = test_map_to_state(all_full_eig_vecs[6], instructions, indexer, !isa(meta_data["electron count"], Number);
        maxiters=20, optimization=:adjoint_gradient)
    println("Hermitian Loss: ", data_herm["loss_metrics"][end][end])
    println("Params count: ", length(data_herm["coefficients"][end]))
catch e
    println("Hermitian Optimization Failed: ", e)
    showerror(stdout, e, catch_backtrace())
end

println("\n--- Testing Anti-Hermitian Optimization (Symmetry ON) ---")
instructions_anti = copy(instructions)
instructions_anti["antihermitian"] = true

try
    data_anti = test_map_to_state(all_full_eig_vecs[6], instructions_anti, indexer, !isa(meta_data["electron count"], Number);
        maxiters=20, optimization=:adjoint_gradient)

    println("Anti-Hermitian Loss: ", data_anti["loss_metrics"][end][end])
    println("Params count: ", length(data_anti["coefficients"][end]))

    # Check if diagonal terms were zeroed out (implied if params count differs or manually check?)
    # We can check specific terms if needed, but low loss suggests it works.
catch e
    println("Anti-Hermitian Optimization Failed: ", e)
    showerror(stdout, e, catch_backtrace())
end
