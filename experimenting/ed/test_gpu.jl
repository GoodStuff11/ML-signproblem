
using Pkg
# Pkg.activate("/home/jek354/research/ML-signproblem")

using CUDA
using SparseArrays
using LinearAlgebra
using JLD2
using Optimization
using OptimizationOptimJL
using Lattices
using Combinatorics
using Random
using Statistics


include("ed_objects.jl")
include("ed_functions.jl")
include("utility_functions.jl")
include("ed_optimization_gpu.jl")

# Load data
println("Loading data...")
dic = load_saved_dict("data/N=(4, 4)_4x2/meta_data_and_E.jld2")

meta_data = dic["meta_data"]
U_values = meta_data["U_values"]
all_full_eig_vecs = dic["all_full_eig_vecs"]
indexer = dic["indexer"]

instructions = Dict(
    "starting state" => Dict("U index" => 1, "levels" => 1),
    "ending state" => Dict("U index" => length(all_full_eig_vecs), "levels" => 1), 
    "optimization_scheme" => [2], 
    "use symmetry" => true
)

start_idx = instructions["starting state"]["U index"]
end_idx = instructions["ending state"]["U index"]
println("Optimizing transition U=$start_idx -> U=$end_idx")

state1 = all_full_eig_vecs[start_idx][1, :]
state2 = all_full_eig_vecs[end_idx][1, :]

println("Starting GPU optimization...")

println("Methods of build_n_body_structure:")
display(methods(build_n_body_structure))
println("State 1 size: $(size(state1))")
println("State 2 size: $(size(state2))")
println("Indexer dimension: $(length(indexer.inv_comb_dict))")

optimize_unitary_gpu(state1, state2, indexer;
    maxiters=50,
    gradient=:adjoint_gradient,
    optimizer=:LBFGS,
    use_symmetry=true,
    spin_conserved=true
)

println("Total time: $(time() - t_start)s")
