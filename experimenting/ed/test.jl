using Lattices
using LinearAlgebra
using Combinatorics
using SparseArrays
using Plots
import Graphs
using LaTeXStrings
using Statistics
using Random
using Zygote
using Optimization, OptimizationOptimisers
using JSON
using OptimizationOptimJL
using JLD2
using ExponentialUtilities


include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("utility_functions.jl")

folder_name = "data/N=(3, 3)_3x2"
dic = load_saved_dict(folder_name * "/meta_data_and_E.jld2")

meta_data = dic["meta_data"]
U_values = meta_data["U_values"]
all_full_eig_vecs = dic["all_full_eig_vecs"]
n_eigs = [parse(Int, x) for x in split(meta_data["sites"], "x")]
indexer = dic["indexer"]

# meta_data = Dict("electron count"=>3, "sites"=>"2x3", "bc"=>"periodic", "basis"=>"adiabatic", 
#                 "U_values"=>U_values, "maxiters"=>10)
instructions = Dict("starting state" => Dict("U index" => 1, "levels" => 1),
     "ending state" => Dict("U index" => 30, "levels" => 1), "optimization_scheme" => [1, 2], "use symmetry" => true)
println("U in [", U_values[instructions["starting state"]["U index"]], ", ", U_values[instructions["ending state"]["U index"]], "]")

# New test using interaction_scan_map_to_state
println("\n--- Running Interaction Scan ---")
scan_instructions = Dict(
     "starting state" => Dict("U index" => 1, "levels" => [1]),
     "ending state" => Dict("U index" => 1, "levels" => [1]), # level index for targets
     "u_range" => 1:length(U_values),
     "optimization_scheme" => [1, 2],
     "use symmetry" => true
)

scan_data = interaction_scan_map_to_state(all_full_eig_vecs[1], scan_instructions, indexer, !isa(meta_data["electron count"], Number);
     maxiters=100, gradient=:adjoint_gradient,
     optimizer=[:GradientDescent, :LBFGS, :GradientDescent, :LBFGS, :GradientDescent, :LBFGS],
     save_folder=folder_name, save_name="test_scan")

println("\nScan complete. Final losses: $(scan_data["loss_metrics"])")
scan_data