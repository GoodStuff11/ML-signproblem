
# using Pkg
# Pkg.activate("/home/jek354/research/ML-signproblem")
# Pkg.update()

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

dic = load_saved_dict("data/N=(4, 4)_4x2/meta_data_and_E.jld2")

meta_data = dic["meta_data"]
U_values = meta_data["U_values"]
all_full_eig_vecs = dic["all_full_eig_vecs"]
n_eigs = [parse(Int,x) for x in split(meta_data["sites"], "x")]
indexer = dic["indexer"]

# meta_data = Dict("electron count"=>3, "sites"=>"2x3", "bc"=>"periodic", "basis"=>"adiabatic", 
#                 "U_values"=>U_values, "maxiters"=>10)
instructions = Dict("starting state"=>Dict("U index"=>1, "levels"=>1),
                "ending state"=>Dict("U index"=>30, "levels"=>1), "optimization_scheme"=>[2,1], "use symmetry"=>true)
println("U in [",U_values[instructions["starting state"]["U index"]], ", ", U_values[instructions["ending state"]["U index"]],"]")

# x = zero(degen_rm_U[1:2,:])
# x[1,1] = 1
# x[2,2] = 1
data_dict_tmp = test_map_to_state(all_full_eig_vecs[6], instructions, indexer, !isa(meta_data["electron count"], Number);
     maxiters=100#meta_data["maxiters"]
     , optimization=:adjoint_gradient)
data_dict_tmp
# save_with_metadata(data_dict_tmp, "data/tmp.jld2")