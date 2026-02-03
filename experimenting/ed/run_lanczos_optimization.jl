
     
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


include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("utility_functions.jl")


function (@main)(ARGS)
    folder = "/home/jek354/research/data/N=6_3x2"
    dic = load_saved_dict(joinpath(folder,"meta_data_and_E.jld2"))

    selected_u_values = 1:length(U_values)

    level1_options = [8]
    level2_options = [58]
    if length(ARGS) >= 2
        level1_options = [parse(Int, ARGS[1])]
        level2_options = [parse(Int, ARGS[2])]
    end
    if length(ARGS) >= 4 # for parallelization
        selected_u_values = parse(Int, ARGS[3]):parse(Int, ARGS[4])
    end

     meta_data = dic["meta_data"]
     U_values = meta_data["U_values"]
     all_full_eig_vecs = dic["all_full_eig_vecs"]
     n_eigs = [parse(Int,x) for x in split(meta_data["sites"], "x")]
     indexer = dic["indexer"]


     instructions = Dict("starting state"=>Dict("U index"=>1, "levels"=>1),
                    "ending state"=>Dict("U index"=>30, "levels"=>1), "max_order"=>2, "use symmetry"=>false)
     println("U in [",U_values[instructions["starting state"]["U index"]], ", ", U_values[instructions["ending state"]["U index"]],"]")

     # x = zero(degen_rm_U[1:2,:])
     # x[1,1] = 1
     # x[2,2] = 1
     data_dict = test_map_to_state(all_full_eig_vecs[6], instructions, indexer, !isa(meta_data["electron count"], Number);
          meta_data["maxiters"], optimization=:adjoint_gradient)

    save_dictionary(folder,"unitary_map_energy_N=$N", data_dict)
    return 0 
end