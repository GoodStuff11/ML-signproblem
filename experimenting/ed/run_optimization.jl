
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
    folder = "data/N=6"

    dict = load_saved_dict(joinpath(folder,"meta_data_and_E.jld2"))
    meta_data = dict["meta_data"]
    indexer = dict["indexer"]
    U_values = meta_data["U_values"]
    degen_rm_U = dict["degen_rm_U"]
    maxiters = meta_data["maxiters"]
    N = meta_data["electron count"]

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

    # optimization
    for level1 in level1_options
        for level2 in level2_options
            for u_index in selected_u_values
                instructions = Dict("starting state"=>Dict("U index"=>1, "levels"=>level1),
                                "ending state"=>Dict("U index"=>u_index, "levels"=>level2), "max_order"=>2)
                data_dict_tmp = test_map_to_state(degen_rm_U, instructions, indexer; maxiters=maxiters, optimization=:gradient)
                save_dictionary(folder,"unitary_map_energy_N=$N", data_dict_tmp)
            end
        end
    end

    return 0 
end