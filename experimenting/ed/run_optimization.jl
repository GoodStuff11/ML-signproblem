
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
using ExponentialUtilities


include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("utility_functions.jl")


function (@main)(ARGS)
    t = 1.0
    U = 6
    μ = 0  # positive incentivises fewer particles (one electron costs this much energy)
    # N_up = 2
    # N_down = 2
    N = 5
    half_filling = false
    lattice_dimension = (2,3)
    bc = "periodic"
    # lattice = Chain(6, Periodic())
    lattice = Square(lattice_dimension, if bc == "periodic" Periodic() else Open() end)
    # lattice = Graphs.cycle_graph(3)

    models = []

    reference_index = 2
    # for _t in t_values
    #     # println(_t)
    #     push!(models,HubbardModel(_t,0.0001,μ,half_filling))
    # end
    # U_values = [0.00001,0.01,0.2, 1,1.001,3,4,5,7,10, 100]
    U_values = [0.00001; LinRange(2.1,9,20)]
    for U in U_values
        # println(t)
        push!(models,HubbardModel(t,U,μ,half_filling))
    end

    subspace = HubbardSubspace(N, lattice)
    # subspace = HubbardSubspace(N_up, N_down, lattice)

    ops = []
    push!(ops,Matrix(create_operator(subspace,:Sx)))
    push!(ops, Matrix(create_operator(subspace,:S2)))
    # op3 = Matrix(create_operator(subspace,:L2))
    push!(ops, Matrix(create_operator(subspace,:T, kind=1)))
    push!(ops, Matrix(create_operator(subspace,:T, kind=2)))
    # push!(ops, Matrix(create_operator(subspace,:σ, kind=1)))
    E = []
    H = []
    V = []
    for model ∈ models
        push!(H, Matrix(create_Hubbard(model, subspace; perturbations=false)))
        e, v = eigen(H[end])
        push!(E, e)
        push!(V, v)
    end

    degen_rm_U = create_consistent_basis(H, ops;reference_index=reference_index)

    # dim = get_subspace_dimension(subspace)
    indexer = CombinationIndexer(reduce(vcat,collect(sites(subspace.lattice))), get_subspace_info(subspace)...)
    # difference_dict = collect_all_conf_differences(indexer)


    # optimization
    meta_data = Dict("electron count"=>N, "sites"=>join(lattice_dimension, "x"), "bc"=>bc, "basis"=>"adiabatic", 
                    "U_values"=>U_values, "maxiters"=>2)
    instructions = Dict("starting state"=>Dict("U index"=>1, "levels"=>16),
                    "ending state"=>Dict("U index"=>10, "levels"=>54), "max_order"=>2)
    data_dict_tmp = test_map_to_state(degen_rm_U, instructions, indexer; maxiters=meta_data["maxiters"], optimization=:gradient)
    data_dict_tmp["meta_data"] = meta_data
    println("Hey")
    append_to_json_files(data_dict_tmp, "data/unitary_map_N=5")
    println("done")
end