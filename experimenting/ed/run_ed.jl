
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
    t = 1.0
    U = 6
    μ = 0  # positive incentivises fewer particles (one electron costs this much energy)
    # N_up = 2
    # N_down = 2
    N = 3
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
    U_values = sort([U_values; 10.0 .^LinRange(-3,2,40)])

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
    indexer = CombinationIndexer(reduce(vcat,collect(sites(subspace.lattice))), get_subspace_info(subspace)...)

    meta_data = Dict(
        "electron count"=>N, 
        "sites"=>join(lattice_dimension, "x"), 
        "bc"=>bc, 
        "basis"=>"adiabatic", 
        "U_values"=>U_values,
        "maxiters"=>200, 
        "optimizer"=>"LBFGS"
        )
    
    dict = Dict(
        "meta_data"=>meta_data,
        "E"=>E,
        "V"=>V,
        "H"=>H,
        "degen_rm_U"=>degen_rm_U,
        "indexer"=>indexer)
    
    
    save_energy_with_metadata("data/test", dict)

    return 0 
end