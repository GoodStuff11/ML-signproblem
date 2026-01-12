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
using KrylovKit
# using ExponentialUtilities


include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("utility_functions.jl")
include("adiabatic_analysis.jl")


function @main(ARGS)
    krylov_times = []
    hamiltonian_times = []
    system_size = []
    eigen_value_count = []
    for lattice_dimension in [(2,3), (4,2),(5,2),(4,3)]
        println("lattice_dimension=$lattice_dimension")
        for N in 2:prod(lattice_dimension)
            println("N=$N")
            t = 1.0
            U = 6.0
            μ = 0  # positive incentivises fewer particles (one electron costs this much energy)
            N_up = 4
            N_down = 4
            # N =  3
            half_filling = false
            # lattice = Chain(6, Periodic())
            bc = "periodic"
            # lattice = Chain(6, Periodic())
            lattice = Square(lattice_dimension, if bc == "periodic" Periodic() else Open() end)
            # lattice = Graphs.cycle_graph(3)

            model = HubbardModel(t,U,μ,half_filling)
            subspace = HubbardSubspace(N, lattice)

            tmp = create_Hubbard(model, subspace; perturbations=false)
            hamiltonian_time = @elapsed H = create_Hubbard(model, subspace; perturbations=false)
            for n_eigs = 1:5
                push!(hamiltonian_times, hamiltonian_time)
                eigsolve(H, normalize(rand(size(H)[1])), n_eigs,:SR)
                push!(krylov_times, @elapsed eigsolve(H, normalize(rand(size(H)[1])), n_eigs,:SR))
                push!(system_size, size(H)[1])
                push!(eigen_value_count, n_eigs)
            end
        end
    end
    @save "data/Krylov_ED_benchmark.jld2" krylov_times system_size eigen_value_count

end