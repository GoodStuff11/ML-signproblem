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
using HDF5
using KrylovKit

using CUDA
CUDA.set_runtime_version!(v"12.8")


include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("utility_functions.jl")



function (@main)(ARGS)
    if length(ARGS) >= 1 && startswith(ARGS[1], "data")
        folder = ARGS[1]
        ARGS = ARGS[2:end]
    else
        folder = "data/N=(2, 2)_3x2"
    end
    U_values, target_vecs, indexer, precomputed_structures, N, spin_conserved, use_symmetry, sign_convention = load_ED_data(folder)

    scan_instructions = Dict(
        "starting level" => 1,
        "ending level" => 1, # level index for targets
        "optimization_scheme" => [2],
        "use symmetry" => use_symmetry,
        "multi_start_iters" => 50, # 30
        "multi_start_samples" => 2, #5
        "initialization_samples" => 10,#20
        "sign_convention" => sign_convention,
    )
    println("ARGS: $(length(ARGS))")
    if length(ARGS) == 1
        v1 = tryparse(Int, ARGS[1])
        if isnothing(v1)
            if ARGS[1] == "forward"
                println("Forward")
                scan_instructions["u_range"] = 26:length(U_values)
            else
                ARGS[1] == "backward"
                println("backward")
                scan_instructions["u_range"] = 18:-1:1
            end
            scan_instructions["load_file"] = joinpath(folder, "unitary_map_energy_symmetry=$(use_symmetry)_N=$(N)_u_19.jld2")
            println("Load: $(scan_instructions["load_file"])")
        else
            println("doing: $v1")
            scan_instructions["u_range"] = v1:v1
            # scan_instructions["load_file"] = joinpath(folder, "unitary_map_energy_symmetry=$(use_symmetry)_N=$(N)_u_$(v1).jld2")
        end
    elseif length(ARGS) == 2
        v1 = tryparse(Int, ARGS[1])
        v2 = tryparse(Int, ARGS[2])
        if v1 > v2
            scan_instructions["u_range"] = v1:-1:v2
            if isfile(joinpath(folder, "unitary_map_energy_symmetry=$(use_symmetry)_N=$(N)_u_$(v1+1).jld2"))
                scan_instructions["load_file"] = joinpath(folder, "unitary_map_energy_symmetry=$(use_symmetry)_N=$(N)_u_$(v1+1).jld2")
            end
        else
            scan_instructions["u_range"] = v1:v2
            if isfile(joinpath(folder, "unitary_map_energy_symmetry=$(use_symmetry)_N=$(N)_u_$(v1-1).jld2"))
                scan_instructions["load_file"] = joinpath(folder, "unitary_map_energy_symmetry=$(use_symmetry)_N=$(N)_u_$(v1-1).jld2")
            end
        end
    else
        scan_instructions["u_range"] = 25:25
    end



    interaction_scan_map_to_state(target_vecs, scan_instructions, indexer,
        spin_conserved;
        maxiters=200, gradient=:adjoint_gradient,
        perturb_optimization=0.01,
        optimizer=[:GradientDescent, :LBFGs],
        save_folder=folder, save_name="unitary_map_energy_symmetry=$(use_symmetry)_N=$N",
        precomputed_structures=precomputed_structures,
        max_time_ratio=5.0)

    return 0
end
