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
try
    if !CUDA.functional()
        @info "CUDA not functional yet — trying local_toolkit mode"
        CUDA.set_runtime_version!(local_toolkit=true)
    end
    if CUDA.functional()
        @info "GPU available: $(CUDA.name(CUDA.CuDevice(0)))"
    end
catch e
    @warn "CUDA setup warning: $e"
end

using Dates


include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("utility_functions.jl")
include("nn_strategy.jl")
include("logging.jl")

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "run_lanczos_scan_optimization")
    with_logging(log_path) do

    if length(ARGS) >= 1 && startswith(ARGS[1], "data")
        folder = ARGS[1]
        ARGS = ARGS[2:end]
    else
        error("Please input a folder. Ex data/N=(2, 2)_2x2")
    end

    # Parse --nn argument if provided
    nn_strategy_file = nothing
    filtered_args = String[]
    for arg in ARGS
        if startswith(arg, "--nn=")
            nn_strategy_file = split(arg, "=")[2]
            println("Using neural network found in $(nn_strategy_file)")
        else
            push!(filtered_args, arg)
        end
    end
    ARGS = filtered_args

    # Parse electrons and dimension from the folder name
    electrons_parsed = (2, 2)
    dim_parsed = [2, 2]
    try
        m_elec = match(r"N=\((?<N>\d+),\s*(?<M>\d+)\)", folder)
        if !isnothing(m_elec)
            electrons_parsed = (parse(Int, m_elec[:N]), parse(Int, m_elec[:M]))
        end
        m_dim = match(r"_(?<W>\d+)x(?<H>\d+)", folder)
        if !isnothing(m_dim)
            dim_parsed = [parse(Int, m_dim[:W]), parse(Int, m_dim[:H])]
        end
    catch e
        @warn "Could not parse electrons or dimensions from folder path, using defaults: (2, 2) and [2, 2]"
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
        max_time_ratio=5.0,
        nn_strategy_file=nn_strategy_file,
        nn_electrons=electrons_parsed,
        nn_dim=dim_parsed,
        nn_U_values=U_values)

    return 0
    end # with_logging
end
