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
    if length(ARGS) >= 1 && startswith(ARGS[1], "data/")
        folder = ARGS[1]
        ARGS = ARGS[2:end]
    else
        folder = "data/N=(5, 5)_4x4"
    end
    file_path = joinpath(folder, "meta_data_and_E.jld2")

    dic = load_saved_dict(file_path)

    meta_data = dic["meta_data"]
    U_values = meta_data["U_values"]
    all_full_eig_vecs = dic["all_full_eig_vecs"]
    all_E = dic["E"] # Needed for energy selection
    indexer = dic["indexer"]
    precomputed_structures = get(dic, "precomputed_structures", Dict())

    println("Meta data:")
    display(meta_data)

    # Extract N for saving
    N = meta_data["electron count"]
    spin_conserved = !isa(meta_data["electron count"], Number) # True if tuple (N_up, N_down)
    use_symmetry = false

    # --- New Logic: Find lowest energy sector ---
    min_E = Inf
    k_min = 1
    for (k, E_vec) in enumerate(all_E)
        # Assuming E_vec is sorted or we check the ground state (first element)
        if !isempty(E_vec)
            E_ground = E_vec[1]
            if E_ground < min_E
                min_E = E_ground
                k_min = k
            end
        end
    end
    println("Selected lowest energy symmetry sector: $k_min with Energy $(min_E)")

    # Select the eigenvectors for this sector
    # all_full_eig_vecs is a list of sectors. each sector is a list of vectors (per U).
    target_vecs = all_full_eig_vecs[k_min]
    if indexer isa Vector
        indexer = indexer[k_min]
    end

    scan_instructions = Dict(
        "starting level" => 1,
        "ending level" => 1, # level index for targets
        "optimization_scheme" => [2],
        "use symmetry" => use_symmetry,
        "multi_start_iters" => 50, # 30
        "multi_start_samples" => 20, #5
        "initialization_samples" => 100,#20
    )
    println("ARGS: $(length(ARGS))")
    if length(ARGS) == 1
        v1 = tryparse(Int, ARGS[1])
        if isnothing(v1)
            if ARGS[1] == "forward"
                println("Forward")
                scan_instructions["u_range"] = 26:length(U_values)
            else ARGS[1] == "backward"
                println("backward")
                scan_instructions["u_range"] = 60:-1:1
            end
            scan_instructions["load_file"] = joinpath(folder, "unitary_map_energy_symmetry=$(use_symmetry)_N=$(N)_u_61.jld2")
            println("Load: $(scan_instructions["load_file"])")
        else
            println("doing: $v1")
            scan_instructions["u_range"] = v1:v1
            # scan_instructions["load_file"] = joinpath(folder, "unitary_map_energy_symmetry=$(use_symmetry)_N=$(N)_u_$(v1).jld2")
        end 
    elseif length(ARGS) == 2
        v1 = tryparse(Int, ARGS[1])
        v2 = tryparse(Int, ARGS[2])
        scan_instructions["u_range"] = v1:-1:v2
    else
        scan_instructions["u_range"] = 25:25
    end



    interaction_scan_map_to_state(target_vecs, scan_instructions, indexer,
        spin_conserved;
        maxiters=20, gradient=:adjoint_gradient,
        perturb_optimization=0.01,
        optimizer=[:GradientDescent, :LBFGS, :GradientDescent, :LBFGS, :GradientDescent, :LBFGS],
        save_folder=folder, save_name="unitary_map_energy_symmetry=$(use_symmetry)_N=$N",
        precomputed_structures=precomputed_structures)

    return 0
end
