
     
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
    folder = "data/N=6_2x3"

    file_path = joinpath(folder, "meta_data_and_E.jld2")

    dic = load_saved_dict(file_path)

    meta_data = dic["meta_data"]
    U_values = meta_data["U_values"]
    all_full_eig_vecs = dic["all_full_eig_vecs"]
    all_E = dic["E"] # Needed for energy selection
    indexer = dic["indexer"]

    # Extract N for saving
    N = meta_data["electron count"]
    spin_conserved = !isa(meta_data["electron count"], Number) # True if tuple (N_up, N_down)
    use_symmetry = ARGS[1] == "true"

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

    # Snippet logic for instructions
    # instructions = Dict("starting state"=>Dict("U index"=>1, "levels"=>1),
    #                "ending state"=>Dict("U index"=>30, "levels"=>1), "max_order"=>2, "use symmetry"=>false)

    # User had 30 explicitly. Assuming they want to optimize to U index 30 (or max avaliable).
    # Let's target the last U value available if 30 is out of bounds, or just use 30 if valid.
    selected_u_values = 1:length(U_values)
    if length(ARGS) >= 2 # for parallelization
        selected_u_values = parse(Int, ARGS[2]):parse(Int, ARGS[3])
    end


    for u_index in selected_u_values
        instructions = Dict(
            "starting state" => Dict("U index" => 1, "levels" => 1), # Level 1 = Ground state
            "ending state" => Dict("U index" => u_index, "levels" => 1),
            "max_order" => 2,
            "use symmetry" => use_symmetry
        )

        println("Running Optimization from U=$(U_values[instructions["starting state"]["U index"]]) to U=$(U_values[instructions["ending state"]["U index"]])")

        data_dict = test_map_to_state(
            target_vecs,
            instructions,
            indexer,
            spin_conserved;
            maxiters=meta_data["maxiters"],
            optimization=:adjoint_gradient
        )
        dic["optimization_results"] = data_dict
        dic["symmetry_sector"] = k_min

        save_dictionary(folder, "unitary_map_energy_N=$N", data_dict)
    end

    return 0
end
