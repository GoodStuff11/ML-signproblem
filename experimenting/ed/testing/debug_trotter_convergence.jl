using Lattices
using LinearAlgebra
using SparseArrays
using JLD2
using HDF5
using Zygote
using Optimization
using OptimizationOptimJL
using Combinatorics

include("../utility_functions.jl")
using .UtilityFunctions
include("../trotter.jl")
using .Trotter
include("../data_path.jl")
include("../logging.jl")
include("../nn_strategy.jl")
include("../ed_objects.jl")
include("../ed_functions.jl")
include("../ed_optimization.jl")
include("trotter_exp_testing_lib.jl")

FOLDER = data_folder("N=(4, 4)_3x3")
TROTTER_ORDERS = [1, 2, 4, 8, 16, 32]
custom_ref_state_arg = "slater"
antihermitian = true
loss_type = :overlap

U_values, gs_energies, exact_exp_energies, exact_exp_overlaps, trotter_energies, trotter_overlaps, trotter_opt_energies, trotter_opt_overlaps, trotter_to_exact_overlaps, n_up_loaded, n_dn_loaded, lvec =
    compute_trotterized_energies_u_sweep(FOLDER, TROTTER_ORDERS, custom_ref_state_arg, antihermitian, loss_type)

println("\n\n================ DIAGNOSTICS ================")
for u_i in [10, 30, 50, 58]
    println("\n--- u_i=$u_i, U=$(U_values[u_i]) ---")
    println("  gs_energy         = $(gs_energies[u_i])")
    println("  exact_exp_energy  = $(exact_exp_energies[u_i])  (diff=$(exact_exp_energies[u_i]-gs_energies[u_i]))")
    for P in TROTTER_ORDERS
        e = trotter_energies[P][u_i]
        println("  P=$P energy = $e  (diff=$(e-gs_energies[u_i]))")
    end
    println("  trotter_opt_energy = $(trotter_opt_energies[u_i])  (diff=$(trotter_opt_energies[u_i]-gs_energies[u_i]))")
end

# Cross check against raw stored loss values the notebook's last cell reads directly
println("\n\n================ RAW STORED LOSSES (notebook cell 5 style) ================")
for u_i in [10, 30, 50, 58]
    exact_file = joinpath(FOLDER, "unitary_map_energy_symmetry=false_N=(4, 4)_ref_slater_antihermitian_u_$(u_i+1).jld2")
    trotter_file = joinpath(FOLDER, "trotter_N=9_ref_slater_antihermitian_u_$(u_i+1).jld2")
    if isfile(exact_file)
        d = load(exact_file)["dict"]
        println("u_i=$u_i exact stored loss = $(d["metrics"]["loss"][end])")
    else
        println("u_i=$u_i exact file missing: $exact_file")
    end
    if isfile(trotter_file)
        d = load(trotter_file)["dict"]
        println("u_i=$u_i trotter stored loss = $(d["metrics"]["loss"][end])")
    else
        println("u_i=$u_i trotter file missing: $trotter_file")
    end
end
