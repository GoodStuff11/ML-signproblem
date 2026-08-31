using Lattices, LinearAlgebra, SparseArrays, JLD2, HDF5, Zygote, Optimization, OptimizationOptimJL, Combinatorics
include("../utility_functions.jl"); using .UtilityFunctions
include("../trotter.jl"); using .Trotter
include("../data_path.jl"); include("../logging.jl"); include("../nn_strategy.jl")
include("../ed_objects.jl"); include("../ed_functions.jl"); include("../ed_optimization.jl")
include("trotter_exp_testing_lib.jl")

FOLDER = data_folder("N=(4, 4)_3x3")
TROTTER_ORDERS = [1, 2, 4, 8]
custom_ref_state_arg = "slater"
antihermitian = true
loss_type = :overlap

U_values, gs_energies, exact_exp_energies, exact_exp_overlaps, trotter_energies, trotter_overlaps, trotter_opt_energies, trotter_opt_overlaps, trotter_to_exact_overlaps, n_up_loaded, n_dn_loaded, lvec =
    compute_trotterized_energies_u_sweep(FOLDER, TROTTER_ORDERS, custom_ref_state_arg, antihermitian, loss_type)

println("\n\n=== recomputed infidelity (1-overlap) vs stored loss (cell5), exact-exp vs trotter_opt ===")
println(rpad("U",6), rpad("stored_exact",14), rpad("recomp_exact_infid",20), rpad("stored_trotter",16), rpad("recomp_trotter_infid",20))
for u_i in 1:length(U_values)
    U = U_values[u_i]
    exact_file = joinpath(FOLDER, "unitary_map_energy_symmetry=false_N=(4, 4)_ref_slater_antihermitian_u_$(u_i).jld2")
    trotter_file = joinpath(FOLDER, "trotter_N=9_ref_slater_antihermitian_u_$(u_i).jld2")
    stored_exact = isfile(exact_file) ? load(exact_file)["dict"]["metrics"]["loss"][end] : NaN
    stored_trotter = isfile(trotter_file) ? load(trotter_file)["dict"]["metrics"]["loss"][end] : NaN
    recomp_exact = 1.0 - exact_exp_overlaps[u_i]
    recomp_trotter = 1.0 - trotter_opt_overlaps[u_i]
    if u_i % 4 == 0 || u_i <= 2
        println(rpad(string(round(U,digits=2)),6), rpad(string(round(stored_exact,sigdigits=4)),14), rpad(string(round(recomp_exact,sigdigits=4)),20), rpad(string(round(stored_trotter,sigdigits=4)),16), rpad(string(round(recomp_trotter,sigdigits=4)),20))
    end
end

# find crossover U in stored losses vs recomputed infidelities
println("\n=== crossover search ===")
function find_crossover(a, b, Uv)
    for i in 2:length(a)
        if !isnan(a[i]) && !isnan(b[i]) && !isnan(a[i-1]) && !isnan(b[i-1])
            if (a[i-1]-b[i-1]) * (a[i]-b[i]) < 0
                println("crossover between U=$(Uv[i-1]) and U=$(Uv[i])")
            end
        end
    end
end
stored_exact_all = fill(NaN, length(U_values))
stored_trotter_all = fill(NaN, length(U_values))
for u_i in 1:length(U_values)
    exact_file = joinpath(FOLDER, "unitary_map_energy_symmetry=false_N=(4, 4)_ref_slater_antihermitian_u_$(u_i).jld2")
    trotter_file = joinpath(FOLDER, "trotter_N=9_ref_slater_antihermitian_u_$(u_i).jld2")
    if isfile(exact_file); stored_exact_all[u_i] = load(exact_file)["dict"]["metrics"]["loss"][end]; end
    if isfile(trotter_file); stored_trotter_all[u_i] = load(trotter_file)["dict"]["metrics"]["loss"][end]; end
end
println("Stored (cell5) crossover:")
find_crossover(stored_exact_all, stored_trotter_all, U_values)
println("Recomputed infidelity crossover:")
find_crossover(1.0 .- exact_exp_overlaps, 1.0 .- trotter_opt_overlaps, U_values)
