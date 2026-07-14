using Lattices
using HDF5
using SparseArrays
using LinearAlgebra
using Combinatorics
using JLD2

# Include the source files
include("../ed_objects.jl")
include("../utility_functions.jl")
include("../ed_functions.jl")

function run_test()
    folder = "/home/jek354/research/ML-signproblem/experimenting/ed/data/N=(2, 2)_3x2"
    println("Loading ED data from folder: ", folder)
    U_values, target_vecs, _, precomputed_structures, N, spin_conserved, use_symmetry, sign_convention = load_ED_data(folder; verbose=true, omit_indexer=true)

    # Let's find k_min (the best sector)
    file_path = joinpath(folder, readdir(folder)[findfirst(f -> occursin("HubbardED", f), readdir(folder))])
    h5open(file_path, "r") do data
        Lvec = read(data, "metadata/Lvec")
        U_values = read(data, "data/uvec")
        key_labels = [parse(Int, k) for k in keys(data["data/energies"])]
        all_E = [real.(read(data, "data/energies/$(k)"))[:, 1] for k in key_labels]
        k_min = find_best_energy_sector(all_E, U_values; labels=key_labels)

        println("Best energy sector: ", k_min)

        # Now run both old and new su2 ground state solvers
        target_S = 0.0 # for N=(2,2), singlet state has S = 0

        println("\n--- Running old get_su2_ground_state_old ---")
        t_old = @elapsed indices_old, coeffs_old = get_su2_ground_state_old(indexer, target_S; tol=1e-8, sign_convention=sign_convention)
        println("Old solver finished in ", round(t_old, digits=4), "s. Found ", length(indices_old), " states.")

        println("\n--- Running new get_su2_ground_state ---")
        t_new = @elapsed indices_new, coeffs_new = get_su2_ground_state(data, k_min, target_S; tol=1e-8, sign_convention=sign_convention)
        println("New solver finished in ", round(t_new, digits=4), "s. Found ", length(indices_new), " states.")

        # Sort and compare
        p_old = sortperm(indices_old)
        p_new = sortperm(indices_new)

        indices_old_sorted = indices_old[p_old]
        coeffs_old_sorted = coeffs_old[p_old]

        indices_new_sorted = indices_new[p_new]
        coeffs_new_sorted = coeffs_new[p_new]

        @assert indices_old_sorted == indices_new_sorted "Indices do not match!"
        println("✓ Indices match perfectly!")

        # They might differ by an overall phase, so let's check overlap / relative phase
        overlap = abs(dot(coeffs_old_sorted, coeffs_new_sorted))
        println("Overlap: ", overlap)
        @assert abs(overlap - 1.0) < 1e-10 "Coefficients do not match (overlap is $overlap instead of 1.0)!"
        println("✓ Coefficients match perfectly!")

        println("\nAll tests passed successfully!")
    end
end

run_test()
