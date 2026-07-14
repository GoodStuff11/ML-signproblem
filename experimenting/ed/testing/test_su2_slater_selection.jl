"""
    test_su2_slater_selection.jl

Verify that `load_ED_data` and helper functions correctly implement the `su2_symmetry`
and `use_slater_reference` features.

Usage:
    julia --project=.. testing/test_su2_slater_selection.jl
"""

using Lattices
using LinearAlgebra
using SparseArrays
using JLD2
using HDF5
using Test

include(joinpath(@__DIR__, "..", "logging.jl"))
include(joinpath(@__DIR__, "..", "utility_functions.jl"))
using .UtilityFunctions
include(joinpath(@__DIR__, "..", "ed_objects.jl"))
include(joinpath(@__DIR__, "..", "ed_functions.jl"))

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "test_su2_slater_selection")
    with_logging(log_path) do
        println("=== Running SU(2) Slater Selection Tests ===")

        # Test 1: Load HDF5 with su2_symmetry=true
        h5_folder = "/home/jek354/research/ML-signproblem/experimenting/ed/data_new_sign/N=(3, 3)_3x3"
        println("\n--- Test 1: Loading H5 with su2_symmetry=true ---")
        U_values_h5, target_vecs_h5, indexer_h5, _, N_h5, spin_conserved_h5, _, sign_convention_h5 =
            load_ED_data(h5_folder; verbose=true, su2_symmetry=true)
        println("H5 Loaded successfully. target_vecs size: ", size(target_vecs_h5))
        @test size(target_vecs_h5, 1) == length(U_values_h5) + 1  # Since use_slater_reference=true by default for H5

        # Verify that the reference state is doubly occupied
        h5open(joinpath(h5_folder, "HubbardED_Slater_3x3_(3,3)_t_1_m_2.h5"), "r") do data
            # Best energy sector (which find_best_energy_sector selected)
            key_labels = [parse(Int, k) for k in keys(data["data/energies"])]
            all_E = [real.(read(data, "data/energies/$(k)"))[:, 1] for k in key_labels]
            U_vals = read(data, "data/uvec")
            k_min = find_best_energy_sector(all_E, U_vals; labels=key_labels, data=data, su2_symmetry=true)
            slater_idx = get_slater_ground_state(data, k_min)
            println("Selected k_min: ", k_min, ", Slater index: ", slater_idx)
            is_double = is_doubly_occupied(data, k_min, slater_idx)
            println("Is doubly occupied: ", is_double)
            @test is_double == true
        end

        # Test 2: Load JLD2 with su2_symmetry=true and use_slater_reference=true
        jld2_folder = "/home/jek354/research/ML-signproblem/experimenting/ed/data/N=(3, 3)_3x2"
        println("\n--- Test 2: Loading JLD2 with su2_symmetry=true and use_slater_reference=true ---")
        U_values_jld, target_vecs_jld, indexer_jld, _, N_jld, spin_conserved_jld, _, sign_convention_jld =
            load_ED_data(jld2_folder; verbose=true, su2_symmetry=true, use_slater_reference=true)
        println("JLD2 Loaded successfully. target_vecs size: ", size(target_vecs_jld))
        @test size(target_vecs_jld, 1) == length(U_values_jld) + 1

        # Verify that the reference state is doubly occupied in JLD2
        jld_file = joinpath(jld2_folder, "meta_data_and_E.jld2")
        dic = load_saved_dict(jld_file)
        k_min_jld = find_best_energy_sector(dic["E"], dic["meta_data"]["U_values"]; data=dic, su2_symmetry=true)
        slater_idx_jld = get_slater_ground_state(dic, k_min_jld)
        is_double_jld = is_doubly_occupied(dic, k_min_jld, slater_idx_jld)
        println("JLD2 Selected k_min: ", k_min_jld, ", Slater index: ", slater_idx_jld)
        println("JLD2 Is doubly occupied: ", is_double_jld)
        @test is_double_jld == true

        # Test 3: Load JLD2 with su2_symmetry=true and use_slater_reference=false (default)
        println("\n--- Test 3: Loading JLD2 with su2_symmetry=true and use_slater_reference=false ---")
        U_values_jld2, target_vecs_jld2, indexer_jld2, _, N_jld2, spin_conserved_jld2, _, sign_convention_jld2 =
            load_ED_data(jld2_folder; verbose=true, su2_symmetry=true, use_slater_reference=false)
        println("JLD2 Loaded successfully (no reference). target_vecs size: ", size(target_vecs_jld2))
        @test size(target_vecs_jld2, 1) == length(U_values_jld2)

        # Test 4: Expect error when loading odd electron counts with su2_symmetry=true
        odd_folder = "/home/jek354/research/ML-signproblem/experimenting/ed/data_new_sign/N=(3, 2)_3x2"
        println("\n--- Test 4: Loading odd electron counts with su2_symmetry=true (Expecting Error) ---")
        try
            load_ED_data(odd_folder; verbose=true, su2_symmetry=true)
            println("FAILED: Did not raise an error for odd electron count.")
            @test false
        catch e
            println("PASSED: Correctly raised error: ", e)
            @test true
        end

        println("\n=== All SU(2) Slater Selection Tests Passed! ===")
    end
end
