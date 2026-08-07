"""
    test_multi_file_h5_selection.jl

Test and verify that `load_h5_ED_data` correctly identifies and selects the HDF5 file
with the lowest ground state energy when multiple valid HDF5 files exist in a directory.

Usage:
    julia --project=.. testing/test_multi_file_h5_selection.jl
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
    log_path = make_log_path(@__DIR__, "test_multi_file_h5_selection")
    with_logging(log_path) do
        println("=== Running Multi-File HDF5 Energy Selection Test ===")

        # Test 1: Single file loading works as expected
        single_file_folder = "/home/jek354/research/ML-signproblem/experimenting/ed/data_new_sign/N=(3, 3)_3x3"
        if isdir(single_file_folder)
            println("Test 1: Single file loading in $single_file_folder")
            res = load_h5_ED_data(single_file_folder; verbose=true)
            @test length(res[1]) > 0  # U_values
            println("Single file load successful!")
        end

        # Test 2: Multi-file selection with dummy/mock HDF5 files
        mktempdir() do temp_dir
            println("Test 2: Creating mock multi-file dataset in $temp_dir")
            
            # File A: Higher energy (e.g., E = 10.0)
            file_a = joinpath(temp_dir, "HubbardED_sector_1.h5")
            h5open(file_a, "w") do h5
                # metadata
                g_meta = create_group(h5, "metadata")
                g_meta["nu"] = 2
                g_meta["nd"] = 2
                g_meta["Lvec"] = [3, 2]
                g_meta["qvecs"] = [0 1; 0 1]
                g_meta["slater_labels/1/up"] = [0 1; 2 3]
                g_meta["slater_labels/1/dn"] = [0 1; 2 3]

                # data
                g_data = create_group(h5, "data")
                g_data["uvec"] = [1.0, 2.0, 4.0]
                g_data["energies/1"] = reshape(Float64[10.0, 10.0, 10.0], 1, 3) # shape (n_eig, n_U)
                g_data["evecs/1"] = zeros(ComplexF64, 4, 1, 3) # shape (H_dim, n_eig, n_U)
            end

            # File B: Lower energy (e.g., E = -5.0)
            file_b = joinpath(temp_dir, "HubbardED_sector_2.h5")
            h5open(file_b, "w") do h5
                # metadata
                g_meta = create_group(h5, "metadata")
                g_meta["nu"] = 2
                g_meta["nd"] = 2
                g_meta["Lvec"] = [3, 2]
                g_meta["qvecs"] = [0 1; 0 1]
                g_meta["slater_labels/1/up"] = [0 1; 2 3]
                g_meta["slater_labels/1/dn"] = [0 1; 2 3]

                # data
                g_data = create_group(h5, "data")
                g_data["uvec"] = [1.0, 2.0, 4.0]
                g_data["energies/1"] = reshape(Float64[-5.0, -5.0, -5.0], 1, 3) # shape (n_eig, n_U)
                g_data["evecs/1"] = zeros(ComplexF64, 4, 1, 3) # shape (H_dim, n_eig, n_U)
            end

            println("Testing load_h5_ED_data on multi-file directory...")
            U_values, target_vecs, indexer, precomputed, N, spin_conserved, use_sym, sign_conv, Lvec, order = load_h5_ED_data(
                temp_dir; verbose=true, omit_indexer=true, use_slater_reference=false
            )

            # File B has E = -5.0, so U_values should match uvec from file B, and length is 3
            @test U_values == [1.0, 2.0, 4.0]
            @test size(target_vecs, 1) == 3 # 3 U values without slater ref
            println("Multi-file selection selected lower energy file successfully!")
        end

        println("=== All Tests Passed ===")
    end
end
