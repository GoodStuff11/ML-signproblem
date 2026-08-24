# test_trotter_save_prefix_and_data_files.jl
#
# Verification test script to validate:
# 1. build_save_name_prefix output with and without grow_from_exponentials
# 2. File resolution logic in data_h5_fixed
# 3. Integrity and readability of renamed files in data_h5_fixed

using Test
using JLD2
using SparseArrays
using LinearAlgebra
using Combinatorics
using Lattices

include("../data_path.jl")
include("../utility_functions.jl")
using .UtilityFunctions
include("../ed_objects.jl")
include("../ed_functions.jl")

@testset "Trotter Save Prefix Logic" begin
    # 1. Standard num_exponentials=1 run (no grow_from_exponentials)
    prefix_std = build_save_name_prefix(
        :trotter;
        sites=9,
        electrons=(4, 4),
        custom_ref_state_arg="slater",
        antihermitian=true,
        loss_type=:overlap,
        num_exponentials=1,
        suffix=nothing
    )
    @test prefix_std == "trotter_N=9_ref_slater_antihermitian"
    @test !occursin("u_build", prefix_std)

    # 2. Grown run (grow_from_exponentials=1 to num_exponentials=2) -> should have u_build
    prefix_grown = build_save_name_prefix(
        :trotter;
        sites=9,
        electrons=(4, 4),
        custom_ref_state_arg="slater",
        antihermitian=true,
        loss_type=:overlap,
        num_exponentials=2,
        suffix="u_build"
    )
    @test prefix_grown == "trotter_N=9_num_exponentials=2_ref_slater_antihermitian_u_build"
    @test occursin("u_build", prefix_grown)

    # 3. Standard pruned run (num_exponentials=1, no grow) -> target_fidelity without u_build
    pruned_suffix_std = "target_fidelity=0.95"
    prefix_pruned_std = build_save_name_prefix(
        :trotter;
        sites=9,
        electrons=(4, 4),
        custom_ref_state_arg="slater",
        antihermitian=true,
        loss_type=:overlap,
        num_exponentials=1,
        suffix=pruned_suffix_std
    )
    @test prefix_pruned_std == "trotter_N=9_ref_slater_antihermitian_target_fidelity=0.95"
    @test !occursin("u_build", prefix_pruned_std)

    # 4. Grown pruned run -> target_fidelity with u_build
    pruned_suffix_grown = "u_build_target_fidelity=0.95"
    prefix_pruned_grown = build_save_name_prefix(
        :trotter;
        sites=9,
        electrons=(4, 4),
        custom_ref_state_arg="slater",
        antihermitian=true,
        loss_type=:overlap,
        num_exponentials=2,
        suffix=pruned_suffix_grown
    )
    @test prefix_pruned_grown == "trotter_N=9_num_exponentials=2_ref_slater_antihermitian_u_build_target_fidelity=0.95"
    @test occursin("u_build", prefix_pruned_grown)
end

@testset "data_h5_fixed File Verification" begin
    data_root = "/home/jek354/research/data/new_data/data_h5_fixed"
    
    # Verify subdirectories exist
    folders = filter(d -> isdir(joinpath(data_root, d)), readdir(data_root))
    @test length(folders) >= 12

    total_jld2 = 0
    u_build_count = 0
    sample_files_loaded = 0

    for folder_name in folders
        folder_path = joinpath(data_root, folder_name)
        jld2_files = filter(f -> endswith(f, ".jld2"), readdir(folder_path))
        total_jld2 += length(jld2_files)
        
        for f in jld2_files
            if occursin("u_build", f)
                u_build_count += 1
                # Must only be in N=(4, 4)_3x3 for num_exponentials=2
                @test folder_name == "N=(4, 4)_3x3"
                @test occursin("num_exponentials=2", f)
            end
        end

        # Test loading a sample trotter optimization file from each folder
        trotter_files = filter(f -> startswith(f, "trotter_") && occursin("_u_", f), jld2_files)
        if !isempty(trotter_files)
            sample_f = joinpath(folder_path, first(trotter_files))
            d = load_saved_dict(sample_f)
            @test haskey(d, "coefficients")
            @test haskey(d, "metrics")
            sample_files_loaded += 1
        end
    end

    println("Total JLD2 files across all folders: $total_jld2")
    println("Total u_build files remaining (kept growth duplicates in N=(4, 4)_3x3): $u_build_count")
    println("Sample trotter files successfully loaded: $sample_files_loaded / $(length(folders))")
    
    @test u_build_count == 47
    @test sample_files_loaded == length(folders)
end
