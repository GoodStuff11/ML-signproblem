"""
test_su2_slater_eigenstate.jl

Test whether the Slater determinant reference state used in `load_ED_data` has SU(2) symmetry
when `su2_symmetry=true` is specified, and verify that it is an exact eigenvector of the S² operator.

Usage:
  julia --project=.. testing/test_su2_slater_eigenstate.jl [options]

Options:
  --data_folder=<path> (optional): Path to a specific ED data folder or subfolder name under data root.
                                  Default: "" (tests a standard suite of datasets).
  --su2_symmetry=<true|false> (optional): Whether to specify su2_symmetry in load_ED_data. Default: true.
  --sign_convention=<spin_first|coordinate_first> (optional): Sign convention to use. Default: "spin_first".
  --verbose=<true|false> (optional): Print detailed output. Default: true.

Examples:
  julia --project=.. testing/test_su2_slater_eigenstate.jl
  julia --project=.. testing/test_su2_slater_eigenstate.jl --data_folder="N=(3, 3)_3x2" --sign_convention=coordinate_first
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
include(joinpath(@__DIR__, "..", "data_path.jl"))

"""
    parse_cli_args(args::Vector{String}) -> Dict{Symbol, Any}

Parse command line arguments for the test script.
"""
function parse_cli_args(args::Vector{String})
    parsed = Dict{Symbol, Any}(
        :data_folder => "",
        :su2_symmetry => true,
        :sign_convention => :spin_first,
        :verbose => true
    )

    for arg in args
        if startswith(arg, "--data_folder=")
            parsed[:data_folder] = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--su2_symmetry=")
            parsed[:su2_symmetry] = parse(Bool, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--sign_convention=")
            parsed[:sign_convention] = Symbol(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--verbose=")
            parsed[:verbose] = parse(Bool, split(arg, "=", limit=2)[2])
        else
            @warn "Unknown option: $arg"
        end
    end

    return parsed
end

"""
    verify_slater_reference_su2(folder::String; su2_symmetry::Bool=true, sign_convention::Symbol=:spin_first, verbose::Bool=true)

Verify that the Slater reference state loaded from `folder` has SU(2) symmetry
and is an exact eigenvector of the S² operator.
"""
function verify_slater_reference_su2(
    folder::String;
    su2_symmetry::Bool=true,
    sign_convention::Symbol=:spin_first,
    verbose::Bool=true
)
    if verbose
        println("\n--------------------------------------------------------------------------------")
        println("Testing dataset: $folder")
        println("  su2_symmetry: $su2_symmetry")
        println("  sign_convention: $sign_convention")
    end

    # Load ED data with use_slater_reference=true
    U_values, target_vecs, indexer, _, N, spin_conserved, _, sign_conv =
        load_ED_data(folder; verbose=false, su2_symmetry=su2_symmetry, use_slater_reference=true, sign_convention=sign_convention)

    # Extract Slater reference state vector (row 1 of target_vecs)
    psi_ref = Vector{ComplexF64}(target_vecs[1, :])
    norm_psi = norm(psi_ref)
    if verbose
        println("  Loaded electron count N = $N")
        println("  Hilbert space dimension H_dim = $(length(psi_ref))")
        println("  Reference state norm = $norm_psi")
    end

    # Construct the S² operator in the subspace basis
    dim = length(indexer.inv_comb_dict)
    rows = Int[]; cols = Int[]; vals = Float64[]
    create_S2!(rows, cols, vals, 1.0, indexer; momentum_basis=true, sign_convention=sign_convention)
    S2_op = sparse(rows, cols, vals, dim, dim)

    # Compute S² |psi_ref>
    S2_psi = S2_op * psi_ref

    # Expectation value <psi_ref| S² |psi_ref>
    s2_exp = real(dot(psi_ref, S2_psi))

    # Eigenstate residual: r = S²|psi_ref> - <S²>|psi_ref>
    residual = S2_psi .- s2_exp .* psi_ref
    residual_norm = norm(residual)

    if verbose
        println("  <S²> expectation value: $s2_exp")
        println("  Residual norm ||S² ψ - <S²> ψ||: $residual_norm")
    end

    # Verification criteria:
    # 1. Eigenstate residual norm must be essentially zero (< 1e-10)
    is_eigenstate = residual_norm < 1e-10

    # 2. When su2_symmetry=true, the eigenvalue should be 0.0 (singlet state, S=0)
    is_singlet = abs(s2_exp) < 1e-10

    if verbose
        if is_eigenstate
            println("  ✓ CONFIRMED: Reference state IS an exact eigenvector of S².")
        else
            println("  ✗ FAILED: Reference state is NOT an eigenvector of S²! Residual norm = $residual_norm")
        end

        if su2_symmetry
            if is_singlet
                println("  ✓ CONFIRMED: Reference state has SU(2) symmetry (S² = 0.0 singlet).")
            else
                println("  ✗ FAILED: Reference state does NOT have S² = 0.0 singlet symmetry! Got S² = $s2_exp")
            end
        end
    end

    return (is_eigenstate=is_eigenstate, is_singlet=is_singlet, s2_exp=s2_exp, residual_norm=residual_norm)
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "test_su2_slater_eigenstate")
    with_logging(log_path) do
        println("================================================================================")
        println("           SU(2) Slater Reference State Eigenstate Verification Test            ")
        println("================================================================================")

        config = parse_cli_args(ARGS)
        data_folder_arg = config[:data_folder]
        su2_sym = config[:su2_symmetry]
        sign_conv = config[:sign_convention]
        verbose = config[:verbose]

        # Determine target folders to test
        folders_to_test = String[]
        if !isempty(data_folder_arg)
            push!(folders_to_test, data_folder(data_folder_arg))
        else
            # Default suite of datasets to test
            default_subfolders = [
                "N=(2, 2)_2x2",
                "N=(2, 2)_3x2",
                "N=(3, 3)_3x2",
                "N=(3, 3)_3x3",
                "N=(4, 4)_3x3"
            ]
            for sub in default_subfolders
                path = data_folder(sub)
                if isdir(path)
                    push!(folders_to_test, path)
                else
                    println("Note: Subfolder $sub not found at $path, skipping.")
                end
            end
        end

        if isempty(folders_to_test)
            error("No valid dataset folders found to test.")
        end

        all_passed = true

        for folder in folders_to_test
            for sc in [sign_conv, sign_conv == :spin_first ? :coordinate_first : :spin_first]
                println("\n================================================================================")
                println("Testing SU(2) symmetry = true (sign_convention = $sc)")
                try
                    res_su2 = verify_slater_reference_su2(folder; su2_symmetry=true, sign_convention=sc, verbose=verbose)

                    @test res_su2.is_eigenstate
                    @test res_su2.is_singlet

                    if !res_su2.is_eigenstate || !res_su2.is_singlet
                        all_passed = false
                    end
                catch e
                    println("Caught expected error for dataset without doubly occupied Slater state: $e")
                    @test occursin("No momentum sector found", string(e)) || occursin("cannot have SU(2)", string(e))
                end

                println("\nComparing with SU(2) symmetry = false (sign_convention = $sc)")
                try
                    res_no_su2 = verify_slater_reference_su2(folder; su2_symmetry=false, sign_convention=sc, verbose=verbose)
                    println("  Without su2_symmetry: <S²> = $(res_no_su2.s2_exp), residual_norm = $(res_no_su2.residual_norm), is_eigenstate = $(res_no_su2.is_eigenstate)")
                catch e
                    println("  Without su2_symmetry error: $e")
                end
            end
        end

        println("\n================================================================================")
        if all_passed
            println("SUCCESS: All Slater reference state SU(2) eigenvector tests PASSED!")
        else
            println("FAILURE: One or more tests failed!")
        end
        println("================================================================================")
    end
end
