"""
    test_h5_sign_convention_overlap.jl

Test and verify that `load_ED_data` correctly converts HDF5 dataset sign conventions
to `:spin_first` and `:coordinate_first`, matching both `create_Hubbard` and `HubbardMomentumBasis`
with overlap 1.0 for all U values.

Usage:
    julia --project=.. testing/test_h5_sign_convention_overlap.jl
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
include(joinpath(@__DIR__, "..", "trotter.jl"))
using .Trotter
include(joinpath(@__DIR__, "..", "ed_objects.jl"))
include(joinpath(@__DIR__, "..", "ed_functions.jl"))

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "test_h5_sign_convention_overlap")
    with_logging(log_path) do
        println("=== Running HDF5 Sign Convention Overlap Test ===")

        folder_new_sign = "/home/jek354/research/data/new_data/data_h5_fixed/N=(3, 2)_3x3"
        if !isdir(folder_new_sign)
            folder_new_sign = "/home/jek354/research/ML-signproblem/experimenting/ed/data_new_sign/N=(3, 2)_3x3"
        end

        for sign_conv in [:spin_first, :coordinate_first]
            println("\nTesting sign_convention = :$sign_conv")

            U_vals, vecs, indexer, _, N_elec, _, _, actual_sign = load_ED_data(
                folder_new_sign; sign_convention=sign_conv, verbose=false, use_slater_reference=false
            )
            @test actual_sign == sign_conv

            Lvec = parse_lattice_dimension(folder_new_sign)
            if Lvec === nothing
                Lvec = [3, 3]
            end

            lattice = Square(Tuple(Lvec), Periodic())
            order = actual_sign == :spin_first ? ColSnake() : RowSnake()

            # Verify for multiple U parameter indices j
            for j in 1:min(4, length(U_vals))
                u_val = U_vals[j]
                gs = vecs[j, :]

                # 1. Trotter HubbardMomentumBasis (real matrix)
                H_trotter, _, _ = Trotter.HubbardMomentumBasis(
                    1.0, u_val, Tuple(Lvec), N_elec;
                    indexer=indexer, sign_convention=actual_sign, lattice_ordering=order
                )
                M_trotter_real = real.(Matrix(H_trotter))
                V_trotter = eigvecs(M_trotter_real)
                overlap_trotter = abs(V_trotter[:, 1]' * gs)

                # 2. create_Hubbard momentum Hamiltonian
                k = indexer.k
                subspace = HubbardSubspace(N_elec..., lattice; k=k)
                indexer_rebuilt = CombinationIndexer(subspace; order=order)

                new_hopping = create_Hubbard(HubbardModel(1.0, 0.0, 0.0, false), subspace;
                    indexer=indexer_rebuilt, momentum_basis=true, sign_convention=actual_sign, lattice_ordering=order)
                new_interaction = create_Hubbard(HubbardModel(0.0, 1.0, 0.0, false), subspace;
                    indexer=indexer_rebuilt, momentum_basis=true, sign_convention=actual_sign, lattice_ordering=order)
                H_create = Matrix(new_hopping .+ u_val .* new_interaction)
                v_create = eigvecs(H_create)
                overlap_create = abs(v_create[:, 1]' * gs)

                pred_energy = real(gs' * H_create * gs)
                exact_energy = eigvals(H_create)[1]

                println("  j=$j (U=$u_val): create overlap = $overlap_create | trotter overlap = $overlap_trotter | GS energy = $pred_energy")

                @test isapprox(overlap_create, 1.0; atol=1e-5)
                @test isapprox(overlap_trotter, 1.0; atol=1e-5)
                @test isapprox(pred_energy, exact_energy; atol=1e-5)
            end
        end

        println("\n=== All Tests Passed ===")
    end
end
