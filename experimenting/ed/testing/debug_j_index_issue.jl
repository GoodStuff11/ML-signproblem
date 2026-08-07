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
    log_path = make_log_path(@__DIR__, "debug_j_index_issue")
    with_logging(log_path) do
        println("=== Testing different U index j ===")

        folder_new_sign = "/home/jek354/research/data/new_data/data_h5_fixed/N=(3, 2)_3x3"
        if !isdir(folder_new_sign)
            folder_new_sign = "/home/jek354/research/ML-signproblem/experimenting/ed/data_new_sign/N=(3, 2)_3x3"
        end

        for sign_conv in [:spin_first, :coordinate_first]
            println("\n==================================================")
            println("Sign Convention: $sign_conv")
            println("==================================================")

            U2, vecs_new, indexer_new, _, N_elec2, _, _, actual_sign = load_ED_data(
                folder_new_sign; sign_convention=sign_conv, verbose=false, use_slater_reference=false
            )
            Lvec2 = parse_lattice_dimension(folder_new_sign)
            if Lvec2 === nothing; Lvec2 = [3, 3]; end
            lattice = Square(Tuple(Lvec2), Periodic())
            order = actual_sign == :spin_first ? ColSnake() : RowSnake()

            k = indexer_new.k
            subspace = HubbardSubspace(N_elec2..., lattice; k=k)
            indexer_rebuilt = CombinationIndexer(subspace; order=order)

            for j in 1:min(4, length(U2))
                u_val2 = U2[j]
                gs_new = vecs_new[j, :]

                # Trotter basis
                H_trotter, _, _ = Trotter.HubbardMomentumBasis(
                    1.0, u_val2, Tuple(Lvec2), N_elec2;
                    indexer=indexer_rebuilt, sign_convention=actual_sign, lattice_ordering=order
                )
                V_trotter = eigvecs(Matrix(H_trotter))
                E_trotter = eigvals(Matrix(H_trotter))

                # create_Hubbard basis
                new_hopping = create_Hubbard(HubbardModel(1.0, 0.0, 0.0, false), subspace; 
                                    indexer=indexer_rebuilt, momentum_basis=true, sign_convention=actual_sign, lattice_ordering=order)
                new_interaction = create_Hubbard(HubbardModel(0.0, 1.0, 0.0, false), subspace; 
                                    indexer=indexer_rebuilt, momentum_basis=true, sign_convention=actual_sign, lattice_ordering=order)
                H = Matrix(new_hopping .+ u_val2 .* new_interaction)
                v = eigvecs(H)

                # Correct column: column 1 (ground state of H at U_j)
                ov_create_col1 = abs(v[:, 1]' * gs_new)
                ov_trotter_col1 = abs(V_trotter[:, 1]' * gs_new)

                # Wrong column: column j (j-th eigenvector of H at U_j)
                ov_create_colj = abs(v[:, j]' * gs_new)
                ov_trotter_colj = abs(V_trotter[:, j]' * gs_new)

                pred_energy = real(gs_new' * H * gs_new)
                exact_gs_energy = E_trotter[1]

                println("j = $j, U = $u_val2")
                println("  Exact GS energy: $exact_gs_energy | gs_new' * H * gs_new: $pred_energy")
                println("  Overlap with col 1 (GS of H): create=$ov_create_col1, trotter=$ov_trotter_col1")
                println("  Overlap with col j (index j): create=$ov_create_colj, trotter=$ov_trotter_colj")

                @test isapprox(pred_energy, exact_gs_energy; atol=1e-5)
                @test isapprox(ov_create_col1, 1.0; atol=1e-5)
                @test isapprox(ov_trotter_col1, 1.0; atol=1e-5)
            end
        end
        println("\n=== All Tests Passed ===")
    end
end
