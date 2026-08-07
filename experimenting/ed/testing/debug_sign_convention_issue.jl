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
    log_path = make_log_path(@__DIR__, "debug_sign_convention_issue")
    with_logging(log_path) do
        println("=== Testing Trotter.HubbardMomentumBasis with sign_convention & lattice_ordering ===")

        folder_new_sign = "/home/jek354/research/data/new_data/data_h5_fixed/N=(3, 2)_3x3"

        for sign_conv in [:spin_first, :coordinate_first]
            println("\n--------------------------------------------------")
            println("Testing sign_convention = :$sign_conv")
            println("--------------------------------------------------")

            U2, vecs_new, indexer_new, _, N_elec2, _, _, actual_sign = load_ED_data(
                folder_new_sign; sign_convention=sign_conv, verbose=true, use_slater_reference=false
            )

            Lvec2 = parse_lattice_dimension(folder_new_sign)
            j = 1
            u_val2 = U2[j]
            println("U = $u_val2")
            gs_new = vecs_new[1, :]
            println("Returned sign_convention: $actual_sign")

            lattice = Square(Tuple(Lvec2), Periodic())
            order = actual_sign == :spin_first ? ColSnake() : RowSnake()

            # Pass sign_convention and lattice_ordering to Trotter.HubbardMomentumBasis
            H_trotter, basis_trotter, counts = Trotter.HubbardMomentumBasis(
                1.0, u_val2, Tuple(Lvec2), N_elec2;
                indexer=indexer_new, sign_convention=actual_sign, lattice_ordering=order
            )
            energy_trotter, V_trotter = eigen(Matrix(H_trotter))

            k = indexer_new.k
            subspace = HubbardSubspace(3, 2, lattice; k=k)
            indexer_rebuilt = CombinationIndexer(subspace; order=order)

            new_hopping = create_Hubbard(HubbardModel(1.0, 0.0, 0.0, false), subspace; 
                                indexer=indexer_rebuilt, momentum_basis=true, sign_convention=actual_sign, lattice_ordering=order)
            new_interaction = create_Hubbard(HubbardModel(0.0, 1.0, 0.0, false), subspace; 
                                indexer=indexer_rebuilt, momentum_basis=true, sign_convention=actual_sign, lattice_ordering=order)
            H_create = Matrix(new_hopping .+ u_val2 .* new_interaction)
            v_create = eigvecs(H_create)

            println("Overlap v_create[:, 1] ' * gs_new: $(v_create[:, 1]' * gs_new) (abs: $(abs(v_create[:, 1]' * gs_new)))")
            println("Overlap V_trotter[:, 1]' * gs_new: $(V_trotter[:, 1]' * gs_new) (abs: $(abs(V_trotter[:, 1]' * gs_new)))")
            println("Predicted energy gs_new' * H_create * gs_new: $(gs_new' * H_create * gs_new)")
        end
    end
end
