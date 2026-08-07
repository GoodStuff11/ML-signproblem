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
    log_path = make_log_path(@__DIR__, "debug_trotter_coord_first")
    with_logging(log_path) do
        println("=== Debugging Trotter vs create_Hubbard in coordinate_first ===")

        folder_new_sign = "/home/jek354/research/data/new_data/data_h5_fixed/N=(3, 2)_3x3"
        sign_conv = :coordinate_first

        U_vals, vecs, indexer, _, N_elec, _, _, actual_sign = load_ED_data(
            folder_new_sign; sign_convention=sign_conv, verbose=false, use_slater_reference=false
        )

        Lvec = parse_lattice_dimension(folder_new_sign)
        u_val = U_vals[1]
        gs = vecs[1, :]
        lattice = Square(Tuple(Lvec), Periodic())
        order = RowSnake()

        println("actual_sign = $actual_sign, order = $order")

        # Trotter
        H_trotter, basis_trotter, _ = Trotter.HubbardMomentumBasis(
            1.0, u_val, Tuple(Lvec), N_elec;
            indexer=indexer, sign_convention=actual_sign, lattice_ordering=order
        )
        V_trotter = eigvecs(Matrix(H_trotter))
        E_trotter = eigvals(Matrix(H_trotter))

        # create_Hubbard
        k = indexer.k
        subspace = HubbardSubspace(N_elec..., lattice; k=k)
        indexer_rebuilt = CombinationIndexer(subspace; order=order)

        new_hopping = create_Hubbard(HubbardModel(1.0, 0.0, 0.0, false), subspace; 
                            indexer=indexer_rebuilt, momentum_basis=true, sign_convention=actual_sign, lattice_ordering=order)
        new_interaction = create_Hubbard(HubbardModel(0.0, 1.0, 0.0, false), subspace; 
                            indexer=indexer_rebuilt, momentum_basis=true, sign_convention=actual_sign, lattice_ordering=order)
        H_create = Matrix(new_hopping .+ u_val .* new_interaction)
        v_create = eigvecs(H_create)
        E_create = eigvals(H_create)

        println("E_trotter[1:3] = $(E_trotter[1:3])")
        println("E_create[1:3]  = $(E_create[1:3])")

        ov_create = abs(v_create[:, 1]' * gs)
        ov_trotter = abs(V_trotter[:, 1]' * gs)
        ov_between = abs(V_trotter[:, 1]' * v_create[:, 1])

        println("Overlap create vs gs: $ov_create")
        println("Overlap trotter vs gs: $ov_trotter")
        println("Overlap trotter vs create: $ov_between")

        # Check H_create vs H_trotter matrix elements
        diff_H = norm(Matrix(H_trotter) .- H_create)
        println("norm(H_trotter - H_create) = $diff_H")
    end
end
