using Lattices
using LinearAlgebra
using SparseArrays
using JLD2
using HDF5

include(joinpath(@__DIR__, "..", "logging.jl"))
include(joinpath(@__DIR__, "..", "utility_functions.jl"))
using .UtilityFunctions
include(joinpath(@__DIR__, "..", "trotter.jl"))
using .Trotter
include(joinpath(@__DIR__, "..", "ed_objects.jl"))
include(joinpath(@__DIR__, "..", "ed_functions.jl"))

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "debug_indexer_difference")
    with_logging(log_path) do
        println("=== Debugging indexer returned by load_ED_data vs indexer_rebuilt ===")

        folder_new_sign = "/home/jek354/research/data/new_data/data_h5_fixed/N=(3, 2)_3x3"
        sign_conv = :coordinate_first

        U_vals, vecs, indexer_from_load, _, N_elec, _, _, actual_sign = load_ED_data(
            folder_new_sign; sign_convention=sign_conv, verbose=false, use_slater_reference=false
        )

        Lvec = parse_lattice_dimension(folder_new_sign)
        lattice = Square(Tuple(Lvec), Periodic())
        order = RowSnake()

        k = indexer_from_load.k
        subspace = HubbardSubspace(N_elec..., lattice; k=k)
        indexer_rebuilt = CombinationIndexer(subspace; order=order)

        gs = vecs[1, :]
        u_val = U_vals[1]

        # Case 1: using indexer_from_load
        H_trotter_1, _, _ = Trotter.HubbardMomentumBasis(
            1.0, u_val, Tuple(Lvec), N_elec;
            indexer=indexer_from_load, sign_convention=actual_sign, lattice_ordering=order
        )
        V1 = eigvecs(Matrix(H_trotter_1))
        ov1 = abs(V1[:, 1]' * gs)

        # Case 2: using indexer_rebuilt
        H_trotter_2, _, _ = Trotter.HubbardMomentumBasis(
            1.0, u_val, Tuple(Lvec), N_elec;
            indexer=indexer_rebuilt, sign_convention=actual_sign, lattice_ordering=order
        )
        V2 = eigvecs(Matrix(H_trotter_2))
        ov2 = abs(V2[:, 1]' * gs)

        println("Overlap using indexer_from_load: $ov1")
        println("Overlap using indexer_rebuilt:   $ov2")

        println("indexer_from_load inv_comb_dict length: $(length(indexer_from_load.inv_comb_dict))")
        println("indexer_rebuilt inv_comb_dict length:   $(length(indexer_rebuilt.inv_comb_dict))")

        if indexer_from_load.inv_comb_dict != indexer_rebuilt.inv_comb_dict
            println("inv_comb_dict differs between indexer_from_load and indexer_rebuilt!")
            diff_count = count(i -> indexer_from_load.inv_comb_dict[i] != indexer_rebuilt.inv_comb_dict[i], 1:length(indexer_from_load.inv_comb_dict))
            println("Number of differing state combinations: $diff_count")
        end
    end
end
