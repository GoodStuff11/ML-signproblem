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
    log_path = make_log_path(@__DIR__, "debug_basis_sector_mismatch")
    with_logging(log_path) do
        folder_new_sign = "/home/jek354/research/data/new_data/data_h5_fixed/N=(3, 2)_3x3"
        
        for sign_conv in [:spin_first, :coordinate_first]
            println("\n=== Testing sign_conv = :$sign_conv ===")
            U_vals, vecs, indexer, _, N_elec, _, _, actual_sign = load_ED_data(
                folder_new_sign; sign_convention=sign_conv, verbose=false, use_slater_reference=false
            )
            Lvec = parse_lattice_dimension(folder_new_sign)
            N_sites = prod(Lvec)

            # Get basis_sector from indexer
            basis_sector = Trotter.TamFermion.get_basis_sector(indexer, Lvec, N_sites)

            order = actual_sign == :spin_first ? ColSnake() : RowSnake()
            full_basis = Trotter.TamFermion.fullSlaterMomBasis(Lvec, N_elec[1], N_elec[2]; 
                            sort_by_momentum=true, sign_convention=actual_sign, lattice_ordering=order)

            # Find matching indices of basis_sector in full_basis["ints"]
            full_ints = full_basis["ints"]
            state_to_idx = Dict(val => idx for (idx, val) in enumerate(full_ints))

            matched = [get(state_to_idx, val, 0) for val in basis_sector]
            println("Number of basis_sector elements: $(length(basis_sector))")
            println("Number of matches found in fullSlaterMomBasis: $(count(>(0), matched))")
            if any(==(0), matched)
                println("Unmatched basis_sector elements exist!")
            end
        end
    end
end
