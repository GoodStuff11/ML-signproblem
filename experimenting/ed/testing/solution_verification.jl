using HDF5, LinearAlgebra, SparseArrays
using Lattices, Combinatorics, KrylovKit

include("../data_path.jl")
include("../utility_functions.jl")
include("../ed_objects.jl")
include("../ed_functions.jl")
include("../trotter.jl")
using .Trotter
using .Trotter.TamFermion

function main()
    folder_new_sign = "/home/jek354/research/data/new_data/data_new_sign/N=(3, 2)_3x2"

    println("=========================================================================")
    println("SOLUTION 1: Standard load_ED_data approach (Recommended)")
    println("=========================================================================")
    U_vals, vecs, indexer, _, N, _, _, sign_convention = load_ED_data(
        folder_new_sign;
        sign_convention=:spin_first,
        use_slater_reference=false,
        verbose=false
    )

    Lvec = [3, 2]
    H_hop, _, _ = Trotter.TamFermion.HubbardMomentumBasis(1.0, 0.0, Lvec, N; indexer=indexer)
    H_int, _, _ = Trotter.TamFermion.HubbardMomentumBasis(0.0, 1.0, Lvec, N; indexer=indexer)

    u_val = U_vals[1] # 0.25
    H = Matrix(H_hop + u_val * H_int)
    state = vecs[1, :] # Ground state at U_vals[1]

    E_state = real(state' * H * state)
    E_eig = eigvals(H)[1]

    println("state' * H * state : ", E_state)
    println("eigvals(H)[1]      : ", E_eig)
    println("Difference          : ", abs(E_state - E_eig))
    println("Fidelity |<s|gs>|^2 : ", abs2(dot(state, eigvecs(H)[:, 1])))

    println("\n=========================================================================")
    println("SOLUTION 2: Barebones direct HDF5 alignment")
    println("=========================================================================")
    valid_files = [f for f in readdir(folder_new_sign) if occursin("HubbardED", f)]
    file_path = joinpath(folder_new_sign, valid_files[1])

    local evecs_dataset, sl_up, sl_dn, kvecs
    sector_key = "2"
    h5open(file_path, "r") do data
        sl_up = read(data, "metadata/slater_labels/$sector_key/up")
        sl_dn = read(data, "metadata/slater_labels/$sector_key/dn")
        evecs_dataset = read(data, "data/evecs/$sector_key")
        kvecs = read(data, "metadata/kvecs")
    end

    raw_gs = evecs_dataset[:, 1, 1]
    H_dim = size(sl_up, 2)
    lattice = Square(tuple(Lvec...), Periodic())
    subspace = HubbardSubspace(N..., lattice; k=nothing)
    idxer = CombinationIndexer(subspace; order=ColSnake())

    map_coord = Dict{Int,Coordinate}(o => Coordinate(kvecs[1, o+1] + 1, kvecs[2, o+1] + 1) for o in 0:5)
    sorted_sites = sort(idxer.a, order=ColSnake())

    perm = Vector{Int}(undef, H_dim)
    signs = Vector{Float64}(undef, H_dim)
    for h5_idx in 1:H_dim
        up_orbs = sl_up[:, h5_idx]
        dn_orbs = sl_dn[:, h5_idx]

        up_set = Set([map_coord[o] for o in up_orbs])
        dn_set = Set([map_coord[o] for o in dn_orbs])
        perm[h5_idx] = index(idxer, up_set, dn_set)

        ops = vcat(
            [(map_coord[o], 1, :create) for o in up_orbs],
            [(map_coord[o], 2, :create) for o in dn_orbs]
        )
        signs[h5_idx] = compute_jw_sign((up_set, dn_set), sorted_sites, ops; sign_convention=:spin_first)
    end

    aligned_state = zeros(ComplexF64, length(idxer.inv_comb_dict))
    for h5_idx in 1:H_dim
        aligned_state[perm[h5_idx]] = raw_gs[h5_idx] * signs[h5_idx]
    end

    H_hop2 = create_Hubbard(HubbardModel(1.0, 0.0, 0.0, false), subspace; indexer=idxer, momentum_basis=true, sign_convention=:spin_first)
    H_int2 = create_Hubbard(HubbardModel(0.0, 1.0, 0.0, false), subspace; indexer=idxer, momentum_basis=true, sign_convention=:spin_first)
    H2 = Matrix(H_hop2 + u_val * H_int2)

    E_state2 = real(aligned_state' * H2 * aligned_state)
    E_eig2 = eigvals(H2)[1]

    println("aligned_state' * H2 * aligned_state : ", E_state2)
    println("eigvals(H2)[1]                     : ", E_eig2)
    println("Difference                         : ", abs(E_state2 - E_eig2))
end

main()
