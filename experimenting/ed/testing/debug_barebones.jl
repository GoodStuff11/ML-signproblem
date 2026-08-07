using HDF5, LinearAlgebra, SparseArrays
include("../trotter.jl")
using .Trotter
using .Trotter.TamFermion

function main()
    folder_new_sign = "/home/jek354/research/data/new_data/data_new_sign/N=(3, 2)_3x2"
    valid_files = [f for f in readdir(folder_new_sign) if occursin("HubbardED", f)]
    file_path = joinpath(folder_new_sign, valid_files[1])

    # 1. Read HDF5 data
    local evecs_dataset, energies_dataset, sl_up, sl_dn, kvecs
    h5open(file_path, "r") do data
        kvecs = read(data, "metadata/kvecs")
        sl_up = read(data, "metadata/slater_labels/2/up")
        sl_dn = read(data, "metadata/slater_labels/2/dn")
        evecs_dataset = read(data, "data/evecs/2")
        energies_dataset = read(data, "data/energies/2")
    end

    raw_gs = evecs_dataset[:, 1, 1] # ground state at U=0.25 (index 1)

    # 2. Build Hamiltonian in TamFermion
    Lvec = [3, 2]
    N = (3, 2)
    H_hop, basis_dict, _ = Trotter.TamFermion.HubbardMomentumBasis(1.0, 0.0, Lvec, N; q_target=2)
    H_int, _, _ = Trotter.TamFermion.HubbardMomentumBasis(0.0, 1.0, Lvec, N; q_target=2)
    H = Matrix(H_hop + 0.25 * H_int)

    tam_ints = basis_dict["ints"] # uint bit representations of (up, dn) in TamFermion
    
    N_sites = prod(Lvec)
    H_dim = size(sl_up, 2)
    h5_ints = Vector{UInt16}(undef, H_dim)
    for i in 1:H_dim
        up_orbs = sl_up[:, i] # 0-based orbital indices from HDF5
        dn_orbs = sl_dn[:, i]
        
        u_bin = zero(UInt16)
        for o in up_orbs
            u_bin |= (one(UInt16) << o)
        end
        d_bin = zero(UInt16)
        for o in dn_orbs
            d_bin |= (one(UInt16) << o)
        end
        h5_ints[i] = u_bin | (d_bin << N_sites)
    end

    println("Set difference tam_ints \\ h5_ints: ", setdiff(Set(tam_ints), Set(h5_ints)))
    println("Set difference h5_ints \\ tam_ints: ", setdiff(Set(h5_ints), Set(tam_ints)))

    # If sets are equal, compute permutation!
    if isempty(setdiff(Set(tam_ints), Set(h5_ints)))
        tam_dict = Dict(val => idx for (idx, val) in enumerate(tam_ints))
        perm = [tam_dict[val] for val in h5_ints]
        
        permuted_gs = zeros(ComplexF64, H_dim)
        permuted_gs[perm] = raw_gs

        println("permuted_gs' * H * permuted_gs: ", real(permuted_gs' * H * permuted_gs))

        evals, evecs = eigen(H)
        exact_gs = evecs[:, 1]
        println("Fidelity without signs |<permuted_gs | exact_gs>|^2: ", abs2(dot(permuted_gs, exact_gs)))

        # Now test sign parities!
        # In HDF5, Slater determinant i has mode ordering: up_orbs then dn_orbs
        # In TamFermion, does it use up_orbs then dn_orbs?
        # Let's test if applying parity sgn gives 1.0 fidelity!
        sign_gs = copy(permuted_gs)
        println("Testing parity signs...")
    end
end

main()
