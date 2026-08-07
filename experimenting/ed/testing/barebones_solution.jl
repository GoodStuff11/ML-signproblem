using HDF5, LinearAlgebra, SparseArrays
include("../trotter.jl")
using .Trotter
using .Trotter.TamFermion

function main()
    folder_new_sign = "/home/jek354/research/data/new_data/data_new_sign/N=(3, 2)_3x2"
    valid_files = [f for f in readdir(folder_new_sign) if occursin("HubbardED", f)]
    file_path = joinpath(folder_new_sign, valid_files[1])

    local evecs_dataset, energies_dataset, sl_up, sl_dn
    h5open(file_path, "r") do data
        sl_up = read(data, "metadata/slater_labels/2/up")
        sl_dn = read(data, "metadata/slater_labels/2/dn")
        evecs_dataset = read(data, "data/evecs/2")
        energies_dataset = read(data, "data/energies/2")
    end

    # Ground state at U=0.25 is index 1 of state_dim, index 1 of U_dim
    raw_gs = evecs_dataset[:, 1, 1]
    println("HDF5 Ground state energy at U=0.25: ", energies_dataset[1, 1])

    Lvec = [3, 2]
    N = (3, 2)
    N_sites = prod(Lvec)

    # Build TamFermion Hamiltonian for q_target=2 at U=0.25
    H_hop, basis_dict, _ = Trotter.TamFermion.HubbardMomentumBasis(1.0, 0.0, Lvec, N; q_target=2)
    H_int, _, _ = Trotter.TamFermion.HubbardMomentumBasis(0.0, 1.0, Lvec, N; q_target=2)
    H = Matrix(H_hop + 0.25 * H_int)
    println("TamFermion exact GS energy:          ", eigvals(H)[1])

    # Build HDF5 basis integer representation
    H_dim = size(sl_up, 2)
    h5_ints = Vector{UInt16}(undef, H_dim)
    for i in 1:H_dim
        u_bin = UInt16(sum(1 << o for o in sl_up[:, i]))
        d_bin = UInt16(sum(1 << o for o in sl_dn[:, i]))
        h5_ints[i] = u_bin | (d_bin << N_sites)
    end

    # Permutation from HDF5 basis order -> TamFermion basis order
    tam_ints = basis_dict["ints"]
    tam_dict = Dict(val => idx for (idx, val) in enumerate(tam_ints))
    perm = [tam_dict[val] for val in h5_ints]

    # Permute raw_gs
    permuted_gs = zeros(ComplexF64, H_dim)
    permuted_gs[perm] = raw_gs

    println("\nExpecting permuted_gs without sign corrections: ", real(permuted_gs' * H * permuted_gs))

    # Now compute Jordan-Wigner sign factors for each HDF5 Slater determinant
    # In HDF5, Slater determinant i is created as:
    # (c^\dagger_{up, o_1} ... c^\dagger_{up, o_nup}) * (c^\dagger_{dn, o'_1} ... c^\dagger_{dn, o'_ndown}) |0>
    # In TamFermion, modes are ordered as spin-first (or sorted by mode index).
    # Let's check parity of sorting target modes!
    aligned_gs = copy(permuted_gs)

    # Let's test permutation parity sign corrections
    signs = ones(Float64, H_dim)
    for i in 1:H_dim
        up_orbs = sl_up[:, i]
        dn_orbs = sl_dn[:, i]
        
        # Initial mode list in HDF5 state: all up orbitals then all down orbitals
        # Modes are 1-based: up orbitals 1..N_sites, down orbitals N_sites+1 .. 2*N_sites
        modes = vcat([o + 1 for o in up_orbs], [o + 1 + N_sites for o in dn_orbs])
        
        # Sort modes and compute permutation parity
        # Parity of sorting `modes`
        p = 0
        m = copy(modes)
        for a in 1:length(m)-1
            for b in a+1:length(m)
                if m[a] > m[b]
                    p += 1
                end
            end
        end
        sgn = iseven(p) ? 1.0 : -1.0
        signs[i] = sgn
    end

    # Apply signs to permuted_gs
    signed_gs = zeros(ComplexF64, H_dim)
    signed_gs[perm] = raw_gs .* signs

    println("Expecting signed_gs with JW parity signs:      ", real(signed_gs' * H * signed_gs))

    evals, evecs = eigen(H)
    exact_gs = evecs[:, 1]
    println("Fidelity |<signed_gs | exact_gs>|^2:           ", abs2(dot(signed_gs, exact_gs)))
end

main()
