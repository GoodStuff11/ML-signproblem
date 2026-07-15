"""
    verify_h5_permutation.jl

Quick verification that load_h5_ED_data correctly permutes target_vecs to match
the CombinationIndexer ordering. Checks that:
1. The Hamiltonian built from the indexer basis produces the correct eigenvalues
   when applied to the permuted eigenvectors.
2. The overlap |⟨ψ_ED|ψ_indexer⟩|² = 1.0 for all U values.

Usage:
    julia --project=.. testing/verify_h5_permutation.jl <folder>
"""

using Lattices
using LinearAlgebra
using SparseArrays
using JLD2
using HDF5

include(joinpath(@__DIR__, "..", "trotter.jl"))
using .Trotter

if !isdefined(Main, :UtilityFunctions)
    include(joinpath(@__DIR__, "..", "utility_functions.jl"))
end
using .UtilityFunctions
include(joinpath(@__DIR__, "..", "ed_objects.jl"))
include(joinpath(@__DIR__, "..", "ed_functions.jl"))
include(joinpath(@__DIR__, "..", "logging.jl"))

function (@main)(ARGS)
    folder = ARGS[1]
    println("=== Verifying H5 permutation for: $folder ===")

    # Load data with the permutation fix
    U_values, target_vecs, indexer, _, N_elec, spin_conserved, _, sign_convention =
        load_h5_ED_data(folder; verbose=true, sign_convention=:spin_first)

    n_up, n_dn = N_elec

    # Parse Lvec from folder name
    Lvec = parse_lattice_dimension(folder)
    N_sites = prod(Lvec)
    println("Lvec = $Lvec, N_sites = $N_sites, N_elec = $N_elec")

    # Helper to convert Coordinate to 0-based site index
    function coord_to_site_idx(coord, Lvec)
        c0 = coord.coordinates .- 1
        return Trotter.ravel_c(c0, Tuple(Lvec))
    end

    function coord_set_to_binary(coord_set, Lvec)
        val = zero(UInt)
        for coord in coord_set
            site_idx = coord_to_site_idx(coord, Lvec)
            val |= (one(UInt) << site_idx)
        end
        return val
    end

    # Build basis_sector from indexer
    basis_sector = Vector{UInt}(undef, length(indexer.inv_comb_dict))
    for (idx, conf) in enumerate(indexer.inv_comb_dict)
        u_bin = coord_set_to_binary(conf[1], Lvec)
        d_bin = coord_set_to_binary(conf[2], Lvec)
        basis_sector[idx] = Trotter.combineSpinInts(u_bin, d_bin, N_sites)
    end
    println("Basis sector size: $(length(basis_sector))")

    # Build Hamiltonian in momentum basis, sliced to sector
    basis_up, _ = Trotter.getReducedHilSpace(N_sites, n_up; returnOcc=true)
    basis_dn, _ = Trotter.getReducedHilSpace(N_sites, n_dn; returnOcc=true)

    edges = Trotter.findLatticeEdges(Lvec; use_pbc=true)
    hop_up = Trotter.fermionNNHopping(basis_up, edges, 1.0)
    hop_dn = Trotter.fermionNNHopping(basis_dn, edges, 1.0)
    I_up = sparse(I, length(basis_up), length(basis_up))
    I_dn = sparse(I, length(basis_dn), length(basis_dn))
    H_hop_real = kron(I_up, hop_dn) + kron(hop_up, I_dn)
    H_int_real = Trotter.fermionOnSiteSpinDensity(basis_up, basis_dn; u=1.0)

    F_up, _ = Trotter.SlaterCOB_RtoK_nparticle(Lvec, n_up)
    F_dn, _ = Trotter.SlaterCOB_RtoK_nparticle(Lvec, n_dn)
    F_total = kron(F_up, F_dn)

    kron_ints = [Trotter.combineSpinInts(u, d, N_sites) for u in basis_up for d in basis_dn]
    state_to_kron_idx = Dict(val => idx for (idx, val) in enumerate(kron_ints))
    sector_indices_in_kron = [state_to_kron_idx[val] for val in basis_sector]

    H_hop_sector = (F_total*H_hop_real*F_total')[sector_indices_in_kron, sector_indices_in_kron]
    H_int_sector = (F_total*H_int_real*F_total')[sector_indices_in_kron, sector_indices_in_kron]

    println("\n=== Eigenvalue comparison ===")
    println("target_vecs size: $(size(target_vecs))")
    # target_vecs: row 1 = reference state, rows 2.. = U-value eigenvectors
    # row index = U-value index, column index = basis state index

    # Load the known H5 eigenvalues for comparison
    valid_files = [f for f in readdir(folder) if occursin("HubbardED", f)]
    h5_file = joinpath(folder, valid_files[1])
    h5open(h5_file, "r") do data
        key_labels = [parse(Int, k) for k in keys(data["data/energies"])]
        all_E = [real.(read(data, "data/energies/$(k)"))[:, 1] for k in key_labels]
        k_min = find_best_energy_sector(all_E, U_values; labels=key_labels)
        E_ref = real.(read(data, "data/energies/$(k_min)"))[1, :]

        # Check a subset of U values
        n_check = min(5, length(U_values))
        check_indices = round.(Int, range(1, length(U_values), length=n_check))

        for u_idx in check_indices
            U = U_values[u_idx]
            H_full = H_hop_sector .+ U .* H_int_sector

            # Get the eigenvector from target_vecs (row u_idx+1, since row 1 is reference)
            psi = Vector{ComplexF64}(target_vecs[u_idx+1, :])

            # Compute energy expectation value
            E_trotter = real(dot(psi, H_full * psi))
            E_ed = E_ref[u_idx]

            # Also compute overlap with the ground eigenstate of H_full
            evals, evecs = eigen(Hermitian(Matrix(H_full)))
            overlap = abs(dot(evecs[:, 1], psi))^2

            println("U=$(round(U, digits=2)): E_trotter=$(round(E_trotter, digits=8)), " *
                    "E_ED=$(round(E_ed, digits=8)), " *
                    "ΔE=$(round(abs(E_trotter - E_ed), digits=10)), " *
                    "overlap=$(round(overlap, digits=10))")
        end
    end
end
