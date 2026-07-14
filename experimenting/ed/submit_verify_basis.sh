#!/usr/bin/env bash
#SBATCH --mem=20g
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00
#SBATCH --partition=kim
#SBATCH --job-name=verify_basis
#SBATCH --output=/home/jek354/research/ML-signproblem/jobs/verify_basis_%j.out
#SBATCH --error=/home/jek354/research/ML-signproblem/jobs/verify_basis_%j.err

cd /home/jek354/research/ML-signproblem/experimenting/ed
julia --project=.. /home/jek354/.gemini/antigravity-ide/brain/297f19dc-a298-4866-b7ba-8370ff5ee6dc/scratch/verify_basis.jl

# verify_basis.jl

# using Lattices
# using LinearAlgebra
# using Combinatorics
# using SparseArrays
# using JLD2
# using HDF5

# # Include trotter.jl
# include("/home/jek354/research/ML-signproblem/experimenting/ed/trotter.jl")
# using .Trotter

# include("/home/jek354/research/ML-signproblem/experimenting/ed/utility_functions.jl")
# include("/home/jek354/research/ML-signproblem/experimenting/ed/ed_objects.jl")
# include("/home/jek354/research/ML-signproblem/experimenting/ed/ed_functions.jl")

# function verify(folder)
#     println("--------------------------------------------------")
#     println("Verifying folder: $folder")
    
#     # 1. Load ED data
#     U_values, target_vecs, indexer, _, N_elec, spin_conserved, _, sign_convention = 
#         load_ED_data(folder; verbose=true)
        
#     n_up, n_dn = N_elec
    
#     Lvec = [3, 3]
#     try
#         m_dim = match(r"_(?<W>\d+)x(?<H>\d+)", folder)
#         if !isnothing(m_dim)
#             Lvec = [parse(Int, m_dim[:W]), parse(Int, m_dim[:H])]
#         end
#     catch e
#     end
#     N_sites = prod(Lvec)
#     println("System: Lvec = $Lvec, N_sites = $N_sites, Ne = $(N_elec)")

#     # 2. Reconstruct sector basis
#     function coord_to_site_idx(coord, Lvec)
#         c0 = coord.coordinates .- 1
#         return Trotter.TamLib.ravel_c(c0, Tuple(Lvec))
#     end

#     function coord_set_to_binary(coord_set, Lvec)
#         val = zero(UInt)
#         for coord in coord_set
#             site_idx = coord_to_site_idx(coord, Lvec)
#             val |= (one(UInt) << site_idx)
#         end
#         return val
#     end

#     # Check JLD2 vs H5
#     basis_sector = nothing
#     k_min = 1
#     if !isnothing(indexer)
#         println("Using indexer from JLD2...")
#         basis_sector = Vector{UInt}(undef, length(indexer.inv_comb_dict))
#         for (idx, conf) in enumerate(indexer.inv_comb_dict)
#             u_bin = coord_set_to_binary(conf[1], Lvec)
#             d_bin = coord_set_to_binary(conf[2], Lvec)
#             basis_sector[idx] = Trotter.combineSpinInts(u_bin, d_bin, N_sites)
#         end
#     else
#         println("Using HDF5 file...")
#         valid_files = [f for f in readdir(folder) if occursin("HubbardED", f)]
#         h5_file = joinpath(folder, valid_files[1])
#         basis_sector, k_min = h5open(h5_file, "r") do data
#             key_labels = [parse(Int, k) for k in keys(data["data/energies"])]
#             all_E = [real.(read(data, "data/energies/$(k)"))[:, 1] for k in key_labels]
#             k_min = find_best_energy_sector(all_E, U_values; labels=key_labels)
            
#             separate_spins_stored = (read(data, "metadata/slater_labels/$k_min") isa Dict)
#             if !separate_spins_stored
#                 slater_labels = read(data, "metadata/slater_labels/$k_min")
#                 H_dim = size(slater_labels, 2)
#             else
#                 slater_labels_up = read(data, "metadata/slater_labels/$k_min/up")
#                 slater_labels_down = read(data, "metadata/slater_labels/$k_min/dn")
#                 H_dim = size(slater_labels_up, 2)
#             end
            
#             # Helper to convert orbital indices to binary representation
#             function orbital_indices_to_binary(indices, N)
#                 val = zero(UInt)
#                 for idx in indices
#                     val |= (one(UInt) << idx)
#                 end
#                 return val
#             end
            
#             basis_sector_h5 = Vector{UInt}(undef, H_dim)
#             for idx in 1:H_dim
#                 up_indices = separate_spins_stored ? slater_labels_up[:, idx] : slater_labels[:, idx, 1]
#                 dn_indices = separate_spins_stored ? slater_labels_down[:, idx] : slater_labels[:, idx, 2]
                
#                 u_bin = orbital_indices_to_binary(up_indices, N_sites)
#                 d_bin = orbital_indices_to_binary(dn_indices, N_sites)
#                 basis_sector_h5[idx] = Trotter.combineSpinInts(u_bin, d_bin, N_sites)
#             end
#             return basis_sector_h5, k_min
#         end
#     end

#     # 3. Supply Hamiltonian components from real-space Hubbard
#     basis_up, _ = Trotter.getReducedHilSpace(N_sites, n_up; returnOcc=true)
#     basis_dn, _ = Trotter.getReducedHilSpace(N_sites, n_dn; returnOcc=true)
#     edges = Trotter.findLatticeEdges(Lvec; use_pbc=true)
#     hop_up = Trotter.fermionNNHopping(basis_up, edges, 1.0)
#     hop_dn = Trotter.fermionNNHopping(basis_dn, edges, 1.0)
#     I_up = sparse(I, length(basis_up), length(basis_up))
#     I_dn = sparse(I, length(basis_dn), length(basis_dn))
#     H_hop_real = kron(I_up, hop_dn) + kron(hop_up, I_dn)
#     H_int_real = Trotter.fermionOnSiteSpinDensity(basis_up, basis_dn; u=1.0)

#     # 4. Transform components to momentum space using Slater COB
#     F_up, _ = Trotter.SlaterCOB_RtoK_nparticle(Lvec, n_up)
#     F_dn, _ = Trotter.SlaterCOB_RtoK_nparticle(Lvec, n_dn)
#     F_total = kron(F_up, F_dn)

#     # Slice to the sector basis
#     kron_ints = [Trotter.combineSpinInts(u, d, N_sites) for u in basis_up for d in basis_dn]
#     state_to_kron_idx = Dict(val => idx for (idx, val) in enumerate(kron_ints))
#     sector_indices_in_kron = [state_to_kron_idx[val] for val in basis_sector]

#     H_hop_sector = (F_total * H_hop_real * F_total')[sector_indices_in_kron, sector_indices_in_kron]
#     H_int_sector = (F_total * H_int_real * F_total')[sector_indices_in_kron, sector_indices_in_kron]

#     # Test for a specific U value
#     u_idx = 25
#     u_val = U_values[u_idx]
#     println("Testing at U index $u_idx (U = $u_val)")

#     H = H_hop_sector + u_val * H_int_sector
#     vals, vecs = eigen(Hermitian(Matrix(H)))

#     # Get the ED eigenvalues for comparison
#     ed_E = nothing
#     if !isnothing(indexer)
#         dic = load(joinpath(folder, "meta_data_and_E.jld2"))
#         all_E = dic["E"]
#         k_min_jld = find_best_energy_sector(all_E, U_values)
#         ed_E = real(all_E[k_min_jld][u_idx])
#     else
#         h5open(h5_file, "r") do data
#             ed_E = real(read(data, "data/energies/$(k_min)"))[u_idx, 1]
#         end
#     end

#     calculated_E = vals[1]
#     println("Calculated Ground State Energy: $calculated_E")
#     println("ED Ground State Energy:         $ed_E")
#     energy_diff = abs(calculated_E - ed_E)
#     println("Energy Difference:              $energy_diff")

#     # Check ground state eigenvector overlap
#     loaded_vec = !isnothing(indexer) ? target_vecs[u_idx, :] : target_vecs[u_idx + 1, :]
#     calculated_vec = vecs[:, 1]
#     overlap = abs(dot(loaded_vec, calculated_vec))
#     println("Overlap (Fidelity) with ED state: $overlap")

#     if energy_diff < 1e-10 && abs(overlap - 1.0) < 1e-10
#         println("SUCCESS: Basis convention and ordering agree perfectly!")
#     else
#         println("FAILURE: Discrepancy found in basis convention or ordering.")
#     end
#     println("--------------------------------------------------")
# end

# # Run verification on both paths if present
# verify("data/N=(2, 2)_2x2")

# h5_folder = "data_new_sign/N=(2, 2)_3x2"
# if isdir(h5_folder)
#     verify(h5_folder)
# end