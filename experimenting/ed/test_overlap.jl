using Lattices
using HDF5
using JLD2, LinearAlgebra, SparseArrays

include("trotter.jl")
using .Trotter
include("utility_functions.jl")
include("ed_objects.jl")
include("ed_functions.jl")
include("data_path.jl")

folder = data_folder("N=(2, 2)_2x2")
u_start = 1
u_end = 25
maxiters = 10
loss_type = :overlap
num_exponentials = 1
antihermitian = true
is_slater_ref = true
# 1. Load ED data (loads indexer if JLD2, or we can use it to build the sector basis)
U_values, state_vecs, indexer, _, N_elec, spin_conserved, _, sign_convention =
    load_ED_data(folder; verbose=true, sign_convention=:spin_first, use_slater_reference=is_slater_ref)

n_up, n_dn = N_elec

# Parse dimension from folder name, default to (3, 3) if fails
Lvec = parse_lattice_dimension(folder)
N_sites = prod(Lvec)

# 2. Computing the basis
# Convert each (up_coords_set, dn_coords_set) entry in the indexer to a combined
# 2N-bit integer that fgateToTauSector expects: up bits in the low N bits, dn bits
# in the upper N bits (via combineSpinInts).
basis_sector = Trotter.get_basis_sector(indexer, Lvec, N_sites)

# 3. Find the Hamiltonian
# Derive the momentum sector from the indexer (same convention as trotter_exp_testing.jl).
# indexer.k is 1-based coordinate tuple; q_target is the C-order flat index (0-based).
q_target = nothing
if !isnothing(indexer.k) && !isnothing(indexer.lattice_dims)
    q_target = Trotter.ravel_c(Tuple(k - 1 for k in indexer.k), Tuple(Lvec))
end
@time H_hop_sector, basis_dict_sector, _ = Trotter.TamFermion.HubbardMomentumBasis(
    1.0, 0.0, Lvec, (n_up, n_dn); indexer=indexer
)
@time H_int_sector, _, _ = Trotter.TamFermion.HubbardMomentumBasis(
    0.0, 1.0, Lvec, (n_up, n_dn); indexer=indexer
)

println("Direct Expectation Value from load_ED_data: ", real(state_vecs[1, :]' * H_hop_sector * state_vecs[1, :]))