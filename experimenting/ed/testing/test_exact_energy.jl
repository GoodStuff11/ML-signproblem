using JLD2
using Lattices
using SparseArrays
using LinearAlgebra
using Combinatorics
using Zygote
using Optimization
using OptimizationOptimJL

# Load the packages in the correct order
include("../trotter.jl")
using .Trotter
using .Trotter.UtilityFunctions
include("../nn_strategy.jl")

include("../ed_objects.jl")
include("../ed_functions.jl")
include("../ed_optimization.jl")

# Setup lattice
lvec = [3, 2]
n_up, n_dn = 3, 3
N_sites = prod(lvec)

# Get momentum basis
H_hop_mom, basis_dict, _ = TamFermion.HubbardMomentumBasis(1.0, 0.0, lvec, (n_up, n_dn); q_target=0)
basis_ints = basis_dict["ints"]

# Load indexer and target vecs from metadata file
folder = "data/N=(3, 3)_3x2"
meta_file = joinpath(folder, "meta_data_and_E.jld2")
meta_data = load(meta_file)
# If indexer is a vector, select the first one (Gamma sector)
indexer = meta_data["dict"]["indexer"]
if indexer isa Vector
    indexer = indexer[1]
end

# Load sectors and select Gamma sector
target_vecs_all = meta_data["dict"]["all_full_eig_vecs"]
k_min = find_best_energy_sector(meta_data["dict"]["E"], meta_data["dict"]["meta_data"]["U_values"]; verbose=false)
target_vecs = target_vecs_all[k_min]

# Reconstruct basis_sector from indexer
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
basis_sector = Vector{UInt}(undef, length(indexer.inv_comb_dict))
for (idx, conf) in enumerate(indexer.inv_comb_dict)
    u_bin = coord_set_to_binary(conf[1], lvec)
    d_bin = coord_set_to_binary(conf[2], lvec)
    basis_sector[idx] = Trotter.combineSpinInts(u_bin, d_bin, N_sites)
end

# Get the ground state energy and Hamiltonian for U_values[u_i]
state_to_idx = Dict(val => idx for (idx, val) in enumerate(basis_ints))
perm = [state_to_idx[val] for val in basis_sector]

H_hop = H_hop_mom[perm, perm]
H_int_mom, _, _ = TamFermion.HubbardMomentumBasis(0.0, 1.0, lvec, (n_up, n_dn); q_target=0)
H_int = H_int_mom[perm, perm]

# Load exact exp data for u_2
u_i = 2
U_values = meta_data["dict"]["meta_data"]["U_values"]
u_val = U_values[u_i]
fpath = joinpath(folder, "unitary_map_energy_symmetry=false_N=(3, 3)_u_$(u_i).jld2")
d = load(fpath)["dict"]

# Find the vector of coefficients (the one that is not nothing)
exact_coeffs = nothing
for coeff in d["coefficients"]
    if coeff isa AbstractArray{<:Number}
        global exact_coeffs = Vector{Float64}(coeff)
        break
    end
end

if exact_coeffs === nothing
    error("Could not find coefficient vector in file!")
end

# Build exact matrix
operator_cache = Dict{Int,Dict{Symbol,Any}}()
struct_data = ensure_operator_structure!(2, operator_cache, indexer, true, false, true, :spin_first, Dict(), false, 1.0)
vals = update_values(struct_data[:signs], struct_data[:param_index_map], exact_coeffs, struct_data[:parameter_mapping], struct_data[:parity])
mat_l = sparse(struct_data[:rows], struct_data[:cols], vals, length(basis_sector), length(basis_sector))
mat_l = make_hermitian(mat_l)

# Prepare exact state: ref_state is the ground state at U = U_values[1] (which is target_vecs[1, :])
ref_state = target_vecs[1, :][perm]

# Evolve the state
psi_exact = exp(1im * Matrix(mat_l)) * ref_state

# Target state is the ground state at U = U_values[2] (which is target_vecs[2, :])
state2 = target_vecs[2, :][perm]

# Compute Hamiltonian at u_val
H_u = H_hop + u_val * H_int
E_exact = real(dot(psi_exact, H_u * psi_exact))

# Find the true ground state energy of H_u
vals_ev, vecs_ev = eigen(Symmetric(Matrix(real(H_u))))
E_gs = vals_ev[1]

# Compute overlap with target state
overlap = abs2(dot(psi_exact, state2))

println("U value: ", u_val)
println("Norm of ref_state: ", norm(ref_state))
println("Norm of psi_exact: ", norm(psi_exact))
println("Norm of state2:    ", norm(state2))
println("Norm of mat_l:     ", norm(mat_l))
println("Is mat_l Hermitian? ", ishermitian(mat_l))
println("Overlap raw (dot product): ", dot(psi_exact, state2))
println("Overlap abs2: ", overlap)
println("Prepared state energy: ", E_exact)
println("True Ground State energy:        ", E_gs)
println("Difference (prepared - GS):      ", E_exact - E_gs)
