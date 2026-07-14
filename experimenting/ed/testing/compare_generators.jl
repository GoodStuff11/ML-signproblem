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

# Load indexer from metadata file
folder = "data/N=(3, 3)_3x2"
meta_file = joinpath(folder, "meta_data_and_E.jld2")
meta_data = load(meta_file)
# If indexer is a vector, select the first one (Gamma sector)
indexer = meta_data["dict"]["indexer"]
if indexer isa Vector
    indexer = indexer[1]
end

# Reconstruct basis_sector from indexer
# Helper to convert Coordinate to 0-based site index
function coord_to_site_idx(coord, Lvec)
    c0 = coord.coordinates .- 1
    return Trotter.ravel_c(c0, Tuple(Lvec))
end

# Helper to convert Coordinate set to binary representation
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

# Enumerate Trotter gates
gates = TamFermion.enumerate_ferm_excitations(2, lvec; conserve_mom=true, conserve_sz=true, include_diagonal=true)

# Build the exact-exp matrices (using the same code as ed_optimization.jl)
operator_cache = Dict{Int,Dict{Symbol,Any}}()
struct_data = ensure_operator_structure!(2, operator_cache, indexer, true, false, true, :spin_first, Dict(), false, 1.0)

# Use struct_data[:t_keys] directly!
t_keys_exact = struct_data[:t_keys]

# Build exact-exp matrices
P = length(t_keys_exact)
exact_mats = []
for i in 1:P
    t_l = zeros(P)
    t_l[i] = 1.0
    vals = update_values(struct_data[:signs], struct_data[:param_index_map], t_l, struct_data[:parameter_mapping], struct_data[:parity])
    mat_l = sparse(struct_data[:rows], struct_data[:cols], vals, length(basis_sector), length(basis_sector))
    mat_l = make_hermitian(mat_l)
    push!(exact_mats, mat_l)
end

# Build Trotter matrices on basis_sector
trotter_mats = [real(Matrix(Trotter.TamFermion.tau_g_operator_sector(g, N_sites, basis_sector))) for g in gates]

# Helper to convert key to canonical
function key_to_canonical(k)
    [(c.coordinates, spin, op) for (c, spin, op) in k]
end
function conjugate_canonical(ck)
    conj_ops = [(c, spin, op == :create ? :annihilate : :create) for (c, spin, op) in ck]
    cre = sort(filter(op -> op[3] == :create,     conj_ops), by = op -> (op[1], op[2]))
    ann = sort(filter(op -> op[3] == :annihilate, conj_ops), by = op -> (op[1], op[2]))
    return [cre; ann]
end

canon_keys = [key_to_canonical(k) for k in t_keys_exact]
canon_to_idx = Dict(k => idx for (idx, k) in enumerate(canon_keys))

mismatches = 0
factors = Float64[]
for (g_idx, g) in enumerate(gates)
    lbl = fgate_to_label(g, lvec)
    ck = key_to_canonical(lbl)
    idx = get(canon_to_idx, ck, 0)
    is_conj = false
    if idx == 0
        idx = get(canon_to_idx, conjugate_canonical(ck), 0)
        is_conj = true
    end
    
    if idx == 0
        println("Gate $g_idx not found!")
        global mismatches += 1
        continue
    end
    
    E_mat = Matrix(real(exact_mats[idx]))
    T_mat = trotter_mats[g_idx]
    
    # Find the first non-zero element in T_mat
    nz_indices = findall(x -> abs(x) > 1e-5, T_mat)
    if isempty(nz_indices)
        println("Gate $g_idx: Trotter matrix is entirely zero!")
        global mismatches += 1
        continue
    end
    
    r, c = nz_indices[1].I
    f = E_mat[r, c] / T_mat[r, c]
    push!(factors, f)
    
    # Verify that E_mat ≈ f * T_mat
    if !(E_mat ≈ f * T_mat)
        println("Gate $g_idx: Matrices not proportional!")
        println("  Factor: ", f)
        println("  Max diff: ", maximum(abs.(E_mat .- f * T_mat)))
        global mismatches += 1
    end
end

println("Total mismatches: ", mismatches)
if mismatches == 0
    println("All gates matched perfectly up to a factor!")
    println("Unique factors found: ", unique(factors))
end
