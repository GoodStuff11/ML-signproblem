using Lattices
using SparseArrays
using LinearAlgebra
using JLD2
using HDF5
using Combinatorics

include("../utility_functions.jl")
using .UtilityFunctions
include("../ed_objects.jl")
include("../ed_functions.jl")
include("../trotter.jl")
using .Trotter
include("../nn_strategy.jl")

function canonical_label(lbl)
    # lbl is a Vector of (coord, spin, op_type)
    return [(c.coordinates, spin, op) for (c, spin, op) in lbl]
end

function conjugate_label(lbl)
    # Swap :create and :annihilate
    conj_ops = [(c, spin, op == :create ? :annihilate : :create) for (c, spin, op) in lbl]
    cre = filter(op -> op[3] == :create, conj_ops)
    ann = filter(op -> op[3] == :annihilate, conj_ops)
    sort!(cre, by = op -> (op[1], op[2]))
    sort!(ann, by = op -> (op[1], op[2]))
    return [cre; ann]
end

# 1. Load data
folder = "data/N=(3, 3)_3x2"
lvec = [3, 2]
n_up, n_dn = 3, 3
N_sites = prod(lvec)

meta = load(joinpath(folder, "meta_data_and_E.jld2"))["dict"]
U_values = meta["meta_data"]["U_values"]
u_i = 41 # Let's test u_i = 41
U_val = U_values[u_i]

# Load sector Hamiltonians
q_target = 0 # Gamma point
H_hop_mom, basis_dict, _ = TamFermion.HubbardMomentumBasis(1.0, 0.0, lvec, (n_up, n_dn); q_target=q_target)
H_int_mom, _, _ = TamFermion.HubbardMomentumBasis(0.0, 1.0, lvec, (n_up, n_dn); q_target=q_target)
basis_ints = basis_dict["ints"]
H_u = H_hop_mom + U_val * H_int_mom

# Load ground state (reference state)
vals, vecs = eigen(Symmetric(Matrix(real(H_hop_mom + U_values[1] * H_int_mom))))
ref_state = ComplexF64.(vecs[:, 1])

# Load exact coefficients
exact_shared = load(joinpath(folder, "unitary_map_energy_symmetry=false_N=(3, 3)_shared.jld2"))["dict"]
t_keys = exact_shared["coefficient_labels"][2]
t_keys_canon = [canonical_label(k) for k in t_keys]

exact_u = load(joinpath(folder, "unitary_map_energy_symmetry=false_N=(3, 3)_u_$(u_i).jld2"))["dict"]
exact_coeffs = exact_u["coefficients"]
# Get the non-nothing coeff vector
A_exact = nothing
for c in exact_coeffs
    if c isa AbstractVector{<:Number}
        global A_exact = Float64.(c)
        break
    end
end

if isnothing(A_exact)
    error("Could not load exact coefficients for u_i=$u_i")
end

println("Exact coefficients length: ", length(A_exact))

# Load Trotter gates
gates = TamFermion.enumerate_ferm_excitations(2, lvec; conserve_mom=true, conserve_sz=true, include_diagonal=true)
num_gates = length(gates)
println("Trotter gates length: ", num_gates)

# 2. Build the mappings
# Map canonical key to index in t_keys
canon_to_idx = Dict(k => idx for (idx, k) in enumerate(t_keys_canon))

# For each gate, find the corresponding key in t_keys
gate_to_exact_idx = Vector{Int}(undef, num_gates)
gate_is_conjugate = Vector{Bool}(undef, num_gates)

for (g_idx, g) in enumerate(gates)
    lbl = fgate_to_label(g, lvec)
    lbl_canon = canonical_label(lbl)
    
    if haskey(canon_to_idx, lbl_canon)
        gate_to_exact_idx[g_idx] = canon_to_idx[lbl_canon]
        gate_is_conjugate[g_idx] = false
    else
        lbl_conj = conjugate_label(lbl_canon)
        if haskey(canon_to_idx, lbl_conj)
            gate_to_exact_idx[g_idx] = canon_to_idx[lbl_conj]
            gate_is_conjugate[g_idx] = true
        else
            error("Gate $g_idx with label $lbl_canon not found in t_keys (or its conjugate)")
        end
    end
end

# 3. Construct Trotter coefficients for both options
# Option A: Divide by 2 for conjugate pairs
A_optA = zeros(num_gates)
counts = zeros(Int, length(A_exact))
for g_idx in 1:num_gates
    counts[gate_to_exact_idx[g_idx]] += 1
end

for g_idx in 1:num_gates
    exact_idx = gate_to_exact_idx[g_idx]
    A_optA[g_idx] = A_exact[exact_idx] / counts[exact_idx]
end

# Option B: Assign full value to the first gate of the pair, 0 to the other
A_optB = zeros(num_gates)
assigned = fill(false, length(A_exact))
for g_idx in 1:num_gates
    exact_idx = gate_to_exact_idx[g_idx]
    if !assigned[exact_idx]
        A_optB[g_idx] = A_exact[exact_idx]
        assigned[exact_idx] = true
    else
        A_optB[g_idx] = 0.0
    end
end

# Option C: Copy full value to both (no division)
A_optC = zeros(num_gates)
for g_idx in 1:num_gates
    exact_idx = gate_to_exact_idx[g_idx]
    A_optC[g_idx] = A_exact[exact_idx]
end

# 4. Evaluate exact energy using JLD2 saved value or exact exp matrix
# Let's see: the saved ground state at u_i is:
target_gs_E = eigen(Symmetric(Matrix(real(H_u)))).values[1]
exact_loss_energy = exact_u["metrics"]["loss"][end]
println("Exact optimized energy (from file): ", exact_loss_energy)
println("Ground state energy (exact ED): ", target_gs_E)

# 5. Evaluate Trotterized energies for different orders P
function compute_energy(psi, H)
    return real(dot(psi, H * psi))
end

for (opt_name, A_base) in [("Option A (divide by 2)", A_optA), ("Option B (first only)", A_optB), ("Option C (no division)", A_optC)]
    println("\n--- Testing $opt_name ---")
    for P in [1, 2, 4, 8, 16]
        A_trotter = repeat(A_base, P) ./ P
        psi = TrotterOptimization.apply_unitary(A_trotter, gates, ref_state, basis_ints, N_sites, P)
        E = compute_energy(psi, H_u)
        println("  P=$P: Energy = $E, Diff = $(E - exact_loss_energy)")
    end
end
