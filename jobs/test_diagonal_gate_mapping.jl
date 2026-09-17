# Regression test for: "Gate N (label [...]) has no matching key in exact-exp t_keys (struct_data)."
#
# Root cause: with antihermitian=true, `ensure_operator_structure!` now omits diagonal
# (density-density) keys, because an antihermitian generator X - X' is identically zero for
# them. The Trotter gate set is still enumerated with include_diagonal=true, and saved JLD2
# key lists still contain those keys, so `build_exact_to_trotter_mapping` hit a diagonal gate
# with no exact-exp counterpart and raised instead of skipping it.
using HDF5, LinearAlgebra, Lattices, SparseArrays, Combinatorics
include("../ed_objects.jl")
include("../utility_functions.jl")
include("../trotter.jl")
include("../ed_functions.jl")
using .Trotter

Lvec = (2, 2)
subspace = HubbardSubspace(2, 2, Square(Lvec, Periodic()))
indexer = CombinationIndexer(subspace; order=ColSnake())

modes(ops) = [(o[1].coordinates..., o[2]) for o in ops]
function is_diagonal(key)
    n = div(length(key), 2)
    modes(key[1:n]) == modes(key[n+1:end])
end

function keys_for(omit_diagonal)
    _, t_keys = create_randomized_nth_order_operator(
        2, indexer, true; magnitude=1.0, omit_H_conj=true, conserve_spin=true,
        normalize_coefficients=false, conserve_momentum=true, omit_diagonal=omit_diagonal)
    return t_keys
end

keys_herm = keys_for(false)   # what an older saved run / the JLD2 key file contains
keys_anti = keys_for(true)    # what build_order_structures rebuilds when antihermitian=true

diag_herm = filter(is_diagonal, keys_herm)
println("order-2 keys with omit_diagonal=false: $(length(keys_herm))  (diagonal: $(length(diag_herm)))")
println("order-2 keys with omit_diagonal=true : $(length(keys_anti))  (diagonal: $(length(filter(is_diagonal, keys_anti))))")

canon_anti = Set(Trotter.key_to_canonical(k) for k in keys_anti)
missing_diag = [k for k in diag_herm
                if !(Trotter.key_to_canonical(k) in canon_anti) &&
                   !(Trotter.conjugate_canonical(Trotter.key_to_canonical(k)) in canon_anti)]
println("diagonal keys present in the hermitian set but MISSING from the antihermitian set: $(length(missing_diag))")

@assert isempty(filter(is_diagonal, keys_anti)) "expected no diagonal keys when omit_diagonal=true"
@assert length(missing_diag) == length(diag_herm) > 0 "expected every diagonal key to be missing"
println("\nREPRODUCED: every diagonal key is dropped when antihermitian=true, so any diagonal")
println("Trotter gate has no counterpart in the rebuilt exact-exp t_keys.\n")

# ── Skipping those gates is physically correct ────────────────────────────────
# A diagonal key builds a hermitian X, so the antihermitian generator X - X' must vanish.
# Verified through the same non-symmetry assembly path build_generator_matrix uses.
basis_sector = Trotter.get_basis_sector(indexer, collect(Int, Lvec), prod(Lvec))
d_dim = length(basis_sector)
t_dict_h, _ = create_randomized_nth_order_operator(
    2, indexer, true; magnitude=1.0, omit_H_conj=true, conserve_spin=true,
    normalize_coefficients=false, conserve_momentum=true, omit_diagonal=false)
rows, cols, signs, ops_list = build_n_body_structure_from_keys(
    keys_herm, indexer, typeof(t_dict_h[keys_herm[1]]);
    sign_convention=:spin_first, lattice_ordering=ColSnake())
param_index_map = build_param_index_map(ops_list, keys_herm)

worst_diag = 0.0
worst_offdiag = 0.0
for (i, k) in enumerate(keys_herm)
    global worst_diag, worst_offdiag
    t_l = zeros(Float64, length(keys_herm)); t_l[i] = 1.0
    vals = update_values(signs, param_index_map, t_l, nothing, nothing)
    A = make_antihermitian(sparse(rows, cols, vals, d_dim, d_dim))
    nrm = isempty(nonzeros(A)) ? 0.0 : maximum(abs, nonzeros(A))
    is_diagonal(k) ? (worst_diag = max(worst_diag, nrm)) : (worst_offdiag = max(worst_offdiag, nrm))
end
println("largest |entry| of the antihermitian generator, over DIAGONAL keys    : $worst_diag")
println("largest |entry| of the antihermitian generator, over off-diagonal keys: $worst_offdiag")
@assert worst_diag < 1e-12 "diagonal keys should give an identically-zero antihermitian generator"
@assert worst_offdiag > 1e-6 "off-diagonal keys should give a nonzero generator (sanity check)"
println("CONFIRMED: diagonal keys contribute exactly nothing to an antihermitian generator,")
println("so skipping such gates (leaving their mapping factor at 0) is the correct behavior.\n")

# ── End-to-end: build_exact_to_trotter_mapping over the real gate set ─────────
include("../nn_strategy.jl")
include("../fig4_lib.jl")

# The exact label from the reported stack trace.
failing_label = [(Coordinate(1, 1), 1, :create), (Coordinate(1, 2), 1, :create),
                 (Coordinate(1, 1), 1, :annihilate), (Coordinate(1, 2), 1, :annihilate)]
@assert is_diagonal_key(failing_label) "the label from the reported traceback must be diagonal"
@assert !is_diagonal_key(keys_anti[1]) "an off-diagonal key must not be flagged diagonal"
println("is_diagonal_key agrees with the label from the reported stack trace.")

gates = Trotter.enumerate_ferm_excitations(2, collect(Int, Lvec);
    conserve_mom=true, conserve_sz=true, include_diagonal=true)
println("enumerated $(length(gates)) Trotter gates (include_diagonal=true)")

mapping_indices, mapping_factors = build_exact_to_trotter_mapping(
    keys_herm, gates, collect(Int, Lvec), indexer, basis_sector;
    antihermitian=true, sign_convention=:spin_first)

n_diag_gates = count(g -> is_diagonal_key(fgate_to_label(g, collect(Int, Lvec))), gates)
println("diagonal gates in the gate set: $n_diag_gates")
@assert n_diag_gates > 0 "the gate set must contain diagonal gates for this test to mean anything"
for (i, g) in enumerate(gates)
    if is_diagonal_key(fgate_to_label(g, collect(Int, Lvec)))
        @assert mapping_indices[i] == 0 && mapping_factors[i] == 0.0 "diagonal gate $i should map to 0"
    end
end
@assert any(!=(0), mapping_indices) "off-diagonal gates should still produce a real mapping"
println("mapped $(count(!=(0), mapping_indices)) non-diagonal gates; all $n_diag_gates diagonal gates map to 0.")
println("\nPASS: build_exact_to_trotter_mapping completes without error.")
