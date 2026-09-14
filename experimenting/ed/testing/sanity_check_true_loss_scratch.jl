using Lattices, LinearAlgebra, SparseArrays, JLD2, HDF5, Combinatorics, Zygote
using Optimization, OptimizationOptimJL, Statistics
include(joinpath(@__DIR__, "..", "data_path.jl")); include(joinpath(@__DIR__, "..", "utility_functions.jl")); using .UtilityFunctions
include(joinpath(@__DIR__, "..", "ed_objects.jl")); include(joinpath(@__DIR__, "..", "ed_functions.jl")); include(joinpath(@__DIR__, "..", "ed_optimization.jl"))
include(joinpath(@__DIR__, "..", "trotter.jl")); using .Trotter

folder = "/home/jek354/research/data/new_data/data_h5_fixed/N=(5, 4)_3x3"
U_values, state_vecs, indexer, _, N_elec, spin_conserved, _, sign_convention =
    load_ED_data(folder; verbose=false, sign_convention=:spin_first, use_slater_reference=false)

state_ref = state_vecs[1, :]
state_target = state_vecs[10, :]

_, _, coefficient_values, _, _, metrics, _ = optimize_unitary(
    state_ref, state_target, indexer;
    spin_conserved=spin_conserved,
    maxiters=5,
    optimization_scheme=[2],
    gradient=:adjoint_gradient,
    antihermitian=true,
    optimizer=:LBFGS,
    initialization_samples=0,
    loss_type=:overlap,
    use_gpu=false,
    num_exponentials=1
)

println("metrics[\"loss\"] = ", metrics["loss"])

# Independently reconstruct psi from the returned coefficients the same way optimize_unitary does internally,
# to confirm metrics["loss"][end] == true unregularized infidelity of the actual state (not optimizer objective).
coeffs = coefficient_values[2]  # order=2 coefficients
gates = Trotter.enumerate_ferm_excitations(2, collect(Int, indexer.lattice_dims); conserve_mom=true, conserve_sz=true, include_diagonal=true)
println("num gates: ", length(gates), " num coeffs: ", length(coeffs))

# Independently reconstruct psi_metric the same way optimize_unitary's internal
# psi_metric loop does, and confirm metrics["loss"][end] == 1-abs2(dot(psi,state2)).
struct_data = ensure_operator_structure!(2, Dict{Int,Dict{Symbol,Any}}(), indexer, spin_conserved, false, false, :spin_first, ColSnake(), Dict(), true, 1.0)
vals = update_values(struct_data[:signs], struct_data[:param_index_map], coeffs, struct_data[:parameter_mapping], struct_data[:parity])
dim = length(indexer.inv_comb_dict)
mat_l = sparse(struct_data[:rows], struct_data[:cols], vals, dim, dim)
mat_l = make_antihermitian(mat_l)
psi_check = apply_exp(mat_l, state_ref, 1.0)
independent_true_loss = 1.0 - abs2(dot(psi_check, state_target))
println("metrics[\"loss\"][end]      = ", metrics["loss"][end])
println("independent recompute      = ", independent_true_loss)
println("match (atol=1e-10)?        = ", isapprox(metrics["loss"][end], independent_true_loss; atol=1e-10))
