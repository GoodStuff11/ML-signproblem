#=
trotter_analysis_lib.jl

Reusable library of data-loading, coefficient-reordering, energy/overlap-computation,
and CairoMakie plot-building functions shared by trotter_exp_testing.jl and other
analysis scripts (e.g. plot_dimH_and_barren_analysis.jl). This file defines no
`(@main)` entry point and has no side effects beyond `include`s — it is meant to be
`include`d from a script, never run directly.

Extracted verbatim (no logic changes) from trotter_exp_testing.jl.
=#

using Lattices
using LinearAlgebra
using SparseArrays
using JLD2
using HDF5
using CairoMakie
using LaTeXStrings
using Zygote
using Optimization
using OptimizationOptimJL
using Combinatorics

if !isdefined(Main, :UtilityFunctions)
    include("utility_functions.jl")
end
using .UtilityFunctions
if !isdefined(Main, :Trotter)
    include("trotter.jl")
end
using .Trotter
include("data_path.jl")
include("logging.jl")
include("nn_strategy.jl")

include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")

cmap1(L) = [Makie.ColorSchemes.roma[z] for z in range(0, 1, length=L)]
cmap2(L) = [Makie.ColorSchemes.managua[z] for z in range(0, 1, length=L)]

function get_figsize(num_cols::Int, height_mm::Float64=55.0)
    px_per_mm = 96 / 25.4
    if num_cols == 1
        width_mm = 89 # mm
    elseif num_cols == 2
        width_mm = 180 # mm
    else
        error("Can only have num_cols == 1 or 2 (given: $num_cols)")
    end
    width_px = width_mm * px_per_mm
    height_px = height_mm * px_per_mm
    return (width_px, height_px)
end

function create_fig(num_cols::Int, height_mm::Float64=55.0; kwargs...)
    fig = Figure(size=get_figsize(num_cols, height_mm), figure_padding=5)
    ax = Axis(fig[1, 1];
        aspect=(1 + sqrt(5)) / 2,
        kwargs...
    )
    return fig, ax
end

const LEGEND_ARGS = Dict(:rowgap => -8, :padding => (3, 3, 0, 0))
set_theme!(theme_latexfonts(), fontsize=10)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    load_exact_exp_coefficients(folder, prefix, u_i) -> Vector{Float64} or nothing

Load the optimized exact-exponential coefficient vector for U-index u_i.

The per-u JLD2 stores "coefficients" as a Vector{Any} accumulated across U steps;
each element is either Nothing (no optimization at that step) or a Float64 array.
This function returns the first non-Nothing numeric sub-array, cast to Float64.
Returns nothing if the file is missing or all elements are Nothing.
"""
function load_exact_exp_coefficients(folder::String, prefix::String, u_i::Int, U_val::Float64=NaN)
    if U_val == 0.0
        return nothing
    end
    fpath = if !isnan(U_val) && U_val > 0
        u_idx = round(Int, U_val / 0.25) + 1
        p = joinpath(folder, "$(prefix)_u_$(u_idx).jld2")
        isfile(p) ? p : (isfile(joinpath(folder, "$(prefix)_u_$(u_i).jld2")) ? joinpath(folder, "$(prefix)_u_$(u_i).jld2") : joinpath(folder, "$(prefix)_u_$(u_i + 1).jld2"))
    else
        p = joinpath(folder, "$(prefix)_u_$(u_i).jld2")
        isfile(p) ? p : joinpath(folder, "$(prefix)_u_$(u_i + 1).jld2")
    end
    isfile(fpath) || return nothing
    d = load(fpath)["dict"]
    raw = d["coefficients"]
    flat_coeffs = Float64[]
    for elem in raw
        if elem isa AbstractArray{<:Number}
            append!(flat_coeffs, elem)
        end
    end
    return isempty(flat_coeffs) ? nothing : flat_coeffs
end

"""
    load_exact_exp_keys(folder, prefix) -> Vector or nothing

Load the coefficient_labels (operator keys) for the exact-exponential optimization
from the shared JLD2 file. Returns the non-nothing key list, or nothing if the
file is missing or no keys are available.
"""
function load_exact_exp_keys(folder::String, prefix::String)
    fpath = joinpath(folder, "$(prefix)_shared.jld2")
    isfile(fpath) || return nothing
    d = load(fpath)["dict"]
    labels = get(d, "coefficient_labels", nothing)
    isnothing(labels) && return nothing
    flat_keys = []
    for lbl in labels
        if !isnothing(lbl)
            append!(flat_keys, lbl)
        end
    end
    return isempty(flat_keys) ? nothing : flat_keys
end

"""
    load_exact_exp_instructions(folder, prefix) -> Dict or nothing

Load the instructions dict from the shared JLD2 file of exact-exponential optimization.
"""
function load_exact_exp_instructions(folder::String, prefix::String)
    fpath = joinpath(folder, "$(prefix)_shared.jld2")
    isfile(fpath) || return nothing
    d = load(fpath)["dict"]
    return get(d, "instructions", nothing)
end

"""
    build_exact_to_trotter_mapping(t_keys_JLD2, gates, lvec, indexer, basis_ints; antihermitian, sign_convention) -> (mapping_indices, mapping_factors)

Precomputes the index mapping and scale factors from exact-exp coefficient labels to Trotter gates.
Uses sparse matrix operations to execute in O(gates * nnz) instead of dense O(gates * dim^2).
"""
function build_exact_to_trotter_mapping(
    t_keys_JLD2::AbstractVector,
    gates::AbstractVector,
    lvec::Vector{Int},
    indexer,
    basis_ints::AbstractVector{<:Integer};
    antihermitian::Bool=false,
    sign_convention::Symbol=:spin_first
)
    N_sites = prod(lvec)
    basis_sector = Trotter.get_basis_sector(indexer, lvec, N_sites)

    present_orders = Int[]
    for k in t_keys_JLD2
        ord = div(length(k), 2)
        if ord ∉ present_orders
            push!(present_orders, ord)
        end
    end
    sort!(present_orders)

    operator_cache = Dict{Int,Dict{Symbol,Any}}()
    order_structures = Dict{Int,Any}()
    t_keys_exact = []

    flat_to_order_and_idx = Dict{Int,Tuple{Int,Int}}()
    flat_idx = 1

    for ord in present_orders
        struct_data = ensure_operator_structure!(ord, operator_cache, indexer, true, false, true, sign_convention, ColSnake(), Dict(), antihermitian, 1.0)
        order_structures[ord] = struct_data
        t_keys_exact_order = struct_data[:t_keys]
        append!(t_keys_exact, t_keys_exact_order)

        for local_idx in 1:length(t_keys_exact_order)
            flat_to_order_and_idx[flat_idx] = (ord, local_idx)
            flat_idx += 1
        end
    end

    canon_keys_JLD2 = [Trotter.key_to_canonical(k) for k in t_keys_JLD2]
    canon_to_idx_JLD2 = Dict(k => idx for (idx, k) in enumerate(canon_keys_JLD2))

    canon_keys_exact = [Trotter.key_to_canonical(k) for k in t_keys_exact]
    canon_to_idx_exact = Dict(k => idx for (idx, k) in enumerate(canon_keys_exact))

    num_gates = length(gates)
    mapping_indices = zeros(Int, num_gates)
    mapping_factors = zeros(Float64, num_gates)

    d_dim = length(basis_sector)

    # Each gate's mapping entry is independent of every other gate's (distinct
    # g_idx slots in mapping_indices/mapping_factors, no shared mutable state
    # aside from read-only caches built above), so this is embarrassingly
    # parallel across gates -- and for large sectors (d_dim ~ 1e4-1e5) each
    # iteration's sparse-matrix/LinearMap construction dominates runtime, so
    # multithreading here gives a near-linear speedup with thread count.
    @safe_threads for g_idx in 1:num_gates
        g = gates[g_idx]
        # v_in/v_out must be allocated per-iteration (not hoisted/shared) since
        # threads run concurrently and would otherwise race on shared buffers.
        v_in = zeros(ComplexF64, d_dim)
        v_out = zeros(ComplexF64, d_dim)

        lbl = fgate_to_label(g, lvec)
        ck = Trotter.key_to_canonical(lbl)

        idx_JLD2 = get(canon_to_idx_JLD2, ck, 0)
        if idx_JLD2 == 0
            idx_JLD2 = get(canon_to_idx_JLD2, Trotter.conjugate_canonical(ck), 0)
        end
        if idx_JLD2 == 0
            continue
        end

        idx_exact = get(canon_to_idx_exact, ck, 0)
        if idx_exact == 0
            idx_exact = get(canon_to_idx_exact, Trotter.conjugate_canonical(ck), 0)
        end
        if idx_exact == 0
            error("Gate $g_idx (label $ck) has no matching key in exact-exp t_keys (struct_data).")
        end

        # Build sparse mat_l
        ord, i = flat_to_order_and_idx[idx_exact]
        struct_data = order_structures[ord]
        P = length(struct_data[:t_keys])
        t_l = zeros(Float64, P)
        t_l[i] = 1.0
        vals = update_values(struct_data[:signs], struct_data[:param_index_map], t_l, struct_data[:parameter_mapping], struct_data[:parity])
        mat_l = sparse(struct_data[:rows], struct_data[:cols], vals, d_dim, d_dim)
        if antihermitian
            mat_l = make_antihermitian(mat_l)
        else
            mat_l = make_hermitian(mat_l)
        end
        I_nz, J_nz, V_nz = findnz(mat_l)
        # `mat_l` is built against the shared sparsity pattern (struct_data[:rows]/[:cols])
        # of the whole order block, so most stored entries are explicit zeros for this
        # particular one-hot gate; grabbing findnz's first entry would almost always land
        # on one of those structural zeros instead of the gate's actual matrix element.
        nz_pos = findfirst(v -> abs(v) > 1e-12, V_nz)
        if isnothing(nz_pos)
            continue
        end
        r, c, val_E = I_nz[nz_pos], J_nz[nz_pos], real(V_nz[nz_pos])

        # Evaluate LinearMap column c
        T_map = Trotter.tau_g_operator_sector(g, N_sites, basis_sector; antihermitian=antihermitian)
        v_in[c] = 1.0
        mul!(v_out, T_map, v_in)
        val_T = real(v_out[r])

        if abs(val_T) > 1e-12
            f = val_E / val_T
            mapping_indices[g_idx] = idx_JLD2
            mapping_factors[g_idx] = f
        end
    end

    return mapping_indices, mapping_factors
end

"""
    apply_exact_to_trotter_mapping(A_exact, mapping_indices, mapping_factors) -> Vector{Float64}
"""
function apply_exact_to_trotter_mapping(A_exact::Vector{Float64}, mapping_indices::Vector{Int}, mapping_factors::Vector{Float64})
    A_reordered = zeros(Float64, length(mapping_indices))
    for g_idx in 1:length(mapping_indices)
        idx = mapping_indices[g_idx]
        if idx > 0 && idx <= length(A_exact)
            A_reordered[g_idx] = A_exact[idx] * mapping_factors[g_idx]
        end
    end
    return A_reordered
end

"""
    reorder_exact_to_trotter_coeffs(A_exact, t_keys_JLD2, gates, lvec, indexer, basis_ints) -> Vector{Float64}

Re-order the exact-exponential coefficient vector to match the ordering of the Trotter gates,
resolving all coordinate snake orderings, Jordon-Wigner signs, and diagonal scaling factors.
"""
function reorder_exact_to_trotter_coeffs(
    A_exact::Vector{Float64},
    t_keys_JLD2::AbstractVector,
    gates::AbstractVector,
    lvec::Vector{Int},
    indexer,
    basis_ints::AbstractVector{<:Integer};
    antihermitian::Bool=false,
    sign_convention::Symbol=:spin_first
)
    mapping_indices, mapping_factors = build_exact_to_trotter_mapping(
        t_keys_JLD2, gates, lvec, indexer, basis_ints; antihermitian=antihermitian, sign_convention=sign_convention
    )
    return apply_exact_to_trotter_mapping(A_exact, mapping_indices, mapping_factors)
end


# ═══════════════════════════════════════════════════════════════════════════════
# ENERGY COMPUTATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    compute_energy(psi, H) -> Float64

Compute the expectation value ⟨ψ|H|ψ⟩.
"""
compute_energy(psi::AbstractVector, H) = real(dot(psi, H * psi))

"""
    compute_gs_states(H_hop, H_int, U_values) -> (gs_states, gs_energies)

Diagonalise H_hop + U*H_int for each U in U_values using the real symmetric
eigensolver. Returns a vector of ground-state vectors and the corresponding
ground-state energies.

Assumes the Hamiltonian is real and Hermitian (valid for Hubbard in momentum basis).
"""
function compute_gs_states(H_hop, H_int, U_values::Vector{Float64})
    n_U = length(U_values)
    dim = size(H_hop, 1)
    gs_states = Vector{Vector{ComplexF64}}(undef, n_U)
    gs_energies = fill(NaN, n_U)
    for u_i in 1:n_U
        H_u = H_hop + U_values[u_i] * H_int
        vals, vecs = eigen(Symmetric(Matrix(real(H_u))))
        gs_states[u_i] = ComplexF64.(vecs[:, 1])
        gs_energies[u_i] = vals[1]
    end
    return gs_states, gs_energies
end

"""
    compute_exact_exp_energies_and_overlaps(exact_coeffs_raw, t_keys_JLD2, indexer, lvec, basis_ints, ref_state, gs_states, H_hop_mom, H_int_mom, U_values; antihermitian, sign_convention)
        -> (Vector{Float64}, Vector{Float64})

Compute the actual Hamiltonian energy and overlap with the ground state for the exact exponential state:
    ψ_exact = exp(i M) |ref_state⟩
for each U value.
"""
function compute_exact_exp_energies_and_overlaps(
    exact_coeffs_raw::Vector,
    t_keys_JLD2::AbstractVector,
    indexer,
    lvec::Vector{Int},
    basis_ints::AbstractVector{<:Integer},
    ref_state::Vector{ComplexF64},
    gs_states::Vector{Vector{ComplexF64}},
    H_hop_mom,
    H_int_mom,
    U_values::Vector{Float64};
    antihermitian::Bool=false,
    sign_convention::Symbol=:coordinate_first
)
    n_U = length(U_values)
    energies = fill(NaN, n_U)
    overlaps = fill(NaN, n_U)
    exact_states = Vector{Union{Nothing,Vector{ComplexF64}}}(fill(nothing, n_U))

    # 1. Reconstruct basis_sector from indexer
    N_sites = prod(lvec)
    basis_sector = Trotter.get_basis_sector(indexer, lvec, N_sites)

    # Build state mappings
    state_to_idx = Dict(val => idx for (idx, val) in enumerate(basis_ints))
    perm = [state_to_idx[val] for val in basis_sector]
    inv_perm = invperm(perm)

    # Map JLD2 keys
    canon_keys_JLD2 = [Trotter.key_to_canonical(k) for k in t_keys_JLD2]
    canon_to_idx_JLD2 = Dict(k => idx for (idx, k) in enumerate(canon_keys_JLD2))

    # Determine present orders
    present_orders = Int[]
    for k in t_keys_JLD2
        ord = div(length(k), 2)
        if ord ∉ present_orders
            push!(present_orders, ord)
        end
    end
    sort!(present_orders)

    operator_cache = Dict{Int,Dict{Symbol,Any}}()
    order_structures = Dict{Int,Any}()
    for ord in present_orders
        order_structures[ord] = ensure_operator_structure!(ord, operator_cache, indexer, true, false, true, sign_convention, ColSnake(), Dict(), antihermitian, 1.0)
    end

    for u_i in 1:n_U
        coeffs = exact_coeffs_raw[u_i]
        H_u = H_hop_mom + U_values[u_i] * H_int_mom
        if isnothing(coeffs)
            if u_i == 1
                energies[1] = real(dot(ref_state, H_u * ref_state))
                overlaps[1] = abs2(dot(ref_state, gs_states[1]))
                exact_states[1] = copy(ref_state)
            end
            continue
        end

        # Map ref_state to the indexer basis
        psi_exact_sector = ref_state[perm]

        # Apply sequential unitaries order-by-order
        for ord in present_orders
            struct_data = order_structures[ord]
            t_keys_exact = struct_data[:t_keys]

            coeffs_exact_order = zeros(Float64, length(t_keys_exact))
            for (idx_exact, key_exact) in enumerate(t_keys_exact)
                ck = Trotter.key_to_canonical(key_exact)
                idx_JLD2 = get(canon_to_idx_JLD2, ck, 0)
                if idx_JLD2 == 0
                    idx_JLD2 = get(canon_to_idx_JLD2, Trotter.conjugate_canonical(ck), 0)
                end
                if idx_JLD2 == 0
                    coeffs_exact_order[idx_exact] = 0.0
                else
                    coeffs_exact_order[idx_exact] = coeffs[idx_JLD2]
                end
            end

            vals = update_values(struct_data[:signs], struct_data[:param_index_map], coeffs_exact_order, struct_data[:parameter_mapping], struct_data[:parity])
            mat_l_order = sparse(struct_data[:rows], struct_data[:cols], vals, length(basis_sector), length(basis_sector))
            if antihermitian
                mat_l_order = make_antihermitian(mat_l_order)
                # apply_exp (ed_optimization.jl) uses expv (Krylov, sparse-matrix-vector-product
                # only) above a size threshold instead of materializing a dense exp(M), since a
                # dense exp(M) is O(dimH^3) and infeasible once dimH reaches the thousands.
                psi_exact_sector = apply_exp(mat_l_order, psi_exact_sector, 1.0)
            else
                mat_l_order = make_hermitian(mat_l_order)
                psi_exact_sector = apply_exp(mat_l_order, psi_exact_sector, 1.0im)
            end
        end

        # Map back to the basis_ints basis
        psi_exact = psi_exact_sector[inv_perm]
        exact_states[u_i] = psi_exact

        # Compute energy and overlap
        energies[u_i] = real(dot(psi_exact, H_u * psi_exact))
        overlaps[u_i] = abs2(dot(psi_exact, gs_states[u_i]))
    end

    return energies, overlaps, exact_states
end

function compute_exact_exp_energies(
    exact_coeffs_raw::Vector,
    t_keys_JLD2::AbstractVector,
    indexer,
    lvec::Vector{Int},
    basis_ints::AbstractVector{<:Integer},
    ref_state::Vector{ComplexF64},
    H_hop_mom,
    H_int_mom,
    U_values::Vector{Float64};
    antihermitian::Bool=false,
    sign_convention::Symbol=:coordinate_first
)
    dummy_gs = [ref_state for _ in 1:length(U_values)]
    energies, _, _ = compute_exact_exp_energies_and_overlaps(
        exact_coeffs_raw, t_keys_JLD2, indexer, lvec, basis_ints, ref_state, dummy_gs,
        H_hop_mom, H_int_mom, U_values; antihermitian=antihermitian, sign_convention=sign_convention
    )
    return energies
end

"""
    compute_trotterized_energies_and_overlaps(exact_coeffs, trotter_orders, gates, ref_state,
                                              basis_ints, gs_states, N_sites, H_hop, H_int,
                                              U_values, num_gates; exact_states, antihermitian=false)
        -> (Dict{Int, Vector{Float64}}, Dict{Int, Vector{Float64}}, Dict{Int, Vector{Float64}})

For each Trotter order P in trotter_orders, compute both the evolved energy, overlap with gs_states,
and overlap with exact_states (Trotterization discretization error).
"""
function compute_trotterized_energies_and_overlaps(
    exact_coeffs::Vector, trotter_orders::Vector{Int},
    gates, ref_state::AbstractVector, basis_ints::AbstractVector,
    gs_states::Vector{Vector{ComplexF64}},
    N_sites::Int, H_hop, H_int, U_values::Vector{Float64}, num_gates::Int;
    exact_states::Union{Nothing,AbstractVector}=nothing,
    antihermitian::Bool=false
)
    n_U = length(U_values)
    energies_dict = Dict{Int,Vector{Float64}}()
    overlaps_dict = Dict{Int,Vector{Float64}}()
    exact_overlaps_dict = Dict{Int,Vector{Float64}}()

    for P in trotter_orders
        energies_P = fill(NaN, n_U)
        overlaps_P = fill(NaN, n_U)
        exact_overlaps_P = fill(NaN, n_U)
        for u_i in 1:n_U
            A_base = exact_coeffs[u_i]
            H_u = H_hop + U_values[u_i] * H_int
            if isnothing(A_base)
                energies_P[u_i] = compute_energy(ref_state, H_u)
                overlaps_P[u_i] = abs2(dot(ref_state, gs_states[u_i]))
                exact_overlaps_P[u_i] = 1.0
                continue
            end
            length(A_base) % num_gates == 0 || continue
            stored_num_exp = length(A_base) ÷ num_gates
            # Each coefficient copy is scaled down by 1/P; P copies applied in sequence
            A_trotter = repeat(A_base, P) ./ P
            total_exp = P * stored_num_exp
            psi = Trotter.apply_unitary(
                A_trotter, gates, ref_state, basis_ints, N_sites, total_exp;
                antihermitian=antihermitian
            )
            energies_P[u_i] = compute_energy(psi, H_u)
            overlaps_P[u_i] = abs2(dot(psi, gs_states[u_i]))
            if !isnothing(exact_states) && !isnothing(exact_states[u_i])
                exact_overlaps_P[u_i] = abs2(dot(psi, exact_states[u_i]))
            end
        end
        energies_dict[P] = energies_P
        overlaps_dict[P] = overlaps_P
        exact_overlaps_dict[P] = exact_overlaps_P
        println("  P=$P: $(sum(!isnan, energies_P)) non-NaN energy & overlap values")
    end

    return energies_dict, overlaps_dict, exact_overlaps_dict
end

function compute_trotterized_energies(
    exact_coeffs::Vector, trotter_orders::Vector{Int},
    gates, ref_state::AbstractVector, basis_ints::AbstractVector,
    N_sites::Int, H_hop, H_int, U_values::Vector{Float64}, num_gates::Int;
    antihermitian::Bool=false
)
    dummy_gs = [ref_state for _ in 1:length(U_values)]
    energies, _ = compute_trotterized_energies_and_overlaps(
        exact_coeffs, trotter_orders, gates, ref_state, basis_ints, dummy_gs,
        N_sites, H_hop, H_int, U_values, num_gates; antihermitian=antihermitian
    )
    return energies
end

"""
    load_trotter_opt_energies_and_overlaps(folder, N_sites, n_U_or_U_values; custom_ref_state_arg, antihermitian, loss_type, gs_energies)
        -> (Vector{Float64}, Vector{Float64})

Load the final overlap-optimized Trotter energies and overlaps from trotter_N=<N_sites>_u_<i>.jld2 files.
"""
function load_trotter_opt_energies_and_overlaps(
    folder::String, N_sites::Int, n_U_or_U_values::Union{Int,Vector{Float64}};
    custom_ref_state_arg::Union{String,Nothing}=nothing,
    antihermitian::Bool=false,
    loss_type::Symbol=:overlap,
    gs_energies::Union{Vector{Float64},Nothing}=nothing
)
    U_values = if n_U_or_U_values isa Vector{Float64}
        n_U_or_U_values
    else
        fill(NaN, n_U_or_U_values)
    end
    n_U = length(U_values)
    prefix = "trotter_N=$(N_sites)"
    if !isnothing(custom_ref_state_arg)
        prefix *= "_ref_$(custom_ref_state_arg)"
    end
    if antihermitian
        prefix *= "_antihermitian"
    end
    if loss_type == :energy
        prefix *= "_loss_energy"
    end
    energies = fill(NaN, n_U)
    overlaps = fill(NaN, n_U)
    for (u_i, U_val) in enumerate(U_values)
        if U_val == 0.0
            if !isnothing(gs_energies) && length(gs_energies) >= u_i
                energies[u_i] = gs_energies[u_i]
            end
            overlaps[u_i] = 1.0
            continue
        end
        fpath = if !isnan(U_val) && U_val > 0
            u_idx = round(Int, U_val / 0.25) + 1
            p = joinpath(folder, "$(prefix)_u_$(u_idx).jld2")
            isfile(p) ? p : (isfile(joinpath(folder, "$(prefix)_u_$(u_i).jld2")) ? joinpath(folder, "$(prefix)_u_$(u_i).jld2") : joinpath(folder, "$(prefix)_u_$(u_i + 1).jld2"))
        else
            p = joinpath(folder, "$(prefix)_u_$(u_i).jld2")
            isfile(p) ? p : joinpath(folder, "$(prefix)_u_$(u_i + 1).jld2")
        end
        if isfile(fpath)
            d = load(fpath)["dict"]
            met = d["metrics"]
            if haskey(met, "energy") && !isempty(met["energy"])
                energies[u_i] = met["energy"][end]
            end
            if haskey(met, "loss") && !isempty(met["loss"])
                loss_val = met["loss"][end]
                if get(d, "loss_type", loss_type) == :overlap
                    overlaps[u_i] = 1.0 - loss_val
                    if isnan(energies[u_i])
                        energies[u_i] = loss_val
                    end
                else
                    energies[u_i] = loss_val
                end
            end
            if haskey(met, "overlap") && !isempty(met["overlap"])
                overlaps[u_i] = met["overlap"][end]
            end
        end
    end
    return energies, overlaps
end

function load_trotter_opt_energies(
    folder::String, N_sites::Int, n_U_or_U_values::Union{Int,Vector{Float64}};
    custom_ref_state_arg::Union{String,Nothing}=nothing,
    antihermitian::Bool=false,
    loss_type::Symbol=:overlap,
    gs_energies::Union{Vector{Float64},Nothing}=nothing
)
    energies, _ = load_trotter_opt_energies_and_overlaps(
        folder, N_sites, n_U_or_U_values;
        custom_ref_state_arg=custom_ref_state_arg,
        antihermitian=antihermitian,
        loss_type=loss_type,
        gs_energies=gs_energies
    )
    return energies
end

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

"""
    build_comparison_plot(U_values, gs_energies, exact_exp_energies,
                          trotter_energies, trotter_opt_energies,
                          trotter_orders, n_up, n_dn, lvec; custom_ref_state_arg, loss_type, num_cols, height_mm, legend_position) -> Figure
"""
function build_comparison_plot(
    U_values::Vector{Float64},
    gs_energies::Vector{Float64},
    exact_exp_energies::Vector{Float64},
    trotter_energies::Dict{Int,Vector{Float64}},
    trotter_opt_energies::Vector{Float64},
    trotter_orders::Vector{Int},
    n_up::Int, n_dn::Int, lvec::Vector{Int};
    custom_ref_state_arg::Union{String,Nothing}=nothing,
    loss_type::Symbol=:overlap,
    num_cols::Int=2,
    height_mm::Float64=70.0,
    legend_position::Symbol=:rb,
    ylim::Tuple=(1e-5, nothing),
)
    fig, ax = create_fig(num_cols, height_mm;
        xlabel=L"U",
        ylabel=L"E - E_0(U)",
        yscale=log10,
        limits=((0, 15), ylim),
    )

    palette = cmap2(length(trotter_orders))

    # Exact exp line
    valid_exact = .!isnan.(exact_exp_energies) .& (U_values .> 0)
    if any(valid_exact)
        lines!(ax, U_values[valid_exact], max.(exact_exp_energies[valid_exact] .- gs_energies[valid_exact], 1e-16);
            label=L"\textrm{Exact exp}",
            color=:black,
            linewidth=2,
            linestyle=:solid
        )
    end

    # Trotterized lines (P = 1, 2, 4, 8)
    for (idx, P) in enumerate(trotter_orders)
        energies_P = trotter_energies[P]
        valid_P = .!isnan.(energies_P) .& (U_values .> 0)
        if any(valid_P)
            lines!(ax, U_values[valid_P], max.(energies_P[valid_P] .- gs_energies[valid_P], 1e-16);
                label=L"P = %$P",
                color=palette[idx],
                linewidth=1.5,
                linestyle=:dash
            )
        end
    end

    # Trotter-optimized line
    valid_opt = .!isnan.(trotter_opt_energies) .& (U_values .> 0)
    if any(valid_opt)
        lines!(ax, U_values[valid_opt], max.(trotter_opt_energies[valid_opt] .- gs_energies[valid_opt], 1e-16);
            label=L"\textrm{Trotter opt}",
            color=:crimson,
            linewidth=2,
            linestyle=:solid
        )
    end

    axislegend(ax; position=legend_position, backgroundcolor=(:white, 0.8), LEGEND_ARGS...)

    return fig
end

"""
    build_overlap_comparison_plot(U_values, exact_exp_overlaps, trotter_overlaps, trotter_opt_overlaps,
                                  trotter_orders, n_up, n_dn, lvec;
                                  trotter_to_exact_overlaps=nothing, custom_ref_state_arg=nothing,
                                  loss_type=:overlap, metric=:infidelity,
                                  num_cols=2, height_mm=70.0, legend_position=:rb) -> Figure

Constructs and returns a CairoMakie figure comparing overlaps across Trotter orders P = 1, 2, 4, 8,
the exact exponential state, and the Trotter-optimized baseline using `axislegend`.

When `metric=:infidelity` (default):
    Plots 1 - |⟨E_0(U)|𝒰|E_0(0)⟩|² on a log10 scale against the target ground state.
    Shows the crossover where Exact Exponential outperforms Trotter-optimized at higher U.
When `metric=:trotter_error` or `metric=:exact_infidelity`:
    Plots 1 - |⟨ψ_exact|𝒰_P|ψ_ref⟩|² on a log10 scale, showing the clear O(1/P²) scaling across Trotter orders.
When `metric=:overlap`:
    Plots |⟨E_0(U)|𝒰|E_0(0)⟩|² directly.
"""
function build_overlap_comparison_plot(
    U_values::Vector{Float64},
    exact_exp_overlaps::Vector{Float64},
    trotter_overlaps::Dict{Int,Vector{Float64}},
    trotter_opt_overlaps::Vector{Float64},
    trotter_orders::Vector{Int},
    n_up::Int, n_dn::Int, lvec::Vector{Int};
    trotter_to_exact_overlaps::Union{Nothing,Dict{Int,Vector{Float64}}}=nothing,
    custom_ref_state_arg::Union{String,Nothing}=nothing,
    loss_type::Symbol=:overlap,
    metric::Symbol=:infidelity,
    num_cols::Int=2,
    height_mm::Float64=70.0,
    legend_position::Symbol=:rb,
    ylim::Tuple=(1e-5, nothing),
)
    palette = cmap2(length(trotter_orders))

    if metric == :infidelity || metric == :ground_state
        fig, ax = create_fig(num_cols, height_mm;
            xlabel=L"U",
            ylabel=L"1 - |\langle E_0(U)|\mathcal{U}|E_0(0)\rangle|^2",
            yscale=log10,
            limits=((0, 15), ylim),
        )

        # Exact exp line
        valid_exact = .!isnan.(exact_exp_overlaps) .& (U_values .> 0)
        if any(valid_exact)
            infid_exact = max.(1.0 .- exact_exp_overlaps[valid_exact], 1e-16)
            lines!(ax, U_values[valid_exact], infid_exact;
                label=L"\textrm{Exact exp}",
                color=:black,
                linewidth=2,
                linestyle=:solid
            )
        end

        # Trotterized lines (P = 1, 2, 4, 8)
        for (idx, P) in enumerate(trotter_orders)
            overlaps_P = trotter_overlaps[P]
            valid_P = .!isnan.(overlaps_P) .& (U_values .> 0)
            if any(valid_P)
                infid_P = max.(1.0 .- overlaps_P[valid_P], 1e-16)
                lines!(ax, U_values[valid_P], infid_P;
                    label=L"P = %$P",
                    color=palette[idx],
                    linewidth=1.5,
                    linestyle=:dash
                )
            end
        end

        # Trotter-optimized line
        valid_opt = .!isnan.(trotter_opt_overlaps) .& (U_values .> 0)
        if any(valid_opt)
            infid_opt = max.(1.0 .- trotter_opt_overlaps[valid_opt], 1e-16)
            lines!(ax, U_values[valid_opt], infid_opt;
                label=L"\textrm{Trotter opt}",
                color=:crimson,
                linewidth=2,
                linestyle=:solid
            )
        end

        axislegend(ax; position=legend_position, backgroundcolor=(:white, 0.8), LEGEND_ARGS...)
        return fig
    elseif metric == :trotter_error || metric == :exact_infidelity
        fig, ax = create_fig(num_cols, height_mm;
            xlabel=L"U",
            ylabel=L"1 - |\langle \psi_{\textrm{exact}}|\mathcal{U}_P|\psi_{\textrm{ref}}\rangle|^2",
            yscale=log10,
            limits=((0, 15), ylim),
        )

        # Plot discretization error for each Trotter order P
        if !isnothing(trotter_to_exact_overlaps)
            for (idx, P) in enumerate(trotter_orders)
                ovlp_exact_P = trotter_to_exact_overlaps[P]
                valid_P = .!isnan.(ovlp_exact_P) .& (U_values .> 0)
                if any(valid_P)
                    infid_exact_P = max.(1.0 .- ovlp_exact_P[valid_P], 1e-16)
                    lines!(ax, U_values[valid_P], infid_exact_P;
                        label=L"P = %$P",
                        color=palette[idx],
                        linewidth=1.5,
                        linestyle=:solid
                    )
                end
            end
        end

        axislegend(ax; position=legend_position, backgroundcolor=(:white, 0.8), LEGEND_ARGS...)
        return fig
    else
        fig, ax = create_fig(num_cols, height_mm;
            xlabel=L"U",
            ylabel=L"|\langle E_0(U)|\mathcal{U}|E_0(0)\rangle|^2",
            limits=((0, 15), (-0.05, 1.05)),
        )

        # Exact exp line
        valid_exact = .!isnan.(exact_exp_overlaps) .& (U_values .> 0)
        if any(valid_exact)
            lines!(ax, U_values[valid_exact], exact_exp_overlaps[valid_exact];
                label=L"\textrm{Exact exp}",
                color=:black,
                linewidth=2,
                linestyle=:solid
            )
        end

        # Trotterized lines (P = 1, 2, 4, 8)
        for (idx, P) in enumerate(trotter_orders)
            overlaps_P = trotter_overlaps[P]
            valid_P = .!isnan.(overlaps_P) .& (U_values .> 0)
            if any(valid_P)
                lines!(ax, U_values[valid_P], overlaps_P[valid_P];
                    label=L"P = %$P",
                    color=palette[idx],
                    linewidth=1.5,
                    linestyle=:dash
                )
            end
        end

        # Trotter-optimized line
        valid_opt = .!isnan.(trotter_opt_overlaps) .& (U_values .> 0)
        if any(valid_opt)
            lines!(ax, U_values[valid_opt], trotter_opt_overlaps[valid_opt];
                label=L"\textrm{Trotter opt}",
                color=:crimson,
                linewidth=2,
                linestyle=:solid
            )
        end

        axislegend(ax; position=legend_position, backgroundcolor=(:white, 0.8), LEGEND_ARGS...)
        return fig
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# MODULAR SYSTEM HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    setup_system(folder, custom_ref_state_arg, antihermitian_arg, loss_type)

Loads ED data, determines the antihermitian flag, constructs the Hamiltonians in
momentum (or coordinate) basis, and enumerates the Trotter gates.
"""
function setup_system(
    folder::String,
    custom_ref_state_arg::Union{String,Nothing},
    antihermitian_arg::Union{Bool,Nothing},
    loss_type::Symbol;
    verbose=true
)
    U_values, target_vecs, indexer, _, N_elec, _, _, sign_convention =
        load_ED_data(folder; verbose=verbose, use_slater_reference=(custom_ref_state_arg == "slater"))
    n_up_loaded, n_dn_loaded = N_elec
    n_U = length(U_values)
    N_sites = prod(indexer.lattice_dims)
    dims_val = collect(Int, indexer.lattice_dims)

    # Derive antihermitian
    local antihermitian
    if isnothing(antihermitian_arg)
        test_prefix = "trotter_N=$N_sites"
        if !isnothing(custom_ref_state_arg)
            test_prefix *= "_ref_$(custom_ref_state_arg)"
        end
        test_prefix *= "_antihermitian"
        if loss_type == :energy
            test_prefix *= "_loss_energy"
        end
        if isfile(joinpath(folder, "$(test_prefix)_shared.jld2"))
            antihermitian = true
        else
            antihermitian = false
        end
    else
        antihermitian = antihermitian_arg
    end

    # Derive q_target
    q_target = nothing
    k_val = try
        indexer.k
    catch
        nothing
    end
    if !isnothing(k_val) && !isnothing(dims_val)
        q_target = Trotter.ravel_c(Tuple(k - 1 for k in k_val), Tuple(dims_val))
    end

    local H_hop_mom, H_int_mom, basis_ints
    if isnothing(q_target)
        verbose && println("Subspace does not conserve momentum. Constructing Hamiltonians in coordinate basis (:spin_first)...")
        basis_ints = Trotter.get_basis_sector(indexer, dims_val, N_sites)
        lattice = Square(Tuple(dims_val), Periodic())
        subspace = HubbardSubspace(n_up_loaded, n_dn_loaded, lattice; k=nothing)
        H_hop_mom, H_int_mom = create_hubbard_matrices(subspace; indexer=indexer, sign_convention=:spin_first)
    else
        verbose && println("\nBuilding sector Hamiltonians in momentum basis...")
        H_hop_mom, basis_dict, _ = Trotter.HubbardMomentumBasis(
            1.0, 0.0, dims_val, (n_up_loaded, n_dn_loaded); indexer=indexer
        )
        H_int_mom, _, _ = Trotter.HubbardMomentumBasis(
            0.0, 1.0, dims_val, (n_up_loaded, n_dn_loaded); indexer=indexer
        )
        basis_ints = basis_dict["ints"]
    end

    verbose && println("Hilbert space sector dim = $(length(basis_ints))")

    verbose && println("\nEnumerating Trotter gates...")
    gates = Trotter.enumerate_ferm_excitations(
        2, dims_val; conserve_mom=true, conserve_sz=true, include_diagonal=true,
    )
    # tau_terms = Trotter.fgateToTauSector(gates, N_sites, basis_ints; antihermitian=antihermitian)
    num_gates = length(gates)

    return U_values, target_vecs, indexer, sign_convention, antihermitian, H_hop_mom, H_int_mom, basis_ints, gates, num_gates, N_sites, n_up_loaded, n_dn_loaded
end

"""
    get_exact_coefficients(folder, U_values, n_up_loaded, n_dn_loaded, custom_ref_state_arg, antihermitian, loss_type, sign_convention, gates, indexer, basis_ints)

Loads the optimized exact-exponential coefficient vector and maps/reorders them to match the Trotter gate ordering.
"""
function get_exact_coefficients(
    folder::String,
    U_values::Vector{Float64},
    n_up_loaded::Int,
    n_dn_loaded::Int,
    custom_ref_state_arg::Union{String,Nothing},
    antihermitian::Bool,
    loss_type::Symbol,
    sign_convention::Symbol,
    gates,
    indexer,
    basis_ints
)
    n_U = length(U_values)
    unitary_prefix = "unitary_map_energy_symmetry=false_N=($n_up_loaded, $n_dn_loaded)"
    if !isnothing(custom_ref_state_arg)
        unitary_prefix *= "_ref_$(custom_ref_state_arg)"
    end
    if antihermitian
        unitary_prefix *= "_antihermitian"
    end
    if loss_type == :energy
        unitary_prefix *= "_loss_energy"
    end

    exact_coeffs_raw = [
        load_exact_exp_coefficients(folder, unitary_prefix, u_i, U_values[u_i]) for u_i in 1:n_U
    ]
    n_loaded = sum(!isnothing, exact_coeffs_raw)
    println("\nLoaded exact-exp coefficients for $n_loaded / $n_U U values")
    if n_loaded > 0
        first_nz = findfirst(!isnothing, exact_coeffs_raw)
        println("  First non-nothing: u_i=$first_nz, length=$(length(exact_coeffs_raw[first_nz]))")
    end

    t_keys = load_exact_exp_keys(folder, unitary_prefix)
    num_gates = length(gates)
    lvec = collect(Int, indexer.lattice_dims)

    exact_coeffs = if !isnothing(t_keys) && n_loaded > 0
        println("  Operator keys loaded: $(length(t_keys)). Precomputing reordering mapping for $num_gates gates...")
        mapping_indices, mapping_factors = build_exact_to_trotter_mapping(
            t_keys, gates, lvec, indexer, basis_ints; antihermitian=antihermitian, sign_convention=sign_convention
        )
        map(exact_coeffs_raw) do A
            isnothing(A) ? nothing : apply_exact_to_trotter_mapping(A, mapping_indices, mapping_factors)
        end
    else
        @warn "Could not load operator keys from shared file; exact-exp evaluation skipped."
        fill(nothing, n_U)
    end

    return exact_coeffs_raw, t_keys, exact_coeffs
end


# ═══════════════════════════════════════════════════════════════════════════════
# HIGH-LEVEL ANALYSES
# ═══════════════════════════════════════════════════════════════════════════════


"""
    compute_trotterized_energies_u_sweep(folder, trotter_orders, custom_ref_state_arg, antihermitian_arg, loss_type)

Runs the sweep over all U values for a single system size/folder, returning the computed data.
"""
function compute_trotterized_energies_u_sweep(
    folder::String,
    trotter_orders::Vector{Int},
    custom_ref_state_arg::Union{String,Nothing},
    antihermitian_arg::Union{Bool,Nothing},
    loss_type::Symbol
)
    # 1. Setup system
    U_values, target_vecs, indexer, sign_convention, antihermitian, H_hop_mom, H_int_mom, basis_ints, gates, num_gates, N_sites, n_up_loaded, n_dn_loaded =
        setup_system(folder, custom_ref_state_arg, antihermitian_arg, loss_type)

    n_U = length(U_values)
    lvec = collect(Int, indexer.lattice_dims)

    # 2. Get exact coefficients
    exact_coeffs_raw, t_keys, exact_coeffs = get_exact_coefficients(
        folder, U_values, n_up_loaded, n_dn_loaded, custom_ref_state_arg, antihermitian, loss_type, sign_convention, gates, indexer, basis_ints
    )

    # 3. Extract ground states and reference state
    has_prepended_ref = (size(target_vecs, 1) == n_U + 1)
    gs_states = [ComplexF64.(vec(target_vecs[has_prepended_ref ? u_i + 1 : u_i, :])) for u_i in 1:n_U]
    gs_energies = [compute_energy(gs_states[u_i], H_hop_mom + U_values[u_i] * H_int_mom) for u_i in 1:n_U]
    println("gs energies: $gs_energies")

    ref_state = ComplexF64.(vec(target_vecs[1, :]))

    # 4. Compute exact-exp energies and overlaps
    println("\nComputing exact-exp energies and overlaps...")
    exact_exp_energies, exact_exp_overlaps, exact_states = if !isnothing(t_keys) && sum(!isnothing, exact_coeffs_raw) > 0
        compute_exact_exp_energies_and_overlaps(
            exact_coeffs_raw, t_keys, indexer, lvec, basis_ints, ref_state, gs_states,
            H_hop_mom, H_int_mom, U_values; antihermitian=antihermitian, sign_convention=sign_convention
        )
    else
        (fill(NaN, n_U), fill(NaN, n_U), Vector{Union{Nothing,Vector{ComplexF64}}}(fill(nothing, n_U)))
    end
    println("  Done ($(sum(!isnan, exact_exp_energies)) non-NaN values)")

    if all(isnan, exact_exp_energies)
        @warn "No exact exponential data found (all exact-exp energies are NaN)."
    end

    # 5. Compute trotterized energies and overlaps
    println("\nComputing trotterized energies and overlaps for orders P = $trotter_orders...")
    trotter_energies, trotter_overlaps, trotter_to_exact_overlaps = compute_trotterized_energies_and_overlaps(
        exact_coeffs, trotter_orders, gates, ref_state, basis_ints, gs_states,
        N_sites, H_hop_mom, H_int_mom, U_values, num_gates; exact_states=exact_states, antihermitian=antihermitian
    )

    # 6. Load trotter-optimized energies and overlaps
    println("\nLoading trotter-optimized results...")
    trotter_opt_energies, trotter_opt_overlaps = load_trotter_opt_energies_and_overlaps(
        folder, N_sites, U_values;
        custom_ref_state_arg=custom_ref_state_arg,
        antihermitian=antihermitian,
        loss_type=loss_type,
        gs_energies=gs_energies
    )
    println("trotter energies: $trotter_opt_energies")
    println("Diff energies: $(trotter_opt_energies - gs_energies)")
    println("  Loaded $(sum(!isnan, trotter_opt_energies)) values")

    return U_values, gs_energies, exact_exp_energies, exact_exp_overlaps, trotter_energies, trotter_overlaps, trotter_opt_energies, trotter_opt_overlaps, trotter_to_exact_overlaps, n_up_loaded, n_dn_loaded, lvec
end

"""
    compute_trotterized_energies_system_size(system_folders, U_target, trotter_orders, custom_ref_state_arg, antihermitian_arg, loss_type)

Computes exact ground states, exact-exp, trotterized, and trotter-optimized energies for a single U value across multiple system sizes.
"""
function compute_trotterized_energies_system_size(
    system_folders::Vector{String},
    U_target::Float64,
    trotter_orders::Vector{Int},
    custom_ref_state_arg::Union{String,Nothing},
    antihermitian_arg::Union{Bool,Nothing},
    loss_type::Symbol
)
    n_systems = length(system_folders)
    system_names = String[]
    num_sites = Int[]
    gs_energies = Float64[]
    exact_exp_energies = Float64[]
    trotter_energies = Dict{Int,Vector{Float64}}()
    for P in trotter_orders
        trotter_energies[P] = fill(NaN, n_systems)
    end
    trotter_opt_energies = fill(NaN, n_systems)

    for (s_idx, folder) in enumerate(system_folders)
        println("\n=======================================================")
        println("Processing system: $(basename(folder))")
        println("=======================================================")

        # 1. Setup system
        U_values, target_vecs, indexer, sign_convention, antihermitian, H_hop_mom, H_int_mom, basis_ints, gates, num_gates, N_sites, n_up_loaded, n_dn_loaded =
            setup_system(folder, custom_ref_state_arg, antihermitian_arg, loss_type)

        push!(system_names, basename(folder))
        push!(num_sites, N_sites)

        # 2. Find the U index closest to U_target
        u_i = findmin(abs.(U_values .- U_target))[2]
        U_actual = U_values[u_i]
        println("Target U = $U_target, closest actual U = $U_actual (index $u_i)")

        # 3. Extract ground states and reference state
        gs_state = ComplexF64.(vec(target_vecs[u_i+1, :]))
        gs_energy = compute_energy(gs_state, H_hop_mom + U_actual * H_int_mom)
        push!(gs_energies, gs_energy)
        println("GS energy: $gs_energy")

        ref_state = ComplexF64.(vec(target_vecs[1, :]))

        # 4. Load exact coefficients
        exact_coeffs_raw, t_keys, exact_coeffs = get_exact_coefficients(
            folder, U_values, n_up_loaded, n_dn_loaded, custom_ref_state_arg, antihermitian, loss_type, sign_convention, gates, indexer, basis_ints
        )

        # 5. Compute exact-exp energy for this U
        if !isnothing(t_keys) && !isnothing(exact_coeffs_raw[u_i])
            # Call compute_exact_exp_energies wrapping just this U
            exact_val = compute_exact_exp_energies(
                [exact_coeffs_raw[u_i]], t_keys, indexer, collect(Int, indexer.lattice_dims), basis_ints, ref_state,
                H_hop_mom, H_int_mom, [U_actual]; antihermitian=antihermitian, sign_convention=sign_convention
            )[1]
            push!(exact_exp_energies, exact_val)
            println("Exact exp energy: $exact_val")
        else
            push!(exact_exp_energies, NaN)
            println("Exact exp energy: NaN")
        end

        # 6. Compute trotterized energies for each P
        if !isnothing(exact_coeffs[u_i])
            trotter_vals = compute_trotterized_energies(
                [exact_coeffs[u_i]], trotter_orders, gates, ref_state, basis_ints,
                N_sites, H_hop_mom, H_int_mom, [U_actual], num_gates; antihermitian=antihermitian
            )
            for P in trotter_orders
                trotter_energies[P][s_idx] = trotter_vals[P][1]
            end
        end

        # 7. Load trotter-optimized energies
        trotter_opt = load_trotter_opt_energies(
            folder, N_sites, length(U_values);
            custom_ref_state_arg=custom_ref_state_arg,
            antihermitian=antihermitian,
            loss_type=loss_type
        )
        trotter_opt_energies[s_idx] = trotter_opt[u_i]
        println("Trotter opt energy: $(trotter_opt[u_i])")
    end

    return system_names, num_sites, gs_energies, exact_exp_energies, trotter_energies, trotter_opt_energies
end

# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM SIZE PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

"""
    build_system_size_comparison_plot(system_sizes, num_sites, gs_energies, exact_exp_energies, trotter_energies, trotter_opt_energies, trotter_orders, U_value, loss_type; num_cols, height_mm, legend_position) -> Figure

Plots energy differences (E - E_gs) vs system sizes at a single U value.
"""
function build_system_size_comparison_plot(
    system_sizes::Vector{String},
    num_sites::Vector{Int},
    gs_energies::Vector{Float64},
    exact_exp_energies::Vector{Float64},
    trotter_energies::Dict{Int,Vector{Float64}},
    trotter_opt_energies::Vector{Float64},
    trotter_orders::Vector{Int},
    U_value::Float64,
    loss_type::Symbol;
    num_cols::Int=2,
    height_mm::Float64=70.0,
    legend_position::Symbol=:rt,
)
    # Sort systems by: 1. Number of sites, 2. GS Energy (as proxy for size/filling uniqueness), 3. System name
    sort_keys = [(num_sites[i], gs_energies[i], system_sizes[i]) for i in 1:length(system_sizes)]
    perm = sortperm(sort_keys)

    sorted_sizes = system_sizes[perm]
    sorted_gs = gs_energies[perm]
    sorted_exact = exact_exp_energies[perm]
    sorted_opt = trotter_opt_energies[perm]

    sorted_trotter = Dict{Int,Vector{Float64}}()
    for P in trotter_orders
        sorted_trotter[P] = trotter_energies[P][perm]
    end

    x_coords = 1:length(system_sizes)

    fig, ax = create_fig(num_cols, height_mm;
        xlabel=L"\textrm{System}",
        ylabel=L"E - E_0(U)",
        yscale=log10,
        xticks=(x_coords, sorted_sizes),
    )

    palette = cmap2(length(trotter_orders))

    valid_exact = .!isnan.(sorted_exact)
    if any(valid_exact)
        scatterlines!(ax, x_coords[valid_exact], max.(sorted_exact[valid_exact] .- sorted_gs[valid_exact], 1e-15);
            label=L"\textrm{Exact exp}",
            color=:black,
            linewidth=2,
            markersize=8,
        )
    end

    for (idx, P) in enumerate(trotter_orders)
        energies_P = sorted_trotter[P]
        valid = .!isnan.(energies_P)
        if any(valid)
            scatterlines!(ax, x_coords[valid], max.(energies_P[valid] .- sorted_gs[valid], 1e-15);
                label=L"P = %$P",
                color=palette[idx],
                linewidth=1.5,
                markersize=6,
                linestyle=:dash,
            )
        end
    end

    valid_opt = .!isnan.(sorted_opt)
    if any(valid_opt)
        scatterlines!(ax, x_coords[valid_opt], max.(sorted_opt[valid_opt] .- sorted_gs[valid_opt], 1e-15);
            label=L"\textrm{Trotter opt}",
            color=:crimson,
            linewidth=2,
            markersize=8,
        )
    end

    axislegend(ax; position=legend_position, backgroundcolor=(:white, 0.8), LEGEND_ARGS...)

    return fig
end
