#=
trotter_exp_testing.jl

Compare Hamiltonian energies across three methods:
  1. Exact exponential  – stored coefficients from unitary_map_energy_symmetry=false files
                          applied as a single unitary (assumes energy optimization).
  2. Trotterized        – same stored coefficients repeated P times, each copy divided
                          by P (approximating the exact exponential at increasing order P).
  3. Trotter-optimized  – overlap-optimized Trotter coefficients from
                          trotter_N=<N_sites>_u files.

The plot is saved to <folder>/trotter_order_comparison.png.

Usage:
  julia --project=.. trotter_exp_testing.jl [folder] [--trotter_orders=<list>]

Arguments:
  folder (optional): Path to the ED data folder containing the JLD2 files.
                     Default: "data/N=(4, 4)_3x3_2".

  --trotter_orders=<list> (optional): Comma-separated list of positive integers
                     specifying the Trotterization repetition counts P to compare.
                     For each P the exact-exp coefficient vector is repeated P times
                     and divided by P before applying the unitary.
                     Default: "1,2,4,8".

  --n_up=<int> (optional): Number of spin-up electrons. Default: 4.

  --n_dn=<int> (optional): Number of spin-down electrons. Default: 4.

  --lvec=<WxH> (optional): Lattice dimensions in the format WxH (e.g. "3x3").
                     Default: "3x3".

  --output=<string> (optional): Name of png to output (will have .png appendend to end). Default: trotter_order_comparison

Examples:
  julia --project=.. trotter_exp_testing.jl
  julia --project=.. trotter_exp_testing.jl "data/N=(4, 5)_3x3"
  julia --project=.. trotter_exp_testing.jl "data/N=(4, 4)_3x3_2" --trotter_orders=1,2,4,8,16
  julia --project=.. trotter_exp_testing.jl "data/N=(4, 4)_3x3_2" --n_up=4 --n_dn=4 --lvec=3x3 --output=trotter_order_comparison
=#

using Lattices
using LinearAlgebra
using SparseArrays
using JLD2
using HDF5
using Plots
using LaTeXStrings
using Zygote
using Optimization
using OptimizationOptimJL
using Combinatorics

if !isdefined(Main, :UtilityFunctions)
    include("utility_functions.jl")
end
using .UtilityFunctions
include("trotter.jl")
using .Trotter
include("logging.jl")
include("nn_strategy.jl")

include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")

# ═══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ═══════════════════════════════════════════════════════════════════════════════

"""
    parse_arguments(args) -> (folder, trotter_orders, n_up, n_dn, lvec)

Parse command-line arguments for the trotter_exp_testing script.

Returns:
  - folder         (String)        : path to the ED data folder.
  - trotter_orders (Vector{Int})   : Trotter repetition counts to sweep over.
  - n_up           (Int)           : number of spin-up electrons.
  - n_dn           (Int)           : number of spin-down electrons.
  - lvec           (Vector{Int})   : lattice dimensions [W, H].
  - output         (String)        : output filename
"""
function parse_arguments(args::Vector{String})
    folder = "data/N=(2, 2)_3x2"
    output = "trotter_order_comparison"
    trotter_orders = [1, 2, 4, 8]
    n_up = 4
    n_dn = 4
    lvec = [3, 3]
    antihermitian = nothing
    loss_type = :overlap
    custom_ref_state_arg = nothing
    positional = String[]

    for arg in args
        if startswith(arg, "--trotter_orders=")
            val = split(arg, "=", limit=2)[2]
            trotter_orders = [parse(Int, s) for s in split(val, ",")]
        elseif startswith(arg, "--n_up=")
            n_up = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--n_dn=")
            n_dn = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--lvec=")
            parts = split(split(arg, "=", limit=2)[2], "x")
            length(parts) == 2 || error("--lvec must be in the form WxH, e.g. 3x3")
            lvec = [parse(Int, parts[1]), parse(Int, parts[2])]
        elseif startswith(arg, "--output")
            output = String(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--antihermitian")
            if occursin("=", arg)
                antihermitian = parse(Bool, split(arg, "=", limit=2)[2])
            else
                antihermitian = true
            end
        elseif startswith(arg, "--loss=")
            val = String(split(arg, "=", limit=2)[2])
            if val == "overlap"
                loss_type = :overlap
            elseif val == "energy"
                loss_type = :energy
            else
                error("Invalid --loss option: '$val'. Valid options are: 'overlap', 'energy'.")
            end
        elseif startswith(arg, "--custom_ref_state=")
            custom_ref_state_arg = String(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--")
            error("Unknown option: $arg")
        else
            push!(positional, arg)
        end
    end

    if length(positional) >= 1
        folder = positional[1]
    end

    return folder, trotter_orders, n_up, n_dn, lvec, output, antihermitian, loss_type, custom_ref_state_arg
end

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
function load_exact_exp_coefficients(folder::String, prefix::String, u_i::Int)
    fpath = joinpath(folder, "$(prefix)_u_$(u_i).jld2")
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
    get_clean_coords(c) -> Tuple

Robustly extract the flat coordinates tuple from a Lattices.Coordinate object,
supporting both in-memory and JLD2-reconstructed structures.
"""
function get_clean_coords(c)
    coords = c.coordinates
    if !isempty(coords) && coords[1] isa Tuple
        return coords[1]
    else
        return coords
    end
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
    N_sites = prod(lvec)
    # Reconstruct basis_sector from indexer
    function coord_to_site_idx(coord, Lvec)
        c0 = get_clean_coords(coord) .- 1
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

    # Determine the present orders from t_keys_JLD2
    present_orders = Int[]
    for k in t_keys_JLD2
        ord = div(length(k), 2)
        if ord ∉ present_orders
            push!(present_orders, ord)
        end
    end
    sort!(present_orders)

    # Build exact-exp keys structures (without pre-computing large matrix lists)
    operator_cache = Dict{Int,Dict{Symbol,Any}}()
    order_structures = Dict{Int,Any}()
    t_keys_exact = []

    flat_to_order_and_idx = Dict{Int,Tuple{Int,Int}}()
    flat_idx = 1

    for ord in present_orders
        struct_data = ensure_operator_structure!(ord, operator_cache, indexer, true, false, true, sign_convention, Dict(), antihermitian, 1.0)
        order_structures[ord] = struct_data
        t_keys_exact_order = struct_data[:t_keys]
        append!(t_keys_exact, t_keys_exact_order)

        for local_idx in 1:length(t_keys_exact_order)
            flat_to_order_and_idx[flat_idx] = (ord, local_idx)
            flat_idx += 1
        end
    end

    # Helper to convert key to canonical
    function key_to_canonical(k)
        [(get_clean_coords(c), spin, op) for (c, spin, op) in k]
    end
    function conjugate_canonical(ck)
        conj_ops = [(c, spin, op == :create ? :annihilate : :create) for (c, spin, op) in ck]
        cre = sort(filter(op -> op[3] == :create, conj_ops), by=op -> (op[1], op[2]))
        ann = sort(filter(op -> op[3] == :annihilate, conj_ops), by=op -> (op[1], op[2]))
        return [cre; ann]
    end

    # Lookups
    canon_keys_JLD2 = [key_to_canonical(k) for k in t_keys_JLD2]
    canon_to_idx_JLD2 = Dict(k => idx for (idx, k) in enumerate(canon_keys_JLD2))

    canon_keys_exact = [key_to_canonical(k) for k in t_keys_exact]
    canon_to_idx_exact = Dict(k => idx for (idx, k) in enumerate(canon_keys_exact))

    A_reordered = zeros(Float64, length(gates))

    for (g_idx, g) in enumerate(gates)
        lbl = fgate_to_label(g, lvec)
        ck = key_to_canonical(lbl)

        # Look up in JLD2 keys
        idx_JLD2 = get(canon_to_idx_JLD2, ck, 0)
        if idx_JLD2 == 0
            idx_JLD2 = get(canon_to_idx_JLD2, conjugate_canonical(ck), 0)
        end
        if idx_JLD2 == 0
            # If the gate is not present in JLD2, it was not part of the optimized ansatz, so its coefficient is 0.0
            A_reordered[g_idx] = 0.0
            continue
        end

        # Look up in exact struct_data keys
        idx_exact = get(canon_to_idx_exact, ck, 0)
        if idx_exact == 0
            idx_exact = get(canon_to_idx_exact, conjugate_canonical(ck), 0)
        end
        if idx_exact == 0
            error("Gate $g_idx (label $ck) has no matching key in exact-exp t_keys (struct_data).")
        end

        # Build E_mat on-the-fly
        ord, i = flat_to_order_and_idx[idx_exact]
        struct_data = order_structures[ord]
        P = length(struct_data[:t_keys])
        t_l = zeros(Float64, P)
        t_l[i] = 1.0
        vals = update_values(struct_data[:signs], struct_data[:param_index_map], t_l, struct_data[:parameter_mapping], struct_data[:parity])
        mat_l = sparse(struct_data[:rows], struct_data[:cols], vals, length(basis_sector), length(basis_sector))
        if antihermitian
            mat_l = make_antihermitian(mat_l)
        else
            mat_l = make_hermitian(mat_l)
        end
        E_mat = Matrix(real(mat_l))

        # Build T_mat on-the-fly
        T_mat = real(Matrix(Trotter.TamFermion.tau_g_operator_sector(g, N_sites, basis_sector; antihermitian=antihermitian)))

        nz_indices = findall(x -> abs(x) > 1e-5, T_mat)
        if isempty(nz_indices)
            A_reordered[g_idx] = 0.0
            continue
        end

        r, c = nz_indices[1].I
        f = E_mat[r, c] / T_mat[r, c]

        A_reordered[g_idx] = A_exact[idx_JLD2] * f
    end

    return A_reordered
end

"""
    derive_q_target(folder, inst, lvec) -> Int

Derive the momentum-sector index q_target from the Trotter shared file's
instructions dict, or fall back to scanning the HubbardED HDF5 file for the
sector with the lowest ground-state energy. Returns 0 (Gamma point) if neither
source is available.
"""
function derive_q_target(folder::String, inst::Dict, lvec::Vector{Int})
    # Try instructions dict first
    q = get(inst, "q_target", nothing)
    !isnothing(q) && return q

    # Fall back to HDF5 file
    h5_files = [f for f in readdir(folder) if occursin("HubbardED", f)]
    if !isempty(h5_files)
        h5_file = joinpath(folder, h5_files[1])
        return h5open(h5_file, "r") do data
            key_labels = [parse(Int, k) for k in keys(data["data/energies"])]
            all_E = [real.(read(data, "data/energies/$(k)"))[:, 1] for k in key_labels]
            k_min = key_labels[argmin([minimum(e) for e in all_E])]
            kvecs = read(data, "metadata/kvecs")
            k_tuple = tuple((kvecs[:, k_min+1] .+ 1)...)
            Trotter.ravel_c(k_tuple .- 1, Tuple(lvec))
        end
    end

    @warn "Could not determine q_target; defaulting to 0 (Gamma point)"
    return 0
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
    compute_exact_exp_energies(exact_coeffs_raw, t_keys_JLD2, indexer, lvec, basis_ints, ref_state, H_hop_mom, H_int_mom, U_values) -> Vector{Float64}

Compute the actual Hamiltonian energy of the state prepared by the exact coefficients:
    ψ_exact = exp(i M) |ref_state⟩
for each U value.
"""
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
    n_U = length(U_values)
    energies = fill(NaN, n_U)

    # 1. Reconstruct basis_sector from indexer
    N_sites = prod(lvec)
    function coord_to_site_idx(coord, Lvec)
        c0 = get_clean_coords(coord) .- 1
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

    # Build state mappings
    state_to_idx = Dict(val => idx for (idx, val) in enumerate(basis_ints))
    perm = [state_to_idx[val] for val in basis_sector]
    inv_perm = invperm(perm)

    # Helpers
    function key_to_canonical(k)
        [(get_clean_coords(c), spin, op) for (c, spin, op) in k]
    end
    function conjugate_canonical(ck)
        conj_ops = [(c, spin, op == :create ? :annihilate : :create) for (c, spin, op) in ck]
        cre = sort(filter(op -> op[3] == :create, conj_ops), by=op -> (op[1], op[2]))
        ann = sort(filter(op -> op[3] == :annihilate, conj_ops), by=op -> (op[1], op[2]))
        return [cre; ann]
    end

    # Map JLD2 keys
    canon_keys_JLD2 = [key_to_canonical(k) for k in t_keys_JLD2]
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
        order_structures[ord] = ensure_operator_structure!(ord, operator_cache, indexer, true, false, true, sign_convention, Dict(), antihermitian, 1.0)
    end

    Threads.@threads for u_i in 1:n_U
        coeffs = exact_coeffs_raw[u_i]
        if isnothing(coeffs)
            if u_i == 1
                H_u = H_hop_mom + U_values[1] * H_int_mom
                energies[1] = real(dot(ref_state, H_u * ref_state))
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
                ck = key_to_canonical(key_exact)
                idx_JLD2 = get(canon_to_idx_JLD2, ck, 0)
                if idx_JLD2 == 0
                    idx_JLD2 = get(canon_to_idx_JLD2, conjugate_canonical(ck), 0)
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
                psi_exact_sector = exp(Matrix(mat_l_order)) * psi_exact_sector
            else
                mat_l_order = make_hermitian(mat_l_order)
                psi_exact_sector = exp(1im * Matrix(mat_l_order)) * psi_exact_sector
            end
        end

        # Map back to the basis_ints basis
        psi_exact = psi_exact_sector[inv_perm]

        # Compute energy
        H_u = H_hop_mom + U_values[u_i] * H_int_mom
        energies[u_i] = real(dot(psi_exact, H_u * psi_exact))
    end

    return energies
end

"""
    compute_trotterized_energies(exact_coeffs, trotter_orders, gates, ref_state,
                                  basis_ints, N_sites, H_hop, H_int, U_values, num_gates)
        -> Dict{Int, Vector{Float64}}

For each Trotter order P in trotter_orders, approximate the exact exponential by
repeating the stored coefficient vector P times and dividing each copy by P, then
apply via apply_unitary with num_exponentials = P * stored_num_exp.

This approximates exp(A·G) ≈ [exp((A/P)·G)]^P. In the limit of large P and
commuting gates this converges to the exact exponential.

Returns a Dict mapping P -> energy vector over U_values.
"""
function compute_trotterized_energies(
    exact_coeffs::Vector, trotter_orders::Vector{Int},
    gates, ref_state::AbstractVector, basis_ints::AbstractVector,
    N_sites::Int, H_hop, H_int, U_values::Vector{Float64}, num_gates::Int;
    antihermitian::Bool=false
)
    n_U = length(U_values)
    result = Dict{Int,Vector{Float64}}()

    for P in trotter_orders
        energies_P = fill(NaN, n_U)
        Threads.@threads for u_i in 1:n_U
            A_base = exact_coeffs[u_i]
            isnothing(A_base) && continue
            length(A_base) % num_gates == 0 || continue
            stored_num_exp = length(A_base) ÷ num_gates
            # Each coefficient copy is scaled down by 1/P; P copies applied in sequence
            A_trotter = repeat(A_base, P) ./ P
            total_exp = P * stored_num_exp
            psi = TrotterOptimization.apply_unitary(
                A_trotter, gates, ref_state, basis_ints, N_sites, total_exp;
                antihermitian=antihermitian
            )
            H_u = H_hop + U_values[u_i] * H_int
            energies_P[u_i] = compute_energy(psi, H_u)
        end
        result[P] = energies_P
        println("  P=$P: $(sum(!isnan, energies_P)) non-NaN values")
    end

    return result
end

"""
    construct_custom_ref_state(custom_ref_state_arg::Union{String,Nothing}, folder::String, H_dim::Int, U_values::Vector{Float64})

Construct a custom Slater determinant reference state vector of length `H_dim` if requested by the command line argument.
Returns a `Vector{ComplexF64}` or `nothing`.
"""
function construct_custom_ref_state(custom_ref_state_arg::Union{String,Nothing}, folder::String, H_dim::Int, U_values::Vector{Float64})
    if isnothing(custom_ref_state_arg)
        return nothing
    end

    local slater_idx
    if custom_ref_state_arg == "slater"
        jld2_path = joinpath(folder, "meta_data_and_E.jld2")
        if isfile(jld2_path)
            println("Finding Slater ground state index from JLD2 file...")
            dic = load_saved_dict(jld2_path)
            all_E = dic["E"]
            U_values_for_sector = dic["meta_data"]["U_values"]
            k_min = find_best_energy_sector(all_E, U_values_for_sector; data=dic)
            slater_idx = get_slater_ground_state(dic, k_min)
        else
            println("Finding Slater ground state index from HDF5 file...")
            valid_files = [f for f in readdir(folder) if occursin("HubbardED", f)]
            if isempty(valid_files)
                error("No HubbardED HDF5 file found in folder: $folder")
            end
            h5_file = joinpath(folder, valid_files[1])
            slater_idx = h5open(h5_file, "r") do data
                key_labels = [parse(Int, k) for k in keys(data["data/energies"])]
                all_E = [real.(read(data, "data/energies/$(k)"))[:, 1] for k in key_labels]
                k_min = find_best_energy_sector(all_E, U_values; labels=key_labels)
                return get_slater_ground_state(data, k_min)
            end
        end
        println("Slater ground state index found: $slater_idx")
    else
        try
            slater_idx = parse(Int, custom_ref_state_arg)
        catch e
            error("Invalid --custom_ref_state value: '$custom_ref_state_arg'. Must be 'slater' or an integer index.")
        end
        if slater_idx < 1 || slater_idx > H_dim
            error("Parsed Slater index $slater_idx is out of bounds (1 to $H_dim).")
        end
        println("Using user-specified Slater index: $slater_idx")
    end

    if slater_idx == -1
        error("No Slater ground state could be found in the current sector.")
    end

    custom_ref = zeros(ComplexF64, H_dim)
    custom_ref[slater_idx] = 1.0
    return custom_ref
end

"""
    load_trotter_opt_energies(folder, N_sites, n_U; custom_ref_state_arg, antihermitian, loss_type) -> Vector{Float64}

Load the final overlap-optimized Trotter energies from trotter_N=<N_sites>_u_<i>.jld2 files.
"""
function load_trotter_opt_energies(
    folder::String, N_sites::Int, n_U::Int;
    custom_ref_state_arg::Union{String,Nothing}=nothing,
    antihermitian::Bool=false,
    loss_type::Symbol=:overlap,
)
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
    for u_i in 1:n_U
        fpath = joinpath(folder, "$(prefix)_u_$(u_i).jld2")
        if isfile(fpath)
            d = load(fpath)["dict"]
            met = d["metrics"]
            if haskey(met, "energy") && !isempty(met["energy"])
                energies[u_i] = met["energy"][end]
            elseif haskey(met, "loss") && !isempty(met["loss"])
                energies[u_i] = met["loss"][end]
            end
        end
    end
    return energies
end

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

"""
    build_comparison_plot(U_values, gs_energies, exact_exp_energies,
                          trotter_energies, trotter_opt_energies,
                          trotter_orders, n_up, n_dn, lvec; custom_ref_state_arg, loss_type) -> Plot
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
)
    W, H = lvec
    ref_name = isnothing(custom_ref_state_arg) ? "GS(U=0)" : custom_ref_state_arg
    p = plot(
        xlabel=L"U",
        ylabel="Energy",
        title="Hamiltonian energy vs Trotterization order\n($(W)×$(H) lattice, N↑=$n_up, N↓=$n_dn, ref=$ref_name, loss=$(loss_type))",
        legend=:outerright,
        size=(1000, 550),
        xlim=(0, 10),
        yticks=10.0 .^ (-16:2),
        margin=5Plots.mm,
        yscale=:log10,
    )

    valid = .!isnan.(exact_exp_energies)
    plot!(p, U_values[valid], max.(exact_exp_energies[valid] .- gs_energies[valid], 1e-15);
        label="Exact exp ($(loss_type)-opt coeffs, applied as-is)",
        color=:blue,
        lw=2,
        marker=:circle,
        markersize=4,
    )

    trotter_palette = [:red, :orange, :purple, :darkgreen, :magenta, :brown]
    for (idx, P) in enumerate(trotter_orders)
        println("trotter order $idx")
        energies_P = trotter_energies[P]
        valid = .!isnan.(energies_P)
        plot!(p, U_values[valid], max.(energies_P[valid] .- gs_energies[valid], 1e-15);
            label="Trotterized P=$P (coeffs/P, repeated P×)",
            color=trotter_palette[mod1(idx, length(trotter_palette))],
            lw=1.5,
            marker=:square,
            markersize=3,
        )
    end

    valid = .!isnan.(trotter_opt_energies)
    plot!(p, U_values[valid], max.(trotter_opt_energies[valid] .- gs_energies[valid], 1e-15);
        label="Trotter opt ($(loss_type)-optimized)",
        color=:cyan,
        lw=2,
        marker=:diamond,
        markersize=4,
    )

    return p
end

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "trotter_exp_testing")
    with_logging(log_path) do
        folder, trotter_orders, n_up, n_dn, lvec, output_file, antihermitian_arg, loss_type, custom_ref_state_arg = parse_arguments(ARGS)

        println("=== Trotter experiment testing ===")
        println("  folder         = $folder")
        println("  lvec           = $lvec")
        println("  (n_up, n_dn)   = ($n_up, $n_dn)")
        println("  trotter_orders = $trotter_orders")
        println("  loss_type      = $loss_type")
        println("  custom_ref     = $custom_ref_state_arg")
        println()

        N_sites = prod(lvec)

        # ── 1. Load U_values ───────────────────────────────────────────────────
        meta = load(joinpath(folder, "meta_data_and_E.jld2"))["dict"]
        U_values = meta["meta_data"]["U_values"]
        n_U = length(U_values)
        println("Loaded U_values: $n_U entries, range $(U_values[1]) to $(U_values[end])")

        # Find best energy sector to get indexer
        k_min = find_best_energy_sector(meta["E"], U_values; verbose=false)
        indexer = meta["indexer"]
        if indexer isa Vector
            indexer = indexer[k_min]
        end

        # ── 2. Derive momentum sector and build Hamiltonians ───────────────────
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

        trotter_prefix = "trotter_N=$N_sites"
        if !isnothing(custom_ref_state_arg)
            trotter_prefix *= "_ref_$(custom_ref_state_arg)"
        end
        if antihermitian
            trotter_prefix *= "_antihermitian"
        end
        if loss_type == :energy
            trotter_prefix *= "_loss_energy"
        end

        shared_trotter_path = joinpath(folder, "$(trotter_prefix)_shared.jld2")
        println("Using antihermitian = $antihermitian")
        println("Loading shared trotter info from $shared_trotter_path")

        shared_trotter = load(shared_trotter_path)["dict"]

        q_target = derive_q_target(folder, shared_trotter["instructions"], lvec)
        println("q_target = $q_target")

        println("\nBuilding sector Hamiltonians...")
        @time H_hop_mom, basis_dict, _ = TamFermion.HubbardMomentumBasis(
            1.0, 0.0, lvec, (n_up, n_dn); q_target=q_target,
        )
        @time H_int_mom, _, _ = TamFermion.HubbardMomentumBasis(
            0.0, 1.0, lvec, (n_up, n_dn); q_target=q_target,
        )
        basis_ints = basis_dict["ints"]
        println("Hilbert space sector dim = $(length(basis_ints))")

        # ── 3. Enumerate gates ─────────────────────────────────────────────────
        println("\nEnumerating Trotter gates...")
        @time gates = TamFermion.enumerate_ferm_excitations(
            2, lvec; conserve_mom=true, conserve_sz=true, include_diagonal=true,
        )
        @time tau_terms = TamFermion.fgateToTauSector(gates, N_sites, basis_ints; antihermitian=antihermitian)
        num_gates = length(gates)
        println("num_gates = $num_gates")

        # ── 4. Load exact-exp coefficients and expand to Trotter gate basis ──
        unitary_prefix = "unitary_map_energy_symmetry=false_N=($n_up, $n_dn)"
        if !isnothing(custom_ref_state_arg)
            unitary_prefix *= "_ref_$(custom_ref_state_arg)"
        end
        if loss_type == :energy
            unitary_prefix *= "_loss_energy"
        end

        exact_coeffs_raw = [
            load_exact_exp_coefficients(folder, unitary_prefix, u_i) for u_i in 1:n_U
        ]
        n_loaded = sum(!isnothing, exact_coeffs_raw)
        println("\nLoaded exact-exp coefficients for $n_loaded / $n_U U values")
        if n_loaded > 0
            first_nz = findfirst(!isnothing, exact_coeffs_raw)
            println("  First non-nothing: u_i=$first_nz, length=$(length(exact_coeffs_raw[first_nz]))")
        end

        # Load the operator keys used by the exact-exp optimization and reorder
        # the coefficients to match the Trotter gate ordering.
        t_keys = load_exact_exp_keys(folder, unitary_prefix)
        sign_convention = :coordinate_first
        println("  Using sign convention: $sign_convention")

        exact_coeffs = if !isnothing(t_keys) && n_loaded > 0
            println("  Operator keys loaded: $(length(t_keys)). Reordering to $num_gates gate basis...")
            map(exact_coeffs_raw) do A
                isnothing(A) ? nothing : reorder_exact_to_trotter_coeffs(A, t_keys, gates, lvec, indexer, basis_ints; antihermitian=antihermitian, sign_convention=sign_convention)
            end
        else
            @warn "Could not load operator keys from shared file; exact-exp evaluation skipped."
            fill(nothing, n_U)
        end

        # ── 5. Compute sector ground states via eigen ──────────────────────────
        println("\nComputing sector ground states ($n_U U values)...")
        gs_states, gs_energies = compute_gs_states(H_hop_mom, H_int_mom, U_values)
        println("gs energies; $gs_energies")
        println("  Done")

        # Determine reference state
        local ref_state
        if !isnothing(custom_ref_state_arg)
            H_dim = length(basis_ints)
            ref_state = construct_custom_ref_state(custom_ref_state_arg, folder, H_dim, U_values)
        else
            ref_state = gs_states[1]
        end

        # ── 6. Exact-exp energies ─────────────────────────────────────────────
        println("\nComputing exact-exp energies...")
        exact_exp_energies = if !isnothing(t_keys) && n_loaded > 0
            compute_exact_exp_energies(
                exact_coeffs_raw, t_keys, indexer, lvec, basis_ints, ref_state,
                H_hop_mom, H_int_mom, U_values; antihermitian=antihermitian, sign_convention=sign_convention
            )
        else
            fill(NaN, n_U)
        end
        println("  Done ($(sum(!isnan, exact_exp_energies)) non-NaN values)")

        if all(isnan, exact_exp_energies)
            error("No exact exponential data found (all exact-exp energies are NaN). The comparison is meaningless without exact exponential data.")
        end

        # ── 7. Trotterized energies for each P ────────────────────────────────
        println("\nComputing trotterized energies for orders P = $trotter_orders...")
        trotter_energies = compute_trotterized_energies(
            exact_coeffs, trotter_orders, gates, ref_state, basis_ints,
            N_sites, H_hop_mom, H_int_mom, U_values, num_gates; antihermitian=antihermitian
        )

        # ── 8. Trotter-optimized energies ─────────────────────────────────────
        println("\nLoading trotter-optimized (energy-loss) results...")
        trotter_opt_energies = load_trotter_opt_energies(
            folder, N_sites, n_U;
            custom_ref_state_arg=custom_ref_state_arg,
            antihermitian=antihermitian,
            loss_type=loss_type
        )
        println("trotter energies: $trotter_opt_energies")
        println("Diff energies: $(trotter_opt_energies - gs_energies)")
        println("  Loaded $(sum(!isnan, trotter_opt_energies)) values")

        # ── 9. Plot and save ──────────────────────────────────────────────────
        println("\nBuilding and saving plot...")
        p = build_comparison_plot(
            U_values, gs_energies, exact_exp_energies,
            trotter_energies, trotter_opt_energies,
            trotter_orders, n_up, n_dn, lvec;
            custom_ref_state_arg=custom_ref_state_arg,
            loss_type=loss_type
        )

        out_path = joinpath(folder, "$output_file.png")
        savefig(p, out_path)
        println("  Saved → $out_path")

        return 0
    end
end
