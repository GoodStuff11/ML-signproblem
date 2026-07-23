#=
trotter_exp_testing.jl

Compare Hamiltonian energies across three methods:
  1. Exact exponential  – stored coefficients from unitary_map_energy_symmetry=false files
                          applied as a single unitary (assumes energy optimization).
  2. Trotterized        – same stored coefficients repeated P times, each copy divided
                          by P (approximating the exact exponential at increasing order P).
  3. Trotter-optimized  – overlap-optimized Trotter coefficients from
                          trotter_N=<N_sites>_u files.

Usage:
  julia --project=.. trotter_exp_testing.jl [folders...] [options]

Arguments/Options:
  folders (positional, optional): Zero, one, or more paths to ED data folders (or system size suffixes like "3x2", "3x3").
                                 - If zero are provided: defaults to "N=(2, 2)_3x2".
                                 - If one is provided and no --u option is specified: runs a U-value sweep analysis
                                   for that single system size, saving the plot to <folder>/<output_file>.png.
                                 - If multiple are provided (or one folder with the --u option is specified): runs a
                                   system-size comparison analysis at the specified U value.
                                   Saves the plot to <first_folder>/<output_file>_system_size.png.

  --u=<float> or --U=<float> (optional): Specify a single U value to perform system size comparison at.
                                         This argument is REQUIRED if multiple folders/system sizes are specified.
                                         If multiple system sizes are inputted but this option is missing, the script
                                         raises an error.

  --trotter_orders=<list> (optional): Comma-separated list of positive integers
                     specifying the Trotterization repetition counts P to compare.
                     For each P the exact-exp coefficient vector is repeated P times
                     and divided by P before applying the unitary.
                     Default: "1,2,4,8".

  --n_up=<int> (optional): Number of spin-up electrons. Default: 4.

  --n_dn=<int> (optional): Number of spin-down electrons. Default: 4.

  --lvec=<WxH> (optional): Lattice dimensions in the format WxH (e.g. "3x3").
                     Default: "3x3".

  --output=<string> (optional): Name of png to output (will have .png or _system_size.png appended). Default: trotter_order_comparison

  --antihermitian (optional): Whether to use antihermitian operators. Can be true/false or --antihermitian. Default: auto-detect from shared.jld2 files.

  --loss=<string> (optional): Loss type. Valid options are:
                              - "overlap": overlap-optimized loss
                              - "energy": energy-optimized loss
                              Default: "overlap".

  --custom_ref_state=<string> (optional): Custom reference state label. Default: nothing (use Slater determinant).

Examples:
  1. Sweep U values for a single system:
     julia --project=.. trotter_exp_testing.jl
     julia --project=.. trotter_exp_testing.jl "N=(2, 2)_3x2"

  2. Compare trotterized energies across system sizes at a single U value:
     julia --project=.. trotter_exp_testing.jl "N=(2, 2)_3x2" "N=(3, 3)_3x3" --u=2.0

  3. System size comparison specifying short system size suffixes:
     julia --project=.. trotter_exp_testing.jl 3x2 3x3 --u=4.0 --trotter_orders=1,2,4
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

include("utility_functions.jl")
using .UtilityFunctions
include("trotter.jl")
using .Trotter
include("data_path.jl")
include("logging.jl")
include("nn_strategy.jl")

include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")

# ═══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ═══════════════════════════════════════════════════════════════════════════════


"""
    resolve_system_folder(input::String) -> String

Resolves the positional input to a full directory path inside get_data_root().
Supports absolute paths, folders within the data root, or strings that end with or match system names.
"""
function resolve_system_folder(input::String)
    if isabspath(input) && isdir(input)
        return input
    end
    root = get_data_root()
    p1 = joinpath(root, input)
    if isdir(p1)
        return p1
    end
    # Search for matching folders
    for item in readdir(root)
        if isdir(joinpath(root, item))
            if item == input || endswith(item, "_" * input)
                return joinpath(root, item)
            end
        end
    end
    error("Could not resolve system size / folder: '$input' in data root '$root'")
end

"""
    parse_arguments(args::Vector{String}) -> (folders, trotter_orders, n_up, n_dn, lvec, output, antihermitian, loss_type, custom_ref_state_arg, u_val)

Parse command-line arguments for the trotter_exp_testing script.

Returns:
  - folders        (Vector{String}) : resolved paths to ED data folders.
  - trotter_orders (Vector{Int})    : Trotter repetition counts to sweep over.
  - n_up           (Int)            : number of spin-up electrons.
  - n_dn           (Int)            : number of spin-down electrons.
  - lvec           (Vector{Int})    : lattice dimensions [W, H].
  - output         (String)         : output filename
  - antihermitian  (Union{Bool,Nothing}) : antihermitian flag or nothing
  - loss_type      (Symbol)         : loss type (:overlap or :energy)
  - custom_ref_state_arg (Union{String,Nothing}) : custom reference state name or nothing
  - u_val          (Union{Float64,Nothing}) : specified U value or nothing
"""
function parse_arguments(args::Vector{String})
    output = "trotter_order_comparison"
    trotter_orders = [1, 2, 4, 8]
    n_up = 4
    n_dn = 4
    lvec = [3, 3]
    antihermitian = nothing
    loss_type = :overlap
    custom_ref_state_arg = nothing
    u_val = nothing
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
        elseif startswith(arg, "--u=") || startswith(arg, "--U=")
            u_val = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--")
            error("Unknown option: $arg")
        else
            push!(positional, arg)
        end
    end

    folders = String[]
    if isempty(positional)
        push!(folders, resolve_system_folder("N=(2, 2)_3x2"))
    else
        for pos in positional
            push!(folders, resolve_system_folder(pos))
        end
    end

    if length(folders) > 1 && isnothing(u_val)
        error("If multiple system sizes are specified, a single U value must be provided via the --u parameter (e.g. --u=2.0).")
    end

    return folders, trotter_orders, n_up, n_dn, lvec, output, antihermitian, loss_type, custom_ref_state_arg, u_val
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
    basis_sector = Trotter.get_basis_sector(indexer, lvec, N_sites)

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

    # Lookups
    canon_keys_JLD2 = [Trotter.key_to_canonical(k) for k in t_keys_JLD2]
    canon_to_idx_JLD2 = Dict(k => idx for (idx, k) in enumerate(canon_keys_JLD2))

    canon_keys_exact = [Trotter.key_to_canonical(k) for k in t_keys_exact]
    canon_to_idx_exact = Dict(k => idx for (idx, k) in enumerate(canon_keys_exact))

    A_reordered = zeros(Float64, length(gates))

    for (g_idx, g) in enumerate(gates)
        lbl = fgate_to_label(g, lvec)
        ck = Trotter.key_to_canonical(lbl)

        # Look up in JLD2 keys
        idx_JLD2 = get(canon_to_idx_JLD2, ck, 0)
        if idx_JLD2 == 0
            idx_JLD2 = get(canon_to_idx_JLD2, Trotter.conjugate_canonical(ck), 0)
        end
        if idx_JLD2 == 0
            # If the gate is not present in JLD2, it was not part of the optimized ansatz, so its coefficient is 0.0
            A_reordered[g_idx] = 0.0
            continue
        end

        # Look up in exact struct_data keys
        idx_exact = get(canon_to_idx_exact, ck, 0)
        if idx_exact == 0
            idx_exact = get(canon_to_idx_exact, Trotter.conjugate_canonical(ck), 0)
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
    loss_type::Symbol
)
    U_values, target_vecs, indexer, _, N_elec, _, _, sign_convention =
        load_ED_data(folder; verbose=true, use_slater_reference=antihermitian_arg == "slater")
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
        println("Subspace does not conserve momentum. Constructing Hamiltonians in coordinate basis (:spin_first)...")
        basis_ints = Trotter.get_basis_sector(indexer, dims_val, N_sites)
        lattice = Square(Tuple(dims_val), Periodic())
        subspace = HubbardSubspace(n_up_loaded, n_dn_loaded, lattice; k=nothing)
        H_hop_mom, H_int_mom = create_hubbard_matrices(subspace; indexer=indexer, sign_convention=:spin_first)
    else
        println("\nBuilding sector Hamiltonians in momentum basis...")
        H_hop_mom, basis_dict, _ = TamFermion.HubbardMomentumBasis(
            1.0, 0.0, dims_val, (n_up_loaded, n_dn_loaded); q_target=q_target,
        )
        H_int_mom, _, _ = TamFermion.HubbardMomentumBasis(
            0.0, 1.0, dims_val, (n_up_loaded, n_dn_loaded); q_target=q_target,
        )
        basis_ints = basis_dict["ints"]
    end

    println("Hilbert space sector dim = $(length(basis_ints))")

    println("\nEnumerating Trotter gates...")
    gates = TamFermion.enumerate_ferm_excitations(
        2, dims_val; conserve_mom=true, conserve_sz=true, include_diagonal=true,
    )
    tau_terms = TamFermion.fgateToTauSector(gates, N_sites, basis_ints; antihermitian=antihermitian)
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
        load_exact_exp_coefficients(folder, unitary_prefix, u_i) for u_i in 1:n_U
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
        println("  Operator keys loaded: $(length(t_keys)). Reordering to $num_gates gate basis...")
        map(exact_coeffs_raw) do A
            isnothing(A) ? nothing : reorder_exact_to_trotter_coeffs(A, t_keys, gates, lvec, indexer, basis_ints; antihermitian=antihermitian, sign_convention=sign_convention)
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
    gs_states = [ComplexF64.(vec(target_vecs[u_i+1, :])) for u_i in 1:n_U]
    gs_energies = [compute_energy(gs_states[u_i], H_hop_mom + U_values[u_i] * H_int_mom) for u_i in 1:n_U]
    println("gs energies: $gs_energies")

    ref_state = ComplexF64.(vec(target_vecs[1, :]))

    # 4. Compute exact-exp energies
    println("\nComputing exact-exp energies...")
    exact_exp_energies = if !isnothing(t_keys) && sum(!isnothing, exact_coeffs_raw) > 0
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

    # 5. Compute trotterized energies
    println("\nComputing trotterized energies for orders P = $trotter_orders...")
    trotter_energies = compute_trotterized_energies(
        exact_coeffs, trotter_orders, gates, ref_state, basis_ints,
        N_sites, H_hop_mom, H_int_mom, U_values, num_gates; antihermitian=antihermitian
    )

    # 6. Load trotter-optimized energies
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

    return U_values, gs_energies, exact_exp_energies, trotter_energies, trotter_opt_energies, n_up_loaded, n_dn_loaded, lvec
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

        # 3. Load ground state and reference state
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
    build_system_size_comparison_plot(system_sizes, num_sites, gs_energies, exact_exp_energies, trotter_energies, trotter_opt_energies, trotter_orders, U_value, loss_type) -> Plot

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
    loss_type::Symbol
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

    p = plot(
        xlabel="System",
        ylabel="Energy Difference (E - E_gs)",
        title="Hamiltonian energy difference vs System Size\n(U = $U_value, loss=$(loss_type))",
        legend=:outerright,
        size=(1000, 550),
        xticks=(x_coords, sorted_sizes),
        yscale=:log10,
        margin=5Plots.mm
    )

    valid_exact = .!isnan.(sorted_exact)
    plot!(p, x_coords[valid_exact], max.(sorted_exact[valid_exact] .- sorted_gs[valid_exact], 1e-15);
        label="Exact exp ($(loss_type)-opt)",
        color=:blue,
        lw=2,
        marker=:circle,
        markersize=5
    )

    trotter_palette = [:red, :orange, :purple, :darkgreen, :magenta, :brown]
    for (idx, P) in enumerate(trotter_orders)
        energies_P = sorted_trotter[P]
        valid = .!isnan.(energies_P)
        plot!(p, x_coords[valid], max.(energies_P[valid] .- sorted_gs[valid], 1e-15);
            label="Trotterized P=$P",
            color=trotter_palette[mod1(idx, length(trotter_palette))],
            lw=1.5,
            marker=:square,
            markersize=4
        )
    end

    valid_opt = .!isnan.(sorted_opt)
    plot!(p, x_coords[valid_opt], max.(sorted_opt[valid_opt] .- sorted_gs[valid_opt], 1e-15);
        label="Trotter opt ($(loss_type)-optimized)",
        color=:cyan,
        lw=2,
        marker=:diamond,
        markersize=5
    )

    return p
end

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "trotter_exp_testing")
    with_logging(log_path) do
        folders, trotter_orders, n_up, n_dn, lvec, output_file, antihermitian, loss_type, custom_ref_state_arg, u_val = parse_arguments(ARGS)

        println("=== Trotter experiment testing ===")
        println("  folders        = $folders")
        println("  trotter_orders = $trotter_orders")
        println("  loss_type      = $loss_type")
        println("  custom_ref     = $custom_ref_state_arg")
        println("  u_val          = $u_val")
        println()

        if !isnothing(u_val)
            # Run system size computation for the given folders
            system_names, num_sites, gs_energies, exact_exp_energies, trotter_energies, trotter_opt_energies =
                compute_trotterized_energies_system_size(
                    folders, u_val, trotter_orders, custom_ref_state_arg, antihermitian, loss_type
                )

            # Plot the results
            println("\nBuilding and saving system size comparison plot...")
            p = build_system_size_comparison_plot(
                system_names, num_sites, gs_energies, exact_exp_energies,
                trotter_energies, trotter_opt_energies,
                trotter_orders, u_val, loss_type
            )

            # Save the plot in the first folder (or a default path)
            out_path = joinpath(folders[1], "$(output_file)_system_size.png")
            savefig(p, out_path)
            println("  Saved → $out_path")
        else
            # Run U sweep for the single folder
            U_values, gs_energies, exact_exp_energies, trotter_energies, trotter_opt_energies, n_up_loaded, n_dn_loaded, lvec =
                compute_trotterized_energies_u_sweep(
                    folders[1], trotter_orders, custom_ref_state_arg, antihermitian, loss_type
                )

            # Plot the results
            println("\nBuilding and saving plot...")
            p = build_comparison_plot(
                U_values, gs_energies, exact_exp_energies,
                trotter_energies, trotter_opt_energies,
                trotter_orders, n_up_loaded, n_dn_loaded, lvec;
                custom_ref_state_arg=custom_ref_state_arg,
                loss_type=loss_type
            )

            # Save the plot
            out_path = joinpath(folders[1], "$output_file.png")
            savefig(p, out_path)
            println("  Saved → $out_path")
        end

        return 0
    end
end
