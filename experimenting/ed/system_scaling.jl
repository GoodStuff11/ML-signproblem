using IJulia
using Lattices
using LinearAlgebra
using Combinatorics
using SparseArrays
using Plots
import Graphs
using LaTeXStrings
using Statistics
using Random
using Zygote
using Optimization, OptimizationOptimisers
using JSON
using OptimizationOptimJL
using JLD2
using ExponentialUtilities
ENV["JULIA_CUDA_USE_COMPAT"] = "false"
using CUDA
# using cuDNN   # Commented out to avoid CUDNNError on activation pullbacks on HPC; CUDA.jl handles MLPs natively and stably
using Printf
using Flux

include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("utility_functions.jl")
include("nn_strategy.jl")

"""
    load_data_coefficients(electrons::Tuple{Int, Int}, system_size::Vector{Int})

Loads the specified quantum many-body system data from JLD2 files based on the given
`electrons` count and `system_size` dimensions. Returns a Dict mapping each available
`u_idx` to its coefficient vector, a vector of corresponding state labels, the grid
dimensions of the system, and the full vector of U values.
"""
function load_data_coefficients(electrons, system_size)
    folder = "data/N=$(electrons)_$(system_size[1])x$(system_size[2])"
    # folder="data/tmp"

    e_metadata = load_saved_dict(joinpath(folder, "meta_data_and_E.jld2"))
    dim = [parse(Int, x) for x in split(e_metadata["meta_data"]["sites"], "x")]
    U_values = e_metadata["meta_data"]["U_values"]
    shared_data = load_saved_dict(joinpath(folder, "unitary_map_energy_symmetry=false_N=$(electrons)_shared.jld2"))
    labels = shared_data["coefficient_labels"][2]
    println(size(labels))

    # Load coefficients for every available u_idx
    coefficients_by_u = Dict{Int,Any}()
    for u_idx in eachindex(U_values)
        u_file = joinpath(folder, "unitary_map_energy_symmetry=false_N=$(electrons)_u_$(u_idx).jld2")
        if isfile(u_file)
            dic = load_saved_dict(u_file)
            val = dic["coefficients"][2]
            if isnothing(val)
                @warn "u_idx=$(u_idx): coefficients[2] is nothing, skipping"
            else
                coefficients_by_u[u_idx] = val
            end
        else
            @warn "Missing u_idx=$(u_idx) file: $u_file"
        end
    end

    return coefficients_by_u, labels, dim, U_values
end

# 5-point Gauss-Legendre quadrature nodes and weights for interval [-1, 1]
const GL_X = [
    0.0,
    1 / 3 * sqrt(5 - 2 * sqrt(10 / 7)),
    -1 / 3 * sqrt(5 - 2 * sqrt(10 / 7)),
    1 / 3 * sqrt(5 + 2 * sqrt(10 / 7)),
    -1 / 3 * sqrt(5 + 2 * sqrt(10 / 7))
]

const GL_W = [
    128 / 225,
    (322 + 13 * sqrt(70)) / 900,
    (322 + 13 * sqrt(70)) / 900,
    (322 - 13 * sqrt(70)) / 900,
    (322 - 13 * sqrt(70)) / 900
]

"""
    legendre(n::Int, x::Float64)

Computes the 1-dimensional Legendre polynomial of degree `n` iteratively evaluated at 
point `x` ∈ [-1, 1].
"""
function legendre(n::Int, x::Float64)
    if n == 0
        return 1.0
    end
    if n == 1
        return x
    end

    p0, p1 = 1.0, x
    for m in 1:(n-1)
        p0, p1 = p1, ((2m + 1) * x * p1 - m * p0) / (m + 1)
    end
    return p1
end

"""
    base_F(r1::Vector{Float64}, r2::Vector{Float64}, r3::Vector{Float64}, r4::Vector{Float64}, idxs::Vector{Int})

Computes the core multi-dimensional Legendre polynomial basis function by multiplying the 
1D Legendre polynomials for each component in the `d`-dimensional wavevectors `r1` to `r4`.
Inputs are mapped from [0, 2π] linearly onto the range [-1, 1] using `r_c / π - 1.0`.
"""
function base_F(r1::Vector{Float64}, r2::Vector{Float64}, r3::Vector{Float64}, r4::Vector{Float64}, idxs::Vector{Int})
    val = 1.0
    d = length(r1)
    for c in 1:d
        val *= legendre(idxs[c], r1[c] / pi - 1.0)
    end
    for c in 1:d
        val *= legendre(idxs[d+c], r2[c] / pi - 1.0)
    end
    for c in 1:d
        val *= legendre(idxs[2d+c], r3[c] / pi - 1.0)
    end
    for c in 1:d
        val *= legendre(idxs[3d+c], r4[c] / pi - 1.0)
    end
    return val
end

"""
    evaluate_basis_raw(r::Vector{Vector{Float64}}, sigma::Vector{Int}, idxs::Vector{Int})

Internal un-symmetrized raw basis function.
"""
function evaluate_basis(r::Vector{Vector{Float64}}, sigma::Vector{Int}, idxs::Vector{Int})
    d = length(r[1])
    is_ij_same = true
    is_kl_same = true
    for c in 1:d
        if idxs[c] != idxs[d+c]
            is_ij_same = false
        end
        if idxs[2d+c] != idxs[3d+c]
            is_kl_same = false
        end
    end


    # Group 1: The Fully Polarized States
    if (sigma == [1, 1, 1, 1]) || (sigma == [2, 2, 2, 2])
        # Must have i_idxs != j_idxs and k_idxs != l_idxs to be non-zero
        if is_ij_same || is_kl_same
            return 0.0
        end

        val1 = base_F(r[1], r[2], r[3], r[4], idxs) - base_F(r[2], r[1], r[3], r[4], idxs) -
               base_F(r[1], r[2], r[4], r[3], idxs) + base_F(r[2], r[1], r[4], r[3], idxs)

        val2 = base_F(r[3], r[4], r[1], r[2], idxs) - base_F(r[4], r[3], r[1], r[2], idxs) -
               base_F(r[3], r[4], r[2], r[1], idxs) + base_F(r[4], r[3], r[2], r[1], idxs)

        return val1 + val2

        # Group 2: The Mixed Spin States
    elseif sigma == [1, 2, 1, 2] || sigma == [2, 1, 2, 1]
        return base_F(r[1], r[2], r[3], r[4], idxs)
        +base_F(r[3], r[4], r[1], r[2], idxs)
        +base_F(r[2], r[1], r[4], r[3], idxs)
        +base_F(r[4], r[3], r[2], r[1], idxs)

    elseif sigma == [1, 2, 2, 1] || sigma == [2, 1, 1, 2]
        return -(base_F(r[1], r[2], r[4], r[3], idxs)
                 + base_F(r[4], r[3], r[1], r[2], idxs)
                 + base_F(r[2], r[1], r[3], r[4], idxs)
                 + base_F(r[3], r[4], r[2], r[1], idxs))

    else
        # Any spin configuration outside the specified 6 yields zero
        return 0.0
    end
end

# label_to_k is now defined in nn_strategy.jl

"""
    integrate_dimension_all_n!(dim_idx::Int, r::Vector{Vector{Float64}}, Ki::Vector{Vector{Float64}}, 
                               delta_k::Vector{Float64}, spins::Vector{Int}, basis_degrees::Vector{Vector{Int}}, 
                               d::Int, out::Vector{Float64}, weight::Float64)

Recursively computes a `3 * d` dimensional Gauss-Legendre quadrature integral bounding 
the unit volume hypercube defined by `Ki` and `delta_k`. This highly-optimized in-place
function concurrently integrates all `N` combined sets of polynomial `basis_degrees` and 
dumps the weighted density evaluations sequentially into `out`, avoiding memory allocations 
for maximum throughput. The wavevector `r_4` is dynamically calculated per quadrature point 
conserving spatial momentum: `k4 = k1 + k2 - k3`.
"""
function integrate_dimension_all_n!(dim_idx::Int, r::Vector{Vector{Float64}}, Ki::Vector{Vector{Float64}}, delta_k::Vector{Float64}, spins::Vector{Int}, basis_degrees::Vector{Vector{Int}}, d::Int, out::Vector{Float64}, weight::Float64)
    if dim_idx > 3 * d
        for c in 1:d
            r[4][c] = mod(r[1][c] + r[2][c] - r[3][c], 2 * pi)
        end
        for n in 1:length(basis_degrees)
            out[n] += weight * evaluate_basis(r, spins, basis_degrees[n])
        end
        return
    end

    r_idx = (dim_idx - 1) ÷ d + 1
    c_idx = (dim_idx - 1) % d + 1

    for (w, pt) in zip(GL_W, GL_X)
        k_val = Ki[r_idx][c_idx] + (delta_k[c_idx] / 2) * pt + (delta_k[c_idx] / 2)
        r[r_idx][c_idx] = k_val
        new_weight = weight * w * (delta_k[c_idx] / 2)
        integrate_dimension_all_n!(dim_idx + 1, r, Ki, delta_k, spins, basis_degrees, d, out, new_weight)
    end
end

"""
    build_P_matrix(labels::Vector, basis_degrees::Vector{Vector{Int}}, dim::Vector{Int})

Constructs the continuous projection matrix `P_{m,n}`, connecting discrete scattering `labels`
(`m`) to the parametrised multi-dimensional Legendre function definitions (`n`). It performs
the density volume element numeric integration evaluating every independent `basis_degrees`
against the label coordinates spanning hyper-volumes scaled by `dim`.
"""
function build_P_matrix(labels, basis_degrees::Vector{Vector{Int}}, dim::Vector{Int})
    N = length(labels)
    if length(basis_degrees) != N
        error("Number of basis degrees ($(length(basis_degrees))) must match number of labels ($N)")
    end

    P = zeros(Float64, N, N)
    K = [label_to_k(lbl, dim) for lbl in labels]
    d = length(dim)
    delta_k = 2 * pi ./ dim

    r = [zeros(Float64, d) for _ in 1:4]

    for m in 1:N
        Ki = [K[m][1], K[m][2], K[m][3]]
        spins = K[m][5]

        out = zeros(Float64, N)
        integrate_dimension_all_n!(1, r, Ki, delta_k, spins, basis_degrees, d, out, 1.0)
        P[m, :] .= out
    end

    return P
end

"""
    generate_bounded_combinations(target_sum::Int, max_degrees::Vector{Int})

Recursive helper function generating integer arrays that sum exactly to `target_sum`
without exceeding the physical lattice degree bounds in `max_degrees`. 
Ensures polynomials do not alias under discrete Brillouin zone integration.
"""
function generate_bounded_combinations(target_sum::Int, max_degrees::Vector{Int})
    if length(max_degrees) == 1
        if target_sum <= max_degrees[1]
            return [[target_sum]]
        else
            return Vector{Vector{Int}}()
        end
    end
    res = Vector{Vector{Int}}()
    for v in 0:min(target_sum, max_degrees[1])
        for sub_comb in generate_bounded_combinations(target_sum - v, max_degrees[2:end])
            push!(res, vcat([v], sub_comb))
        end
    end
    return res
end

"""
    generate_basis_degrees(N::Int, dim::Vector{Int}; labels=nothing)

Sequentially generates `N` combinations of multi-dimensional polynomial indicator lists
of variables bounded strictly under `dim - 1` to prevent aliasing. It selectively filters
combinations that produce structural zeros against the provided dataset `labels` spin properties.
"""
function generate_basis_degrees(N::Int, dim::Vector{Int}; labels=nothing)
    d = length(dim)
    max_degrees = repeat([d_i - 1 for d_i in dim], 4)

    dataset_spins = nothing
    if labels !== nothing
        dataset_spins = [label_to_k(lbl, dim)[5] for lbl in labels]
    end

    degrees = Vector{Vector{Int}}()
    sum_val = 0
    max_possible_sum = sum(max_degrees)

    # Generic random momentum to test structural non-zeros
    r_test = [rand(Float64, d) .* 2pi for _ in 1:4]

    while length(degrees) < N && sum_val <= max_possible_sum
        combs = generate_bounded_combinations(sum_val, max_degrees)
        for comb in combs
            is_viable = true
            if dataset_spins !== nothing
                evals_all_zero = true
                for spins in dataset_spins
                    if abs(evaluate_basis(r_test, spins, comb)) > 1e-10
                        evals_all_zero = false
                        break
                    end
                end
                is_viable = !evals_all_zero
            end

            if is_viable
                push!(degrees, comb)
                if length(degrees) == N
                    return degrees
                end
            end
        end
        sum_val += 1
    end

    if length(degrees) < N
        println("Warning: Only found $(length(degrees)) linearly independent basis degrees under grid bounds, requested $N.")
    end
    return degrees
end


"""
    get_interpolator(coefficients::AbstractVector{T}, labels::Vector, 
                     basis_degrees::Vector{Vector{Int}}, dim::Vector{Int}; 
                     tol::Float64=1e-10) where T

Constructs a continuous `CoefficientInterpolator` structural evaluator for a specific
system `dim`. Computes the inverted projection matrix mapping pseudo-inverses from linearly 
dependent redundant dimensions filtered by the Singular Value Decomposition (SVD) cutoff 
`tol`, successfully generating smooth least-squares coefficient estimates mathematically.
"""
function get_interpolator(coefficients::AbstractVector{T}, labels, basis_degrees::Vector{Vector{Int}}, dim::Vector{Int}; tol=1e-10) where T
    P = build_P_matrix(labels, basis_degrees, dim)
    println("Rank of P: $(rank(P))  dimension of P: $(size(P))")
    F = svd(P)
    cond_P = iszero(F.S[end]) ? Inf : F.S[1] / F.S[end]
    println("Info: Matrix P condition number: ", cond_P)

    S_inv = [s > tol * F.S[1] ? 1 / s : 0.0 for s in F.S]
    P_inv = F.V * Diagonal(S_inv) * F.U'

    interpolated_c = P_inv * coefficients

    return CoefficientInterpolator{eltype(interpolated_c)}(interpolated_c, basis_degrees, length(dim))
end

function (interp::CoefficientInterpolator)(k1::Vector{Float64}, k2::Vector{Float64}, k3::Vector{Float64}, spins::Vector{Int})
    d = interp.d
    k4 = zeros(Float64, d)
    for c in 1:d
        k4[c] = mod(k1[c] + k2[c] - k3[c], 2 * pi)
    end

    r = [k1, k2, k3, k4]

    val = zero(eltype(interp.interpolated_c))
    for n in 1:length(interp.interpolated_c)
        val += interp.interpolated_c[n] * evaluate_basis(r, spins, interp.basis_degrees[n])
    end
    return val
end

function interpolate_coefficients(new_labels, target_dim::Vector{Int}, interp::CoefficientInterpolator)
    N = length(new_labels)
    new_coefficients = similar(interp.interpolated_c, N)
    for m in 1:N
        k1, k2, k3, _, spins = label_to_k(new_labels[m], target_dim)
        new_coefficients[m] = interp(k1, k2, k3, spins)
    end
    return new_coefficients
end
function old_process()
    coefficients_by_u, labels, dim, U_values = load_data_coefficients((3, 3), (3, 2))
    # Use the first available u_idx for the legacy test
    u_idx = first(sort(collect(keys(coefficients_by_u))))
    coeffs = coefficients_by_u[u_idx]
    # Mathematically restrict data to only symmetric unique elements (First Spin == 1)
    # 1111, 1212, 1221 etc. The flipped states 2222, 2121, 2112 are structurally redundant.
    valid_idxs = [label_to_k(lbl, dim)[5][1] == 1 for lbl in labels]
    coeffs = coeffs[valid_idxs]
    labels = labels[valid_idxs]
    N = length(labels)
    d = length(dim)
    @time bd = generate_basis_degrees(N, dim; labels=labels)
    # println("Basis degrees (4d indices): ", bd)
    # Build Interpolator
    @time interp = get_interpolator(coeffs, labels, bd, dim)
    # println("Interpolated C: ", interp.interpolated_c)
    # Test continuous evaluation
    k1 = [0.1, 0.2]
    k2 = [0.3, 0.4]
    k3 = [0.5, 0.6]
    spins = [1, 1, 1, 1]
    flipped_spins = [2, 1, 2, 1]
    @time println("A(k1,k2,k3, 1212) = ", interp(k1, k2, k3, spins))
    @time println("A(k1,k2,k3, 2121) = ", interp(k1, k2, k3, flipped_spins))
    @time println("A(k2,k1,k3, 1212) = ", interp(k2, k1, k3, spins))
    @time println("A(k2,k1,k3, 2112) = ", interp(k2, k1, k3, [2, 1, 1, 2]))
    # Test interpolation to new lattice
    new_labels = [
        [(Coordinate((1, 1)), 1), (Coordinate((2, 2)), 1), (Coordinate((1, 1)), 1), (Coordinate((2, 2)), 1)]
    ]
    @time new_coeffs = interpolate_coefficients(new_labels, [8, 8], interp)
    println("New coeffs: ", new_coeffs)
    println("\n--- Testing Identity Interpolation ---")
    # If we interpolate back onto the exact same original discrete lattice points,
    # it should reconstruct the original coefficients (subject to SVD tolerances).
    @time recovered_coeffs = interpolate_coefficients(labels, dim, interp)
    diff_norm = norm(recovered_coeffs - coeffs)
    max_diff = maximum(abs.(recovered_coeffs - coeffs))
    println("Recovered coefficients norm difference: ", diff_norm)
    println("Recovered coefficients max absolute difference: ", max_diff)

    if max_diff < 1e-10
        println("SUCCESS: Interpolation identity verified!")
    else
        println("WARNING: Interpolation identity showed deviations above 1e-10. This is expected if the P matrix is heavily rank-deficient.")
    end
end
function (@main)(ARGS)

    # -------------------------------------------------------------------------
    # Configuration: small system (source of coefficients) and large system (target)
    # -------------------------------------------------------------------------
    test_folders = [
        "data/N=(3, 3)_3x2",
        "data/N=(3, 3)_4x3",
        "data/N=(3, 3)_3x3",
        "data/N=(4, 4)_3x3_2",
        "data/N=(2, 2)_3x2",
        "data/N=(4, 4)_4x2",
        "data/N=(2, 2)_2x2",
    ]
    nn_folder_specs = [
        ((3, 3), [3, 2], 2:52, ""),
        ((2, 2), [3, 2], 2:52, ""),
        ((3, 3), [3, 3], 2:52, ""),
        # ((4, 4), [3, 3], 2:52, "_2"),
        ((4, 4), [4, 2], 2:52, ""),
        ((2, 2), [2, 2], 2:52, ""),
    ]

    small_electrons = (3, 3)
    small_size = [3, 2]
    order = 2

    if CUDA.functional()
        println("Using CUDA")
    else
        println("Using CPU")
    end

    # Parse --strategy=legendre (default) or --strategy=neural from ARGS
    strategy_flag = "legendre"
    for arg in ARGS
        if startswith(arg, "--strategy=")
            strategy_flag = split(arg, "=")[2]
        end
    end
    println("\n=== Using interpolation strategy: $strategy_flag ===")

    # -------------------------------------------------------------------------
    # Step 1: Build interpolation strategy
    # -------------------------------------------------------------------------
    strategy = if strategy_flag == "neural"
        # Neural-network strategy: train across multiple datasets.
        # Edit folder_specs / hyperparameters to match your training data.
        nn_filepath = "trained_neural_network.jld2"
        for arg in ARGS
            if startswith(arg, "--name=")
                nn_filepath = "trained_neural_network_" * split(arg, "=")[2] * ".jld2"
            end
        end

        @time get_or_train_neural_strategy(nn_filepath, nn_folder_specs;
            U_max=10.0,
            include_dim=true,
            include_electrons=true,
            dim_max=4,
            n_epochs=200,
            batch_size=256,
            lr=1e-3,
            use_gpu=CUDA.functional(),
            use_scale_head=true,
        )
    else
        # Legendre polynomial strategy: fit one interpolator per U value from the small system.
        println("\n=== Loading small system coefficients (all U values) ===")
        coeffs_by_u_small, labels_small, dim_small, U_values_small = load_data_coefficients(small_electrons, small_size)
        println("Small system: dim=$(dim_small), N=$(length(labels_small)), U_values=$(length(U_values_small))")
        @time LegendreStrategy(coeffs_by_u_small, labels_small, dim_small, U_values_small)
    end

    # -------------------------------------------------------------------------
    # Step 2: Load ED data for the large system
    # -------------------------------------------------------------------------
    for large_folder in test_folders
        regex = r"N=\((?<N>\d+), (?<M>\d+)\)"
        m = match(regex, large_folder)
        large_electrons = (parse(Int, m[:N]), parse(Int, m[:M]))

        println("\n===================================")
        println("Testing on folder: $large_folder")
        println("===================================")
        println("\n=== Loading large system ED data ===")
        U_values_large, target_vecs_large, indexer_large, precomputed_structures_large,
        N_large, spin_conserved_large, use_symmetry_large, sign_convention_large = load_ED_data(large_folder; verbose=false)

        dim_large = length(indexer_large.inv_comb_dict)
        println("Large system Hilbert space dim: $dim_large")

        # -------------------------------------------------------------------------
        # Step 3: Build the operator structure for the large system
        # -------------------------------------------------------------------------
        println("\n=== Building operator structure for large system (order=$order) ===")
        momentum_basis = false
        initial_loss = 1.0

        @time t_dict_large, t_keys_large = create_randomized_nth_order_operator(
            order, indexer_large, true;
            magnitude=initial_loss * 100,
            omit_H_conj=!use_symmetry_large,
            conserve_spin=spin_conserved_large,
            normalize_coefficients=false,
            conserve_momentum=momentum_basis
        )
        @time rows_large, cols_large, signs_large, ops_list_large = build_n_body_structure_from_keys(
            t_keys_large, indexer_large, typeof(t_dict_large[t_keys_large[1]]);
            sign_convention=sign_convention_large
        )
        param_index_map_large = build_param_index_map(ops_list_large, t_keys_large)

        indices_by_param = [Int[] for _ in 1:length(t_keys_large)]
        for k in eachindex(param_index_map_large)
            push!(indices_by_param[param_index_map_large[k]], k)
        end
        ops_large = [] # this parameter is actually unused by adjoint_loss (without gradient)

        # -------------------------------------------------------------------------
        # Step 4: Prepare for coefficient prediction
        # -------------------------------------------------------------------------
        dim_large_vec = [parse(Int, x) for x in split(
            load_saved_dict(joinpath(large_folder, "meta_data_and_E.jld2"))["meta_data"]["sites"], "x")]

        # -------------------------------------------------------------------------
        # Step 5: Evaluate adjoint_loss for each U in the large system
        # -------------------------------------------------------------------------
        println("\n=== Evaluating adjoint_loss per U-index ===")
        println(@sprintf("%-8.4s  %-12s  %-12s  %-8s  %-15s  %-15s  %-15s  %-15s", "U-value", "Stored loss", "Pred loss", "Ratio", "MeanAbs Stored", "MeanAbs Pred", "RMS Stored", "RMS Pred"))
        println("-"^125)

        save_name = "unitary_map_energy_symmetry=false_N=$(large_electrons)"
        ref_u_idx = 1
        state1 = target_vecs_large[ref_u_idx, :]

        for u_idx in 2:55
            # Re-predict coefficients for this U value — both strategies now take a U context.
            new_coeffs = if isa(strategy, NeuralNetStrategy)
                ctx = NeuralNetContext(U_values_large[u_idx], large_electrons, strategy.U_max)
                interpolate_coefficients(strategy, ctx, t_keys_large, dim_large_vec)
            else
                ctx = LegendreContext(U_values_large[u_idx])
                interpolate_coefficients(strategy, ctx, t_keys_large, dim_large_vec)
            end

            state2 = target_vecs_large[u_idx, :]
            # println("coef magnitude: $(sum(abs.(new_coeffs))/length(new_coeffs))")
            pred_loss = adjoint_loss(
                real.(new_coeffs), ops_large,
                rows_large, cols_large, signs_large,
                param_index_map_large, nothing, nothing,
                dim_large, state1, state2, nothing,
                !use_symmetry_large, false
            )

            u_file = joinpath(large_folder, "$(save_name)_u_$(u_idx).jld2")
            stored_loss = NaN
            stored_coeffs = nothing
            if isfile(u_file)
                d = load_saved_dict(u_file)
                stored_loss = d["metrics"]["loss"][1]
                stored_coeffs = d["coefficients"][2]
            else
                error("File not found: $(u_file)")
            end

            stored_loss_recalc = isnothing(stored_coeffs) ? NaN : adjoint_loss(
                real.(stored_coeffs), ops_large,
                rows_large, cols_large, signs_large,
                param_index_map_large, nothing, nothing,
                dim_large, state1, state2, nothing,
                !use_symmetry_large, false
            )

            mean_abs_pred = mean(abs.(new_coeffs))
            mean_abs_stored = isnothing(stored_coeffs) ? NaN : mean(abs.(stored_coeffs))
            rms_pred = sqrt(mean(new_coeffs .^ 2))
            rms_stored = isnothing(stored_coeffs) ? NaN : sqrt(mean(stored_coeffs .^ 2))

            ratio = isnan(stored_loss_recalc) || stored_loss_recalc == 0.0 ? NaN : pred_loss / stored_loss_recalc
            println(@sprintf("%-8.4g  %-12.6g  %-12.6g  %-8.4g  %-15.6g  %-15.6g  %-15.6g  %-15.6g", U_values_large[u_idx], stored_loss_recalc, pred_loss, ratio, mean_abs_stored, mean_abs_pred, rms_stored, rms_pred))
        end

        println("\nDone.")
    end
end