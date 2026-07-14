#=
system_scaling.jl

Train a neural network or Legendre strategy to scale unitary map coefficients across system sizes.

Usage:
  julia --project=.. system_scaling.jl [options]

Options:
  --strategy=<strategy> (optional): Legendre or neural. Default: "legendre".
                        Valid options:
                        - "legendre": Use Legendre polynomial strategy (interpolator per U value).
                        - "neural": Train/load a Neural Network strategy.
  --weighting=<weighting> (optional): Weighting scheme. Default: "low_u".
                        Valid options:
                        - "low_u_mild": Mild preference for small U (weight ~ U^(-0.25)).
                        - "low_u": Standard preference for small U (weight ~ U^(-0.5)).
                        - "low_u_strong": Strong preference for small U (weight ~ U^(-1.5)).
                        - "low_u_very_strong": Very aggressive preference for small U (weight ~ U^(-3.0)).
                        - "uniform": Equal weighting for all U values.
                        - "high_u": Preference for large U (weight ~ U^(0.5)).
                        - "high_u_strong": Heavy priority for large U (weight ~ U^(1.5)).
                        - "focus_u8": Gaussian centered at log10(8) ≈ 0.903.
                        - "loss_mild": Mildly scaling by log10 overlap loss.
                        - "loss_std": Standard scaling by squared log10 overlap loss.
                        - "loss_power_mild": Power-law overlap loss scaling with exponent -0.07.
                        - "loss_power_std": Power-law overlap loss scaling with exponent -0.12.
                        - "loss_power_neg03": Power-law loss^alpha scaling with alpha = -0.3.
                        - "loss_power_neg04": Power-law loss^alpha scaling with alpha = -0.4.
                        - "loss_power_neg05": Power-law loss^alpha scaling with alpha = -0.5.
                        - "one_minus_loss_power_03": (1-loss)^alpha scaling with alpha = 0.3.
                        - "one_minus_loss_power_05": (1-loss)^alpha scaling with alpha = 0.5.
                        - "one_minus_loss_power_07": (1-loss)^alpha scaling with alpha = 0.7.
                        - "one_minus_loss_power_10": (1-loss)^alpha scaling with alpha = 1.0.
                        - "one_minus_loss_power_15": (1-loss)^alpha scaling with alpha = 1.5.
                        - "one_minus_loss_power_20": (1-loss)^alpha scaling with alpha = 2.0.
                        - "one_minus_loss_power_25": (1-loss)^alpha scaling with alpha = 2.5.
  --u-range=<u_range> (optional): Range of U indices (e.g. "2:52"). Default: "2:52".
  --folder-set=<folder_set> (optional): Dataset set. Default: "all".
                        Valid options:
                        - "all": Default set with all shapes.
                        - "2x2_only": 2x2 systems only.
                        - "small_only": 2x2 and 3x2 small systems.
                        - "exclude_2x2": Exclude 2x2 systems.
                        - "large_only": 3x3 and 4x2 systems.
                        - "small_nonsplit", "medium_nonsplit", "mixed_nonsplit", "square_nonsplit": Non-overlapping training sets.
                        - "square_pure": Pure square geometries (3x3, 4x5).
                        - "square_extended": Pure squares including third filling.
                        - "square_with_2x2": Square systems with 2x2 anchor.
                        - "square_and_rect": Square systems and small rectangular systems.
                        - "multiscale_3x3": Square systems and 4x2 rectangles.
  --name=<name> (optional): Filename suffix for trained NN. Default: "".
  --base-hidden=<hidden> (optional): Hidden layer sizes (comma-separated). Default: "128,128".
  --embed-dim=<dim> (optional): Embedding dimension. Default: 64.
  --context-hidden=<hidden> (optional): Context layers. Default: "64,32".
  --scale-hidden=<hidden> (optional): Scaling layers. Default: "32,16".
  --scale-loss-weight=<weight> (optional): Weight of the auxiliary scale loss term. Default: 3.0.
  --use-gpu=<true|false> (optional): Whether to use GPU acceleration (CUDA). If set to false, it runs entirely on CPU without loading the CUDA package. Default: true.

Examples:
  julia --project=.. system_scaling.jl --strategy=neural --folder-set=square_pure --name=pure_run
=#

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
using KrylovKit

# Pre-scan ARGS for --use-gpu before loading CUDA
_use_gpu = let val = nothing
    for arg in ARGS
        if startswith(arg, "--use-gpu=")
            val = parse(Bool, split(arg, "=", limit=2)[2])
        end
    end
    val
end

if _use_gpu !== false
    try
        ENV["JULIA_CUDA_USE_COMPAT"] = "true"
        using CUDA
        if CUDA.functional()
            @info "GPU available: $(CUDA.name(CUDA.CuDevice(0)))"
        else
            @warn "CUDA not functional. CPU fallback will be used."
        end
    catch e
        @warn "CUDA loading or initialization failed: $e. CPU fallback will be used."
    end
end

# using cuDNN   # Commented out to avoid CUDNNError on HPC; CUDA.jl handles MLPs natively and stably
using Printf
using Flux
using Dates

include("logging.jl")
include("utility_functions.jl")
include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("trotter.jl")
using .Trotter
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
"""
    sample_coefficients_from_histogram(source_coeffs::Vector{Float64}, n::Int; n_bins::Int=50)

Draws `n` independent samples whose distribution matches `source_coeffs`.

The empirical density of `source_coeffs` is estimated via a histogram with `n_bins` bins.
Bin counts are used to build a piecewise-linear (trapezoidal) approximation of the PDF,
from which an unnormalised CDF is computed over the bin-edge grid. Each sample is then
drawn via inverse-CDF transform: a uniform [0,1) variate is mapped to the value axis by
linear interpolation on the normalised CDF grid. This handles bimodal (and any
multi-modal) distributions correctly, unlike Gaussian approximations or MCMC.
"""
function sample_coefficients_from_histogram(source_coeffs::Vector{Float64}, n::Int; n_bins::Int=50, reps::Int=1)
    lo, hi = minimum(source_coeffs), maximum(source_coeffs)
    # Edge case: all values identical
    if lo ≈ hi
        return fill(lo, n)
    end

    # Build histogram: counts per bin
    edges = range(lo, hi; length=n_bins + 1)
    counts = zeros(Float64, n_bins)
    bin_width = step(edges)
    for v in source_coeffs
        idx = clamp(floor(Int, (v - lo) / bin_width) + 1, 1, n_bins)
        counts[idx] += 1.0
    end

    # Piecewise-linear PDF on bin-edge grid: linearly interpolate between bin-centre
    # densities. Represent the density at each edge as the average of adjacent bins.
    pdf_edges = zeros(Float64, n_bins + 1)
    pdf_edges[1] = counts[1]
    pdf_edges[end] = counts[end]
    for i in 2:n_bins
        pdf_edges[i] = (counts[i-1] + counts[i]) / 2.0
    end

    # Trapezoidal CDF at each edge
    cdf_edges = zeros(Float64, n_bins + 1)
    for i in 2:(n_bins+1)
        cdf_edges[i] = cdf_edges[i-1] + (pdf_edges[i-1] + pdf_edges[i]) / 2.0 * bin_width
    end
    cdf_edges ./= cdf_edges[end]  # normalise to [0, 1]

    edge_vals = collect(edges)

    # Inverse-CDF sampling: for each uniform draw, find position in cdf_edges via
    # linear interpolation (binary search for the bracketing interval).
    all_samples = []
    for r in 1:reps
        samples = Vector{Float64}(undef, n)
        for i in 1:n
            u = rand()
            # Find j such that cdf_edges[j] <= u < cdf_edges[j+1]
            j = searchsortedlast(cdf_edges, u)
            j = clamp(j, 1, n_bins)
            # Linear interpolation within the interval
            Δcdf = cdf_edges[j+1] - cdf_edges[j]
            t = Δcdf ≈ 0.0 ? 0.0 : (u - cdf_edges[j]) / Δcdf
            samples[i] = edge_vals[j] + t * bin_width
        end
        push!(all_samples, samples)
    end

    if reps > 1
        return all_samples
    end
    return all_samples[1]
end

"""
    parse_arguments(args::Vector{String})

Parse command line arguments for running the system scaling interpolation study.
"""
function parse_arguments(args::Vector{String})
    weighting_flag = "low_u"
    u_range_str = "2:52"
    folder_set_flag = "all"
    base_hidden_str = "128,128"
    embed_dim_val = 64
    context_hidden_str = "64,32"
    scale_hidden_str = "32,16"
    strategy_flag = "neural"
    name_val = ""
    scale_loss_weight_val = 3.0f0
    is_trotter_val = false
    k_max_val = 10
    loss_type_val = :overlap
    k_eval_val = 2

    for arg in args
        if startswith(arg, "--weighting=")
            weighting_flag = String(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--u-range=")
            u_range_str = String(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--folder-set=")
            folder_set_flag = String(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--base-hidden=")
            base_hidden_str = String(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--embed-dim=")
            embed_dim_val = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--context-hidden=")
            context_hidden_str = String(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--scale-hidden=")
            scale_hidden_str = String(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--strategy=")
            strategy_flag = String(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--name=")
            name_val = String(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--scale-loss-weight=")
            scale_loss_weight_val = parse(Float32, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--is-trotter=")
            is_trotter_val = parse(Bool, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--k-max=")
            k_max_val = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--loss-type=")
            val = String(split(arg, "=", limit=2)[2])
            loss_type_val = val == "energy" ? :energy : :overlap
        elseif startswith(arg, "--k-eval=")
            k_eval_val = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--use-gpu=")
            continue
        end
    end

    # Parse u_range range
    parts = split(u_range_str, ":")
    u_range = parse(Int, parts[1]):parse(Int, parts[2])

    # Parse hidden layers
    base_hidden_val = [parse(Int, x) for x in split(base_hidden_str, ",")]
    context_hidden_val = [parse(Int, x) for x in split(context_hidden_str, ",")]
    scale_hidden_val = [parse(Int, x) for x in split(scale_hidden_str, ",")]

    return strategy_flag, weighting_flag, u_range, folder_set_flag, name_val, base_hidden_val, embed_dim_val, context_hidden_val, scale_hidden_val, scale_loss_weight_val, is_trotter_val, k_max_val, loss_type_val, k_eval_val
end


function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "system_scaling")
    with_logging(log_path) do

        strategy_flag, weighting_flag, u_range, folder_set_flag, name_val, base_hidden_val, embed_dim_val, context_hidden_val, scale_hidden_val, scale_loss_weight_val, is_trotter_val, k_max_val, loss_type_val, k_eval_val = parse_arguments(ARGS)

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
            "data/N=(4, 5)_3x3",
            # "data/N=(3, 2)_3x2",
            "nn_test_data/N=(5, 5)_4x3",
            "nn_test_data/N=(3, 4)_4x3"
        ]
        # Build folder specs based on parsed options
        # NOTE: test_folders below includes N=(3,3)_3x2, N=(3,3)_4x3, N=(3,3)_3x3,
        # N=(4,4)_3x3_2, N=(2,2)_3x2, N=(4,4)_4x2, N=(2,2)_2x2.
        # The "_nonsplit" variants below are disjoint from all those test folders.
        raw_specs = if folder_set_flag == "small_only"
            [
                ((2, 2), [3, 2], ""),
                ((2, 2), [2, 2], ""),
            ]
        elseif folder_set_flag == "2x2_only"
            [
                ((2, 2), [2, 2], ""),
            ]
        elseif folder_set_flag == "exclude_2x2"
            [
                ((3, 3), [3, 2], ""),
                ((3, 3), [3, 3], ""),
                ((4, 4), [4, 2], ""),
            ]
        elseif folder_set_flag == "large_only"
            [
                ((3, 3), [3, 3], ""),
                ((4, 4), [4, 2], ""),
            ]
            # --- New non-overlapping training sets (disjoint from test_folders) ---
            # Small: only the duplicate 3x2 folders (replications of the smallest system)
        elseif folder_set_flag == "small_nonsplit"
            [
                ((3, 3), [3, 2], "_2"),
                ((3, 3), [3, 2], "_3"),
            ]
            # Medium: duplicate 3x2 + 4x2 variants not in test set
        elseif folder_set_flag == "medium_nonsplit"
            [
                ((3, 3), [3, 2], "_2"),
                ((3, 3), [3, 2], "_3"),
                ((4, 4), [4, 2], "_2"),
            ]
            # Larger: 3x2 duplicates + 3x3 non-test + 4x2 duplicate
        elseif folder_set_flag == "mixed_nonsplit"
            [
                ((3, 3), [3, 2], "_2"),
                ((3, 3), [3, 2], "_3"),
                ((3, 3), [4, 2], ""),
                ((4, 4), [4, 2], "_2"),
            ]
            # --- Aspect-ratio-based sets (same geometry, different filling) ---
            # Square geometry only: targets the 3x3 test systems
            # N=(3,3)_3x3_newsign (60 files), N=(4,4)_3x3 (13 files, will be clamped)
            # N=(4,5)_3x3 (61 files) and N=(4,5)_3x3_3 (60 files) — extra fillings
        elseif folder_set_flag == "square_nonsplit"
            [
                ((3, 3), [3, 3], "_newsign"),
                ((4, 4), [3, 3], ""),
                ((4, 5), [3, 3], ""),
            ]
            # Minimal square set: single clean copy of each filling, no duplicates.
            # N=(3,3)_3x3_newsign (63 files), N=(4,5)_3x3 (64 files).
            # Drops the sparse N=(4,4)_3x3 (only 3 u-files) used in square_nonsplit.
        elseif folder_set_flag == "square_pure"
            [
                ((3, 3), [3, 3], ""),
                ((4, 5), [3, 3], ""),
            ]
            # Extended square set: adds N=(4,5)_3x3_3 (63 files) — a third filling with full data.
            # Still no duplicates; each entry is a distinct electron count.
        elseif folder_set_flag == "square_extended"
            [
                ((3, 3), [3, 3], ""),
                ((4, 5), [3, 3], ""),
                ((4, 5), [3, 3], "_3"),
            ]
            # Square set + 2x2 lattice: adds N=(2,2)_2x2 (61 files) to give the network
            # a small-system anchor at the same square geometry.
        elseif folder_set_flag == "square_with_2x2"
            [
                ((3, 3), [3, 3], ""),
                ((4, 5), [3, 3], ""),
                ((2, 2), [2, 2], ""),
            ]
            # Square geometry + small rectangular systems
        elseif folder_set_flag == "square_and_rect"
            [
                ((3, 3), [3, 3], "_newsign"),
                ((4, 4), [3, 3], ""),
                ((4, 5), [3, 3], ""),
                ((3, 3), [3, 2], "_2"),
                ((3, 3), [3, 2], "_3"),
            ]
            # Square geometry + 4x2 rectangles — multiple aspect ratios, all clean
        elseif folder_set_flag == "multiscale_3x3"
            [
                ((3, 3), [3, 3], "_newsign"),
                ((4, 4), [3, 3], ""),
                ((4, 5), [3, 3], ""),
                ((4, 4), [4, 2], "_2"),
                ((3, 3), [4, 2], ""),
            ]
        else # default "all"
            [
                ((3, 3), [3, 2], ""),
                ((2, 2), [3, 2], ""),
                ((3, 3), [3, 3], ""),
                ((4, 4), [4, 2], ""),
                ((2, 2), [2, 2], ""),
            ]
        end

        nn_folder_specs = [(spec[1], spec[2], u_range, spec[3]) for spec in raw_specs]

        small_electrons = (3, 3)
        small_size = [3, 2]
        order = 2

        if @isdefined(CUDA) && CUDA.functional()
            println("Using CUDA")
        else
            println("Using CPU")
        end

        println("\n=== Using interpolation strategy: $strategy_flag ===")

        # -------------------------------------------------------------------------
        # Step 1: Build interpolation strategy
        # -------------------------------------------------------------------------
        strategy = if strategy_flag == "neural"
            # Neural-network strategy: train across multiple datasets.
            # Edit folder_specs / hyperparameters to match your training data.
            nn_filepath = isempty(name_val) ? "trained_neural_networks/trained_neural_network.jld2" : "trained_neural_networks/trained_neural_network_$(name_val).jld2"

            println("Training NN with weighting scheme: $weighting_flag")
            println("Training NN with U index range: $u_range")
            println("Training NN with folder set: $folder_set_flag")

            @time get_or_train_neural_strategy(nn_filepath, nn_folder_specs;
                U_max=20.0,
                include_dim=true,
                include_electrons=true,
                dim_max=4,
                n_epochs=200,
                batch_size=256,
                lr=1e-3,
                use_gpu=@isdefined(CUDA) && CUDA.functional(),
                use_scale_head=true,
                weighting_scheme=weighting_flag,
                base_hidden=base_hidden_val,
                embed_dim=embed_dim_val,
                context_hidden=context_hidden_val,
                scale_hidden=scale_hidden_val,
                scale_loss_weight=scale_loss_weight_val,
                is_trotter=is_trotter_val,
                loss_type=loss_type_val,
                k_max=k_max_val
            )
        else
            # Legendre polynomial strategy: fit one interpolator per U value from the small system.
            println("\n=== Loading small system coefficients (all U values) ===")
            coeffs_by_u_small, labels_small, dim_small, U_values_small = load_data_coefficients(small_electrons, small_size)
            println("Small system: dim=$(dim_small), N=$(length(labels_small)), U_values=$(length(U_values_small))")
            @time LegendreStrategy(coeffs_by_u_small, labels_small, dim_small, U_values_small)
        end

        # -------------------------------------------------------------------------
        # Step 2: Test accuracy of interpolation strategy
        # -------------------------------------------------------------------------
        function evaluate_coefficients_metrics(coeffs, rows, cols, signs, param_index_map, dim, state1, state2, H, use_symmetry)
            vals = update_values(signs, param_index_map, coeffs, nothing, nothing)
            A = sparse(rows, cols, vals, dim, dim)
            if !use_symmetry
                A = make_hermitian(A)
            end

            # Propagate states using optimized exponentiation (dense exp for small matrices, expv for large)
            if dim > 128
                psi_loss = expv(1.0im, A, state2)
                psi_energy = expv(1.0im, A, state1)
            else
                dense_A = Matrix(A)
                U_mat = exp(1.0im * dense_A)
                psi_loss = U_mat * state2
                psi_energy = U_mat * state1
            end

            loss = 1.0 - abs2(dot(state1, psi_loss))
            energy = real(dot(psi_energy, H * psi_energy))

            return loss, energy
        end

        for large_folder in test_folders
            regex = r"N=\((?<N>\d+), (?<M>\d+)\)"
            m = match(regex, large_folder)
            large_electrons = (parse(Int, m[:N]), parse(Int, m[:M]))

            println("\n===================================")
            println("Testing on folder: $large_folder")
            println("===================================")

            if !isfile(joinpath(large_folder, "meta_data_and_E.jld2"))
                @warn "Skipping $large_folder: no meta_data_and_E.jld2 found"
                continue
            end

            println("\n=== Loading large system ED data ===")
            U_values_large, target_vecs_large, indexer_large, precomputed_structures_large,
            N_large, spin_conserved_large, use_symmetry_large, sign_convention_large = load_ED_data(large_folder; verbose=false)

            dim_large = length(indexer_large.inv_comb_dict)
            println("Large system Hilbert space dim: $dim_large")

            if is_trotter_val
                # -------------------------------------------------------------------------
                # Step 3 (Trotter): Build basis and gate structures for Trotter
                # -------------------------------------------------------------------------
                metadata_large = load_saved_dict(joinpath(large_folder, "meta_data_and_E.jld2"))
                dim_large_vec = [parse(Int, x) for x in split(metadata_large["meta_data"]["sites"], "x")]
                all_E_large = metadata_large["E"]
                k_min_large = find_best_energy_sector(all_E_large, U_values_large; verbose=false)

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

                local basis_sector_large
                if !isnothing(indexer_large)
                    println("Reconstructing basis sector from indexer...")
                    basis_sector_large = Vector{UInt}(undef, length(indexer_large.inv_comb_dict))
                    for (idx, conf) in enumerate(indexer_large.inv_comb_dict)
                        u_bin = coord_set_to_binary(conf[1], dim_large_vec)
                        d_bin = coord_set_to_binary(conf[2], dim_large_vec)
                        basis_sector_large[idx] = Trotter.combineSpinInts(u_bin, d_bin, prod(dim_large_vec))
                    end
                else
                    error("No indexer loaded for large system: $large_folder")
                end

                q_target = Trotter.ravel_c(indexer_large.k .- 1, Tuple(indexer_large.lattice_dims))
                println("Constructing sector Hamiltonians directly using HubbardMomentumBasis...")
                H_hop_mom, basis_dict, _ = Trotter.TamFermion.HubbardMomentumBasis(1.0, 0.0, dim_large_vec, large_electrons; q_target=q_target)
                H_int_mom, _, _ = Trotter.TamFermion.HubbardMomentumBasis(0.0, 1.0, dim_large_vec, large_electrons; q_target=q_target)

                state_to_idx = Dict(val => idx for (idx, val) in enumerate(basis_dict["ints"]))
                perm = [state_to_idx[val] for val in basis_sector_large]

                H_hop_sector_large = H_hop_mom[perm, perm]
                H_int_sector_large = H_int_mom[perm, perm]

                println("Enumerating Trotter gates...")
                gates_large = Trotter.enumerate_ferm_excitations(2, dim_large_vec; conserve_mom=true, conserve_sz=true, include_diagonal=true)
                t_keys_large = [fgate_to_label(g, dim_large_vec) for g in gates_large]

                function evaluate_trotter_coefficients_metrics(coeffs, gates, basis, N_sites, k_steps, state1, state2, H)
                    ref_evolved = Trotter.TrotterOptimization.apply_unitary(coeffs, gates, state1, basis, N_sites, k_steps)
                    overlap = abs2(dot(state2, ref_evolved))
                    loss = 1.0 - overlap
                    energy = real(dot(ref_evolved, H * ref_evolved))
                    return loss, energy
                end

                prefix_large = loss_type_val == :energy ? "trotter_N=$(prod(dim_large_vec))_loss_energy" : "trotter_N=$(prod(dim_large_vec))"
                
                println("\n=== Evaluating Trotter adjoint_loss and energy per U-index (k=$k_eval_val) ===")
                println(@sprintf(
                    "%-8.8s  %-12.12s  %-14.14s  %-14.14s  %-12.12s  %-12.12s  %-14.14s  %-14.14s",
                    "U-value", "True E", "Base loss", "Base E", "Pred loss", "Pred E", "pr/ba loss", "pr-ba E"
                ))
                println("-"^110)

                results_strings = Vector{String}(undef, 55)
                state1 = target_vecs_large[1, :]

                @time begin
                    @safe_threads for u_idx in 2:55
                        u_val = U_values_large[u_idx]
                        
                        coeffs_pred = Float64[]
                        for l in 1:k_eval_val
                            ctx = NeuralNetContext(u_val, large_electrons, strategy.U_max, k_eval_val, l, k_max_val)
                            c_l = interpolate_coefficients(strategy, ctx, t_keys_large, dim_large_vec)
                            append!(coeffs_pred, c_l)
                        end
                        
                        H_large = H_hop_sector_large + u_val * H_int_sector_large
                        true_E = all_E_large[k_min_large][u_idx]
                        state2 = target_vecs_large[u_idx, :]
                        
                        pred_loss, pred_energy = evaluate_trotter_coefficients_metrics(
                            coeffs_pred, gates_large, basis_sector_large, prod(dim_large_vec), k_eval_val,
                            state1, state2, H_large
                        )

                        u_file = joinpath(large_folder, "$(prefix_large)_u_$(u_idx).jld2")
                        stored_loss, stored_energy = NaN, NaN
                        if isfile(u_file)
                            d = load_saved_dict(u_file)
                            stored_coeffs = d["coefficients"]
                            k_stored = length(stored_coeffs) ÷ length(gates_large)
                            stored_loss, stored_energy = evaluate_trotter_coefficients_metrics(
                                stored_coeffs, gates_large, basis_sector_large, prod(dim_large_vec), k_stored,
                                state1, state2, H_large
                            )
                        end
                        
                        ratio = isnan(stored_loss) || stored_loss == 0.0 ? NaN : pred_loss / stored_loss
                        diff_E = isnan(stored_energy) || isnan(pred_energy) ? NaN : pred_energy - stored_energy

                        results_strings[u_idx] = @sprintf(
                            "%-8.4g  %-12.6g  %-14.6g  %-14.6g  %-12.6g  %-12.6g  %-14.4g  %-14.4g",
                            u_val, true_E, stored_loss, stored_energy, pred_loss, pred_energy, ratio, diff_E
                        )
                    end
                end

                for u_idx in 2:55
                    if isassigned(results_strings, u_idx)
                        println(results_strings[u_idx])
                    end
                end
                println("\nDone.")
            else
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
                # Step 4: Prepare for coefficient prediction and energy evaluation
                # -------------------------------------------------------------------------
                metadata_large = load_saved_dict(joinpath(large_folder, "meta_data_and_E.jld2"))
                dim_large_vec = [parse(Int, x) for x in split(metadata_large["meta_data"]["sites"], "x")]
                all_E_large = metadata_large["E"]
                k_min_large = find_best_energy_sector(all_E_large, U_values_large; verbose=false)

                subspace_large = reconstruct_subspace(indexer_large, spin_conserved_large)
                hopping_model_large = HubbardModel(1.0, 0.0, 0.0, false)
                interaction_model_large = HubbardModel(0.0, 1.0, 0.0, false)
                H_hopping_large = create_Hubbard(hopping_model_large, subspace_large; indexer=indexer_large, sign_convention=sign_convention_large)
                H_interaction_large = create_Hubbard(interaction_model_large, subspace_large; indexer=indexer_large, sign_convention=sign_convention_large)

                # -------------------------------------------------------------------------
                # Step 5: Evaluate adjoint_loss and adjoint_energy_loss for each U in the large system
                # -------------------------------------------------------------------------
                println("\n=== Evaluating adjoint_loss and energy per U-index ===")
                println(@sprintf(
                    "%-8.8s  %-12.12s  %-14.14s  %-14.14s  %-12.12s  %-12.12s  %-14.14s  %-14.14s  %-15.15s  %-15.15s  %-15.15s  %-15.15s  %-16.16s  %-16.16s  %-14.14s  %-14.14s  %-14.14s  %-14.14s  %-14.14s  %-14.14s  %-14.14s  %-14.14s",
                    "U-value", "True E", "Base loss", "Base E", "Pred loss", "Pred E", "pr/ba loss", "pr-ba E",
                    "MeanAbs Stor", "MeanAbs Pred", "RMS Stored", "RMS Pred",
                    "rand min loss", "rand min E", "mean loss", "mean E", "max loss", "max E", "pr/rd loss", "pr-rd E", "rd/ba loss", "rd-ba E"
                ))
                println("-"^310)

                all_stored_coeffs = Float64[]
                save_name_pre = "unitary_map_energy_symmetry=false_N=$(large_electrons)"
                for ui in 2:55
                    uf = joinpath(large_folder, "$(save_name_pre)_u_$(ui).jld2")
                    if isfile(uf)
                        d_tmp = load_saved_dict(uf)
                        c_tmp = d_tmp["coefficients"][2]
                        if !isnothing(c_tmp)
                            append!(all_stored_coeffs, real.(c_tmp))
                        end
                    end
                end

                save_name = save_name_pre
                ref_u_idx = 1
                state1 = target_vecs_large[ref_u_idx, :]

                results_strings = Vector{String}(undef, 55)

                @time begin
                    @safe_threads for u_idx in 2:55
                        u_val = U_values_large[u_idx]
                        new_coeffs = if isa(strategy, NeuralNetStrategy)
                            ctx = NeuralNetContext(u_val, large_electrons, strategy.U_max)
                            interpolate_coefficients(strategy, ctx, t_keys_large, dim_large_vec)
                        else
                            ctx = LegendreContext(u_val)
                            interpolate_coefficients(strategy, ctx, t_keys_large, dim_large_vec)
                        end

                        H_large = H_hopping_large + u_val * H_interaction_large
                        true_E = all_E_large[k_min_large][u_idx]

                        state2 = target_vecs_large[u_idx, :]
                        pred_loss, pred_energy = evaluate_coefficients_metrics(
                            real.(new_coeffs), rows_large, cols_large, signs_large,
                            param_index_map_large, dim_large, state1, state2, H_large,
                            use_symmetry_large
                        )

                        u_file = joinpath(large_folder, "$(save_name)_u_$(u_idx).jld2")
                        stored_coeffs = nothing
                        if isfile(u_file)
                            d = load_saved_dict(u_file)
                            stored_coeffs = d["coefficients"][2]
                        else
                            error("File not found: $(u_file)")
                        end

                        stored_loss_recalc, stored_energy_recalc = isnothing(stored_coeffs) ? (NaN, NaN) : evaluate_coefficients_metrics(
                            real.(stored_coeffs), rows_large, cols_large, signs_large,
                            param_index_map_large, dim_large, state1, state2, H_large,
                            use_symmetry_large
                        )

                        mean_abs_pred = mean(abs.(new_coeffs))
                        mean_abs_stored = isnothing(stored_coeffs) ? NaN : mean(abs.(stored_coeffs))
                        rms_pred = sqrt(mean(new_coeffs .^ 2))
                        rms_stored = isnothing(stored_coeffs) ? NaN : sqrt(mean(stored_coeffs .^ 2))

                        ratio = isnan(stored_loss_recalc) || stored_loss_recalc == 0.0 ? NaN : pred_loss / stored_loss_recalc
                        diff_E = isnan(stored_energy_recalc) || isnan(pred_energy) ? NaN : pred_energy - stored_energy_recalc

                        rand_coeffs = sample_coefficients_from_histogram(all_stored_coeffs, length(new_coeffs); reps=10)
                        losses = Vector{Float64}(undef, length(rand_coeffs))
                        energies = Vector{Float64}(undef, length(rand_coeffs))
                        for idx in eachindex(rand_coeffs)
                            l, e = evaluate_coefficients_metrics(
                                rand_coeffs[idx], rows_large, cols_large, signs_large,
                                param_index_map_large, dim_large, state1, state2, H_large,
                                use_symmetry_large
                            )
                            losses[idx] = l
                            energies[idx] = e
                        end
                        rand_loss = minimum(losses)
                        mean_loss = sum(losses) / length(losses)
                        maximum_loss = maximum(losses)

                        rand_energy_min = minimum(energies)
                        rand_energy_mean = sum(energies) / length(energies)
                        rand_energy_max = maximum(energies)

                        rand_ratio = isnan(stored_loss_recalc) || stored_loss_recalc == 0.0 ? NaN : pred_loss / rand_loss
                        rand_diff_E = isnan(rand_energy_min) || isnan(pred_energy) ? NaN : pred_energy - rand_energy_min

                        rand_baseline_ratio = isnan(stored_loss_recalc) || stored_loss_recalc == 0.0 ? NaN : rand_loss / stored_loss_recalc
                        rand_baseline_diff_E = isnan(stored_energy_recalc) || isnan(rand_energy_min) ? NaN : rand_energy_min - stored_energy_recalc

                        results_strings[u_idx] = @sprintf(
                            "%-8.4g  %-12.6g  %-14.6g  %-14.6g  %-12.6g  %-12.6g  %-14.4g  %-14.4g  %-15.6g  %-15.6g  %-15.6g  %-15.6g  %-16.6g  %-16.6g  %-14.4g  %-14.4g  %-14.4g  %-14.4g  %-14.4g  %-14.4g  %-14.4g  %-14.4g",
                            u_val, true_E, stored_loss_recalc, stored_energy_recalc, pred_loss, pred_energy, ratio, diff_E,
                            mean_abs_stored, mean_abs_pred, rms_stored, rms_pred,
                            rand_loss, rand_energy_min, mean_loss, rand_energy_mean, maximum_loss, rand_energy_max, rand_ratio, rand_diff_E, rand_baseline_ratio, rand_baseline_diff_E
                        )
                    end
                end

                for u_idx in 2:55
                    if isassigned(results_strings, u_idx)
                        println(results_strings[u_idx])
                    end
                end
                println("\nDone.")
            end
        end
    end
end