#=
test_plot_pruning_and_sparsity_threshold.jl

Test script to verify plot_pruning_and_sparsity_at_threshold function added to final_analysis.ipynb.
=#

using Lattices
using LinearAlgebra
using Combinatorics
using SparseArrays
using CairoMakie
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
using LsqFit
using HDF5

if !isdefined(Main, :UtilityFunctions)
    include("../utility_functions.jl")
end
include("../ed_objects.jl")
include("../ed_functions.jl")
include("../ed_optimization.jl")
include("../data_path.jl")
include("../logging.jl")

cmap1(L) = [Makie.ColorSchemes.roma[z] for z in range(0, 1, length=L)]
cmap2(L) = [Makie.ColorSchemes.managua[z] for z in range(0, 1, length=L)]

const FILE_LABEL_PAIRS = [
    (L"3\times 2\;(2,2)",   "N=(2, 2)_3x2", (2,2)),
    (L"3\times 2,\;(3,2)",  "N=(3, 2)_3x2", (3,2)),
    (L"3\times 2,\;(3,3)",  "N=(3, 3)_3x2", (3,3)),
    (L"3\times3,\;(3,2)",   "N=(3, 2)_3x3", (3,2)),
    (L"4\times2,\;(3,3)",   "N=(3, 3)_4x2", (3,3)),
    (L"3\times3,\;(3,3)",   "N=(3, 3)_3x3", (3,3)),
    (L"3\times3,\;(4,3)",   "N=(4, 3)_3x3", (4,3)),
    (L"3\times3,\;(4,4)",   "N=(4, 4)_3x3", (4,4)),
    (L"3\times3,\;(5,4)",   "N=(5, 4)_3x3", (5,4)),
    (L"4\times3,\;(5,4)",   "N=(5, 4)_4x3", (5,4)),
]

const FOLDER = get_data_root()

const THICKNESS_SCALE = 1.3
const BASE_FONTSIZE   = 16
const BASE_LINEWIDTH  = 2.5
const BASE_MARKERSIZE = 12

function get_figsize(num_cols::Int, height_mm::Float64=55.0)
    px_per_mm = 96 / 25.4
    φ = (1+sqrt(5))/2
    if num_cols == 1
        width_mm = 89
    elseif num_cols == 2
        width_mm = 180
    else
        error("Can only have num_cols == 1 or 2 (given: $num_cols)")
    end
    width_px = width_mm * px_per_mm
    height_px = height_mm * px_per_mm
    println("Using width=$(width_mm/25.4)\" and height=$(height_mm/25.4)\"")
    return (width_px, height_px)
end

function create_fig(num_cols::Int, height_mm::Float64; kwargs...)
    fig = Figure(size=get_figsize(num_cols, height_mm), figure_padding=5)
    ax = Axis(fig[1, 1]; 
        aspect = (1+sqrt(5))/2,
        kwargs...
    )
    return fig, ax
end

const LEGEND_ARGS = Dict(:rowgap=>-8, :padding=>(3,3,0,0))
set_theme!(theme_latexfonts(), fontsize=10)

function find_root_bisection(f, a, b; tol=1e-5, max_iters=100)
    fa = f(a)
    fb = f(b)
    if sign(fa) == sign(fb)
        return NaN
    end
    for _ in 1:max_iters
        c = (a + b) / 2
        fc = f(c)
        if abs(fc) < tol || (b - a) / 2 < tol
            return c
        end
        if sign(fc) == sign(fa)
            a = c
            fa = fc
        else
            b = c
            fb = fc
        end
    end
    return (a + b) / 2
end

softplus(x, b) = max(x, zero(x)) + log(b) + log1p(exp(-abs(x)) / b)
scaled_softplus(x, p1, p2) = 1 / log(1 + p2) * softplus(p1 * x, p2)
model(x, p) = @. (1 - p[1] * (x - p[2]) / (1 + abs(p[1] * (x - p[2]))^p[3])^(1 / p[3])) / 2
rescale(x, p) = @. x * (p[2] - p[1]) + p[1]

"""
    plot_pruning_and_sparsity_at_threshold(u_values, u_indices, file_label_pair, fit_params, overlap_bounds, sparsity_data, norm_overlap_data, hilbert_space_sizes, output_dir, antihermitian_val, custom_ref_state_arg_val, cmap; selected_U=8, threshold=0.95)

Plot pruning curves for `selected_U` (left panel) and maximum sparsity at overlap `threshold` (right panel) in a single figure using `create_fig`.
"""
function plot_pruning_and_sparsity_at_threshold(
    u_values,
    u_indices,
    file_label_pair,
    fit_params,
    overlap_bounds,
    sparsity_data,
    norm_overlap_data,
    hilbert_space_sizes,
    output_dir::String,
    antihermitian_val::Bool,
    custom_ref_state_arg_val,
    cmap;
    selected_U=8,
    threshold=0.95
)
    fig, ax1 = create_fig(2, 70.0;
        xlabel = L"\textrm{Sparsity}",
        ylabel = L"|\langle E_0(U)|\mathcal{U}|E_0(0)\rangle|^2",
        yticks = 0:0.1:1,
        xticks = 0:0.25:1,
    )

    ax2 = Axis(fig[1, 2];
        aspect = (1+sqrt(5))/2,
        xlabel = L"U",
        ylabel = L"$\textrm{Max\;sparsity\;at\;threshold\;}$\\$|\langle E_0(U)|\mathcal{U}|E_0(0)\rangle|^2 \geq %$(threshold)$",
        limits = ((0, 15), (0, 1)),
        yticks = 0:0.1:1,
    )

    # Panel 1: Pruning curves for selected U
    selected_u_idx = argmin(abs.(u_values[u_indices] .- selected_U)) + u_indices[1] - 1
    plot_idx_local = findfirst(==(selected_u_idx), u_indices)

    for (color_i, (label, _, _)) in enumerate(file_label_pair)
        if !isnothing(plot_idx_local)
            fit_p = fit_params[color_i][plot_idx_local]
            bounds = overlap_bounds[color_i][plot_idx_local]
            sparsity_pts = sparsity_data[color_i][plot_idx_local]
            norm_overlap_pts = norm_overlap_data[color_i][plot_idx_local]
            overlap_pts = rescale(norm_overlap_pts, bounds)

            fit_sparsity = LinRange(0, 1, 200)
            fit_overlap = rescale(model(fit_sparsity, fit_p), bounds)

            lines!(ax1, fit_sparsity, fit_overlap;
                color = cmap[color_i], linestyle = :dash)
            scatter!(ax1, sparsity_pts, overlap_pts;
                color = cmap[color_i], label = string(label))
        end
    end

    # Panel 2: Max sparsity at threshold
    sparsity_at_threshold = []
    for (i, (param_u, (label, _, _), bounds_u, hs_size)) in enumerate(zip(fit_params, file_label_pair, overlap_bounds, hilbert_space_sizes))
        push!(sparsity_at_threshold, Float64[])
        for (param, bounds) in zip(param_u, bounds_u)
            sparsity_val = find_root_bisection(s -> rescale(model(s, param), bounds) - threshold, -1.0, 2.0)
            push!(sparsity_at_threshold[end], sparsity_val)
        end
        scatterlines!(ax2, u_values[u_indices], sparsity_at_threshold[end];
            label = string(label), linewidth = 2, markersize = 8, color = cmap[i])
    end

    Legend(fig[1, 3], ax2; LEGEND_ARGS...)
    colgap!(fig.layout, 10)

    u_val_str = round(u_values[selected_u_idx], digits=2)
    out_filename = build_save_name_prefix(
        "pruning_and_sparsity_threshold_U=$(u_val_str)_th=$(threshold)";
        antihermitian=antihermitian_val,
        custom_ref_state_arg=custom_ref_state_arg_val
    )
    save(joinpath(output_dir, "$(out_filename).pdf"), fig)
    save(joinpath(output_dir, "$(out_filename).png"), fig)
    println("Saved combined plot to: ", joinpath(output_dir, "$(out_filename).png"))
    return fig
end

function main()
    log_path = make_log_path(@__DIR__, "test_plot_pruning_and_sparsity_threshold")
    with_logging(log_path) do
        antihermitian_val, custom_ref_state_arg_val = true, "slater"
        u_values = 0.25:0.25:15.0

        hilbert_space_sizes = Int[]
        fit_params = []
        fit_errors = []
        overlap_bounds = []
        sparsity_data = []
        norm_overlap_data = []
        u_indices = 2:60

        for (color_i, (label, file_label, nelectrons)) in enumerate(FILE_LABEL_PAIRS)
            nsites = prod(parse_lattice_dimension(file_label))
            filename = build_save_name_prefix(
                "pruning_analysis_trotter";
                sites=nsites,
                antihermitian=antihermitian_val,
                custom_ref_state_arg=custom_ref_state_arg_val
            )
            pruning_analysis_path = joinpath(FOLDER, file_label, "$(filename).jld2")
            system_size = parse_lattice_dimension(file_label)
            meta_data_path = joinpath(FOLDER, file_label, "HubbardBasis_XDiag_$(system_size[1])x$(system_size[2])_nu_$(nelectrons[1])_nd_$(nelectrons[2]).h5")

            h5open(meta_data_path, "r") do data
                push!(hilbert_space_sizes, length(data["sectors/0/R_up_rep"][:]))
            end

            d = load(pruning_analysis_path)

            curr_fit_params = Vector{Any}(undef, length(u_indices))
            curr_fit_errors = Vector{Any}(undef, length(u_indices))
            curr_overlap_bounds = Vector{Any}(undef, length(u_indices))
            curr_sparsity_data = Vector{Any}(undef, length(u_indices))
            curr_norm_overlap_data = Vector{Any}(undef, length(u_indices))

            for (idx, i) in collect(enumerate(u_indices))
                filt = d["removed_terms"][:, i] .> 0
                err = max.(abs.(d["error_data"][:, i][filt]), 1e-16)
                overlap = 1 .- err

                sparsity = d["removed_terms"][:, i][filt] ./ maximum(d["removed_terms"][:, i][filt])
                norm_overlap = (overlap .- overlap[end]) ./ (overlap[1] .- overlap[end])
                curr_overlap_bounds[idx] = [overlap[end], overlap[1]]

                filt2 = norm_overlap .> norm_overlap[end]
                weight = min.(1 ./ (1 .- overlap) .^ 2, 1e6)

                fit = curve_fit(
                    model,
                    sparsity[filt2],
                    norm_overlap[filt2],
                    weight[filt2],
                    [1.0, 1.0, 1.0],
                    lower=[-Inf, -Inf, 0.1],
                    upper=[Inf, Inf, 10.0]
                )

                errs = try
                    stderror(fit)
                catch
                    zeros(length(fit.param))
                end

                curr_fit_params[idx] = copy(fit.param)
                curr_fit_errors[idx] = copy(errs)
                curr_sparsity_data[idx] = sparsity[filt2]
                curr_norm_overlap_data[idx] = norm_overlap[filt2]
            end

            push!(fit_params, curr_fit_params)
            push!(fit_errors, curr_fit_errors)
            push!(overlap_bounds, curr_overlap_bounds)
            push!(sparsity_data, curr_sparsity_data)
            push!(norm_overlap_data, curr_norm_overlap_data)
        end

        output_dir = joinpath(@__DIR__, "..", "good_images", "final")
        if !isdir(output_dir)
            mkpath(output_dir)
        end

        cmap = cmap2(length(FILE_LABEL_PAIRS))

        fig = plot_pruning_and_sparsity_at_threshold(
            u_values, u_indices, FILE_LABEL_PAIRS, fit_params,
            overlap_bounds, sparsity_data, norm_overlap_data, hilbert_space_sizes, output_dir,
            antihermitian_val, custom_ref_state_arg_val, cmap; selected_U=8, threshold=0.95
        )

        @assert fig isa Figure "Output must be a Makie Figure"
        println("TEST PASSED SUCCESSFULLY: plot_pruning_and_sparsity_at_threshold returned a valid Figure.")
    end
end

main()
