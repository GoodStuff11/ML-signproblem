#=
pruning_plots.jl

Generates three pruning analysis plots across physical systems for interaction strength U:
  1. Single-U pruning curves (overlap decay vs. sparsity with model fits).
  2. Maximum sparsity at a specified overlap threshold vs. U.
  3. Sparsity curve inflection points vs. U.

Usage:
  julia --project=.. pruning_plots.jl [--antihermitian=<true|false>] [--custom_ref_state=<true|false|slater|none>]

Command-Line Options:
  --antihermitian (optional):
    Specify whether antihermitian generators were used when saving pruning analysis files.
    - true (default): Append '_antihermitian' to the build_save_name_prefix.
    - false: Omit '_antihermitian' from the prefix.
    Can be specified as `--antihermitian` (sets true), `--antihermitian=true`, or `--antihermitian=false`.

  --custom_ref_state (optional):
    Specify whether custom reference state 'slater' was used during pruning analysis.
    - true / slater: Set custom_ref_state_arg = "slater" (appends '_ref_slater' to prefix).
    - false / none (default): Set custom_ref_state_arg = nothing.
    Can be specified as `--custom_ref_state` (sets true), `--custom_ref_state=true`, `--custom_ref_state=false`,
    `--custom_ref_state=slater`, or `--custom_ref_state=none`.
=#

using Lattices
using LinearAlgebra
using Combinatorics
using SparseArrays
using Plots
# using CairoMakie
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
# using CUDA
using HDF5


include("utility_functions.jl")
include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("data_path.jl")
include("logging.jl")


file_label_pair = [
    (L"3\times 2\;(2,2)", "N=(2, 2)_3x2", (2, 2)),
    (L"3\times 2,\;(3,3)", "N=(3, 3)_3x2_3", (3, 3)),
    # (L"3\times 2\;(N_\uparrow, N_\downarrow)=(3,3)","N=(3, 3)_3x2_2", (3,3)),
    (L"4\times2,\;(3,3)", "N=(3, 3)_4x2", (3, 3)),
    (L"3\times3,\;(3,3)", "N=(3, 3)_3x3", (3, 3)),
    # (L"4\times3,\;(3,3)", "N=(4, 3)_4x3", (3, 3)), # to be added
    (L"4\times2,\;(4,4)", "N=(4, 4)_4x2_2", (4, 4)),
    (L"3\times3,\;(4,4)", "N=(4, 4)_3x3_2", (4, 4)),
    # (L"3\times3,\;(N_\uparrow, N_\downarrow)=(4,5)", "N=(4, 5)_3x3_3", (4,5)),
    (L"3\times3,\;(4,5)", "N=(4, 5)_3x3", (4, 5)),
]

softplus(x, b) = max(x, zero(x)) + log(b) + log1p(exp(-abs(x)) / b)
scaled_softplus(x, p1, p2) = 1 / log(1 + p2) * softplus(p1 * x, p2)
# model(x,p) = @. 1 - (p[6]*scaled_softplus(p[5]*(x-1), p[1], p[2]) + (1-p[6])*scaled_softplus(p[5]*(x-1), p[3], p[4]))
# model(x,p) = @. (1-tanh(p[1]*(x-p[2])))/2
model(x, p) = @. (1 - p[1] * (x - p[2]) / (1 + abs(p[1] * (x - p[2]))^p[3])^(1 / p[3])) / 2

rescale(x, p) = @. x * (p[2] - p[1]) + p[1]

"""
    find_root_bisection(f, a, b; tol=1e-5, max_iters=100)

Find a root of scalar function `f` in range `[a, b]` using bisection. Returns `NaN` if `f(a)` and `f(b)` share the same sign.
"""
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

"""
    parse_arguments(args::Vector{String}) -> (Bool, Union{String, Nothing})

Parse command line arguments for `pruning_plots.jl`. Returns a tuple: `(antihermitian, custom_ref_state_arg)`.
"""
function parse_arguments(args::Vector{String})
    antihermitian::Bool = true
    custom_ref_state::Bool = false

    for arg in args
        if startswith(arg, "--antihermitian")
            if occursin("=", arg)
                val_str = String(split(arg, "=", limit=2)[2])
                antihermitian = parse(Bool, val_str)
            else
                antihermitian = true
            end
        elseif startswith(arg, "--custom_ref_state")
            if occursin("=", arg)
                val_str = lowercase(String(split(arg, "=", limit=2)[2]))
                if val_str in ("true", "1", "slater")
                    custom_ref_state = true
                elseif val_str in ("false", "0", "none", "nothing")
                    custom_ref_state = false
                else
                    error("Invalid value for --custom_ref_state: '$val_str'. Options: true/false, slater/none")
                end
            else
                custom_ref_state = false
            end
        elseif startswith(arg, "--")
            error("Unknown option: $arg")
        end
    end

    custom_ref_state_arg = custom_ref_state ? "slater" : nothing
    return antihermitian, custom_ref_state_arg
end

"""
    plot_pruning_curves(u_values, u_indices, file_label_pair, fit_params, overlap_bounds, sparsity_data, norm_overlap_data, output_dir, antihermitian_val, custom_ref_state_arg_val, cmap; selected_U=8)

Calculation 1: Generate and save pruning curve plot at `selected_U` across all physical systems.
"""
function plot_pruning_curves(
    u_values,
    u_indices,
    file_label_pair,
    fit_params,
    overlap_bounds,
    sparsity_data,
    norm_overlap_data,
    output_dir::String,
    antihermitian_val::Bool,
    custom_ref_state_arg_val,
    cmap;
    selected_U=8
)
    selected_u_idx = argmin(abs.(u_values[u_indices] .- selected_U)) + u_indices[1] - 1
    pruning_plot = plot(
        xlabel=L"\textrm{Sparsity}",
        ylabel=L"|\langle E_0(U)|\mathcal{U}|E_0(0)\rangle|^2",
        thickness_scaling=1.3,
        framestyle=:box,
        dpi=200,
        legend=:bottomleft
    )

    plot_idx_local = findfirst(==(selected_u_idx), u_indices)

    for (color_i, (label, _, _)) in enumerate(file_label_pair)
        if !isnothing(plot_idx_local)
            fit_p = fit_params[color_i][plot_idx_local]
            bounds = overlap_bounds[color_i][plot_idx_local]
            sparsity_pts = sparsity_data[color_i][plot_idx_local]
            norm_overlap_pts = norm_overlap_data[color_i][plot_idx_local]
            overlap_pts = rescale(norm_overlap_pts, bounds)

            plot!(
                pruning_plot,
                LinRange(0, 1, 200),
                rescale(model(LinRange(0, 1, 200), fit_p), bounds),
                label=nothing,
                color=cmap[color_i],
                linestyle=:dash
            )
            scatter!(
                pruning_plot,
                sparsity_pts,
                overlap_pts,
                color=cmap[color_i],
                label=label,
                legend=:left
            )
        end
    end

    println("U=$(u_values[selected_u_idx])")

    u_val_str = round(u_values[selected_u_idx], digits=2)
    out_filename = build_save_name_prefix(
        "U=$(u_val_str)_pruning_curve";
        antihermitian=antihermitian_val,
        custom_ref_state_arg=custom_ref_state_arg_val
    )
    savefig(pruning_plot, joinpath(output_dir, "$(out_filename).pdf"))
    savefig(pruning_plot, joinpath(output_dir, "$(out_filename).png"))

    display(pruning_plot)
    return pruning_plot
end

"""
    plot_sparsity_at_threshold(u_values, u_indices, file_label_pair, fit_params, overlap_bounds, hilbert_space_sizes, output_dir, antihermitian_val, custom_ref_state_arg_val, cmap; threshold=0.998)

Calculation 2: Calculate and plot maximum sparsity achieved at overlap `threshold` across interaction strength U.
"""
function plot_sparsity_at_threshold(
    u_values,
    u_indices,
    file_label_pair,
    fit_params,
    overlap_bounds,
    hilbert_space_sizes,
    output_dir::String,
    antihermitian_val::Bool,
    custom_ref_state_arg_val,
    cmap;
    threshold=0.998
)
    sparsity_at_threshold = []
    p = plot(
        xlim=(0, 15),
        ylim=(0, 1),
        xlabel=L"U",
        ylabel=L"\textrm{Max\;sparsity\;at\;threshold\;}" * "\n" * L"|\langle E_0(U)|\mathcal{U}|E_0(0)\rangle|^2 \geq %$(threshold)",
        legend=:outerright,
        thickness_scaling=1.3,
        framestyle=:box,
        dpi=200
    )

    for (i, (param_u, (label, _, _), bounds_u, hs_size)) in enumerate(zip(fit_params, file_label_pair, overlap_bounds, hilbert_space_sizes))
        push!(sparsity_at_threshold, Float64[])
        for (param, bounds) in zip(param_u, bounds_u)
            sparsity_val = find_root_bisection(s -> rescale(model(s, param), bounds) - threshold, -1.0, 2.0)
            push!(sparsity_at_threshold[end], sparsity_val)
        end
        plot!(
            p,
            u_values[u_indices],
            sparsity_at_threshold[end],
            label=label,
            linewidth=2,
            markershape=:circle,
            c=cmap[i],
            markersize=3
        )
    end

    out_filename = build_save_name_prefix(
        "pruning_data_overlap_threshold_$(threshold)";
        antihermitian=antihermitian_val,
        custom_ref_state_arg=custom_ref_state_arg_val
    )
    savefig(p, joinpath(output_dir, "$(out_filename).pdf"))
    savefig(p, joinpath(output_dir, "$(out_filename).png"))

    display(p)
    return p
end

"""
    plot_sparsity_inflection_points(u_values, u_indices, file_label_pair, fit_params, fit_errors, overlap_bounds, hilbert_space_sizes, output_dir, antihermitian_val, custom_ref_state_arg_val, cmap; use_ribbon=false, max_error=0.8, min_u_spacing=0.5)

Calculation 3: Plot sparsity inflection points (fitted parameter `p[2]`) with standard error bars (`stderror`) across interaction strength U.
Strides error bars based on minimum physical distance in U space (`min_u_spacing`) to ensure clean spacing on the linear U axis.
"""
function plot_sparsity_inflection_points(
    u_values,
    u_indices,
    file_label_pair,
    fit_params,
    fit_errors,
    overlap_bounds,
    hilbert_space_sizes,
    output_dir::String,
    antihermitian_val::Bool,
    custom_ref_state_arg_val,
    cmap;
    use_ribbon::Bool=false,
    max_error::Float64=0.8,
    min_u_spacing::Float64=0.1
)
    p = plot(
        xlim=(0, 15),
        ylim=(0, 1),
        xlabel=L"U",
        ylabel=L"\textrm{sparsity\;\,inflection\;\,point}",
        legend=:bottomright,
        thickness_scaling=1.3,
        framestyle=:box,
        dpi=200
    )

    for (i, (param_u, err_u, (label, _, _), bounds_u, hs_size)) in enumerate(zip(fit_params, fit_errors, file_label_pair, overlap_bounds, hilbert_space_sizes))
        u_vals = u_values[u_indices]
        sparsity_inflections = Float64[]
        sparsity_inflection_errors = Float64[]

        for (param, errs) in zip(param_u, err_u)
            push!(sparsity_inflections, param[2])
            val_err = (length(errs) >= 2 && !isnan(errs[2]) && !isinf(errs[2])) ? errs[2] : 0.0
            # Clamp unphysically large covariance errors from ill-conditioned fits at low U
            val_err = min(val_err, max_error)
            push!(sparsity_inflection_errors, val_err)
        end

        if use_ribbon
            plot!(
                p,
                u_vals,
                sparsity_inflections,
                ribbon=sparsity_inflection_errors,
                fillalpha=0.3,
                label=label,
                linewidth=2,
                markershape=:circle,
                c=cmap[i],
                markersize=3
            )
        else
            # Draw primary trend curve
            plot!(
                p,
                u_vals,
                sparsity_inflections,
                label=label,
                linewidth=2,
                c=cmap[i]
            )

            # Subsample error bars based on linear distance in U space
            sub_indices = Int[]
            last_u = -Inf
            for (k, u_val) in enumerate(u_vals)
                if u_val - last_u >= min_u_spacing
                    push!(sub_indices, k)
                    last_u = u_val
                end
            end

            scatter!(
                p,
                u_vals[sub_indices],
                sparsity_inflections[sub_indices],
                yerror=sparsity_inflection_errors[sub_indices],
                label=nothing,
                c=cmap[i],
                markerstrokecolor=cmap[i],
                markershape=:circle,
                markersize=3,
                markeralpha=0.9,
                linewidth=1.2
            )
        end
    end

    out_filename = build_save_name_prefix(
        "pruning_data_inflection_point";
        antihermitian=antihermitian_val,
        custom_ref_state_arg=custom_ref_state_arg_val
    )
    savefig(p, joinpath(output_dir, "$(out_filename).pdf"))
    savefig(p, joinpath(output_dir, "$(out_filename).png"))

    display(p)
    return p
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "pruning_plots")
    with_logging(log_path) do
        antihermitian_val, custom_ref_state_arg_val = parse_arguments(ARGS)
        folder = get_data_root()

        # Load reference metadata for U values
        ref_file_label = file_label_pair[1][2]
        e_metadata = load_saved_dict(joinpath(folder, ref_file_label, "meta_data_and_E.jld2"))
        u_values = e_metadata["meta_data"]["U_values"]

        hilbert_space_sizes = Int[]
        fit_params = []
        fit_errors = []
        overlap_bounds = []
        sparsity_data = []
        norm_overlap_data = []
        u_indices = 15:55

        for (color_i, (label, file_label, _)) in enumerate(file_label_pair)
            nsites = prod(parse_lattice_dimension(file_label))
            filename = build_save_name_prefix(
                "pruning_analysis_trotter";
                sites=nsites,
                antihermitian=antihermitian_val,
                custom_ref_state_arg=custom_ref_state_arg_val
            )
            pruning_analysis_path = joinpath(folder, file_label, "$(filename).jld2")
            meta_data_path = joinpath(folder, file_label, "meta_data_and_E.jld2")

            d_meta = load_saved_dict(meta_data_path)
            hilbert_space_size = size(d_meta["all_full_eig_vecs"][1], 2)
            push!(hilbert_space_sizes, hilbert_space_size)

            d = load(pruning_analysis_path)

            curr_fit_params = Vector{Any}(undef, length(u_indices))
            curr_fit_errors = Vector{Any}(undef, length(u_indices))
            curr_overlap_bounds = Vector{Any}(undef, length(u_indices))
            curr_sparsity_data = Vector{Any}(undef, length(u_indices))
            curr_norm_overlap_data = Vector{Any}(undef, length(u_indices))

            @safe_threads for (idx, i) in collect(enumerate(u_indices))
                filt = d["removed_terms"][:, i] .> 0
                if abs(u_values[i] - 8) < 0.1 && file_label == "N=(4, 4)_3x3_2"
                    println("ERROR: $((1 .- abs.(d["error_data"][:, i][filt])) .* 100)")
                end

                err = max.(abs.(d["error_data"][:, i][filt]), 1e-16)
                overlap = 1 .- err

                sparsity = d["removed_terms"][:, i][filt] ./ maximum(d["removed_terms"][:, i][filt])
                norm_overlap = (overlap .- overlap[end]) ./ (overlap[1] .- overlap[end])
                curr_overlap_bounds[idx] = [overlap[end], overlap[1]]

                filt2 = norm_overlap .>= norm_overlap[end]

                weight = 1 ./ (1 .- overlap) .^ 2

                fit = curve_fit(
                    model,
                    sparsity[filt2],
                    norm_overlap[filt2],
                    weight[filt2],
                    [1.0, 1.0, 1.0],
                    lower=[-Inf, -Inf, 0.1],
                    upper=[Inf, Inf, 10]
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

        subfolder = antihermitian_val ? "antihermitian" : "extras"
        output_dir = joinpath(@__DIR__, "good_images", subfolder)
        if !isdir(output_dir)
            mkpath(output_dir)
        end

        cmap = palette(:managua, length(file_label_pair))

        # Calculation 1: Plot pruning curves for selected U
        plot_pruning_curves(
            u_values, u_indices, file_label_pair, fit_params,
            overlap_bounds, sparsity_data, norm_overlap_data, output_dir,
            antihermitian_val, custom_ref_state_arg_val, cmap; selected_U=8
        )

        # Calculation 2: Plot max sparsity at overlap threshold
        plot_sparsity_at_threshold(
            u_values, u_indices, file_label_pair, fit_params,
            overlap_bounds, hilbert_space_sizes, output_dir,
            antihermitian_val, custom_ref_state_arg_val, cmap; threshold=0.99
        )

        # Calculation 3: Plot sparsity inflection points
        plot_sparsity_inflection_points(
            u_values, u_indices, file_label_pair, fit_params,
            fit_errors, overlap_bounds, hilbert_space_sizes, output_dir,
            antihermitian_val, custom_ref_state_arg_val, cmap, use_ribbon=true,
        )

        return 0
    end
end
