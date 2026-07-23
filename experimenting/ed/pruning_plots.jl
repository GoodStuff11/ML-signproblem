#=
pruning_plots.jl

Generates pruning analysis plots across physical systems for interaction strength U.
Fits pruning overlap decay using a non-linear scaling model and visualizes fitted curves
alongside empirical data points.

Usage:
  julia --project=.. pruning_plots.jl [--antihermitian=<true|false>] [--slater_custom_ref=<true|false|slater|none>]

Command-Line Options:
  --antihermitian (optional):
    Specify whether antihermitian generators were used when saving pruning analysis files.
    - true (default): Append '_antihermitian' to the build_save_name_prefix.
    - false: Omit '_antihermitian' from the prefix.
    Can be specified as `--antihermitian` (sets true), `--antihermitian=true`, or `--antihermitian=false`.

  --slater_custom_ref (optional):
    Specify whether custom reference state 'slater' was used during pruning analysis.
    - true / slater: Set custom_ref_state_arg = "slater" (appends '_ref_slater' to prefix).
    - false / none (default): Set custom_ref_state_arg = nothing.
    Can be specified as `--slater_custom_ref` (sets true), `--slater_custom_ref=true`, `--slater_custom_ref=false`,
    `--slater_custom_ref=slater`, or `--slater_custom_ref=none`.
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
    parse_arguments(args::Vector{String}) -> (Bool, Union{String, Nothing})

Parse command line arguments for `pruning_plots.jl`. Returns a tuple: `(antihermitian, custom_ref_state_arg)`.
"""
function parse_arguments(args::Vector{String})
    antihermitian::Bool = true
    slater_custom_ref::Bool = false

    for arg in args
        if startswith(arg, "--antihermitian")
            if occursin("=", arg)
                val_str = String(split(arg, "=", limit=2)[2])
                antihermitian = parse(Bool, val_str)
            else
                antihermitian = true
            end
        elseif startswith(arg, "--slater_custom_ref")
            if occursin("=", arg)
                val_str = lowercase(String(split(arg, "=", limit=2)[2]))
                if val_str in ("true", "1", "slater")
                    slater_custom_ref = true
                elseif val_str in ("false", "0", "none", "nothing")
                    slater_custom_ref = false
                else
                    error("Invalid value for --slater_custom_ref: '$val_str'. Options: true/false, slater/none")
                end
            else
                slater_custom_ref = false
            end
        elseif startswith(arg, "--")
            error("Unknown option: $arg")
        end
    end

    custom_ref_state_arg = slater_custom_ref ? "slater" : nothing
    return antihermitian, custom_ref_state_arg
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "pruning_plots")
    with_logging(log_path) do
        antihermitian_val, custom_ref_state_arg_val = parse_arguments(ARGS)
        folder = get_data_root()

        # Load reference metadata for U values
        ref_file_label = file_label_pair[1][2]
        e_metadata = load_saved_dict(joinpath(folder, ref_file_label, "meta_data_and_E.jld2"))
        interaction_data = e_metadata["meta_data"]["U_values"]

        hilbert_space_sizes = []
        fit_params2 = []
        rescaling_vals = []
        u_indices = 15:55

        selected_U = 8
        pruning_plot_u_idx = argmin(abs.(interaction_data[u_indices] .- selected_U)) + u_indices[1] - 1
        pruning_plot = plot(
            xlabel=L"\textrm{Sparsity}",
            ylabel=L"|\langle E_0(U)|\mathcal{U}|E_0(0)\rangle|^2",
            thickness_scaling=1.3,
            framestyle=:box,
            dpi=200,
            legend=:bottomleft
        )
        cmap = palette(:managua, length(file_label_pair))

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
            line_width = sqrt(hilbert_space_size) / 20
            push!(hilbert_space_sizes, hilbert_space_size)

            d = load(pruning_analysis_path)

            sys_fit_params = Vector{Any}(undef, length(u_indices))
            sys_rescaling_vals = Vector{Any}(undef, length(u_indices))
            sys_x_filt2 = Vector{Any}(undef, length(u_indices))
            sys_y_filt2 = Vector{Any}(undef, length(u_indices))

            @safe_threads for (idx, i) in collect(enumerate(u_indices))
                filt = d["removed_terms"][:, i] .> 0
                if abs(interaction_data[i] - 8) < 0.1 && file_label == "N=(4, 4)_3x3_2"
                    println("ERROR: $((1 .- abs.(d["error_data"][:, i][filt])) .* 100)")
                end

                err = max.(abs.(d["error_data"][:, i][filt]), 1e-16)
                overlap = 1 .- err

                x = d["removed_terms"][:, i][filt] ./ maximum(d["removed_terms"][:, i][filt])
                y = (overlap .- overlap[end]) ./ (overlap[1] .- overlap[end])
                sys_rescaling_vals[idx] = [overlap[end], overlap[1]]

                filt2 = y .>= y[end]

                weight = 1 ./ (1 .- overlap) .^ 2

                fit = curve_fit(
                    model,
                    x[filt2],
                    y[filt2],
                    weight[filt2],
                    [1.0, 1.0, 1.0],
                    lower=[-Inf, -Inf, 0.1],
                    upper=[Inf, Inf, 10]
                )

                sys_fit_params[idx] = copy(fit.param)
                sys_x_filt2[idx] = x[filt2]
                sys_y_filt2[idx] = y[filt2]
            end

            push!(fit_params2, sys_fit_params)
            push!(rescaling_vals, sys_rescaling_vals)

            plot_idx_local = findfirst(==(pruning_plot_u_idx), u_indices)
            if !isnothing(plot_idx_local)
                fit_p = sys_fit_params[plot_idx_local]
                rescale_p = sys_rescaling_vals[plot_idx_local]
                x_pts = sys_x_filt2[plot_idx_local]
                y_pts = sys_y_filt2[plot_idx_local]

                plot!(
                    pruning_plot,
                    LinRange(0, 1, 200),
                    rescale(model(LinRange(0, 1, 200), fit_p), rescale_p),
                    label=nothing,
                    color=cmap[color_i],
                    linestyle=:dash
                )
                scatter!(
                    pruning_plot,
                    x_pts,
                    rescale(y_pts, rescale_p),
                    color=cmap[color_i],
                    label=label,
                    legend=:left
                )
            end
        end

        println("U=$(interaction_data[pruning_plot_u_idx])")

        subfolder = antihermitian_val ? "antihermitian" : "extras"
        output_dir = joinpath(@__DIR__, "good_images", subfolder)
        if !isdir(output_dir)
            mkpath(output_dir)
        end
        u_val_str = round(interaction_data[pruning_plot_u_idx], digits=2)
        out_filename = build_save_name_prefix(
            "U=$(u_val_str)_pruning_curve";
            antihermitian=antihermitian_val,
            custom_ref_state_arg=custom_ref_state_arg_val
        )
        savefig(pruning_plot, joinpath(output_dir, "$(out_filename).pdf"))
        savefig(pruning_plot, joinpath(output_dir, "$(out_filename).png"))

        display(pruning_plot)
    end
end
