#=
verify_pruning_plots_notebook_merge.jl

Standalone verification (no Plots/CairoMakie/Zygote — kept light for this host's
tight per-user memory cgroup) that the data-loading logic being merged into
final_analysis.ipynb correctly falls back from the (often absent, in the
currently configured ED_DATA_ROOT) meta_data_and_E.jld2 to reading
HubbardBasis_XDiag_*.h5 / HubbardED_XDiag_*.h5 directly, and that the pruning
curve-fitting loop still runs end-to-end against real data.
=#

using Lattices
using LinearAlgebra
using SparseArrays
using HDF5
using JLD2
using LsqFit
using Statistics

include("../utility_functions.jl")
include("../ed_objects.jl")
include("../ed_functions.jl")
include("../data_path.jl")

const PRUNING_FILE_LABEL_PAIRS = [
    ("3x2 (2,2)", "N=(2, 2)_3x2", (2, 2)),
    ("3x2 (3,3)", "N=(3, 3)_3x2_3", (3, 3)),
    ("4x2 (3,3)", "N=(3, 3)_4x2", (3, 3)),
    ("3x3 (3,3)", "N=(3, 3)_3x3", (3, 3)),
    ("4x2 (4,4)", "N=(4, 4)_4x2_2", (4, 4)),
    ("3x3 (4,4)", "N=(4, 4)_3x3_2", (4, 4)),
    ("3x3 (4,5)", "N=(4, 5)_3x3", (4, 5)),
]

const FOLDER = get_data_root()
println("FOLDER = ", FOLDER)

softplus(x, b) = max(x, zero(x)) + log(b) + log1p(exp(-abs(x)) / b)
model(x, p) = @. (1 - p[1] * (x - p[2]) / (1 + abs(p[1] * (x - p[2]))^p[3])^(1 / p[3])) / 2

antihermitian_val = true
custom_ref_state_arg_val = nothing
u_indices = 15:55

# --- Resolve the shared U grid (interaction_data), tolerating missing meta_data_and_E.jld2 ---
interaction_data = nothing
interaction_data_source = nothing
for (_, ref_label, _) in PRUNING_FILE_LABEL_PAIRS
    ref_dir = joinpath(FOLDER, ref_label)
    isdir(ref_dir) || continue
    meta_path = joinpath(ref_dir, "meta_data_and_E.jld2")
    if isfile(meta_path)
        global interaction_data = load_saved_dict(meta_path)["meta_data"]["U_values"]
        global interaction_data_source = meta_path
        break
    end
    ed_files = [f for f in readdir(ref_dir) if occursin("HubbardED", f)]
    if !isempty(ed_files)
        global interaction_data = h5open(joinpath(ref_dir, ed_files[1]), "r") do d
            U = Float64.(read(d, "data/uvec"))
            pushfirst!(U, 0.0)
            U
        end
        global interaction_data_source = joinpath(ref_dir, ed_files[1])
        break
    end
end
@assert interaction_data !== nothing "could not resolve interaction_data from any configured system"
println("interaction_data resolved from: ", interaction_data_source)
println("length(interaction_data) = ", length(interaction_data), " first 3 = ", interaction_data[1:3])

hilbert_space_sizes = Int[]
fit_params2 = []
n_systems_processed = 0

for (label, file_label, _) in PRUNING_FILE_LABEL_PAIRS
    sys_dir = joinpath(FOLDER, file_label)
    if !isdir(sys_dir)
        println("SKIP (missing dir): ", file_label)
        continue
    end

    nsites = prod(parse_lattice_dimension(file_label))
    filename = build_save_name_prefix(
        "pruning_analysis_trotter";
        sites=nsites,
        antihermitian=antihermitian_val,
        custom_ref_state_arg=custom_ref_state_arg_val
    )
    pruning_analysis_path = joinpath(sys_dir, "$(filename).jld2")
    if !isfile(pruning_analysis_path)
        println("SKIP (no pruning_analysis file): ", file_label, " -> ", pruning_analysis_path)
        continue
    end

    meta_data_path = joinpath(sys_dir, "meta_data_and_E.jld2")
    local hilbert_space_size, hs_source
    if isfile(meta_data_path)
        d_meta = load_saved_dict(meta_data_path)
        hilbert_space_size = size(d_meta["all_full_eig_vecs"][1], 2)
        hs_source = "meta_data_and_E.jld2"
    else
        basis_files = [f for f in readdir(sys_dir) if occursin("HubbardBasis", f)]
        if isempty(basis_files)
            println("SKIP (no meta_data_and_E.jld2 or HubbardBasis file): ", file_label)
            continue
        end
        hilbert_space_size = h5open(joinpath(sys_dir, basis_files[1]), "r") do basis_data
            length(read(basis_data["sectors/0/M_sigma"]))
        end
        hs_source = basis_files[1]
    end
    push!(hilbert_space_sizes, hilbert_space_size)

    d = load(pruning_analysis_path)

    curr_fit_params = Vector{Any}(undef, length(u_indices))
    @safe_threads for (idx, i) in collect(enumerate(u_indices))
        filt = d["removed_terms"][:, i] .> 0
        err = max.(abs.(d["error_data"][:, i][filt]), 1e-16)
        overlap = 1 .- err

        x = d["removed_terms"][:, i][filt] ./ maximum(d["removed_terms"][:, i][filt])
        y = (overlap .- overlap[end]) ./ (overlap[1] .- overlap[end])

        filt2 = y .>= y[end]
        weight = 1 ./ (1 .- overlap) .^ 2

        fit = curve_fit(
            model, x[filt2], y[filt2], weight[filt2],
            [1.0, 1.0, 1.0], lower=[-Inf, -Inf, 0.1], upper=[Inf, Inf, 10]
        )
        curr_fit_params[idx] = copy(fit.param)
    end
    push!(fit_params2, curr_fit_params)
    global n_systems_processed += 1

    println("OK: ", file_label, "  hilbert_space_size=", hilbert_space_size, " (source: ", hs_source, ")",
        "  n_fits=", length(curr_fit_params), "  sample_fit_params[1]=", curr_fit_params[1])
end

println()
println("SUMMARY: processed $n_systems_processed / $(length(PRUNING_FILE_LABEL_PAIRS)) configured systems")
@assert n_systems_processed > 0 "no systems were successfully processed — adaptation is broken"
println("VERIFICATION PASSED")
