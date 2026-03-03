# using Pkg
# Pkg.activate("/home/jek354/research/ML-signproblem")
# Pkg.update()

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

include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("utility_functions.jl")

folder = "data/N=(3, 3)_3x2"
# folder="data/tmp"

e_metadata = load_saved_dict(joinpath(folder, "meta_data_and_E.jld2"))
U_values = e_metadata["meta_data"]["U_values"]
dim = [parse(Int, x) for x in split(e_metadata["meta_data"]["sites"], "x")]
shared_data = load_saved_dict(joinpath(folder, "unitary_map_energy_symmetry=false_N=(3, 3)_shared.jld2"))
coefficient_data = []
interaction_data = []
loss = []
initial_loss = []
for i = 6:61#2:61
    dic = load_saved_dict(joinpath(folder, "unitary_map_energy_symmetry=false_N=(3, 3)_u_$i.jld2"))
    push!(coefficient_data, dic["coefficients"][2])
    push!(interaction_data, U_values[i])
    push!(loss, dic["metrics"]["loss"][2])
    push!(initial_loss, dic["metrics"]["loss"][1])
end


order = 2
x = []
y = []
z = []
z1 = []
z2 = []
for i in eachindex(if shared_data["coefficient_labels"][order] !== nothing
    shared_data["coefficient_labels"][order]
else
    []
end)
    # k1 + k2 -> k1' + k2'
    s = sum(abs2, (collect((shared_data["coefficient_labels"][2][i][1][1] + shared_data["coefficient_labels"][2][i][2][1]).coordinates) .- 2)) # (k1 + k2)^2
    t = sum(abs2, (shared_data["coefficient_labels"][2][i][1][1] - shared_data["coefficient_labels"][2][i][1+order][1]).coordinates) # (k1 - k1')^2
    u = sum(abs2, (shared_data["coefficient_labels"][2][i][1][1] - shared_data["coefficient_labels"][2][i][2+order][1]).coordinates) # (k1 - k2')^2
    # println("($s, $t, $u): $(u_data["coefficient_values"][2][i])")

    # push!(x, (u -t)+(rand()*2-1)*0.2)
    push!(x, s)
    push!(y, t)
    push!(z, u)
    # push!(y, s+(rand()*2-1)*0.2)
end

max_val = maximum([maximum(abs, filter(!isnan, c)) for c in coefficient_data])
clim_val = (-max_val, max_val)

anim = @animate for (idx, z_data) in enumerate(coefficient_data)
    # max_z = maximum(abs, filter(!isnan, z_data))
    # max_z = max_z == 0 ? 1.0 : max_z
    sizes = 1 .+ 8 .* abs.(z_data) ./ max_val

    scatter(x, y, z,
        xlabel=L"$(k_1+k_2)^2$",
        ylabel=L"$(k_1-k_1')^2$",
        zlabel=L"$(k_1-k_2')^2$",
        marker_z=z_data,
        ms=sizes,
        color=:balance,
        clim=clim_val,
        alpha=0.4,
        legend=false,
        aspect_ratio=:equal,
        title="U = $(round(interaction_data[idx], digits=2))"
    )
    # scatter(y, z, marker_z=z_data, xlim=(-0.2, 5.2), ylim=(-0.2, 5.2), xticks=0:5, yticks=0:5, xlabel=L"$(k_1-k_1')^2$", ylabel=L"$(k_1-k_2')^2$", ms=sizes, color=:balance, clim=clim_val, alpha=0.4, legend=false, aspect_ratio=:equal, title="U = $(round(interaction_data[idx], digits=2))", dpi=200)
end

gif(anim, "coefficients_animation3d.gif", fps=5)
