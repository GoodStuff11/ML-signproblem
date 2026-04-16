using LsqFit
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


include("../ed_objects.jl")
include("../ed_functions.jl")
include("../ed_optimization.jl")
include("../utility_functions.jl")

softplus(x,b) = max(x, zero(x)) + log(b) + log1p(exp(-abs(x))/b)
scaled_softplus(x, p1, p2) = 1/log(1+p2)*softplus(p1*x, p2)
model(x,p) = @. 1-( p[6]*scaled_softplus(p[5]*(x-1), p[1], p[2]) + (1-p[6])*scaled_softplus(p[5]*(x-1), p[3], p[4]))
println(scaled_softplus.([1,0.5], 1, 1.0))

electrons = (3,3)
file_label = "N=$(electrons)_3x2"
folder = joinpath(@__DIR__, "..", "data", "$(file_label)_3")
# folder="data/tmp"

e_metadata = load_saved_dict(joinpath(folder, "meta_data_and_E.jld2"))
interaction_data = e_metadata["meta_data"]["U_values"]

file_label_pair = [
    (L"N_\uparrow=3, N_\downarrow=3, 3\times 2","N=(3, 3)_3x2_3"), 
    (L"N_\uparrow=3, N_\downarrow=3, 3\times2","N=(3, 3)_3x2_2"),
    (L"N_\uparrow=4, N_\downarrow=4, 3\times3", "N=(4, 4)_3x3_2"),
    (L"N_\uparrow=4, N_\downarrow=4, 4\times2", "N=(4, 4)_4x2_2"),
    (L"N_\uparrow=4, N_\downarrow=5, 3\times3", "N=(4, 5)_3x3_3"),
    (L"N_\uparrow=4, N_\downarrow=5, 3\times3", "N=(4, 5)_3x3"),
    ]

# fit_params2 = []


for (label,file_label) in file_label_pair
    # push!(fit_params2, [])
    for i = 22:22

        folder = "/home/jek354/research/ML-signproblem/experimenting/ed/data/$(file_label)/pruning_analysis.jld2"
        folder2 = "/home/jek354/research/ML-signproblem/experimenting/ed/data/$(file_label)/meta_data_and_E.jld2"
        d = load(folder2)["dict"]
        hilbert_space_size = size(d["all_full_eig_vecs"][1],2)
        line_width = sqrt(hilbert_space_size)/20

        d = load(folder)
        filt = d["removed_terms"][:,i] .> 0

        err = max.(abs.(d["error_data"][:,i][filt]),1e-16)
        overlap = 1 .- err
        # rel_err = (err .- err[1]) ./ err
        # println(d["removed_terms"][:,i])
        # println(d["removed_terms"][:,i][filt])
        x = d["removed_terms"][:,i][filt]./maximum(d["removed_terms"][:,i][filt])
        y = (overlap .- overlap[end])./(overlap[1] .- overlap[end])


        filt2 = y .>= y[end] #.||(x .< 0.9 .|| x .>= 0.99)

        lb = [-Inf, 1e-5, -Inf, 1e-5, -Inf, 0.0]
        ub = [Inf, Inf, Inf, Inf, Inf, 1.0]

        fit = nothing
        for iter in 1:4
            # Fit the model using the current inliers
            fit = curve_fit(model, x[filt2], y[filt2], y[filt2].^8, [1.0, 1.0, 1.0, 1.0, 1.0, 0.5], lower=lb, upper=ub)
            
            # Calculate residuals for ALL data points
            residuals = abs.(y .- model(x, fit.param))
            
            # Use the standard deviation of the current inliers' residuals as a threshold basis
            inlier_std = max(std(residuals[filt2]), 1e-8)
            
            # Keep points that are within 2.0 standard deviations (to remove noise of later data)
            new_filt2 = residuals .<= 3.0 * inlier_std
            
            # Stop if the filter has converged
            if new_filt2 == filt2
                break
            end
            
            # Safety check to avoid dropping too many points
            if sum(new_filt2) < length(fit.param) + 2
                break
            end
            
            filt2 = new_filt2
        end
        fit = curve_fit(model, x[filt2], y[filt2], y[filt2].^8, [1.0, 1.0, 1.0, 1.0, 1.0, 0.5], lower=lb, upper=ub)
        
        p = scatter(x[filt2], y[filt2], label="inliers", color=:blue, ylim=(-0.01,1.01))
        if any(.!filt2)
            scatter!(p, x[.!filt2], y[.!filt2], label="outliers", color=:red)
        end
        plot!(p, LinRange(0,1,200), model(LinRange(0, 1,200),fit.param), label="fit", color=:black)
        display(p)
        # push!(fit_params2[end], fit.param)

        
    end
end

# savefig("good_images/extras/U=$(interaction_data[i])_relative_loss.png")
# savefig("good_images/extras/U=$(interaction_data[i])_relative_loss.pdf")

