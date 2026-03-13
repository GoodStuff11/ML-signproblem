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
u_idx = 60
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
coefficient_data = reduce(hcat, coefficient_data);




order = 2

labels = if shared_data["coefficient_labels"][order] !== nothing
    shared_data["coefficient_labels"][order]
else
    []
end

# 1. Gather all scatterings and their categories
categories_dict = Dict() # category -> list of indices

for i in eachindex(labels)
    # Extract states: K_j = (kx, ky), S_j = spin
    K = []
    S = []
    for term in 1:4
        # Note: term contains ((kx, ky), spin) or similar
        kx = labels[i][term][1].coordinates[1]
        ky = labels[i][term][1].coordinates[2]
        spin = labels[i][term][2]
        push!(K, (kx, ky))
        push!(S, spin)
    end

    function get_partition(A)
        partition = []
        assigned = falses(4)
        for j in 1:4
            if !assigned[j]
                part = [j]
                assigned[j] = true
                for k in j+1:4
                    if A[k] == A[j]
                        push!(part, k)
                        assigned[k] = true
                    end
                end
                push!(partition, part)
            end
        end
        return partition
    end

    # 2. Compute independent partitions for K and S
    K_part = get_partition(K)
    S_part = get_partition(S)

    # Exchanging k1, k2 and k3, k4 shouldn't affect categories
    perms = [
        [1, 2, 3, 4],
        [2, 1, 3, 4],
        [1, 2, 4, 3],
        [2, 1, 4, 3]
    ]

    canonical_forms = []
    for p in perms
        p_inv = invperm(p)
        mapped_K_part = sort([sort([p_inv[x] for x in part]) for part in K_part])
        mapped_S_part = sort([sort([p_inv[x] for x in part]) for part in S_part])
        push!(canonical_forms, (mapped_K_part, mapped_S_part))
    end

    cat = sort(canonical_forms)[1]

    if !haskey(categories_dict, cat)
        categories_dict[cat] = Int[]
    end
    push!(categories_dict[cat], i)
end

# 3. Print average and std deviations of their coefficient_data by category
println("============ CATEGORY STATISTICS ============")
for (cat, indices) in categories_dict
    cat_data = coefficient_data[indices, :]

    cat_mean_at_end = mean(abs.(cat_data[:, end]))
    cat_std_at_end = std(abs.(cat_data[:, end]))

    println("Category $cat :")
    println("  Count = $(length(indices))")
    println("  Mean (last col) = $cat_mean_at_end")
    if length(indices) > 1
        println("  Std  (last col) = $cat_std_at_end")
    else
        println("  Std  (last col) = NaN (only 1 element)")
    end
    println()
end
println("=============================================")

# 4. Iterate through all k and spin values and associate each with a category
# and print what the scattering corresponds to, as was done before.
for (cat, indices) in categories_dict
    println()

    # Sort indices in this category by their coefficient values
    val_idx = size(coefficient_data, 2) > 0 ? min(u_idx, size(coefficient_data, 2)) : 0
    if val_idx > 0
        sort!(indices, by=i -> coefficient_data[i, val_idx])
    end

    prev_coeff_val = NaN
    group_elems = []

    function print_group()
        if !isempty(group_elems)
            println("    --- Coeff Val: $(prev_coeff_val) ---")
            for elem in group_elems
                println("      $elem")
            end
            empty!(group_elems)
        end
    end

    for i in indices
        S = []
        kx_vals = Int[]
        ky_vals = Int[]
        spins = Int[]
        for term in 1:4
            push!(kx_vals, labels[i][term][1].coordinates[1] - 1)
            push!(ky_vals, labels[i][term][1].coordinates[2] - 1)
            push!(spins, labels[i][term][2])
        end

        # Difference between input (1, 2) and output (3, 4) in momentum
        # Considering conservation of momentum: k1 + k2 - k3 - k4 (mod grid size, but simply difference here)
        kx_diff = [kx_vals[3+l] - kx_vals[3+l] for l in [0, 1]]
        ky_diff = [ky_vals[1+l] - ky_vals[3+l] for l in [0, 1]]

        # Their logic for the old labels:
        dd = kx_vals[1] == kx_vals[3] && ky_vals[1] == ky_vals[3] && kx_vals[2] == kx_vals[4] && ky_vals[2] == ky_vals[4]
        onsite = (kx_vals[1] == kx_vals[3] == kx_vals[2] == kx_vals[4]) && (ky_vals[1] == ky_vals[2] == ky_vals[3] == ky_vals[4])
        pairing = (kx_vals[1] == kx_vals[2] && ky_vals[1] == ky_vals[2]) && (kx_vals[3] == kx_vals[4] && ky_vals[3] == ky_vals[4])
        double_to_split = ((kx_vals[1] == kx_vals[2] && ky_vals[1] == ky_vals[2]) || (kx_vals[3] == kx_vals[4] && ky_vals[3] == ky_vals[4])) && !dd && !pairing

        label = begin
            if dd
                spin_flip = !(spins[1] == spins[3] && spins[2] == spins[4])
                "dd($(abs(kx_vals[1] - kx_vals[2])),$(abs(ky_vals[1] - ky_vals[2]))) " * (spin_flip ? "spin_flip " : "")
            else
                ""
            end *
            if onsite
                "onsite "
            else
                ""
            end *
            if pairing
                "pairing "
            else
                ""
            end *
            if double_to_split
                "double_to_split"
            else
                ""
            end
        end

        # Note: the user was printing index 60 of the row. We'll use size(coefficient_data, 2) to be safe.
        coeff_val = val_idx > 0 ? coefficient_data[i, val_idx] : NaN

        elem_str = "σ=$spins, kx=$kx_vals, ky=$ky_vals, Δkx=$kx_diff, Δky=$ky_diff  $label $cat "

        if isnan(prev_coeff_val) || abs(coeff_val - prev_coeff_val) > 1e-3
            print_group()
            prev_coeff_val = coeff_val
        end
        push!(group_elems, elem_str)
    end
    print_group()
end

