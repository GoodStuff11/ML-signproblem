using CairoMakie
using JLD2
using HDF5
using LaTeXStrings

include("../data_path.jl")

const FOLDER = get_data_root()

# Helper for JLD2 dictionary loading
function load_saved_dict(filepath::String)
    return load(filepath)["dict"]
end

# ----------------- PLOT 1 -----------------
const SYSTEMS_PLOT1 = [
    ("3x3 (5,4)", "N=(5, 4)_3x3", 9),
    ("4x3 (5,4)", "N=(5, 4)_4x3", 12),
    ("3x3 (4,4)", "N=(4, 4)_3x3", 9),
    ("4x3 (4,4)", "N=(4, 4)_4x3", 12)
]

fig1 = Figure(size=(1000, 600))
ax1 = Axis(fig1[1, 1],
    xlabel = L"U",
    ylabel = L"1 - |\langle E_0(U)|\mathcal{U}|\psi_{ref}\rangle|^2",
    yscale = log10,
    title = "Interaction Loss vs U"
)

colors1 = [:blue, :forestgreen, :crimson, :purple]
u_indices = 2:60

for (idx, (name, dir, sites)) in enumerate(SYSTEMS_PLOT1)
    sys_dir = joinpath(FOLDER, dir)
    if !isdir(sys_dir)
        continue
    end
    
    # 1. Slater reference run (solid)
    prefix_slater = "trotter_N=$(sites)_ref_slater_antihermitian"
    u_vals_s = Float64[]
    losses_s = Float64[]
    for u_idx in u_indices
        f = joinpath(sys_dir, "$(prefix_slater)_u_$(u_idx).jld2")
        if isfile(f)
            d = load_saved_dict(f)
            push!(u_vals_s, (u_idx - 1) * 0.25)
            push!(losses_s, max(length(d["metrics"]["loss"]) >= 2 ? d["metrics"]["loss"][2] : d["metrics"]["loss"][1], 1e-12))
        end
    end
    
    # 2. Non-Slater reference run (dashed)
    prefix_noslater = "trotter_N=$(sites)_antihermitian"
    u_vals_n = Float64[]
    losses_n = Float64[]
    for u_idx in u_indices
        f = joinpath(sys_dir, "$(prefix_noslater)_u_$(u_idx).jld2")
        if isfile(f)
            d = load_saved_dict(f)
            push!(u_vals_n, (u_idx - 1) * 0.25)
            push!(losses_n, max(length(d["metrics"]["loss"]) >= 2 ? d["metrics"]["loss"][2] : d["metrics"]["loss"][1], 1e-12))
        end
    end
    
    # Plot Slater (solid)
    if !isempty(losses_s)
        lines!(ax1, u_vals_s, losses_s, color=colors1[idx], linewidth=2.5, linestyle=:solid, label="$name (Slater)")
    end
    # Plot non-Slater (dashed)
    if !isempty(losses_n)
        lines!(ax1, u_vals_n, losses_n, color=colors1[idx], linewidth=2.5, linestyle=:dash, label="$name (No Slater)")
    end
    
    # Run garbage collection to free memory
    GC.gc()
end

axislegend(ax1, position=:rt, backgroundcolor=(:white, 0.8))
mkpath("good_images/final")
save("good_images/final/interaction_loss_vs_U.png", fig1)
save("good_images/final/interaction_loss_vs_U.pdf", fig1)
println("Plot 1 saved successfully.")

# ----------------- PLOT 2 -----------------
target_u_idx = 33 # corresponds to U = 8.0
target_u_val = (target_u_idx - 1) * 0.25

const SYSTEMS_PLOT2 = [
    ("3x3 (5,4)", "N=(5, 4)_3x3", 9, 1764, 504, true),
    ("4x3 (5,4)", "N=(5, 4)_4x3", 12, 32670, 1368, true),
    ("3x3 (4,4)", "N=(4, 4)_3x3", 9, 1764, 504, false),
    ("4x3 (4,4)", "N=(4, 4)_4x3", 12, 20439, 1368, false)
]

fig2 = Figure(size=(1000, 600))
ax2 = Axis(fig2[1, 1],
    xlabel = "Hilbert Space Size",
    ylabel = L"1 - |\langle E_0(U)|\mathcal{U}|\psi_{ref}\rangle|^2",
    yscale = log10,
    xscale = log10,
    title = "Loss vs Hilbert Space Size at U = $target_u_val"
)
xlims!(ax2, 1000, 120000)


slater_exists_x_s = Float64[]
slater_exists_y_s = Float64[]
slater_noexists_x_s = Float64[]
slater_noexists_y_s = Float64[]

slater_exists_x_n = Float64[]
slater_exists_y_n = Float64[]
slater_noexists_x_n = Float64[]
slater_noexists_y_n = Float64[]

for (name, dir, sites, hs_size, dof, exists) in SYSTEMS_PLOT2
    sys_dir = joinpath(FOLDER, dir)
    if !isdir(sys_dir)
        continue
    end
    
    # 1. Slater reference
    f_s = joinpath(sys_dir, "trotter_N=$(sites)_ref_slater_antihermitian_u_$(target_u_idx).jld2")
    if isfile(f_s)
        d = load_saved_dict(f_s)
        loss_val = max(length(d["metrics"]["loss"]) >= 2 ? d["metrics"]["loss"][2] : d["metrics"]["loss"][1], 1e-12)
        if exists
            push!(slater_exists_x_s, hs_size)
            push!(slater_exists_y_s, loss_val)
        else
            push!(slater_noexists_x_s, hs_size)
            push!(slater_noexists_y_s, loss_val)
        end
        text!(ax2, hs_size, loss_val; text = "  $name (dof=$dof)", align = (:left, :center), fontsize = 12)
    end
    
    # 2. No Slater reference
    f_n = joinpath(sys_dir, "trotter_N=$(sites)_antihermitian_u_$(target_u_idx).jld2")
    if isfile(f_n)
        d = load_saved_dict(f_n)
        loss_val = max(length(d["metrics"]["loss"]) >= 2 ? d["metrics"]["loss"][2] : d["metrics"]["loss"][1], 1e-12)
        if exists
            push!(slater_exists_x_n, hs_size)
            push!(slater_exists_y_n, loss_val)
        else
            push!(slater_noexists_x_n, hs_size)
            push!(slater_noexists_y_n, loss_val)
        end
        text!(ax2, hs_size, loss_val; text = "  $name (dof=$dof)", align = (:left, :center), fontsize = 12)
    end
    
    GC.gc()
end

# Draw lines
if length(slater_exists_x_s) == 2
    p = sortperm(slater_exists_x_s)
    lines!(ax2, slater_exists_x_s[p], slater_exists_y_s[p], color=:blue, linestyle=:dash, linewidth=2.5)
end
if length(slater_noexists_x_s) == 2
    p = sortperm(slater_noexists_x_s)
    lines!(ax2, slater_noexists_x_s[p], slater_noexists_y_s[p], color=:blue, linestyle=:solid, linewidth=2.5)
end

if length(slater_exists_x_n) == 2
    p = sortperm(slater_exists_x_n)
    lines!(ax2, slater_exists_x_n[p], slater_exists_y_n[p], color=:crimson, linestyle=:dash, linewidth=2.5)
end
if length(slater_noexists_x_n) == 2
    p = sortperm(slater_noexists_x_n)
    lines!(ax2, slater_noexists_x_n[p], slater_noexists_y_n[p], color=:crimson, linestyle=:solid, linewidth=2.5)
end

scatter!(ax2, [slater_exists_x_s; slater_noexists_x_s], [slater_exists_y_s; slater_noexists_y_s], color=:blue, marker=:circle, markersize=10)
scatter!(ax2, [slater_exists_x_n; slater_noexists_x_n], [slater_exists_y_n; slater_noexists_y_n], color=:crimson, marker=:circle, markersize=10)

elem_blue = LineElement(color=:blue, linestyle=:solid, linewidth=2.5)
elem_red = LineElement(color=:crimson, linestyle=:solid, linewidth=2.5)
elem_solid = LineElement(color=:black, linestyle=:solid, linewidth=2.5)
elem_dash = LineElement(color=:black, linestyle=:dash, linewidth=2.5)

Legend(fig2[1, 2],
    [elem_blue, elem_red, elem_solid, elem_dash],
    ["Slater Reference", "No Slater Reference", "Slater State Doesn't Exist", "Slater State Exists"]
)

save("good_images/final/loss_vs_hilbert_space_size.png", fig2)
save("good_images/final/loss_vs_hilbert_space_size.pdf", fig2)
println("Plot 2 saved successfully.")
