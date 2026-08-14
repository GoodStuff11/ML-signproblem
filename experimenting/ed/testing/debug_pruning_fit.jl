using LsqFit, JLD2

FOLDER = "/home/jek354/research/data/new_data/data_h5_fixed"
file_label = "N=(3, 2)_3x2"
filename = "pruning_analysis_trotter_N=6_ref_slater_antihermitian"
pruning_analysis_path = joinpath(FOLDER, file_label, "$(filename).jld2")
println("loading: ", pruning_analysis_path, " exists=", isfile(pruning_analysis_path))

d = load(pruning_analysis_path)
i = 37
filt = d["removed_terms"][:, i] .> 0
err = max.(abs.(d["error_data"][:, i][filt]), 1e-16)
overlap = 1 .- err
x = d["removed_terms"][:, i][filt] ./ maximum(d["removed_terms"][:, i][filt])
y = (overlap .- overlap[end]) ./ (overlap[1] .- overlap[end])
filt2 = y .> y[end]
weight = 1 ./ (1 .- overlap) .^ 2

xf = x[filt2]; yf = y[filt2]; wf = weight[filt2]
println("n points = ", length(xf))
println("x range = ", extrema(xf))
println("y range = ", extrema(yf))

# --- tanh model (2 params), as currently active in the notebook ---
tanh_model(x,p) = @. (1-tanh(p[1]*(x-p[2])))/2
fit_tanh = curve_fit(tanh_model, xf, yf, wf, [10.0, 1.1], lower=[-100.0, -100.0], upper=[100.0, 100.0])
println("tanh fit params: ", fit_tanh.param, " converged=", fit_tanh.converged)

# --- rational-sigmoid model (3 params), as in pruning_plots.jl ---
rational_model(x, p) = @. (1 - p[1] * (x - p[2]) / (1 + abs(p[1] * (x - p[2]))^p[3])^(1 / p[3])) / 2

println("\n--- attempt 1: reusing the notebook's current 2-length p0/bounds (this is likely the bug) ---")
try
    fit_bad = curve_fit(rational_model, xf, yf, wf, [10.0, 1.1], lower=[-100.0, -100.0], upper=[100.0, 100.0])
    println("unexpectedly succeeded: ", fit_bad.param)
catch e
    println("ERROR: ", sprint(showerror, e))
end

println("\n--- attempt 2: proper 3-length p0/bounds matching pruning_plots.jl's original ([-Inf,-Inf,0.1] to [Inf,Inf,10]) ---")
try
    fit_good = curve_fit(rational_model, xf, yf, wf, [1.0, 1.0, 1.0], lower=[-Inf, -Inf, 0.1], upper=[Inf, Inf, 10.0])
    println("params: ", fit_good.param, " converged=", fit_good.converged)
    resid_tanh = sum(abs2, tanh_model(xf, fit_tanh.param) .- yf)
    resid_rational = sum(abs2, rational_model(xf, fit_good.param) .- yf)
    println("SSE tanh = ", resid_tanh, "   SSE rational = ", resid_rational)
catch e
    println("ERROR: ", sprint(showerror, e))
end

println("\n--- attempt 3: better initial guess (p2 inside data domain, p1 order-matched to tanh fit) ---")
for p0 in ([10.0, 0.5, 1.0], [18.0, 0.93, 1.0], [5.0, 0.9, 2.0], [10.0, 0.9, 0.5])
    try
        fit_try = curve_fit(rational_model, xf, yf, wf, p0, lower=[-Inf, -Inf, 0.1], upper=[Inf, Inf, 10.0])
        resid = sum(abs2, rational_model(xf, fit_try.param) .- yf)
        println("p0=$p0 -> params=$(fit_try.param) converged=$(fit_try.converged) SSE=$resid")
    catch e
        println("p0=$p0 -> ERROR: ", sprint(showerror, e))
    end
end

println("\n--- inspecting actual curve shape ---")
perm = sortperm(xf)
for k in perm[1:5]
    println("x=$(xf[k])  y=$(yf[k])")
end
println("...")
for k in perm[end-9:end]
    println("x=$(xf[k])  y=$(yf[k])")
end

println("\n--- weight diagnostics ---")
println("weight range = ", extrema(wf))
println("num points with weight > 1e10 = ", count(>(1e10), wf))
println("overlap range (unfiltered) = ", extrema(overlap))
println("err range = ", extrema(err))

println("\n--- attempt 4: same as attempt 2/3 but with a sane weight cap ---")
wf_capped = min.(wf, 1e6)
println("capped weight range = ", extrema(wf_capped))
for p0 in ([1.0, 1.0, 1.0], [18.0, 0.9, 1.0])
    try
        fit_try = curve_fit(rational_model, xf, yf, wf_capped, p0, lower=[-Inf, -Inf, 0.1], upper=[Inf, Inf, 10.0])
        resid = sum(abs2, rational_model(xf, fit_try.param) .- yf)
        println("p0=$p0 -> params=$(fit_try.param) converged=$(fit_try.converged) SSE=$resid")
    catch e
        println("p0=$p0 -> ERROR: ", sprint(showerror, e))
    end
end

println("\n--- attempt 5: unweighted fit (sanity check) ---")
for p0 in ([1.0, 1.0, 1.0], [18.0, 0.9, 1.0])
    try
        fit_try = curve_fit(rational_model, xf, yf, p0, lower=[-Inf, -Inf, 0.1], upper=[Inf, Inf, 10.0])
        resid = sum(abs2, rational_model(xf, fit_try.param) .- yf)
        println("p0=$p0 -> params=$(fit_try.param) converged=$(fit_try.converged) SSE=$resid")
    catch e
        println("p0=$p0 -> ERROR: ", sprint(showerror, e))
    end
end

resid_tanh_full = sum(abs2, tanh_model(xf, fit_tanh.param) .- yf)
println("\ntanh SSE for comparison = ", resid_tanh_full)
