using JLD2

FOLDER = ARGS[1]
shared_candidates = filter(f -> occursin("trotter_N=", f) && occursin("shared", f), readdir(FOLDER))
println("shared files: ", shared_candidates)
for f in shared_candidates
    d = load(joinpath(FOLDER, f))["dict"]
    println(f, " => instructions: ", get(d, "instructions", "MISSING"))
end

exact_shared = filter(f -> occursin("unitary_map_energy_symmetry=false", f) && occursin("shared", f), readdir(FOLDER))
println("\nexact shared files: ", exact_shared)
for f in exact_shared
    d = load(joinpath(FOLDER, f))["dict"]
    println(f, " => instructions: ", get(d, "instructions", "MISSING"))
end

# inspect a u file for metrics keys
for u_i in [10, 30]
    tf = joinpath(FOLDER, "trotter_N=9_ref_slater_antihermitian_u_$(u_i+1).jld2")
    if isfile(tf)
        d = load(tf)["dict"]
        println("\n$tf metrics keys: ", keys(d["metrics"]))
        for k in keys(d["metrics"])
            v = d["metrics"][k]
            println("  $k[end] = ", isempty(v) ? "EMPTY" : v[end])
        end
        println("  loss_type in dict: ", get(d, "loss_type", "MISSING"))
    end
end
