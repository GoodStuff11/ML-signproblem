#=
verify_grow_coefficients.jl

Verifies grow_coefficients() from trotter_optimization.jl for real, by extracting its exact
source text at runtime (not a hand-copied duplicate) and evaluating it directly. Done this way
(rather than `include("../trotter.jl")`) to avoid pulling in Zygote/Optimization/ChainRulesCore,
which this host's tight per-user memory cgroup makes risky to load just for a pure-function test.
=#

src = read("../trotter_optimization.jl", String)
m = match(r"function grow_coefficients.*?\nend"s, src)
@assert m !== nothing "could not find grow_coefficients in trotter_optimization.jl - extraction pattern is stale"
println("--- extracted source ---")
println(m.match)
println("--- end extracted source ---\n")

eval(Meta.parse(m.match))

# --- growing preserves the old (earlier) layers unchanged, zero-pads the new (later) ones ---
old = [1.0, 2.0, 3.0, 4.0]  # old_num_exponentials=2, num_gates=2
grown = grow_coefficients(old, 2, 3, 2)
println("grow_coefficients([1,2,3,4], 2, 3, 2) = ", grown)
@assert grown == [1.0, 2.0, 3.0, 4.0, 0.0, 0.0] "expected old layers preserved + 1 new zero layer appended, got $grown"
@assert length(grown) == 3 * 2

# --- growing to the same size is a no-op (edge case: new == old) ---
same = grow_coefficients(old, 2, 2, 2)
@assert same == old "growing to the same num_exponentials should return the coefficients unchanged"

# --- growing across multiple new layers at once ---
grown_multi = grow_coefficients([1.0, 2.0], 1, 4, 2)
println("grow_coefficients([1,2], 1, 4, 2)      = ", grown_multi)
@assert grown_multi == [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# --- error paths ---
threw_shrink = try
    grow_coefficients(old, 2, 1, 2)
    false
catch e
    println("shrink correctly errored: ", sprint(showerror, e))
    true
end
@assert threw_shrink "shrinking (new < old) should error"

threw_mismatch = try
    grow_coefficients(old, 3, 4, 2)  # old has 4 elements but claims old_num_exponentials=3 (needs 6)
    false
catch e
    println("length mismatch correctly errored: ", sprint(showerror, e))
    true
end
@assert threw_mismatch "mismatched length(old_coeffs) vs old_num_exponentials*num_gates should error"

println("\nVERIFICATION PASSED: grow_coefficients")
