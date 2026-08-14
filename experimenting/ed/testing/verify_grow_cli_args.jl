#=
verify_grow_cli_args.jl

Verifies parse_arguments()'s new --grow_from_exponentials/--grow_mode flags in
run_trotter_scan_optimization.jl for real, by extracting its exact source text at runtime
(not a hand-copied duplicate) and evaluating it directly alongside the (self-contained,
dependency-free) data_path.jl it calls into - avoiding the heavy Zygote/CUDA/HDF5 imports
at the top of the real file, which aren't needed to test argument parsing itself.
=#

include("../data_path.jl")

src = read("../run_trotter_scan_optimization.jl", String)
m = match(r"function parse_arguments.*?\nend"s, src)
@assert m !== nothing "could not find parse_arguments in run_trotter_scan_optimization.jl - extraction pattern is stale"
eval(Meta.parse(m.match))

# --- default: no grow flags -> nothing/:chain, fully backward compatible ---
result = parse_arguments(["myfolder", "25"])
folder, u_start, u_end, maxiters, loss_type, num_exponentials, antihermitian, custom_ref_state_arg, use_gpu, datatype, grow_from_exponentials, grow_mode = result
println("no grow flags: grow_from_exponentials=$grow_from_exponentials grow_mode=$grow_mode")
@assert grow_from_exponentials === nothing
@assert grow_mode === :chain

# --- --grow_from_exponentials parses to an Int, default grow_mode stays :chain ---
result = parse_arguments(["myfolder", "forward", "--num_exponentials=2", "--grow_from_exponentials=1"])
_, _, _, _, _, num_exponentials, _, _, _, _, grow_from_exponentials, grow_mode = result
println("--grow_from_exponentials=1 --num_exponentials=2: grow_from_exponentials=$grow_from_exponentials ($(typeof(grow_from_exponentials))) grow_mode=$grow_mode num_exponentials=$num_exponentials")
@assert grow_from_exponentials === 1
@assert grow_from_exponentials isa Int
@assert grow_mode === :chain
@assert num_exponentials == 2

# --- --grow_mode=per_u parses correctly ---
result = parse_arguments(["myfolder", "25", "40", "--num_exponentials=3", "--grow_from_exponentials=1", "--grow_mode=per_u"])
_, _, _, _, _, _, _, _, _, _, grow_from_exponentials, grow_mode = result
println("--grow_mode=per_u: grow_mode=$grow_mode")
@assert grow_mode === :per_u

# --- invalid --grow_mode value errors ---
threw_bad_mode = try
    parse_arguments(["myfolder", "25", "--grow_from_exponentials=1", "--grow_mode=bogus"])
    false
catch e
    println("invalid --grow_mode correctly errored: ", sprint(showerror, e))
    true
end
@assert threw_bad_mode

# --- validation: grow_from_exponentials must be < num_exponentials ---
threw_not_smaller = try
    parse_arguments(["myfolder", "25", "--num_exponentials=2", "--grow_from_exponentials=2"])
    false
catch e
    println("grow_from_exponentials >= num_exponentials correctly errored: ", sprint(showerror, e))
    true
end
@assert threw_not_smaller "grow_from_exponentials equal to num_exponentials should be rejected"

threw_larger = try
    parse_arguments(["myfolder", "25", "--num_exponentials=2", "--grow_from_exponentials=3"])
    false
catch e
    println("grow_from_exponentials > num_exponentials correctly errored: ", sprint(showerror, e))
    true
end
@assert threw_larger "grow_from_exponentials greater than num_exponentials should be rejected"

println("\nVERIFICATION PASSED: parse_arguments grow_from_exponentials/grow_mode CLI flags")
