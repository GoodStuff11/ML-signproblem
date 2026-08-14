#=
verify_grow_exponentials_feature.jl

Verifies the num_exponentials=X filename support in build_save_name_prefix
(ed_functions.jl), used by the new --grow_from_exponentials/--grow_mode feature
in run_trotter_scan_optimization.jl.
=#
using HDF5, LinearAlgebra, Lattices, SparseArrays
include("../utility_functions.jl")
include("../ed_objects.jl")
include("../ed_functions.jl")

# --- num_exponentials=1 (default) must be byte-identical to the pre-existing filename
#     scheme, so every already-computed num_exponentials=1 file on disk still matches. ---
name_default = build_save_name_prefix(:trotter; sites=6, antihermitian=true, custom_ref_state_arg="slater")
name_explicit_1 = build_save_name_prefix(:trotter; sites=6, antihermitian=true, custom_ref_state_arg="slater", num_exponentials=1)
println("default (no num_exponentials kwarg):     ", name_default)
println("explicit num_exponentials=1:             ", name_explicit_1)
@assert name_default == name_explicit_1 "num_exponentials=1 must match the no-kwarg default"
@assert name_default == "trotter_N=6_ref_slater_antihermitian" "unexpected default filename: $name_default"

# --- num_exponentials=2 must produce a distinct filename, so it never collides with
#     (or silently overwrites) num_exponentials=1 results in the same folder. ---
name_2 = build_save_name_prefix(:trotter; sites=6, antihermitian=true, custom_ref_state_arg="slater", num_exponentials=2)
println("num_exponentials=2:                      ", name_2)
@assert name_2 != name_default "num_exponentials=2 must differ from the num_exponentials=1 filename"
@assert name_2 == "trotter_N=6_num_exponentials=2_ref_slater_antihermitian" "unexpected filename: $name_2"

println("\nVERIFICATION PASSED: build_save_name_prefix num_exponentials support")
