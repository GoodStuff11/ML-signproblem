#=
test_escape_minima.jl

Tests for the optimizer options added to get out of local minima, and for the warm-start
fix (coefficients saved under an older gate order being silently misapplied):
  1. unit tests: parse_stage_spec / normalize_stages / is_stalled / remap_coefficients;
  2. legacy (pre-2026-08-31, unsorted) and current gate orders are the same gate set;
  3. align_warm_start recovers a legacy-ordered vector and rejects a shuffled one (3x2);
  4. optimize_unitary with a loss-swap stage and basin hops returns the best primary loss (3x2);
  5. informational: which alignment reproduces the saved loss of the real 3x3 u_33 file.

Read-only on the data folder. Run from experimenting/ed:
  julia --project=.. testing/test_escape_minima.jl
=#

using Lattices
using LinearAlgebra
using SparseArrays
using Random
using JLD2
using HDF5

include(joinpath(@__DIR__, "..", "data_path.jl"))
include(joinpath(@__DIR__, "..", "utility_functions.jl"))
using .UtilityFunctions
include(joinpath(@__DIR__, "..", "trotter.jl"))
using .Trotter
include(joinpath(@__DIR__, "..", "ed_objects.jl"))
include(joinpath(@__DIR__, "..", "ed_functions.jl"))

println("--- 1. unit tests ---")
@assert Trotter.parse_stage_spec("LBFGS") == (opt=:LBFGS,)
@assert Trotter.parse_stage_spec("GD:energy") == (opt=:GD, loss=:energy)
@assert Trotter.parse_stage_spec("LBFGS:overlap:50") == (opt=:LBFGS, loss=:overlap, maxiters=50)
@assert (try Trotter.parse_stage_spec("LBFGS:bogus"); false catch; true end)
st = Trotter.normalize_stages([:LBFGS, (opt=:LBFGS, loss=:energy, maxiters=7), (loss=:energy,)], :overlap, 100)
@assert st == [(opt=:LBFGS, loss=:overlap, maxiters=100), (opt=:LBFGS, loss=:energy, maxiters=7), (opt=:LBFGS, loss=:energy, maxiters=100)]
@assert Trotter.normalize_stages(:LBFGS, :energy, 3) == [(opt=:LBFGS, loss=:energy, maxiters=3)]

steady = collect(range(1.0, 0.0, length=200))                   # constant progress: never stalls
@assert !any(n -> Trotter.is_stalled(steady[1:n], 20, 0.05), 1:200)
plateau = vcat(collect(range(1.0, 0.5, length=50)), fill(0.5, 60)) # flat after 50: stalls
k = findfirst(n -> Trotter.is_stalled(plateau[1:n], 20, 0.01), 1:length(plateau))
@assert !isnothing(k) && 70 <= k <= 72 "expected stall right after 20 flat iterations, got $k"
@assert !Trotter.is_stalled(plateau, 0, 0.01) "window=0 disables the rule"

keys_a = [(UInt64(i), UInt64(0), UInt64(0), UInt64(0)) for i in 1:4]
keys_b = keys_a[[3, 1, 4, 2]]
c = [10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0]                 # 2 layers
r = Trotter.remap_coefficients(c, keys_a, keys_b)
@assert r == [30.0, 10.0, 40.0, 20.0, 3.0, 1.0, 4.0, 2.0] "per-layer remap wrong: $r"
@assert Trotter.remap_coefficients(r, keys_b, keys_a) == c
@assert isnothing(Trotter.remap_coefficients(c, keys_a, [keys_a[1:3]; [(UInt64(9), UInt64(0), UInt64(0), UInt64(0))]]))
@assert isnothing(Trotter.remap_coefficients(c[1:7], keys_a, keys_b))
println("unit tests passed")

println("--- 2. legacy vs current gate order ---")
for Lvec in ([3, 2], [3, 3], [4, 4])
    cur = Trotter.gate_keys(Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=false))
    leg = Trotter.gate_keys(Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=false, sort_gates=false))
    @assert Set(cur) == Set(leg) && length(unique(cur)) == length(cur)
    println("  $Lvec: $(length(cur)) gates, order differs: $(cur != leg)")
end

function load_system(name, u_idx)
    folder = data_folder(name)
    U_values, state_vecs, indexer, _, _, _, _, _ =
        load_ED_data(folder; verbose=false, sign_convention=:spin_first, use_slater_reference=true)
    Lvec = parse_lattice_dimension(folder)
    N = prod(Lvec)
    basis = Trotter.get_basis_sector(indexer, Lvec, N)
    gates = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=false)
    tau = Trotter.fgateToTauSector(gates, N, basis; antihermitian=true)
    # Same convention as interaction_scan_map_to_state: row 1 is the prepended reference state.
    t_idx = size(state_vecs, 1) == length(U_values) + 1 ? u_idx + 1 : u_idx
    return (; folder, Lvec, N, basis, gates, tau, ref=state_vecs[1, :], target=state_vecs[t_idx, :])
end

println("--- 3. align_warm_start (3x2) ---")
Random.seed!(7)
s = load_system("N=(3, 3)_3x2", 33)
ov(A) = adjoint_loss(A, s.gates, s.tau, s.ref, s.target, s.basis, s.N; num_exponentials=1, antihermitian=true)
A_opt, loss_opt, _ = optimize_unitary(s.gates, s.tau, s.ref, s.target, s.basis, s.N;
    antihermitian=true, maxiters=60, initialization_samples=4, multi_start_samples=2, multi_start_iters=10)
cur = Trotter.gate_keys(s.gates)
leg = Trotter.gate_keys(Trotter.enumerate_ferm_excitations(2, s.Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=false, sort_gates=false))
A_legacy = Trotter.remap_coefficients(A_opt, cur, leg)            # what a pre-08-31 run would have saved
saved = Dict{String,Any}("coefficients" => A_legacy, "metrics" => Dict("loss" => [1.0, loss_opt]))
println("  optimized loss $loss_opt; legacy vector read naively: $(ov(A_legacy))")
@assert ov(A_legacy) > loss_opt + 1e-3 "test is vacuous: legacy order gives the same loss"
legacy_orders = ["pre-2026-08-31 unsorted gate order" => leg]
got = align_warm_start(A_legacy, saved, s.gates, ov; expected_loss=loss_opt, legacy_gate_orders=legacy_orders)
@assert !isnothing(got) && isapprox(got, A_opt) "legacy vector not recovered"
# grown (2-layer, zero-padded) legacy vector
ov2(A) = adjoint_loss(A, s.gates, s.tau, s.ref, s.target, s.basis, s.N; num_exponentials=2, antihermitian=true)
got2 = align_warm_start(grow_coefficients(A_legacy, 1, 2, length(s.gates)), saved, s.gates, ov2;
    expected_loss=loss_opt, legacy_gate_orders=legacy_orders)
@assert !isnothing(got2) && isapprox(ov2(got2), loss_opt; atol=1e-8)
# a file with recorded gate_keys is remapped by them
saved_k = Dict{String,Any}("coefficients" => A_legacy, "gate_keys" => leg, "metrics" => Dict("loss" => [1.0, loss_opt]))
@assert isapprox(align_warm_start(A_legacy, saved_k, s.gates, ov; expected_loss=loss_opt), A_opt)
# an unrecoverable (shuffled) vector is discarded
A_bad = A_opt[randperm(length(A_opt))]
@assert isnothing(align_warm_start(A_bad, saved, s.gates, ov; expected_loss=loss_opt, legacy_gate_orders=legacy_orders))
println("align_warm_start tests passed")

println("--- 4. stages with a loss swap + basin hopping (3x2) ---")
Random.seed!(3)
A0 = 0.05 .* randn(length(s.gates))
# Energy stage objective: H = -|target><target| + 0.5*diag(random) is minimized near, but not
# exactly at, the target, so it really is a different landscape from the overlap loss.
Hproj = -sparse(s.target * s.target') + 0.5 * spdiagm(rand(length(s.target)))
A_h, loss_h, m = optimize_unitary(s.gates, s.tau, s.ref, s.target, s.basis, s.N;
    antihermitian=true, H=Hproj, initial_coefficients=A0, maxiters=15,
    optimizer=[:LBFGS, (loss=:energy, maxiters=5), :LBFGS],
    stall_window=5, stall_rtol=0.01, basin_hops=2, hop_iters=10, hop_scale=0.2, hop_seed=11)
ci = m["convergence_info"][end]
@assert [c["loss"] for c in ci] == ["overlap", "energy", "overlap"]
hops = m["basin_hops"][end]
@assert length(hops) == 2
best_seen = minimum(vcat(ov(A0), [c["primary_loss"] for c in ci], [h["final_loss"] for h in hops]))
@assert isapprox(loss_h, best_seen; atol=1e-12) "returned loss $loss_h is not the best seen $best_seen"
@assert isapprox(ov(A_h), loss_h; atol=1e-8) "returned coefficients do not reproduce the returned loss"
println("  stages: ", [(c["loss"], c["primary_loss"], c["stalled"]) for c in ci])
println("  hops: ", [(h["start_loss"], h["final_loss"], h["accepted"]) for h in hops])
println("stage/hop tests passed (final loss $loss_h)")

# Adam stage (regression: `Adam` was ambiguous between Optim and Optimisers)
A_a, loss_a, m_a = optimize_unitary(s.gates, s.tau, s.ref, s.target, s.basis, s.N;
    antihermitian=true, H=Hproj, initial_coefficients=A0, maxiters=15,
    optimizer=[:LBFGS, Trotter.parse_stage_spec("Adam:energy:20"), :LBFGS], stall_window=5)
ci_a = m_a["convergence_info"][end]
@assert [(c["optimizer"], c["loss"]) for c in ci_a] == [("LBFGS", "overlap"), ("Adam", "energy"), ("LBFGS", "overlap")]
@assert isapprox(ov(A_a), loss_a; atol=1e-8) && loss_a <= ov(A0)
println("  Adam stage: ", [(c["optimizer"], c["loss"], c["primary_loss"], c["iterations"]) for c in ci_a])
println("Adam stage test passed (final loss $loss_a)")

println("--- 5. real 3x3 u_33 file (informational) ---")
f33 = joinpath(data_folder("N=(3, 3)_3x3"), "trotter_N=9_ref_slater_antihermitian_u_33.jld2")
if isfile(f33)
    d = JLD2.load(f33)["dict"]
    s3 = load_system("N=(3, 3)_3x3", 33)
    ov3(A) = adjoint_loss(A, s3.gates, s3.tau, s3.ref, s3.target, s3.basis, s3.N; num_exponentials=1, antihermitian=true)
    leg3 = Trotter.gate_keys(Trotter.enumerate_ferm_excitations(2, s3.Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=false, sort_gates=false))
    expected = haskey(d, "metrics") ? Float64(d["metrics"]["loss"][end]) : nothing
    println("  file has gate_keys: $(haskey(d, "gate_keys")), saved loss: $expected")
    got3 = align_warm_start(d["coefficients"], d, s3.gates, ov3; expected_loss=expected,
        legacy_gate_orders=["pre-2026-08-31 unsorted gate order" => leg3])
    println("  recovered: $(!isnothing(got3)) (a file that no alignment reproduces is discarded)")
else
    println("  (skipped: $f33 not found)")
end

println("\nALL TESTS PASSED: test_escape_minima")
