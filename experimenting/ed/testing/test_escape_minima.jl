#=
test_escape_minima.jl

Tests for the optimizer options added to get out of local minima, and for the warm-start
fix (coefficients saved under an older gate order being silently misapplied):
  1. unit tests: parse_stage_spec / normalize_stages / is_stalled / gate_permutation;
  2. legacy (pre-2026-08-31, unsorted) and current gate orders are the same gate set;
  3. align_warm_start finds the gate order of a vector genuinely optimized in the legacy
     order (not a remapped one) and rejects a shuffled one; a resumed scan keeps optimizing
     in that order and reproduces the saved loss (3x2);
  4. optimize_unitary with a loss-swap stage and basin hops returns the best primary loss (3x2);
  5. real files: the 3x3 u_33 file (informational) and the 4x3 (5,4) u_60 file, saved
     2026-08-25 in the legacy order, must reproduce its saved loss.

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
perm = Trotter.gate_permutation(keys_b, keys_a)
@assert keys_a[perm] == keys_b "gate_permutation wrong: $perm"
@assert Trotter.gate_permutation(keys_a, keys_a) == 1:4
@assert isnothing(Trotter.gate_permutation([keys_a[1:3]; [(UInt64(9), UInt64(0), UInt64(0), UInt64(0))]], keys_a))
@assert isnothing(Trotter.gate_permutation(keys_a[1:3], keys_a))
@assert isnothing(Trotter.gate_permutation(keys_a[[1, 1, 2, 3]], keys_a))
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
cur = Trotter.gate_keys(s.gates)
leg = Trotter.gate_keys(Trotter.enumerate_ferm_excitations(2, s.Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=false, sort_gates=false))
p_leg = Trotter.gate_permutation(leg, cur)                        # s.gates[p_leg] is the legacy order
@assert cur[p_leg] == leg && p_leg != 1:length(cur)
# What a pre-08-31 run really saved: coefficients optimized with the gates in the legacy order.
A_leg, loss_leg, _ = optimize_unitary(s.gates[p_leg], s.tau[p_leg], s.ref, s.target, s.basis, s.N;
    antihermitian=true, maxiters=60, initialization_samples=4, multi_start_samples=2, multi_start_iters=10)
ovp(A, perm; P=1) = adjoint_loss(A, s.gates[perm], s.tau[perm], s.ref, s.target, s.basis, s.N; num_exponentials=P, antihermitian=true)
ov(A) = ovp(A, 1:length(s.gates))
# Neither reading of the vector in the current order reproduces it: not as saved, and not
# with each coefficient moved onto its gate's new position (the old, wrong "remap" fix).
A_remapped = similar(A_leg); A_remapped[p_leg] = A_leg
println("  legacy-order optimum $loss_leg; read in current order: as saved $(ov(A_leg)), remapped $(ov(A_remapped))")
@assert ov(A_leg) > loss_leg + 1e-3 && ov(A_remapped) > loss_leg + 1e-3 "test is vacuous: gate order does not matter here"
@assert isapprox(ovp(A_leg, p_leg), loss_leg; atol=1e-10)
saved = Dict{String,Any}("coefficients" => A_leg, "metrics" => Dict("loss" => [1.0, loss_leg]))
legacy_orders = ["pre-2026-08-31 unsorted gate order" => leg]
eval1 = (A, perm) -> ovp(A, perm)
got = align_warm_start(A_leg, saved, s.gates, eval1; expected_loss=loss_leg, legacy_gate_orders=legacy_orders)
@assert !isnothing(got) && got.perm == p_leg && got.coefficients == A_leg "legacy gate order not recovered"
# grown (2-layer, zero-padded) legacy vector
eval2 = (A, perm) -> ovp(A, perm; P=2)
got2 = align_warm_start(grow_coefficients(A_leg, 1, 2, length(s.gates)), saved, s.gates, eval2;
    expected_loss=loss_leg, legacy_gate_orders=legacy_orders)
@assert !isnothing(got2) && got2.perm == p_leg && isapprox(eval2(got2.coefficients, got2.perm), loss_leg; atol=1e-8)
# a file with recorded gate_keys uses that order (no legacy candidates needed)
saved_k = Dict{String,Any}("coefficients" => A_leg, "gate_keys" => leg, "metrics" => Dict("loss" => [1.0, loss_leg]))
got_k = align_warm_start(A_leg, saved_k, s.gates, eval1; expected_loss=loss_leg)
@assert !isnothing(got_k) && got_k.perm == p_leg
# neighbouring-U style (no expected loss): best candidate order wins if it beats zero coefficients
got_n = align_warm_start(A_leg, saved, s.gates, eval1; legacy_gate_orders=legacy_orders)
@assert !isnothing(got_n) && got_n.perm == p_leg
# an unrecoverable (shuffled) vector is discarded
A_bad = A_leg[randperm(length(A_leg))]
@assert isnothing(align_warm_start(A_bad, saved, s.gates, eval1; expected_loss=loss_leg, legacy_gate_orders=legacy_orders))
println("align_warm_start tests passed")

println("--- 3b. resumed scan keeps the legacy gate order (3x2) ---")
mktempdir() do dir
    U_values, state_vecs, _, _, _, _, _, _ =
        load_ED_data(s.folder; verbose=false, sign_convention=:spin_first, use_slater_reference=true)
    f = joinpath(dir, "legacy_u_33.jld2")
    JLD2.jldsave(f; dict=Dict{String,Any}("u_idx" => 33, "coefficients" => A_leg,
        "metrics" => Dict{String,Any}("loss" => Any[1.0, loss_leg], "optimization_losses" => Any[[loss_leg]])))
    instr = Dict{String,Any}("u_range" => 33:33, "load_file" => f, "num_exponentials" => 1)
    out = Trotter.interaction_scan_map_to_state(state_vecs, instr, s.gates, s.tau, s.basis, s.N;
        maxiters=2, optimizer=:LBFGS, U_values=U_values, antihermitian=true,
        save_folder=dir, save_name="resumed", legacy_gate_orders=legacy_orders)
    final = Float64(last(out["loss_metrics"][end]))   # per-U entry is the loss history
    d = JLD2.load(joinpath(dir, "resumed_u_33.jld2"))["dict"]
    println("  resumed from $loss_leg -> $final after 2 iterations; saved in legacy order: $(d["gate_keys"] == leg)")
    @assert final <= loss_leg + 1e-4 "resumed scan did not start from the saved optimum"
    @assert d["gate_keys"] == leg "resumed run must keep (and record) the legacy gate order"
    # and the file it wrote resumes again, via its recorded gate_keys
    got_r = align_warm_start(d["coefficients"], d, s.gates, eval1; expected_loss=Float64(d["metrics"]["loss"][end]))
    @assert !isnothing(got_r) && got_r.perm == p_leg
end
println("resumed scan test passed")

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

println("--- 5. real files ---")
function check_real_file(name, fname, u_idx; must_recover)
    f = joinpath(data_folder(name), fname)
    if !isfile(f)
        println("  (skipped: $f not found)")
        return
    end
    d = JLD2.load(f)["dict"]
    sr = load_system(name, u_idx)
    evalr = (A, perm) -> adjoint_loss(A, sr.gates[perm], sr.tau[perm], sr.ref, sr.target, sr.basis, sr.N; num_exponentials=1, antihermitian=true)
    legr = Trotter.gate_keys(Trotter.enumerate_ferm_excitations(2, sr.Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=false, sort_gates=false))
    expected = haskey(d, "metrics") ? Float64(d["metrics"]["loss"][end]) : nothing
    println("  $fname: has gate_keys: $(haskey(d, "gate_keys")), saved loss: $expected")
    got = align_warm_start(d["coefficients"], d, sr.gates, evalr; expected_loss=expected,
        legacy_gate_orders=["pre-2026-08-31 unsorted gate order" => legr])
    println("  recovered: $(isnothing(got) ? false : got.name)")
    must_recover && @assert !isnothing(got) "$fname should resume (it reproduces its saved loss in the legacy gate order)"
end
check_real_file("N=(3, 3)_3x3", "trotter_N=9_ref_slater_antihermitian_u_33.jld2", 33; must_recover=false)
check_real_file("N=(5, 4)_4x3", "trotter_N=12_ref_slater_antihermitian_u_60.jld2", 60; must_recover=true)

println("\nALL TESTS PASSED: test_escape_minima")
