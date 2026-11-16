"""
    test_excited_state_targets.jl

Backs the excited-state targeting added to `load_h5_ED_data` / `get_slater_ground_state`
and the `--sector`, `--target_state`, `--reference_target`, `--zero_mode_tol` options of
run_trotter_scan_optimization.jl.

Usage (from experimenting/ed):
    julia --project=.. testing/test_excited_state_targets.jl [--skip-old-vs-new] [--skip-excited]

Parts:
  A. For every HubbardED_*.h5 in data_h5_fixed: reference Slater index of the ground-state
     sector under the NEW rule (argmax |c|^2 of the weak-coupling ground state) vs the OLD
     rule (min kinetic energy among |c|^2 > 0.1 max, tie -> max |c|^2), reimplemented here.
     Reported, not asserted: it shows which existing `_ref_slater` runs would get a
     different reference. (--skip-old-vs-new skips it.)
  B. 4x3 (4,4) ground state (defaults): sector 0, target energies = lowest stored energies,
     weak_ground_state and weak_connected references coincide, every target vector is an
     eigenvector of H(U) = H_hop + U H_int with the stored energy.
  C. 4x3 (4,4) slightly excited state (sector=7, q=(2,1), target_state=:lowest): target
     energies = sector 7's lowest, above the ground state at every U; the lowest level of
     sector 7 is non-degenerate (gap to the second stored state, whose eigen-residual is
     checked too, since XDiag's Lanczos can return a spurious second value); residuals.
  D. CLI: parse_arguments + target_prefix_parts; ground-state defaults leave the prefix
     unchanged, excited choices add markers.
  E. "N=(4, 4)_4x3_excited_sigma0" (if present; --skip-excited skips it): target_state=
     :min_abs_energy picks the smallest non-zero |E| per U, zero-mode counts, residuals,
     weak_ground_state refused, weak_connected reference.
"""

using Lattices, LinearAlgebra, Combinatorics, SparseArrays, Statistics, Random, JLD2, HDF5, Zygote, Test

include(joinpath(@__DIR__, "..", "data_path.jl"))
include(joinpath(@__DIR__, "..", "logging.jl"))
include(joinpath(@__DIR__, "..", "utility_functions.jl"))
using .UtilityFunctions
include(joinpath(@__DIR__, "..", "trotter.jl"))
using .Trotter
include(joinpath(@__DIR__, "..", "ed_objects.jl"))
include(joinpath(@__DIR__, "..", "ed_functions.jl"))

# The driver is a standalone @main script: include it inside a module so its main never runs here.
module Driver
include(joinpath(@__DIR__, "..", "run_trotter_scan_optimization.jl"))
end

const ROOT = "/home/jek354/research/data/new_data/data_h5_fixed"
const F43 = joinpath(ROOT, "N=(4, 4)_4x3")
const F43X = joinpath(ROOT, "N=(4, 4)_4x3_excited_sigma0")

"""OLD reference rule (removed from ed_functions.jl), for comparison only."""
function old_reference_index(data, sector)
    L = read(data, "metadata/Lvec")
    prob = abs.(data["data/evecs/$sector"][:, 1, 1])
    labels = read(data, "metadata/basis_labels/$sector")       # (2, dim) packed bit strings
    eps = compute_single_spin_energies(L; ordering=:row_major)
    Ekin(i) = sum(Float64.(digits(labels[1, i], base=2, pad=prod(L))) .* eps) +
              sum(Float64.(digits(labels[2, i], base=2, pad=prod(L))) .* eps)
    best, minE, maxp = -1, Inf, -1.0
    for i in findall(prob .> 0.1 * maximum(prob))
        E = Ekin(i)
        if E < minE - 1e-5 || (abs(E - minE) <= 1e-5 && prob[i] > maxp)
            best, minE, maxp = i, min(E, minE), prob[i]
        end
    end
    return best
end

"""H(U) = H_hop + U H_int in the indexer's momentum basis (same construction as the driver)."""
function momentum_hamiltonian_parts(indexer, Lvec, N)
    Hhop, _, _ = Trotter.TamFermion.HubbardMomentumBasis(1.0, 0.0, Lvec, N; indexer=indexer)
    Hint, _, _ = Trotter.TamFermion.HubbardMomentumBasis(0.0, 1.0, Lvec, N; indexer=indexer)
    return Hhop, Hint
end
residual(H, v) = (v = v / norm(v); E = real(dot(v, H * v)); (E, norm(H * v - E * v)))

function check_targets(label, U, vecs, Ehop, Eint, expected_E; uidx)
    worst_r, worst_dE = 0.0, 0.0
    for u in uidx
        E, r = residual(Ehop + U[u] * Eint, vecs[u, :])
        worst_r = max(worst_r, r); worst_dE = max(worst_dE, abs(E - expected_E[u]))
    end
    println("  $label: max eigen-residual ‖Hψ-Eψ‖ = $worst_r, max |<H> - E_stored| = $worst_dE over U indices $(collect(uidx))")
    @test worst_r < 1e-5   # stored data were converged to ~5e-7; a spurious Ritz vector would be O(0.1)
    @test worst_dE < 1e-8
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "test_excited_state_targets")
    with_logging(log_path) do
        @testset "excited-state targets" begin
            # ── A ───────────────────────────────────────────────────────────────────────
            if !("--skip-old-vs-new" in ARGS)
                println("\n=== A. reference Slater index, OLD rule vs NEW rule (ground-state sector) ===")
                n_same = n_diff = 0
                for folder in sort(readdir(ROOT; join=true)), f in sort(readdir(folder))
                    (occursin("HubbardED", f) && endswith(f, ".h5")) || continue
                    h5open(joinpath(folder, f), "r") do data
                        keys_ = [parse(Int, k) for k in keys(data["data/energies"])]
                        U = Float64.(read(data, "data/uvec"))
                        allE = [real.(read(data, "data/energies/$k"))[1, :] for k in keys_]
                        k0 = find_best_energy_sector(allE, U; labels=keys_)
                        new = get_slater_ground_state(data, k0; verbose=false)
                        old = old_reference_index(data, k0)
                        w = abs2.(data["data/evecs/$k0"][:, 1, argmin(U)])
                        same = new == old
                        same ? (n_same += 1) : (n_diff += 1)
                        println("  $(basename(folder))/$f  sector $k0: old=$old new=$new  " *
                                "|c|^2 old=$(round(w[old]; sigdigits=4)) new=$(round(w[new]; sigdigits=4))  " *
                                (same ? "SAME" : "DIFFERENT"))
                    end
                end
                println("  summary: $n_same same, $n_diff different")
            end

            # ── B ───────────────────────────────────────────────────────────────────────
            println("\n=== B. 4x3 (4,4) ground state, default options ===")
            U, sv, idxr, _, N, _, _, _ = load_ED_data(F43; verbose=false, sign_convention=:spin_first, use_slater_reference=true)
            Lvec = [4, 3]
            Ehop, Eint = momentum_hamiltonian_parts(idxr, Lvec, N)
            E0 = h5open(joinpath(F43, "HubbardED_XDiag_momSlater_4x3_nu_4_nd_4_t_1_m_2.h5")) do d
                Dict(k => real.(read(d, "data/energies/$k")) for k in 0:11)
            end
            ref_gs = argmax(abs2.(sv[1, :]))
            h5open(joinpath(F43, "HubbardED_XDiag_momSlater_4x3_nu_4_nd_4_t_1_m_2.h5")) do d
                @test get_slater_ground_state(d, 0; verbose=true) == ref_gs
                @test get_slater_ground_state(d, 0; verbose=false, reference_target=:weak_connected) == ref_gs
            end
            println("  sector 0 dim=$(size(sv, 2)), reference index $ref_gs (identical for weak_ground_state and weak_connected)")
            check_targets("ground state", U, sv, Ehop, Eint, vcat(NaN, E0[0][1, :]); uidx=2:length(U))

            # ── C ───────────────────────────────────────────────────────────────────────
            println("\n=== C. 4x3 (4,4) slightly excited: sector 7, q=(2,1), lowest state ===")
            Ux, svx, idxx, _, _, _, _, _ = load_ED_data(F43; verbose=true, sign_convention=:spin_first,
                use_slater_reference=true, sector=7, target_state=:lowest)
            @test idxx.k == (3, 2)                                        # 1-based (q_x+1, q_y+1) = q (2,1)
            Ehx, Eix = momentum_hamiltonian_parts(idxx, Lvec, N)
            check_targets("sector-7 targets", Ux, svx, Ehx, Eix, vcat(NaN, E0[7][1, :]); uidx=2:length(Ux))
            exc = E0[7][1, :] .- E0[0][1, :]
            gap = E0[7][2, :] .- E0[7][1, :]
            println("  E_target - E_ground over U: min $(minimum(exc)) (U=$(Ux[1 + argmin(exc)])), max $(maximum(exc))")
            println("  sector-7 gap E1 - E0 over U: min $(minimum(gap)) (U=$(Ux[1 + argmin(gap)])), max $(maximum(gap))")
            @test all(exc .> 0)
            @test minimum(gap) > 1e-3
            # the second stored state of sector 7 must be a true eigenvector for the gap to mean anything
            h5open(joinpath(F43, "HubbardED_XDiag_momSlater_4x3_nu_4_nd_4_t_1_m_2.h5")) do d
                ev = d["data/evecs/7"]
                r2 = maximum(residual(Ehx + Ux[1 + u] * Eix, ComplexF64.(ev[:, 2, u]))[2] for u in (1, 16, 32, 60))
                println("  second stored state of sector 7: max eigen-residual at U=0.25,4,8,15: $r2")
                @test r2 < 1e-5
            end
            ref7 = argmax(abs2.(svx[1, :]))
            weak7 = h5open(d -> d["data/evecs/7"][:, 1, 1], joinpath(F43, "HubbardED_XDiag_momSlater_4x3_nu_4_nd_4_t_1_m_2.h5"))
            println("  reference index $ref7 = argmax |c|^2 of sector-7 state at U=$(minimum(Ux[2:end])): weight $(abs2(weak7[ref7]) / sum(abs2, weak7))")
            @test ref7 == argmax(abs2.(weak7))
            @test_throws ErrorException load_ED_data(F43; sector=99, omit_indexer=true)
            @test_throws ErrorException load_ED_data(F43; target_state=:min_abs_energy, omit_indexer=true)  # lowest-state file refused

            # ── D ───────────────────────────────────────────────────────────────────────
            println("\n=== D. CLI parsing and saved-file prefix markers ===")
            base = Driver.parse_arguments(["N=(4, 4)_4x3", "2", "--custom_ref_state=slater"])
            exc_args = Driver.parse_arguments(["N=(4, 4)_4x3", "2", "--custom_ref_state=slater", "--sector=7"])
            vx = Driver.parse_arguments(["N=(4, 4)_4x3_excited_sigma0", "2", "--custom_ref_state=slater", "--sector=0",
                "--target_state=min_abs_energy", "--reference_target=weak_connected"])
            for (name, a) in (("default", base), ("slightly excited", exc_args), ("very excited", vx))
                println("  $name: target_cli=$(a[end])  prefix parts=$(Driver.target_prefix_parts(a[end]))")
            end
            @test isempty(Driver.target_prefix_parts(base[end]))
            @test Driver.target_prefix_parts(exc_args[end]) == ["sector=7"]
            @test Driver.target_prefix_parts(vx[end]) == ["sector=0", "target=min_abs_energy", "reftarget=weak_connected"]
            @test_throws ErrorException Driver.parse_arguments(["x", "--target_state=bogus"])

            # ── E ───────────────────────────────────────────────────────────────────────
            if !("--skip-excited" in ARGS) && isdir(F43X) && any(f -> startswith(f, "HubbardED"), readdir(F43X))
                println("\n=== E. 4x3 (4,4) very excited: smallest non-zero |E| (sigma = 0 file) ===")
                fx = joinpath(F43X, only(filter(f -> startswith(f, "HubbardED"), readdir(F43X))))
                h5open(fx) do d
                    println("  file $(basename(fx)): eig_target=$(read(d, "metadata/eig_target")), eig_sigma=$(read(d, "metadata/eig_sigma"))")
                    for k in (0, 7)
                        E = real.(read(d, "data/energies/$k"))
                        nz = [count(abs.(E[:, u]) .<= 1e-8) for u in axes(E, 2)]
                        println("  sector $k: zero modes (|E|≤1e-8) per U: $(nz)")
                        @test all(nz .< size(E, 1))
                    end
                    @test_throws ErrorException get_slater_ground_state(d, 0; verbose=false)   # weak_ground_state unavailable
                end
                @test_throws ErrorException load_ED_data(F43X; omit_indexer=true)            # target_state=:lowest refused
                @test_throws ErrorException load_ED_data(F43X; omit_indexer=true, sector=0, target_state=:min_abs_energy)  # weak_ground_state refused
                Uv, svv, idxv, _, _, _, _, _ = load_ED_data(F43X; verbose=true, sign_convention=:spin_first, use_slater_reference=true,
                    sector=0, target_state=:min_abs_energy, reference_target=:weak_connected)
                Ehv, Eiv = momentum_hamiltonian_parts(idxv, Lvec, N)
                Ev = h5open(d -> real.(read(d, "data/energies/0")), fx)
                expected = [begin
                    dd = abs.(Ev[:, u]); dd[dd .<= 1e-8] .= Inf; Ev[argmin(dd), u]
                end for u in axes(Ev, 2)]
                println("  target energies (U=0.25, 4, 8, 15): $(expected[[1, 16, 32, 60]])")
                println("  |E_target| over U: min $(minimum(abs.(expected))), max $(maximum(abs.(expected)))")
                check_targets("sigma=0 targets", Uv, svv, Ehv, Eiv, vcat(NaN, expected); uidx=2:length(Uv))
                refv = argmax(abs2.(svv[1, :]))
                println("  reference (weak_connected) index $refv; weight in the U=0.25 target: $(abs2(svv[2, refv]))")
                @test abs2(svv[2, refv]) ≈ maximum(abs2.(svv[2, :]))
                # including zero modes changes the target where they exist
                Uz, svz, _, _, _, _, _, _ = load_ED_data(F43X; verbose=false, sign_convention=:spin_first, omit_indexer=true,
                    use_slater_reference=false, sector=0, target_state=:min_abs_energy, zero_mode_tol=0.0)
                Ez = [real(dot(svz[u, :], (Ehv + Uz[u] * Eiv) * svz[u, :])) for u in (1, 16, 39)]
                println("  zero_mode_tol=0: target energies at U=0.25, 4, 9.75: $Ez")
            else
                println("\n=== E. skipped (excited-state folder not present yet) ===")
            end
        end
    end
end
