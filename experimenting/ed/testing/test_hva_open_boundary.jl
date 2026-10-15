#=
test_hva_open_boundary.jl

Tests for the --target_bc=open option of ../run_hva_parameter_sweep.jl, which replaces the periodic
momentum-sector ED target by ground states of the open-boundary Hubbard model computed in the
real-space (n_up, n_dn) sector. Checks:

  1. The HVA's open-boundary gate set (enumerate_ferm_excitations_HVA(...; use_pbc=false)) contains
     exactly the hopping bonds of the open-boundary Hamiltonian (TamFermion.findLatticeEdges(...;
     use_pbc=false)) and one on-site gate per site, so the ansatz is built from the target's own terms.
  2. The open- and periodic-boundary Hamiltonians differ (the open one has (Lx-1)Ly + Lx(Ly-1) bonds).
  3. Running the sweep with --target_bc=open (3x2 (2,2) and 3x3 (3,3), U = 2 and 15): the target and
     reference ground energies it prints agree with
       (a) dense diagonalization of TamFermion.HubbardRealSpace(...; use_pbc=false) in Julia, and
       (b) Tamra's Python UCC_Hubbard code, TamFermion.Hubbard(t, u, Lvec, nvec, use_pbc=False),
           diagonalized densely with NumPy (an independent implementation),
     and the run writes one CSV row per layer count.
  4. Invalid combinations are rejected: --target_bc=open with --ref=slater, or with --hva_pbc=true.

Usage (from experimenting/ed):
  julia --project=.. testing/test_hva_open_boundary.jl
No command-line arguments. Prints PASS/FAIL per check and exits with status 1 if any check fails.
Output is also written to logs/<date>/test_hva_open_boundary_<timestamp>_<pid>.log.
=#

using LinearAlgebra
using SparseArrays

include(joinpath(@__DIR__, "..", "logging.jl"))
include(joinpath(@__DIR__, "..", "trotter.jl"))
using .Trotter
using .Trotter.TamFermion: HubbardRealSpace, findLatticeEdges

const ED_DIR = normpath(joinpath(@__DIR__, ".."))
const PROJECT = normpath(joinpath(@__DIR__, "..", ".."))
const UCC_HUBBARD = "/home/jek354/research/UCC_Hubbard"
const PYTHON = "/home/jek354/research/.venv/bin/python"
const N_FAILED = Ref(0)

function check(name, ok::Bool, detail="")
    println(ok ? "PASS" : "FAIL", "  ", name, isempty(detail) ? "" : "  ($detail)")
    ok || (N_FAILED[] += 1)
    return ok
end

dense_lowest(Lvec, nvec, u; pbc=false) =
    eigvals(Symmetric(Matrix(HubbardRealSpace(1.0, Float64(u), Lvec, nvec; use_pbc=pbc, returnBasis=false))))[1:3]

function python_lowest(Lvec, nvec, u)
    code = """
import sys; sys.path.insert(0, '$UCC_HUBBARD')
import numpy as np
from TamFermion import Hubbard
H = Hubbard(1.0, $u, ($(Lvec[1]), $(Lvec[2])), ($(nvec[1]), $(nvec[2])), use_pbc=False, returnBasis=False)
print(' '.join(repr(float(x)) for x in np.linalg.eigvalsh(H.toarray())[:3]))
"""
    out = read(`$PYTHON -c $code`, String)
    return parse.(Float64, split(strip(split(strip(out), '\n')[end])))
end

function test_hva_terms_match_open_hamiltonian()
    println("\n== 1-2. Open-boundary HVA terms = open-boundary Hamiltonian terms ==")
    ok = true
    for Lvec in ((3, 2), (2, 3), (3, 3), (4, 3), (4, 2))
        N = prod(Lvec)
        gates, _ = enumerate_ferm_excitations_HVA(Lvec; use_pbc=false, tie=:full)
        bonds_up = Set{Tuple{Int,Int}}()
        n_onsite = 0
        for g in gates
            if g.cre_up == g.ann_up == g.cre_dn == g.ann_dn
                n_onsite += 1
            elseif g.cre_dn == 0 && g.ann_dn == 0
                push!(bonds_up, minmax(trailing_zeros(g.cre_up) + 1, trailing_zeros(g.ann_up) + 1))
            end
        end
        edges = Set(minmax(e...) for e in findLatticeEdges(collect(Lvec); use_pbc=false))
        n_open = (Lvec[1] - 1) * Lvec[2] + Lvec[1] * (Lvec[2] - 1)
        ok &= bonds_up == edges && length(edges) == n_open && n_onsite == N
        println("    $(Lvec[1])x$(Lvec[2]): $(length(bonds_up)) HVA bonds per spin, $(length(edges)) open-boundary edges, $n_onsite on-site gates")
    end
    check("every open-boundary HVA hopping gate is an edge of the open-boundary Hamiltonian and vice versa; one on-site gate per site", ok)
    Lvec, nvec = (3, 3), (3, 3)
    Ho = HubbardRealSpace(1.0, 2.0, Lvec, nvec; use_pbc=false, returnBasis=false)
    Hp = HubbardRealSpace(1.0, 2.0, Lvec, nvec; use_pbc=true, returnBasis=false)
    check("open and periodic 3x3 Hamiltonians differ (ground energies $(round(dense_lowest(Lvec, nvec, 2.0)[1], digits=6)) vs $(round(dense_lowest(Lvec, nvec, 2.0; pbc=true)[1], digits=6)))",
        norm(Matrix(Ho - Hp)) > 1.0)
end

function test_sweep_open_targets()
    println("\n== 3. run_hva_parameter_sweep.jl --target_bc=open: ground energies vs dense Julia and Python UCC_Hubbard ==")
    jl = Base.julia_cmd()
    tmp = mktempdir()
    for (folder, Lvec, nvec) in (("N=(2, 2)_3x2", (3, 2), (2, 2)), ("N=(3, 3)_3x3", (3, 3), (3, 3)))
        for (u_idx, U) in ((8, 2.0), (60, 15.0))
            out = joinpath(tmp, "obc_$(Lvec[1])x$(Lvec[2])_$u_idx.csv")
            log = read(ignorestatus(`$jl --project=$PROJECT $(joinpath(ED_DIR, "run_hva_parameter_sweep.jl")) $folder $u_idx
                --ref=eigenstate --target_bc=open --tie=full --layers=1,2 --runs=1 --maxiters=20 --out=$out`), String)
            m_t = match(r"Exact open-boundary ground energy at U=([0-9.]+): (\S+)", log)
            m_r = match(r"open-boundary reference, U = ([0-9.]+): lowest eigenvalues \[([^\]]+)\]", log)
            if isnothing(m_t) || isnothing(m_r)
                check("$(Lvec[1])x$(Lvec[2]) $nvec U=$U: sweep printed its open-boundary energies", false,
                    "log tail: " * join(last(split(log, '\n'), 8), " | "))
                continue
            end
            u_t, e_t = parse(Float64, m_t[1]), parse(Float64, m_t[2])
            u_r, e_r = parse(Float64, m_r[1]), parse(Float64, strip(split(m_r[2], ',')[1]))
            jd_t, jd_r = dense_lowest(Lvec, nvec, u_t)[1], dense_lowest(Lvec, nvec, u_r)[1]
            py_t, py_r = python_lowest(Lvec, nvec, u_t)[1], python_lowest(Lvec, nvec, u_r)[1]
            println("    $(Lvec[1])x$(Lvec[2]) $nvec: target U=$u_t  sweep $e_t  dense $jd_t  Python $py_t;  reference U=$u_r  sweep $e_r  dense $jd_r  Python $py_r")
            n_rows = isfile(out) ? count(!isempty, readlines(out)) - 1 : 0
            check("$(Lvec[1])x$(Lvec[2]) $nvec, U=$U: target and reference energies match dense Julia and Python UCC_Hubbard to 1e-8; 2 CSV rows",
                u_t == U && abs(e_t - jd_t) < 1e-8 && abs(e_t - py_t) < 1e-8 &&
                abs(e_r - jd_r) < 1e-8 && abs(e_r - py_r) < 1e-8 && n_rows == 2, "rows $n_rows")
        end
    end
end

function test_rejections()
    println("\n== 4. Invalid combinations ==")
    jl = Base.julia_cmd()
    script = joinpath(ED_DIR, "run_hva_parameter_sweep.jl")
    tmp = mktempdir()
    p1 = run(pipeline(ignorestatus(`$jl --project=$PROJECT $script "N=(2, 2)_3x2" 8 --ref=slater --target_bc=open --layers=1 --runs=1 --out=$(joinpath(tmp, "a.csv"))`); stdout=devnull, stderr=devnull))
    p2 = run(pipeline(ignorestatus(`$jl --project=$PROJECT $script "N=(2, 2)_3x2" 8 --ref=eigenstate --target_bc=open --hva_pbc=true --layers=1 --runs=1 --out=$(joinpath(tmp, "b.csv"))`); stdout=devnull, stderr=devnull))
    check("--target_bc=open with --ref=slater exits with an error", p1.exitcode != 0)
    check("--target_bc=open with --hva_pbc=true exits with an error", p2.exitcode != 0)
end

function (@main)(ARGS)
    log_path = make_log_path(ED_DIR, "test_hva_open_boundary")
    with_logging(log_path) do
        test_hva_terms_match_open_hamiltonian()
        test_sweep_open_targets()
        test_rejections()
        println("\n", N_FAILED[] == 0 ? "ALL CHECKS PASSED" : "$(N_FAILED[]) CHECK(S) FAILED")
    end
    return N_FAILED[] == 0 ? 0 : 1
end
