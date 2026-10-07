ENV["JULIA_CUDA_USE_COMPAT"] = "true"
using CUDA
using LinearAlgebra, SparseArrays, Random, Printf, Zygote, Lattices, HDF5, JLD2, Combinatorics, Statistics
using Optimization, OptimizationOptimJL, OptimizationOptimisers

const ED = "/home/jek354/research/ML-signproblem/experimenting/ed"
include(joinpath(ED, "data_path.jl"))
include(joinpath(ED, "utility_functions.jl"))
using .UtilityFunctions
include(joinpath(ED, "trotter.jl"))
include(joinpath(ED, "ed_objects.jl"))
include(joinpath(ED, "ed_functions.jl"))

println("CUDA functional: ", CUDA.functional(), "  device: ", CUDA.name(CUDA.device()))

folder = data_folder("N=(3, 3)_3x3")
U_values, state_vecs, indexer, _, N_elec, _, _, _ =
    load_ED_data(folder; verbose=false, sign_convention=:spin_first, use_slater_reference=false)
Lvec = parse_lattice_dimension(folder); N = prod(Lvec); n_up, n_dn = N_elec
basis_mom = Trotter.get_basis_sector(indexer, Lvec, N)
basis_real, _, _ = Trotter.realspace_basis(Lvec, (n_up, n_dn))
vecs_r = Trotter.momentum_sector_to_realspace(state_vecs, basis_mom, Lvec, (n_up, n_dn))
s1 = vecs_r[1, :]; s2 = vecs_r[8, :]
H = Trotter.TamFermion.HubbardRealSpace(1.0, U_values[8], Lvec, (n_up, n_dn); use_pbc=true, returnBasis=false)
println("real-space dim = $(length(s1)), U = $(U_values[8])")

worst = 0.0
for tie in (:full, :spin, :none), P in (1, 3)
    gates, pmap = Trotter.enumerate_ferm_excitations_HVA(Lvec; use_pbc=false, tie=tie)
    taus = Trotter.fgateToTauSector(gates, N, basis_real; antihermitian=false)
    M = P * Trotter.num_shared_params(pmap, length(gates))
    Random.seed!(1); A = 0.5 .* randn(M)
    for (name, lossfn) in (
        ("overlap", (A, g) -> Trotter.adjoint_loss(A, gates, taus, s1, s2, basis_real, N;
            num_exponentials=P, antihermitian=false, use_gpu=g, param_map=pmap)),
        ("energy", (A, g) -> Trotter.energy_loss(A, gates, taus, H, s1, basis_real, N;
            num_exponentials=P, antihermitian=false, use_gpu=g, param_map=pmap)))
        lc, gc = Zygote.withgradient(a -> lossfn(a, false), A)
        lg, gg = Zygote.withgradient(a -> lossfn(a, true), A)
        dl = abs(lc - lg); dg = maximum(abs.(gc[1] .- gg[1]))
        global worst = max(worst, dl, dg)
        @printf("tie=%-5s P=%d %-8s M=%-4d loss cpu=%.12f gpu=%.12f |dL|=%.1e  max|dgrad|=%.1e  |grad|=%.3e\n",
            tie, P, name, M, lc, lg, dl, dg, norm(gc[1]))
    end
    psi_c = Trotter.apply_unitary(A, gates, s1, basis_real, N, P; use_gpu=false, param_map=pmap)
    psi_g = Array(Trotter.apply_unitary(A, gates, s1, basis_real, N, P; use_gpu=true, param_map=pmap))
    d = norm(psi_c - psi_g); global worst = max(worst, d)
    @printf("tie=%-5s P=%d apply_unitary |psi_cpu - psi_gpu| = %.1e\n", tie, P, d)
end
println("WORST DIFF = $worst")
