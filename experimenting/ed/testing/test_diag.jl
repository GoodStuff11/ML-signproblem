# test_diag.jl
include("../trotter.jl")
using .Trotter
using LinearAlgebra
using SparseArrays

Lvec = [3, 2]
N_sites = 6
n_up = 2

basis_up, occ = Trotter.getReducedHilSpace(N_sites, n_up; returnOcc=true)
edges = Trotter.findLatticeEdges(Lvec; use_pbc=true)
hop_up = Trotter.fermionNNHopping(basis_up, edges, 1.0)

F_up, _ = Trotter.SlaterCOB_RtoK_nparticle(Lvec, n_up)
H_hop_k = F_up * hop_up * adjoint(F_up)

println("Is H_hop_k diagonal? ", all(abs.(H_hop_k - Diagonal(H_hop_k)) .< 1e-10))
println("Diagonal of H_hop_k:")
for i in 1:length(basis_up)
    println("state ", occ[i, :], " diagonal: ", real(H_hop_k[i, i]))
end
