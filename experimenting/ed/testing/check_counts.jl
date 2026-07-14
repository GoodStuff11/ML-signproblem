using Lattices
using SparseArrays
using LinearAlgebra
using Combinatorics

include("../utility_functions.jl")
using .UtilityFunctions
include("../ed_objects.jl")
include("../ed_functions.jl")
include("../trotter.jl")
using .Trotter

# Set up lattice and indexer
lvec = [3, 2]
n_up, n_dn = 3, 3
lattice = Square(tuple(lvec...), Periodic())
subspace = HubbardSubspace(n_up, n_dn, lattice; k=(0, 0))
indexer = CombinationIndexer(subspace)

println("--- create_randomized_nth_order_operator count ---")
for conserve_mom in [true, false]
    for omit_H_conj in [true, false]
        t_dict, t_keys = create_randomized_nth_order_operator(2, indexer, true;
            magnitude=0.01, omit_H_conj=omit_H_conj, conserve_spin=true, conserve_momentum=conserve_mom)
        println("conserve_mom=$conserve_mom, omit_H_conj=$omit_H_conj -> length: ", length(t_keys))
    end
end

println("\n--- enumerate_ferm_excitations count ---")
for include_diag in [true, false]
    gates = Trotter.TamFermion.enumerate_ferm_excitations(2, lvec; conserve_mom=true, conserve_sz=true, include_diagonal=include_diag)
    println("include_diagonal=$include_diag -> gates: ", length(gates))
end
