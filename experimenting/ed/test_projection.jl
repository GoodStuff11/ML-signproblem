
using Lattices
using LinearAlgebra
using Combinatorics
using SparseArrays
using Plots
import Graphs
using LaTeXStrings
using Statistics
using Random
using JSON
using JLD2
using KrylovKit
using ExponentialUtilities

include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("utility_functions.jl")

function test_projection()
    println("Setting up simple lattice (2x2) for testing...")
    lattice_dimension = (2, 2)
    lattice = Square(lattice_dimension, Periodic())

    # 2 electrons, 2x2 = 4 sites. N_up=1, N_down=1 ideally for S2 sectors
    # run_ed_lanczos uses (4,4) or 6. Let's use (2,2) for speed.
    N = 4
    subspace = HubbardSubspace(N, lattice)

    println("Subspace dim: $(get_subspace_dimension(subspace))")

    # Create operators
    Sx = create_operator(subspace, :Sx)
    S2 = create_operator(subspace, :S2)

    dim = size(Sx, 1)

    # Random state
    psi = normalize(rand(ComplexF64, dim))

    println("Initial random state.")
    println("Sx expect: $(real(psi' * Sx * psi))")
    println("S2 expect: $(real(psi' * S2 * psi))")

    # Allowed eigenvalues
    # Sx eigenvalues are integers/half-integers: -N/2 ... N/2 etc.
    particle_n = N
    eig_values_Sx = collect(-particle_n/2:1.0:particle_n/2)
    # S2 eigenvalues s(s+1)
    eig_values_S2 = [s * (s + 1) for s in (particle_n%2)/2:1:particle_n/2]

    println("Allowed Sx: $eig_values_Sx")
    println("Allowed S2: $eig_values_S2")

    # Pick a target. Let's say Sx = 0, S2 = 0 (singlet)
    # Be careful if state exists.
    # Actually, let's try to project to something effectively.

    target_Sx = 1
    target_idx_Sx = findfirst(x -> abs(x - target_Sx) < 1e-5, eig_values_Sx)

    target_S2 = 2.0 # Singlet
    target_idx_S2 = findfirst(x -> abs(x - target_S2) < 1e-5, eig_values_S2)

    println("\nProjecting to Sx = $target_Sx ...")
    psi_Sx = project_hermitian(Sx, psi, target_idx_Sx, eig_values_Sx)
    val_Sx = real(psi_Sx' * Sx * psi_Sx)
    println("After Sx proj: Sx = $val_Sx, error = $(abs(val_Sx - target_Sx))")
    println("Norm: $(norm(psi_Sx))")

    println("\nProjecting the result to S2 = $target_S2 ...")
    psi_final = project_hermitian(S2, psi_Sx, target_idx_S2, eig_values_S2)
    val_S2 = real(psi_final' * S2 * psi_final)
    val_Sx_final = real(psi_final' * Sx * psi_final)

    println("Final State:")
    println("Sx = $val_Sx_final")
    println("S2 = $val_S2")

    if abs(val_Sx_final - target_Sx) < 1e-2 && abs(val_S2 - target_S2) < 1e-2
        println("SUCCESS: Projection worked.")
    else
        println("FAILURE: Projection failed.")
    end

end

test_projection()
