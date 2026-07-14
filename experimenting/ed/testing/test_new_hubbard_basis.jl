# testing/test_new_hubbard_basis.jl
include("../trotter.jl")
using .Trotter
using LinearAlgebra
using Test

function run_test_case(Lvec, nvec, t, u)
    println("==================================================")
    println("Testing Lvec = $Lvec, nvec = $nvec, t = $t, u = $u")
    
    # 1. Real space Hamiltonian
    H_real_default, _ = Trotter.Hubbard(t, u, Lvec, nvec; use_pbc=true, returnBasis=true)
    E_real = sort(real.(eigen(Hermitian(Matrix(H_real_default))).values))
    
    # 2. Orbit momentum basis Hamiltonian
    H_orbit, P_reordered, counts_orbit = Trotter.Hubbard(t, u, Lvec, nvec, :orbit; use_pbc=true)
    E_orbit = sort(real.(eigen(Hermitian(Matrix(H_orbit))).values))
    
    is_unitary = all(abs.(P_reordered' * P_reordered - I) .< 1e-10)
    println("  Orbit CoB Unitary? ", is_unitary)
    @test is_unitary
    
    diff_orbit = maximum(abs.(E_real - E_orbit))
    println("  Orbit eigenvalues max diff: ", diff_orbit)
    @test diff_orbit < 1e-11

    # 3. Slater momentum basis Hamiltonian (Full space)
    H_mom, basis_dict, counts_mom = Trotter.Hubbard(t, u, Lvec, nvec, :momentum; use_pbc=true)
    E_mom = sort(real.(eigen(Hermitian(Matrix(H_mom))).values))
    
    diff_mom = maximum(abs.(E_real - E_mom))
    println("  Slater momentum eigenvalues max diff: ", diff_mom)
    @test diff_mom < 1e-11
    
    @test counts_orbit == counts_mom
    
    # 4. Subspace tests (early vs late filtering)
    println("  Testing momentum subspace (q_target) early vs late filtering:")
    for q in basis_dict["qtot_unique"]
        # Early-filtered Slater momentum basis Hamiltonian
        H_mom_q, basis_dict_q, counts_mom_q = Trotter.Hubbard(t, u, Lvec, nvec, :momentum; use_pbc=true, q_target=q)
        E_mom_q = sort(real.(eigen(Hermitian(Matrix(H_mom_q))).values))
        
        # Late-filtered (sliced) from full H_mom
        subspace_indices = findall(v -> v == q, basis_dict["qtot"])
        H_mom_sliced = H_mom[subspace_indices, subspace_indices]
        E_mom_sliced = sort(real.(eigen(Hermitian(Matrix(H_mom_sliced))).values))
        
        # Check that eigenvalues match
        diff_sliced = maximum(abs.(E_mom_q - E_mom_sliced))
        println("    Sector q = $q: size = $(length(subspace_indices)), early vs late max diff = $diff_sliced")
        @test diff_sliced < 1e-11
        @test counts_mom_q == [length(subspace_indices)]
        
        # Compare with orbit basis subspace
        H_orbit_q, P_reordered_q, counts_orbit_q = Trotter.Hubbard(t, u, Lvec, nvec, :orbit; use_pbc=true, q_target=q)
        E_orbit_q = sort(real.(eigen(Hermitian(Matrix(H_orbit_q))).values))
        
        diff_orbit_q = maximum(abs.(E_mom_q - E_orbit_q))
        @test diff_orbit_q < 1e-11
    end
    println("==================================================")
end

@testset "Hubbard Basis Tests" begin
    # Test case 1: 2x2 lattice, 2 up, 2 down
    run_test_case([2, 2], (2, 2), 1.0, 2.0)
    
    # Test case 2: 3x2 lattice, 2 up, 1 down
    run_test_case([3, 2], (2, 1), 1.0, 4.0)

    # Test case 3: 4x1 lattice, 1 up, 1 down
    run_test_case([4], (1, 1), 1.5, 0.0)
end
