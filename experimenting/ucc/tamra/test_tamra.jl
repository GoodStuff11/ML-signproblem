using Test
using LinearAlgebra
include("TamLib.jl")
include("TamFermion.jl")

using .TamLib
using .TamFermion

@testset "TamLib Tests" begin
    # C-order helpers
    @test ravel_c((1, 2), (3, 4)) == 1 * 4 + 2
    @test unravel_c(6, (3, 4)) == (1, 2)
    
    # Permutations
    @test perm_parity_cyc([0, 1, 2]) == 1
    @test perm_parity_cyc([1, 0, 2]) == -1
    @test cycle_decomp([1, 0, 2]; include_fixed=true) == [[0, 1], [2]]
    
    # Bit manipulations
    @test fastXOR(3) == 0
    @test fastXOR(7) == 1
    @test dec2bin(5, 4) == [0, 1, 0, 1]
    
    # General utility
    @test pascal(3) == [1, 3, 3, 1]
    
    # Lattice coords
    coords = getLatticeCoord([2, 2])
    @test size(coords) == (4, 2)
    @test coords[1, :] == [0, 0]
    @test coords[2, :] == [0, 1]
    @test coords[3, :] == [1, 0]
    @test coords[4, :] == [1, 1]
end

@testset "TamFermion Tests" begin
    # Reduced Hilbert Space
    ints, occ = getReducedHilSpace(4, 2)
    @test length(ints) == 6
    @test size(occ) == (6, 2)
    @test ints[1] == 3 # 0011
    @test occ[1, :] == [1, 2]
    
    # int2occ mapping
    @test int2occ(3, 4) == [1, 2]
    
    # Fermion gates
    cu, au = singleSpinChannelCombos(4, 1, 1)
    # (4 choose 1) * (4 choose 1) = 16 combos where support size = 2 (if 1 cre, 1 ann), but support must be from states. 
    # Actually support size is 1+1=2, so 4C2 = 6 supports. Within support, 2C1 = 2 ways. Total 12.
    # Plus support size 1 (diagonal) where cre=ann. 4C1 = 4 supports. Within support, 1C1=1. Total 4.
    # Sum = 16.
    @test length(cu) == 12
    
    # Hubbard
    H, basis_up, basis_dn = Hubbard(1.0, 2.0, [2], [1, 1])
    @test size(H) == (4, 4) # 2C1 * 2C1 = 2 * 2 = 4
    # Energy levels for 2-site Hubbard with t=1, u=2
    # E = 0 (triplet, degeneracy 3)
    # E = 1 ± sqrt(1^2 + 16) = 1 ± sqrt(17) (from hopping amplitude 2t)
    eigs = eigvals(Matrix(H))
    @test isapprox(eigs[1], 1 - sqrt(17), atol=1e-10)
    @test isapprox(eigs[2], 0.0, atol=1e-10)
    @test isapprox(eigs[3], 2.0, atol=1e-10)
    @test isapprox(eigs[4], 1 + sqrt(17), atol=1e-10)
end

@testset "Python Interface Alignment Tests" begin
    # 1. TamLib
    @test unique_in_sorted([1, 1, 2, 3], true) == ([1, 2, 3], [2, 1, 1])
    @test perm_parity_cyc([1, 0, 2], true) == -1
    @test cycle_decomp([1, 0, 2], true, true, true) == ([[0, 1], [2]], -1)
    
    mat = [1 2; 3 4]
    @test fill_diagonal_offset(mat, 10, 1) == [1 10; 3 4]
    @test mat == [1 10; 3 4] # verify mutated
    
    @test logspace(3, 1, 100) == [1.0, 10.0, 100.0]
    
    # 2. getReducedHilSpace with returnOcc=false
    @test getReducedHilSpace(4, 2, false) == DtMb[3, 5, 6, 9, 10, 12]
    
    # 3. singleSpinChannelCombos with returnMomTransfer
    cre, ann, delta = singleSpinChannelCombos(4, 1, 1, true, [4])
    @test length(cre) == 12
    @test length(delta) == 12
    
    # 4. _match_mom with vectors
    iup, idn = TamFermion._match_mom(Int[0, 1, 1, 3], Int[1, 1, 2, 3], 4)
    @test iup == [2, 2, 3, 3, 4]
    @test idn == [1, 2, 1, 2, 4]

    # 5. sectorTotMom Dict return
    res_stm = sectorTotMom([4], 2)
    @test res_stm isa Dict{String, Any}
    @test issetequal(keys(res_stm), ["ints", "occ", "qtot", "sortOrder", "qtot_unique", "counts"])
    
    # 6. fullSlaterMomBasis Dict return
    res_fsm = fullSlaterMomBasis([2], 1, 1)
    @test res_fsm isa Dict{String, Any}
    @test issetequal(keys(res_fsm), ["ints", "qtot_up", "qtot_dn", "qtot", "qtot_unique", "counts", "sortOrder"])
    
    # 7. translOpnD (Python style)
    Tops, signs = translOpnD([4], "fermion")
    @test size(Tops) == (1, 16)
    @test size(signs) == (1, 16)
    
    # 8. buildOrbitsnD (Python style)
    orbits = buildOrbitsnD([2], "fermion")
    @test orbits[0][:signed_orbit] == [0]
    
    # 9. translnDCOB (Python style)
    P, bSizes = translnDCOB([2], "fermion")
    @test size(P) == (4, 4)
    
    # 10. Hubbard keyword arguments
    H1 = Hubbard(1.0, 2.0, [2], [1, 1], true, false)
    @test size(H1) == (4, 4)
end

