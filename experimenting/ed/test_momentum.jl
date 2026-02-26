using SparseArrays
using LinearAlgebra
using Combinatorics

# Mock lattice to resolve dependency issues
abstract type AbstractLattice end
struct MockLattice <: AbstractLattice
    dims::Tuple{Vararg{Int}}
end
Base.size(l::MockLattice) = l.dims

struct Coordinate{N, T}
    coordinates::NTuple{N, T}
end
Coordinate(coords...) = Coordinate(coords)

function sites(l::MockLattice)
    # just return an array of Coordinate for each site
    coords = []
    for d in Iterators.product((1:dim for dim in l.dims)...)
        push!(coords, Coordinate(d...)) # Assuming Coordinate exists in ed_objects
    end
    return reshape(coords, l.dims)
end

include("ed_objects.jl")
include("ed_functions.jl")

function test_momentum_basis()
    println("Testing momentum basis...")
    dims = (4,)
    
    # 1. Create a simple Hubbard subspace 
    N_up = 2
    N_down = 2
    
    # K = 1 sector (0 momentum)
    lattice = MockLattice(dims)
    Hs_k0 = HubbardSubspace(N_up, N_down, lattice; k=(1,))
    
    println("Lattice size: ", size(Hs_k0.lattice))
    dim_k0 = get_subspace_dimension(Hs_k0)
    println("Subspace dimension (k=1): ", dim_k0)
    
    # Full position basis for reference
    Hs_pos = HubbardSubspace(N_up, N_down, lattice)
    dim_pos = get_subspace_dimension(Hs_pos)
    println("Subspace dimension (position): ", dim_pos)
    
    # Check that sum of momentum dimensions equals full dimension
    total_dim = 0
    for k in 1:dims[1]
        Hs_k = HubbardSubspace(N_up, N_down, lattice; k=(k,))
        dim_k = get_subspace_dimension(Hs_k)
        total_dim += dim_k
        println("  Dim(k=", k, ") = ", dim_k)
    end
    println("Sum of k-sector dimensions: ", total_dim)
    println("Matches position basis: ", total_dim == dim_pos)
    
    # 2. Test operator generation
    m_Hm = HubbardModel(1.0, 4.0, 0.0, false)
    
    H_k0 = create_Hubbard(m_Hm, Hs_k0; momentum_basis=true)
    
    println("H_k0 is symmetric: ", issymmetric(H_k0))
    println("H_k0 is hermitian: ", ishermitian(H_k0))
    
    # 3. Test operator creation for Sz
    Sz_k0 = create_operator(Hs_k0, :Sz; momentum_basis=true)
    println("Sz_k0 is hermitian: ", ishermitian(Sz_k0))
    
    println("Test completed successfully.")
end

test_momentum_basis()
