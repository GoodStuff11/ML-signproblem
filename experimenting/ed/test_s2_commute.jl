using SparseArrays
using LinearAlgebra
using Combinatorics

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
    coords = []
    for d in Iterators.product((1:dim for dim in l.dims)...)
        push!(coords, Coordinate(d...))
    end
    return reshape(coords, l.dims)
end
function neighbors(l::MockLattice, site::Coordinate, order=1)
    if order == 1
        return [
            Coordinate(mod1(site.coordinates[1] + 1, l.dims[1])),
            Coordinate(mod1(site.coordinates[1] - 1, l.dims[1]))
        ]
    end
    return []
end

include("ed_objects.jl")
include("ed_functions.jl")

function test_commutation()
    N_sites = 4
    lat = MockLattice((N_sites,))
    
    Hs = HubbardSubspace(2, 2, lat; k=(1,))
    
    Hm = HubbardModel(1.0, 4.0, 0.0, false)
    H = create_Hubbard(Hm, Hs; momentum_basis=true)
    S2 = create_operator(Hs, :S2; momentum_basis=true)
    
    comm = H * S2 - S2 * H
    norm_comm = norm(comm)
    println("Norm of [H, S2] in momentum basis: ", norm_comm)
    if norm_comm > 1e-10
        println("Matrices do not commute!")
        # Let's check which part does not commute
        # Is it Hopping or U?
        H_t = create_Hubbard(HubbardModel(1.0, 0.0, 0.0, false), Hs; momentum_basis=true)
        H_U = create_Hubbard(HubbardModel(0.0, 4.0, 0.0, false), Hs; momentum_basis=true)
        println("Norm of [H_t, S2]: ", norm(H_t * S2 - S2 * H_t))
        println("Norm of [H_U, S2]: ", norm(H_U * S2 - S2 * H_U))
    end
end

test_commutation()
