# lattice #
struct Lattice
    lattice::AbstractGraph
    type::String
    dimensions::Vector
end
function create_rectangular_lattice(n, m)
    return Lattice(Graphs.grid((n,m)), "rectangular", [n,m])
end
function create_square_lattice(n)
    return Lattice(Graphs.grid((n,n)), "square", [n, n])
end
function neighbors(lattice::Lattice, v)
    return Graphs.neighbors(lattice.lattice, v)
end
function nv(lattice::Lattice)
    return Graphs.nv(lattice.lattice)
end
function reflection_mapping(lattice::Lattice, kind::Int)
    # kind is an integer specifying which axis the reflection is on. 
    # this number ranges from 1 to (whatever the rotational symmetry of lattice)
    d = Dict()
    if lattice.type == "rectangular" || (lattice.type == "square" && kind <= 2)
        xdim, ydim = lattice.dimensions
        for iy ∈ 0:ydim-1
            for ix ∈ 0:xdim-1
                if kind == 1 # horizontal reflection
                    d[iy*xdim + ix + 1] = iy*xdim + (xdim - ix)
                else # vertical reflection
                    d[iy*xdim + ix + 1] = (ydim-1 - iy)*xdim + ix + 1
                end
            end
        end
    elseif lattice.type == "square"
        dim, _ = lattice.dimensions
        for iy ∈ 0:dim-1
            for ix ∈ 0:dim-1
                if kind == 3 # diagonal reflection
                    d[iy*dim + ix + 1] = ix*dim + iy + 1
                else 
                    d[iy*dim + ix + 1] = (dim-1 - ix)*dim + dim - iy
                end
            end
        end
    end


    return d
end
function rotation_mapping(lattice::Lattice)
    d = Dict()
    # rotates by 
    if lattice.type == "rectangular"

    end
    return d
end

struct HubbardSubspace
    N_up
    N_down
    N::Int
    lattice::Lattice

    function HubbardSubspace(N_up, N_down, lattice::Lattice)
        N = -1
        new(N_up, N_down, N, lattice)
    end
    function HubbardSubspace(N::Int, lattice::Lattice)
        new(-1, -1, N, lattice)
    end
end

function get_subspace_info(hs::HubbardSubspace)
    if hs.N == -1
        return (hs.N_up, hs.N_down)
    end
    return hs.N
end
# hubbard model #
struct HubbardModel
    t::Float64
    U::Float64
    μ::Float64
    half_filling::Bool
end

function _combination_indexer!(a::Vector{T}, b::T, c::T, idx::Int, comb_dict, inv_comb_dict) where T
    for comb1 in combinations(a, b)
        set1 = Set(comb1)
        for comb2 in combinations(a, c)
            set2 = Set(comb2)
            pair = (set1, set2)
            comb_dict[pair] = idx
            push!(inv_comb_dict, pair)
            idx += 1
        end
    end
    return idx
end
struct CombinationIndexer{T}
    # Given a vector of labels a (typically these would correspond to site numberings),
    # this object associates each permutation of spin up and spin down dirac fermions
    # with an index. comb_dict maps a permutation to an index, and inv_comb_dict maps
    # indices to the permutation.
    a::Vector{T}
    comb_dict::Dict{Tuple{Set{T}, Set{T}}, Int}
    inv_comb_dict::Vector{Tuple{Set{T}, Set{T}}}

   
    function CombinationIndexer(a::Vector{T}, N) where T
        # Constructor for a fock space of exactly N fermions. Goes from N_up=0 N_down=N to N_up=N N_down=0
        comb_dict = Dict{Tuple{Set{T}, Set{T}}, Int}()
        inv_comb_dict = Vector{Tuple{Set{T}, Set{T}}}()
        
        # Generate combinations and populate comb_dict and inv_comb_dict directly
        idx = 1
        for b_val=0:N
            c_val = N - b_val
            idx = _combination_indexer!(a, b_val, c_val, idx, comb_dict, inv_comb_dict)
        end
        
        new{T}(a, comb_dict, inv_comb_dict)
    end
    function CombinationIndexer(a::Vector{T}, b, c) where T
        # Constructor for a fock space with b spin up fermions and c spin down fermions. b and c can be iterables,
        # and if they are, the fock space includes the cartesian product of these iterables.
        comb_dict = Dict{Tuple{Set{T}, Set{T}}, Int}()
        inv_comb_dict = Vector{Tuple{Set{T}, Set{T}}}()
        
        # Generate combinations and populate comb_dict and inv_comb_dict directly
        idx = 1
        for b_val in b
            for c_val in c
                idx = _combination_indexer!(a, b_val, c_val, idx, comb_dict, inv_comb_dict)
            end
        end
        
        new{T}(a, comb_dict, inv_comb_dict)
    end
end
function get_subspace_dimension(Hs::HubbardSubspace)
    L = nv(Hs.lattice)
    if Hs.N == -1
        total = 0 
        for n_up in Hs.N_up
            for n_down in Hs.N_down
                total += binomial(L,n_up) * binomial(L,n_down)
            end
        end
        return total
    end
    total = 0
    for n_up in 0:Hs.N
        n_down = Hs.N - n_up
            total += binomial(L,n_up) * binomial(L,n_down)
        end
    return total
end

# function index(CI::CombinationIndexer, comb::Vector)
#     return CI.comb_dict[Set(comb)]
# end

function combination(CI::CombinationIndexer, idx::Int)
    return CI.inv_comb_dict[idx]
end


function index(self::CombinationIndexer, comb1::Set, comb2::Set)
    return self.comb_dict[(comb1, comb2)]  # Use tuple as key
end



