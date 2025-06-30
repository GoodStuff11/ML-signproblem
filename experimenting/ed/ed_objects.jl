# lattice #
function reflection_mapping(lattice::AbstractLattice, dims)
    # kind is an integer specifying which axis the reflection is on. 
    # this number ranges from 1 to (whatever the rotational symmetry of lattice)
    d = Dict()
    for (s, sp) ∈ zip(sites(lattice), reverse(collect(sites(lattice)), dims=dims))
        d[s] = sp
    end
    return d
end
function rotation_mapping(lattice::AbstractLattice)
    d = Dict()
    for (s,sp) ∈ zip(_s, mapslices(rotr90, collect(sites(lattice)), dims=[1,2]))
        d[s] = sp
    end
    return d
end
function translation_mapping(lattice::AbstractLattice, dims)
    d = Dict()
    shift = zeros(length(size(sites(lattice))))
    shift[dims] = 1
    for (s,sp) ∈ zip(sites(lattice), circshift(collect(sites(lattice)), shift))
        d[s] = sp
    end
    return d
end
# defining comparing of coordinates
for (k,op) in enumerate([:>, :isless, :<, :>=, :<=])
    @eval function Base.$op(x::Coordinate{N}, y::Coordinate{N}) where N 
        for (a,b) ∈ zip(x.coordinates, y.coordinates)
            if a != b
                return $op(a, b)
            end
        end
        return $(k>3) # true if >= or <= otherwise not
    end
end
struct HubbardSubspace
    N_up
    N_down
    N::Int
    lattice::AbstractLattice

    function HubbardSubspace(N_up, N_down, lattice::AbstractLattice)
        N = -1
        new(N_up, N_down, N, lattice)
    end
    function HubbardSubspace(N::Int, lattice::AbstractLattice)
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
    t::Union{Float64, AbstractArray{Float64}}
    U::Float64
    μ::Float64
    half_filling::Bool
end

function _combination_indexer!(a::Vector{T}, b::Int, c::Int, idx::Int, comb_dict, inv_comb_dict) where T
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
    L = prod(size(Hs.lattice))
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




