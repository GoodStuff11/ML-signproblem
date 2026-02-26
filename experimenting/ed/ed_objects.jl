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
    for (s, sp) ∈ zip(_s, mapslices(rotr90, collect(sites(lattice)), dims=[1, 2]))
        d[s] = sp
    end
    return d
end
function translation_mapping(lattice::AbstractLattice, dims)
    d = Dict()
    shift = zeros(length(size(sites(lattice))))
    shift[dims] = 1
    for (s, sp) ∈ zip(sites(lattice), circshift(collect(sites(lattice)), shift))
        d[s] = sp
    end
    return d
end
# defining comparing of coordinates
for (k, op) in enumerate([:>, :isless, :<, :>=, :<=])
    @eval function Base.$op(x::Coordinate{N}, y::Coordinate{N}) where N
        for (a, b) ∈ zip(x.coordinates, y.coordinates)
            if a != b
                return $op(a, b)
            end
        end
        return $(k > 3) # true if >= or <= otherwise not
    end
end
struct HubbardSubspace
    N_up
    N_down
    N::Int
    lattice::AbstractLattice
    k::Union{Nothing,Tuple{Vararg{Int}}}

    function HubbardSubspace(N_up, N_down, lattice::AbstractLattice; k=nothing)
        N = -1
        new(N_up, N_down, N, lattice, k)
    end
    function HubbardSubspace(N::Int, lattice::AbstractLattice; k=nothing)
        new(-1, -1, N, lattice, k)
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
    t::Union{Float64,AbstractArray{Float64},Dict}
    U::Float64
    μ::Float64
    half_filling::Bool
end

function _combination_indexer!(a::Vector{T}, b::Int, c::Int, idx::Int, comb_dict, inv_comb_dict; k=nothing, lattice_dims=nothing) where T
    for comb1 in combinations(a, b)
        set1 = Set(comb1)
        for comb2 in combinations(a, c)
            set2 = Set(comb2)
            if !isnothing(k) && !isnothing(lattice_dims)
                # check if total momentum matches k
                # assuming objects in combination are Coordinate
                # total momentum is sum of momenta mod lattice_dims
                tot_k = zeros(Int, length(lattice_dims))
                for coord in comb1
                    tot_k .+= (coord.coordinates .- 1)
                end
                for coord in comb2
                    tot_k .+= (coord.coordinates .- 1)
                end
                tot_k = (tot_k .% lattice_dims) .+ 1
                if tuple(tot_k...) != k
                    continue
                end
            end
            pair = (set1, set2)
            comb_dict[pair] = idx
            push!(inv_comb_dict, pair)
            idx += 1
        end
    end
    return idx
end

function _build_indexer(a::Vector{T}, iter; k=nothing, lattice_dims=nothing) where T
    comb_dict = Dict{Tuple{Set{T},Set{T}},Int}()
    inv_comb_dict = Vector{Tuple{Set{T},Set{T}}}()
    idx = 1
    for (b_val, c_val) in iter
        idx = _combination_indexer!(a, b_val, c_val, idx, comb_dict, inv_comb_dict; k=k, lattice_dims=lattice_dims)
    end
    return comb_dict, inv_comb_dict
end

struct CombinationIndexer{T}
    # Given a vector of labels a (typically these would correspond to site numberings),
    # this object associates each permutation of spin up and spin down dirac fermions
    # with an index. comb_dict maps a permutation to an index, and inv_comb_dict maps
    # indices to the permutation.
    a::Vector{T}
    comb_dict::Dict{Tuple{Set{T},Set{T}},Int}
    inv_comb_dict::Vector{Tuple{Set{T},Set{T}}}
    lattice_dims::Union{Nothing,Tuple{Vararg{Int}}}
    k::Union{Nothing,Tuple{Vararg{Int}}}

    function CombinationIndexer(a::Vector{T}, N::Integer) where T
        # Constructor for a fock space of exactly N fermions. Goes from N_up=0 N_down=N to N_up=N N_down=0
        comb_dict, inv_comb_dict = _build_indexer(a, ((b, N - b) for b in 0:N))
        new{T}(a, comb_dict, inv_comb_dict, nothing, nothing)
    end
    function CombinationIndexer(a::Vector{T}, b, c) where T
        # Constructor for a fock space with b spin up fermions and c spin down fermions. b and c can be iterables,
        # and if they are, the fock space includes the cartesian product of these iterables.
        comb_dict, inv_comb_dict = _build_indexer(a, Iterators.product(b, c))
        new{T}(a, comb_dict, inv_comb_dict, nothing, nothing)
    end
    function CombinationIndexer(Hs::HubbardSubspace)
        a = reduce(vcat, collect(sites(Hs.lattice)))
        lattice_dims = size(Hs.lattice)
        k = Hs.k
        iter = Hs.N == -1 ? Iterators.product((Hs.N_up isa Number ? (Hs.N_up:Hs.N_up) : Hs.N_up),
            (Hs.N_down isa Number ? (Hs.N_down:Hs.N_down) : Hs.N_down)) :
               ((n_up, Hs.N - n_up) for n_up in 0:Hs.N)
        comb_dict, inv_comb_dict = _build_indexer(a, iter; k=k, lattice_dims=lattice_dims)
        new{eltype(a)}(a, comb_dict, inv_comb_dict, lattice_dims, k)
    end
    function CombinationIndexer(
        a::Vector{T},
        comb_dict::Dict{Tuple{Set{T},Set{T}},Int},
        inv_comb_dict::Vector{Tuple{Set{T},Set{T}}},
        lattice_dims::Union{Nothing,Tuple{Vararg{Int}}},
        k::Union{Nothing,Tuple{Vararg{Int}}}
    ) where T
        new{T}(a, comb_dict, inv_comb_dict, lattice_dims, k)
    end
end

function get_subspace_dimension(Hs::HubbardSubspace)
    L = prod(size(Hs.lattice))
    get_n(n) = n isa Number ? (n:n) : n
    iter_pairs = Hs.N == -1 ? Iterators.product(get_n(Hs.N_up), get_n(Hs.N_down)) : ((n_up, Hs.N - n_up) for n_up in 0:Hs.N)

    if isnothing(Hs.k)
        return sum(binomial(L, n_up) * binomial(L, n_down) for (n_up, n_down) in iter_pairs)
    end

    # computing dimension with momentum constraint
    dims = size(Hs.lattice)
    total_dim = 0.0
    for r in CartesianIndices(dims)
        r_vec = Tuple(r) .- 1
        m = isempty(dims) ? 1 : reduce(lcm, (dims[i] ÷ gcd(dims[i], r_vec[i]) for i in 1:length(dims)), init=1)
        C = L ÷ m
        tr = sum(n_up % m == 0 && n_down % m == 0 ?
                 ((m % 2 == 0 ? (-1)^(n_up ÷ m + n_down ÷ m) : 1) * binomial(C, n_up ÷ m) * binomial(C, n_down ÷ m)) : 0
                 for (n_up, n_down) in iter_pairs)

        phase = isempty(dims) ? 0.0 : sum(2 * π * (Hs.k[i] - 1) * r_vec[i] / dims[i] for i in 1:length(dims))
        total_dim += tr * cos(phase)
    end
    return round(Int, total_dim / L)
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




