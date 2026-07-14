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
import Base.Order: Ordering, lt

struct RowSnake <: Ordering end
struct ColSnake <: Ordering end

function Base.isless(x::Coordinate{N,I}, y::Coordinate{N,I}) where {N, I}
    for i in eachindex(x.coordinates)
        if x.coordinates[i] != y.coordinates[i]
            return x.coordinates[i] < y.coordinates[i]
        end
    end
    return false
end

function Base.Order.lt(::RowSnake, x::Coordinate{N,I}, y::Coordinate{N,I}) where {N, I}
    for i in eachindex(x.coordinates)
        if x.coordinates[i] != y.coordinates[i]
            return x.coordinates[i] < y.coordinates[i]
        end
    end
    return false
end

function Base.Order.lt(::ColSnake, x::Coordinate{N,I}, y::Coordinate{N,I}) where {N, I}
    for i in eachindex(x.coordinates)
        if x.coordinates[i] != y.coordinates[i]
            return x.coordinates[i] < y.coordinates[i]
        end
    end
    return false
end

function Base.Order.lt(o::Union{RowSnake, ColSnake}, x::Tuple, y::Tuple)
    for i in 1:min(length(x), length(y))
        if x[i] != y[i]
            return Base.Order.lt(o, x[i], y[i])
        end
    end
    return length(x) < length(y)
end

function Base.Order.lt(o::Union{RowSnake, ColSnake}, x::AbstractVector, y::AbstractVector)
    for i in 1:min(length(x), length(y))
        if x[i] != y[i]
            return Base.Order.lt(o, x[i], y[i])
        end
    end
    return length(x) < length(y)
end

function Base.Order.lt(::Union{RowSnake, ColSnake}, x, y)
    return isless(x, y)
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

"""
    _each_comb(f, a, b)

Call `f(view)` for every size-`b` combination of `a` in strictly ascending index order
`a[i₁], a[i₂], …, a[iᵦ]` with `1 ≤ i₁ < i₂ < … < iᵦ ≤ length(a)`.

Uses a single pre-allocated index buffer — zero per-combination heap allocations.
"""
@inline function _each_comb(f::F, a::Vector{T}, b::Int) where {F,T}
    n = length(a)
    b == 0 && (f(T[]); return)  # empty combination
    b > n  && return             # no combinations exist
    # indices[j] = current 1-based position in `a` for the j-th chosen element
    # Ordering: leftmost index is fastest-changing (increments first),
    # rightmost index is slowest-changing. Example b=3:
    #   (1,2,3),(1,2,4),(1,3,4),(2,3,4),(1,2,5),(1,3,5),...
    indices = collect(1:b)
    buf = Vector{T}(undef, b)   # reused across all calls — one alloc total
    while true
        @inbounds for j in 1:b
            buf[j] = a[indices[j]]
        end
        f(buf)
        # Advance: find the LEFTMOST index that can be incremented
        j = 1
        while j <= b
            max_val = j < b ? indices[j+1] - 1 : n
            if indices[j] < max_val
                indices[j] += 1
                # Reset everything to the left back to 1, 2, ..., j-1
                @inbounds for k in 1:j-1
                    indices[k] = k
                end
                break
            end
            j += 1
        end
        j > b && break  # no index could advance → done
    end
end

function _combination_indexer!(a::Vector{T}, b::Int, c::Int, idx::Int, comb_dict, inv_comb_dict; k=nothing, lattice_dims=nothing) where T
    check_k = !isnothing(k) && !isnothing(lattice_dims)
    ndims_k  = check_k ? length(lattice_dims) : 0
    tot_k    = check_k ? zeros(Int, ndims_k) : Int[]  # one alloc, reused
    _each_comb(a, b) do comb1
        set1 = Set(comb1)
        _each_comb(a, c) do comb2
            if check_k
                @inbounds for d in 1:ndims_k
                    tot_k[d] = 0
                end
                for coord in comb1
                    @inbounds tot_k .+= (coord.coordinates .- 1)
                end
                for coord in comb2
                    @inbounds tot_k .+= (coord.coordinates .- 1)
                end
                @inbounds for d in 1:ndims_k
                    tot_k[d] = (tot_k[d] % lattice_dims[d]) + 1
                end
                tuple(tot_k...) != k && return
            end
            set2 = Set(comb2)
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
    function CombinationIndexer(Hs::HubbardSubspace; order::Ordering=RowSnake())
        a = sort(reduce(vcat, collect(sites(Hs.lattice))), order=order) # sort to standardize the order of sites
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




