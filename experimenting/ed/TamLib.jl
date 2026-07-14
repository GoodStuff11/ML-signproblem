module TamLib

using LinearAlgebra
using SparseArrays
using Combinatorics: combinations
using Random

# ═══════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════

# C-order helpers
export ravel_c, unravel_c, reshape_c, flatten_c
# Permutations
export perm_parity_cyc, cycle_decomp, parity_sort, num_inversions
# General utility
export pascal, linmap, find_sublist_index, fill_diagonal_offset
export normalize, logspace, pwlin, pwlin2, group_by_row, cartesian_prod
# Comparison checks
export maxposdiff, isapproxeq, isdiag_approx, isreal_approx
export isherm, issymm, isunitary, isBlockDiag
# Combinatorial generation
export getBinCombos, genStrings, genTokenCombos
# Bit manipulations
export fastXOR, dec2bin, dec2binnD, binnD2dec
export flipBits, circshiftBits, circshiftBitsnD, apply_perm_to_bitstring
# Ring / periodic methods
export circshift_list, shiftarr, levicivita, levicivita_tensor
export getThetaRel, enumerateCwiseInterval, repAsCwiseInterval
export sortCwiseFrom, cwise_dist, checkInIntervalRing, intersectionOnRing
# Lattice coords
export getLatticeCoord, pairwiseAngle, pairwiseDist
# Sampling
export Haarvec, COE, CUE, GOE, GUE
export gaussianFunc, getRandomGaussComplex, getUniform
# Linear algebra
export biorthogonal_eig, densetoBlockDiag, gm_basis
# Helpers
export unique_in_sorted


# ═══════════════════════════════════════════════════════════════════════
# C-ORDER (ROW-MAJOR) ARRAY HELPERS
# These replicate NumPy's default C-order reshape/ravel/multi-index so
# that bit-string ↔ nD-lattice conversions match the Python code exactly.
# All indices are **0-based** (matching bit positions).
# ═══════════════════════════════════════════════════════════════════════

"""
    ravel_c(coords, dims; mode=:raise) → 0-based linear index

C-order (row-major) multi-index → linear index.
`coords`: tuple of coordinate values/arrays (each 0-based).
`dims`:   tuple of dimension sizes.
`mode`:   `:raise` (default) or `:wrap` for periodic wrapping.
"""
function ravel_c(coords, dims; mode=:raise)
    nd = length(dims)
    @assert length(coords) == nd
    # C-order strides: stride[k] = ∏ dims[k+1 : end]
    strides = Vector{Int}(undef, nd)
    strides[nd] = 1
    for k in nd-1:-1:1
        strides[k] = strides[k+1] * dims[k+1]
    end
    if mode == :wrap
        return sum(mod.(coords[k], dims[k]) .* strides[k] for k in 1:nd)
    else
        return sum(coords[k] .* strides[k] for k in 1:nd)
    end
end

"""
    unravel_c(idx, dims) → tuple of 0-based coordinates

C-order (row-major) linear index → tuple of coordinates.
Works on both scalar and array `idx`.
"""
function unravel_c(idx::Integer, dims)
    nd = length(dims)
    coords = Vector{Int}(undef, nd)
    rem = idx
    for k in nd:-1:1
        coords[k] = rem % dims[k]
        rem = div(rem, dims[k])
    end
    return Tuple(coords)
end

function unravel_c(idx::AbstractArray{<:Integer}, dims)
    nd = length(dims)
    result = [similar(idx, Int) for _ in 1:nd]
    rem = Int.(idx)
    for k in nd:-1:1
        result[k] .= rem .% dims[k]
        rem .= div.(rem, dims[k])
    end
    return Tuple(result)
end

"""
    reshape_c(flat, dims) → nD array

Reshape a flat vector into an nD array with C-order (row-major) layout
(last dimension varies fastest in `flat`).
"""
function reshape_c(flat, dims)
    nd = length(dims)
    if nd == 1
        return reshape(flat, dims[1])
    end
    return permutedims(reshape(flat, reverse(dims)...), nd:-1:1)
end

"""
    flatten_c(arr) → Vector

Flatten an nD array in C-order (row-major): last dimension varies fastest.
"""
function flatten_c(arr)
    nd = ndims(arr)
    if nd == 1
        return vec(arr)
    end
    return vec(permutedims(arr, nd:-1:1))
end


# ═══════════════════════════════════════════════════════════════════════
# PERMUTATIONS
# ═══════════════════════════════════════════════════════════════════════

"""
    perm_parity_cyc(perm, validate::Bool=false) → ±1

Parity of a **0-based** permutation via cycle decomposition.
`perm[i+1]` (Julia 1-based array) holds the image of element `i`.
"""
function perm_parity_cyc(perm::AbstractVector{<:Integer}, pos_validate::Bool=false; validate::Bool=pos_validate)
    N = length(perm)
    if validate && sort(perm) != collect(0:N-1)
        error("Invalid permutation.")
    end
    visited = falses(N)
    n_cyc = 0
    for i in 1:N
        if !visited[i]
            n_cyc += 1
            j = i
            while !visited[j]
                visited[j] = true
                j = perm[j] + 1          # 0-based value → 1-based index
            end
        end
    end
    return (N - n_cyc) & 1 == 1 ? -1 : 1
end

"""
    cycle_decomp(perm, include_fixed::Bool=false, returnParity::Bool=false, validate::Bool=false)

Cycle decomposition of a 0-based permutation.
Returns a vector of cycles (each a `Vector{Int}` of 0-based elements).
"""
function cycle_decomp(perm::AbstractVector{<:Integer}, pos_include_fixed::Bool=false, pos_returnParity::Bool=false, pos_validate::Bool=false;
                      include_fixed::Bool=pos_include_fixed, returnParity::Bool=pos_returnParity, validate::Bool=pos_validate)
    N = length(perm)
    if validate && sort(perm) != collect(0:N-1)
        error("Invalid permutation.")
    end
    visited = falses(N)
    cycles = Vector{Vector{Int}}()
    n_cyc = 0
    for i in 1:N
        if !visited[i]
            n_cyc += 1
            j = i
            cyc = Int[]
            while !visited[j]
                visited[j] = true
                push!(cyc, j - 1)        # store 0-based
                j = perm[j] + 1
            end
            if include_fixed || length(cyc) > 1
                push!(cycles, cyc)
            end
        end
    end
    if returnParity
        return cycles, ((N - n_cyc) & 1 == 1 ? -1 : 1)
    end
    return cycles
end

"""
    parity_sort(arr) → ±1

Parity of the permutation that sorts `arr`.
"""
function parity_sort(arr)
    order = sortperm(arr)                 # 1-based
    perm = zeros(Int, length(arr))
    for (new_pos, old_pos) in enumerate(order)
        perm[old_pos] = new_pos - 1       # build 0-based inverse perm
    end
    return perm_parity_cyc(perm)
end

"""
    num_inversions(arr) → ±1

Parity via counting inversions (O(n²)).
"""
function num_inversions(arr)
    inv = 0
    for i in eachindex(arr)
        inv += count(arr[i] .> @view(arr[i+1:end]))
    end
    return inv % 2 == 1 ? -1 : 1
end


# ═══════════════════════════════════════════════════════════════════════
# GENERAL UTILITY
# ═══════════════════════════════════════════════════════════════════════

"""Nth row of Pascal's triangle (0-indexed row)."""
pascal(n::Integer) = [binomial(n, k) for k in 0:n]

"""Linearly map values of `x` into the range of `y`."""
function linmap(x, y)
    x0, xm = minimum(x), maximum(x)
    y0, ym = minimum(y), maximum(y)
    return @. (x - x0) / (xm - x0) * (ym - y0) + y0
end

"""Find index of sublist in a nested list (vector of vectors) that contains `elem`."""
function find_sublist_index(elem, nestedList)
    for (i, sub) in enumerate(nestedList)
        if elem in sub
            return i
        end
    end
    return nothing
end

"""
    fill_diagonal_offset(arr, val, offset::Integer=0)

Fill the diagonal at `offset` (0 = main) of a 2D array with `val`.
Positive offset → upper diagonal, negative → lower.
Mutates the array in-place and returns it.
"""
function fill_diagonal_offset(arr::AbstractMatrix, val, pos_offset::Integer=0; offset::Integer=pos_offset)
    rows, cols = size(arr)
    if offset >= 0
        dlen = min(rows, cols - offset)
        for k in 1:dlen
            arr[k, k + offset] = val
        end
    else
        dlen = min(rows + offset, cols)
        for k in 1:dlen
            arr[k - offset, k] = val
        end
    end
    return arr
end

"""
    normalize(X::AbstractArray)

Normalize the last axis of an array by dividing it by its norm.
"""
function normalize(X::AbstractArray)
    nrm = sqrt.(sum(abs2, X; dims=ndims(X)))
    return X ./ nrm
end

"""
    logspace(n::Integer, x0=2π, xm=2π * 1e4)

`n` geometrically spaced samples from `x0` to `xm`.
"""
function logspace(n::Integer, pos_x0=2π, pos_xm=2π * 1e4; x0=pos_x0, xm=pos_xm)
    b = (xm / x0)^(1 / (n - 1))
    return x0 .* b .^ (0:n-1)
end

"""
    pwlin(x, y0, tlist, mlist)

Piecewise-linear function with arbitrary transition points.
`y0`: y-intercept of the first segment.
`tlist`: x-positions of transition points.
`mlist`: slopes of each segment (length = `length(tlist) + 1`).
"""
function pwlin(x, y0, tlist, mlist)
    length(mlist) == length(tlist) + 1 ||
        error("len(mlist) must equal len(tlist)+1")
    y = similar(x, Float64)
    for (idx, xi) in enumerate(x)
        seg = searchsortedlast(tlist, xi) + 1
        seg = clamp(seg, 1, length(mlist))
        y[idx] = y0 + mlist[seg] * xi
        for j in 1:seg-1
            y[idx] += (mlist[j] - mlist[j+1]) * tlist[j]
        end
    end
    return y
end

"""Two-segment piecewise linear: slope `m1` for x<t, slope `m2` for x≥t."""
function pwlin2(x, y0, t, m1, m2)
    return @. (x < t) * (y0 + m1 * x) + (x >= t) * (m2 * x + (m1 - m2) * t + y0)
end

"""
    group_by_row(A, r)

Group columns of 2D array `A` by unique values in row `r`.
Returns `Dict(key => submatrix)` where row `r` is removed.
"""
function group_by_row(A::AbstractMatrix, r::Integer)
    row = A[r, :]
    other = A[setdiff(1:size(A, 1), r), :]
    order = sortperm(row; alg=MergeSort)
    srow = row[order]
    change = findall(srow[2:end] .!= srow[1:end-1]) .+ 1
    starts = vcat(1, change)
    ends = vcat(change .- 1, length(srow))
    groups = [other[:, order[starts[i]:ends[i]]] for i in eachindex(starts)]
    keys_arr = srow[starts]
    return Dict(zip(keys_arr, groups))
end

"""
    cartesian_prod(x, y) → (repeated_x, tiled_y)

Cartesian product of two 1-D arrays, returned as two flat vectors.
"""
function cartesian_prod(x, y)
    return repeat(x; inner=length(y)), repeat(y, length(x))
end


# ═══════════════════════════════════════════════════════════════════════
# COMPARISON CHECKS
# ═══════════════════════════════════════════════════════════════════════

maxposdiff(a, b) = maximum(abs.(a .- b))

isapproxeq(a, b; tol=1e-15) = maxposdiff(a, b) <= tol

isdiag_approx(A; tol=1e-15) = isapproxeq(A, diagm(diag(A)); tol=tol)

isreal_approx(a; tol=1e-15) = isapproxeq(a, real.(a); tol=tol)

isherm(a; tol=1e-15) = isapproxeq(a, a'; tol=tol)

issymm(a; tol=1e-15) = isapproxeq(a, transpose(a); tol=tol)

isunitary(U; tol=1e-15) =
    isapproxeq(U * U', Matrix{eltype(U)}(I, size(U, 1), size(U, 1)); tol=tol)

"""
    isBlockDiag(A, bs; tol=1e-15)

Check if `A` is block-diagonal with block sizes `bs`.
"""
function isBlockDiag(A::AbstractMatrix, bs; tol=1e-15)
    sum(bs) == size(A, 1) && size(A, 1) == size(A, 2) ||
        error("Block sizes must sum to matrix dimension, and A must be square.")
    labels = vcat([fill(i, bs[i]) for i in eachindex(bs)]...)
    for i in axes(A, 1), j in axes(A, 2)
        if labels[i] != labels[j] && abs(A[i, j]) > tol
            return false
        end
    end
    return true
end


# ═══════════════════════════════════════════════════════════════════════
# COMBINATORIAL GENERATION
# ═══════════════════════════════════════════════════════════════════════

"""All 2^N binary strings of length N as vectors of 0/1 (MSB first)."""
getBinCombos(N::Integer) = [dec2bin(x, N) for x in 0:2^N-1]

"""All length-N strings over the given alphabet."""
function genStrings(alphabet, N::Integer)
    return [join(reverse(p))
            for p in Iterators.product(ntuple(_ -> alphabet, N)...)] |> vec
end

"""All length-N token combinations from `alphabet` as a collected array."""
genTokenCombos(alphabet, N::Integer) =
    collect(Iterators.product(ntuple(_ -> alphabet, N)...))


# ═══════════════════════════════════════════════════════════════════════
# BIT MANIPULATIONS
# ═══════════════════════════════════════════════════════════════════════

"""
    fastXOR(x) → 0 or 1

Parity of all bits of `x` (XOR-fold).  Uses hardware `count_ones`.
Broadcast with `fastXOR.(arr)` for arrays.
"""
fastXOR(x::Integer) = count_ones(x) & 1

"""
    dec2bin(x::Integer, N::Integer, returnType="list")

Convert integer `x` to its `N`-bit binary representation (MSB first).
Returns `Vector{Int}` (`"list"`) or `String` (`"string"`).
"""
function dec2bin(x::Integer, N::Integer, pos_returnType="list"; returnType=pos_returnType)
    x > 0 && floor(Int, log2(x)) >= N && error("insufficient bit width N")
    rt = returnType isa AbstractString ? Symbol(returnType) : returnType
    if rt == :list
        return [Int((x >> (N - 1 - i)) & 1) for i in 0:N-1]
    else
        return string(x; base=2, pad=N)
    end
end

"""
    dec2binnD(x, Lvec)

Convert integer `x` to an nD binary array on lattice with dimensions `Lvec`.
Bit 0 (LSB) maps to site at C-order index 0.
"""
function dec2binnD(x::Integer, Lvec)
    N = prod(Lvec)
    flat = [Int((x >> b) & 1) for b in 0:N-1]
    return reshape_c(flat, Tuple(Lvec))
end

"""
    binnD2dec(xarr)

Convert an nD binary array (from `dec2binnD`) back to an integer.
"""
function binnD2dec(xarr)
    flat = flatten_c(xarr)
    x = 0
    for b in 0:length(flat)-1
        if flat[b+1] != 0
            x |= 1 << b
        end
    end
    return x
end

"""Flip all `n` bits of integer `a` (complement within n-bit width)."""
function flipBits(a::Integer, n::Integer)
    return a ⊻ ((1 << n) - 1)
end

"""
    circshiftBits(x, s, N)

Circular-shift the N-bit representation of `x` by `s` positions.
Positive `s` → shift right; negative → shift left.
"""
function circshiftBits(x::Integer, s::Integer, N::Integer)
    x = x % (1 << N)
    s == 0 && return x
    mask = (1 << N) - 1
    if s < 0
        a = abs(s)
        return ((x << a) & mask) | ((x & mask) >> (N - a))
    else
        return ((x & mask) >> s) | ((x << (N - s)) & mask)
    end
end

"""
    circshiftBitsnD(x::Integer, t::Integer, a::Integer, Lvec, returnBitRep::Bool=false)

Circular-shift the nD bit pattern of integer `x` by `t` positions along
axis `a` (1-based) on a lattice with dimensions `Lvec`.
"""
function circshiftBitsnD(x::Integer, t::Integer, a::Integer, Lvec, pos_returnBitRep::Bool=false; returnBitRep::Bool=pos_returnBitRep)
    bitrep = dec2binnD(x, Lvec)
    shift = ntuple(d -> d == a ? t : 0, length(Lvec))
    shifted = circshift(bitrep, shift)
    xnew = binnD2dec(shifted)
    return returnBitRep ? (xnew, shifted) : xnew
end

"""
    apply_perm_to_bitstring(s, ss_perm)

Apply a **0-based** permutation `ss_perm` to the N-bit string `s`.
`ss_perm[i+1]` = new location (0-based) for bit `i`.
"""
function apply_perm_to_bitstring(s::Integer, ss_perm::AbstractVector{<:Integer})
    out = 0
    for i in 0:length(ss_perm)-1
        if (s >> i) & 1 == 1
            out |= 1 << ss_perm[i+1]   # ss_perm is 0-based, stored in 1-based array
        end
    end
    return out
end


# ═══════════════════════════════════════════════════════════════════════
# PERIODIC RING / SHIFT METHODS
# ═══════════════════════════════════════════════════════════════════════

"""
    circshift_list(arr, n)

Circularly shift a list/vector by `n` positions to the right.
Negative `n` shifts left.  (Julia's built-in `circshift` works on arrays;
this variant operates on generic iterables via conversion.)
"""
circshift_list(arr, n::Integer) = circshift(collect(arr), n)

"""
    shiftarr(x, n; axis=ndims(x), fill=0)

Shift array `x` by `n` places along `axis`, padding with `fill` (no wrap).
"""
function shiftarr(x::AbstractArray, n::Integer;
                  axis::Integer=ndims(x), fill=zero(eltype(x)))
    n == 0 && return copy(x)
    L = size(x, axis)
    abs(n) >= L && return Base.fill!(similar(x), fill)

    xs = similar(x)
    nd = ndims(x)
    # Build slice objects
    if n > 0
        out_idx = ntuple(d -> d == axis ? (n+1:L) : Colon(), nd)
        in_idx  = ntuple(d -> d == axis ? (1:L-n) : Colon(), nd)
        pad_idx = ntuple(d -> d == axis ? (1:n) : Colon(), nd)
    else
        a = abs(n)
        out_idx = ntuple(d -> d == axis ? (1:L-a) : Colon(), nd)
        in_idx  = ntuple(d -> d == axis ? (a+1:L) : Colon(), nd)
        pad_idx = ntuple(d -> d == axis ? (L-a+1:L) : Colon(), nd)
    end
    xs[out_idx...] .= x[in_idx...]
    xs[pad_idx...] .= fill
    return xs
end

"""Levi-Civita symbol for an index tuple (from `combinations`)."""
function levicivita(idx)
    p = 1
    for (a, b) in combinations(idx, 2)
        if b > a
            continue
        elseif b < a
            p = -p
        else
            return 0
        end
    end
    return p
end

"""Full Levi-Civita tensor of dimension `n` (n×n×…×n with n axes)."""
function levicivita_tensor(n::Integer)
    g = Array{Int}(undef, ntuple(_ -> n, n)...)
    for ci in CartesianIndices(g)
        g[ci] = levicivita(Tuple(ci))
    end
    return g
end


# ═══════════════════════════════════════════════════════════════════════
# RING INTERVAL METHODS
# ═══════════════════════════════════════════════════════════════════════

"""Relative angle between two angular positions on a circle."""
getThetaRel(th1, th2) = abs(mod(th1, 2π) - mod(th2, 2π))

"""
    enumerateCwiseInterval(iL, iR, L)

Enumerate all sites in the clockwise interval `[iL, iR]` on a ring of
size `L`.  Sites are **0-based** (`0` to `L-1`).
"""
function enumerateCwiseInterval(iL::Integer, iR::Integer, L::Integer)
    (0 <= iL < L && 0 <= iR < L) ||
        error("iL and iR must be in [0, L-1]")
    if iL <= iR
        return collect(iL:iR)
    else
        return vcat(collect(iL:L-1), collect(0:iR))
    end
end

"""
    repAsCwiseInterval(sites, L) → (start, end)

Represent a set of contiguous sites as a clockwise interval `(start, end)`
on a ring of size `L`.
"""
function repAsCwiseInterval(sites, L::Integer)
    s = Set(mod.(collect(sites), L))
    sz = length(s)
    for start in s
        candidate = Set(mod.(start:start+sz-1, L))
        if s == candidate
            return (start, mod(start + sz - 1, L))
        end
    end
    error("Sites don't form one contiguous interval.")
end

"""
    sortCwiseFrom(sites, L::Integer, ref::Integer=0)

Sort sites clockwise from `ref` on a ring of size `L`.
"""
function sortCwiseFrom(sites, L::Integer, pos_ref::Integer=0; ref::Integer=pos_ref)
    s = [mod(x, L) for x in sites]
    return sort(s; by = x -> mod(x - ref, L))
end

"""Clockwise distance from `iL` to `iR` on a ring of size `L`."""
cwise_dist(iL::Integer, iR::Integer, L::Integer) = mod(iR - iL, L)

"""Check if site `x` lies inside clockwise interval `(iL, iR)` on ring of size `L`."""
checkInIntervalRing(x::Integer, interval, L::Integer) =
    cwise_dist(interval[1], x, L) <= cwise_dist(interval[1], interval[2], L)

"""
    intersectionOnRing(interval1, interval2, L)

Intersection of two clockwise intervals on a ring of size `L`.
Returns `nothing` (empty), a single interval `(iL, iR)`, or a pair of
disjoint intervals `((iL1,iR1), (iL2,iR2))`.
"""
function intersectionOnRing(interval1, interval2, L::Integer)
    iv1, iv2 = Tuple(interval1), Tuple(interval2)
    iL1, iR1 = iv1
    iL2, iR2 = iv2
    iv1 == iv2 && return iv1
    cwise_dist(iL1, iR1, L) == L - 1 && return iv2
    cwise_dist(iL2, iR2, L) == L - 1 && return iv1
    L1in2 = checkInIntervalRing(iL1, iv2, L)
    L2in1 = checkInIntervalRing(iL2, iv1, L)
    !(L1in2 || L2in1) && return nothing
    R1in2 = checkInIntervalRing(iR1, iv2, L)
    R2in1 = checkInIntervalRing(iR2, iv1, L)
    in2 = L1in2 && R1in2
    in1 = L2in1 && R2in1
    if in2 && in1
        return ((iL1, iR2), (iL2, iR1))
    elseif in2 && !in1
        return iv1
    elseif in1 && !in2
        return iv2
    elseif R1in2 && L2in1 && !L1in2 && !R2in1
        return (iL2, iR1)
    else
        return (iL1, iR2)
    end
end


# ═══════════════════════════════════════════════════════════════════════
# LATTICE COORDINATE METHODS
# ═══════════════════════════════════════════════════════════════════════

"""
    getLatticeCoord(Lvec; a=1) → Matrix{Int}

Return `(N × d)` matrix of 0-based lattice coordinates for an nD lattice
with dimensions `Lvec`, using C-order site numbering.
"""
function getLatticeCoord(Lvec; a=1)
    N = prod(Lvec)
    nd = length(Lvec)
    coords = Matrix{Int}(undef, N, nd)
    dims = Tuple(Lvec)
    for b in 0:N-1
        c = unravel_c(b, dims)
        for d in 1:nd
            coords[b+1, d] = c[d]
        end
    end
    return a .* coords
end

"""Pairwise cosine-angle matrix between row-vectors of `coord` (N × d)."""
function pairwiseAngle(coord::AbstractMatrix)
    dp = coord * coord'
    return acos.(clamp.(dp, -1, 1))
end

"""Pairwise Euclidean distance matrix between row-vectors of `coord` (N × d)."""
function pairwiseDist(coord::AbstractMatrix)
    sqdist = sum([(c .- c') .^ 2 for c in eachcol(coord)])
    return sqrt.(sqdist)
end


# ═══════════════════════════════════════════════════════════════════════
# SAMPLING
# ═══════════════════════════════════════════════════════════════════════

"""
    Haarvec(d::Integer, n::Integer=1, real::Bool=false, seed=nothing)

Sample `n` vectors from Haar measure on U(d) (complex) or O(d) (real).
Returns `(n × d)` matrix.
"""
function Haarvec(d::Integer, pos_n::Integer=1, pos_real::Bool=false, pos_seed=nothing;
                 n::Integer=pos_n, real::Bool=pos_real, seed=pos_seed)
    seed !== nothing && Random.seed!(seed)
    X = real ?
        randn(n, d) :
        randn(ComplexF64, n, d)
    nrm = sqrt.(sum(abs2, X; dims=2))
    return X ./ nrm
end

"""
    COE(d::Integer, n::Integer=1, seed=nothing)

Sample `n` orthogonal matrices from Haar measure on O(d).
Returns `(d × d)` if `n==1`, else `(n × d × d)`.
"""
function COE(d::Integer, pos_n::Integer=1, pos_seed=nothing; n::Integer=pos_n, seed=pos_seed)
    seed !== nothing && Random.seed!(seed)
    results = Array{Float64}(undef, n, d, d)
    for i in 1:n
        M = randn(d, d)
        F = qr(M)
        Q = Matrix(F.Q) * Diagonal(sign.(diag(F.R)))
        results[i, :, :] .= Q
    end
    return n == 1 ? results[1, :, :] : results
end

"""
    CUE(d::Integer, n::Integer=1, seed=nothing)

Sample `n` unitary matrices from Haar measure on U(d).
Returns `(d × d)` if `n==1`, else `(n × d × d)`.
"""
function CUE(d::Integer, pos_n::Integer=1, pos_seed=nothing; n::Integer=pos_n, seed=pos_seed)
    seed !== nothing && Random.seed!(seed)
    results = Array{ComplexF64}(undef, n, d, d)
    for i in 1:n
        M = randn(ComplexF64, d, d)
        F = qr(M)
        Q = Matrix(F.Q) * Diagonal(sign.(diag(F.R)))
        results[i, :, :] .= Q
    end
    return n == 1 ? results[1, :, :] : results
end

"""
    GOE(N::Integer, sigsq, n::Integer=1, seed=nothing)

Sample `n` real symmetric (GOE) random matrices of size `N × N`.
"""
function GOE(N::Integer, sigsq, pos_n::Integer=1, pos_seed=nothing; n::Integer=pos_n, seed=pos_seed)
    seed !== nothing && Random.seed!(seed)
    if n == 1
        M = randn(N, N) .* sqrt(sigsq / (4N))
        return M + M'
    else
        results = Array{Float64}(undef, n, N, N)
        for i in 1:n
            M = randn(N, N) .* sqrt(sigsq / (4N))
            results[i, :, :] .= M + M'
        end
        return results
    end
end

"""
    GUE(N::Integer, sigsq, n::Integer=1, seed=nothing)

Sample `n` complex Hermitian (GUE) random matrices of size `N × N`.
"""
function GUE(N::Integer, sigsq, pos_n::Integer=1, pos_seed=nothing; n::Integer=pos_n, seed=pos_seed)
    seed !== nothing && Random.seed!(seed)
    σ = sqrt(sigsq / (8N))
    if n == 1
        M = randn(ComplexF64, N, N) .* σ
        return M + M'
    else
        results = Array{ComplexF64}(undef, n, N, N)
        for i in 1:n
            M = randn(ComplexF64, N, N) .* σ
            results[i, :, :] .= M + M'
        end
        return results
    end
end

"""Gaussian function: A * exp(-(x-mean)² / (2*var))."""
gaussianFunc(x, mean, A, var) = @. A * exp(-(x - mean)^2 / (2 * var))

"""
    getRandomGaussComplex(params, shape, seed=nothing)

Random complex numbers: Cartesian `(μ_re, σ²_re, μ_im, σ²_im)` or
Polar `(μ_R, σ²_R)` with uniform angle.
"""
function getRandomGaussComplex(params, shape, pos_seed=nothing; seed=pos_seed)
    seed !== nothing && Random.seed!(seed)
    if length(params) == 2
        R = randn(shape...) .* sqrt(params[2]) .+ params[1]
        θ = rand(shape...) .* 2π .- π
        return R .* exp.(1im .* θ)
    else
        re = randn(shape...) .* sqrt(params[2]) .+ params[1]
        im_part = randn(shape...) .* sqrt(params[4]) .+ params[3]
        return re .+ 1im .* im_part
    end
end

"""
    getUniform(params, shape, seed=nothing)

Uniform complex numbers: Polar `(lo_R, hi_R)` with uniform angle, or
Cartesian `(lo_re, hi_re, lo_im, hi_im)`.
"""
function getUniform(params, shape, pos_seed=nothing; seed=pos_seed)
    seed !== nothing && Random.seed!(seed)
    if length(params) == 2
        R = rand(shape...) .* (params[2] - params[1]) .+ params[1]
        θ = rand(shape...) .* 2π .- π
        return R .* exp.(1im .* θ)
    else
        re = rand(shape...) .* (params[2] - params[1]) .+ params[1]
        im_part = rand(shape...) .* (params[4] - params[3]) .+ params[3]
        return re .+ 1im .* im_part
    end
end


# ═══════════════════════════════════════════════════════════════════════
# LINEAR ALGEBRA
# ═══════════════════════════════════════════════════════════════════════

"""
    biorthogonal_eig(A) → (eigs, vL, vR)

Bi-orthogonal eigendecomposition: `A = vR * Diagonal(eigs) * vL'`,
with `vL[:, i]' * vR[:, j] = δ_{ij}`.
"""
function biorthogonal_eig(A::AbstractMatrix)
    F = eigen(A)
    eigs = F.values
    vR = F.vectors
    Fl = eigen(collect(A'))
    vL = Fl.vectors
    # Reorder vL to match eigenvalues and normalize biorthogonality
    # Match left eigenvectors to right by maximum overlap
    matched = falses(length(eigs))
    perm = zeros(Int, length(eigs))
    for i in eachindex(eigs)
        best_j = 0
        best_val = 0.0
        for j in eachindex(eigs)
            matched[j] && continue
            val = abs(dot(vL[:, j], vR[:, i]))
            if val > best_val
                best_val = val
                best_j = j
            end
        end
        perm[i] = best_j
        matched[best_j] = true
    end
    vL = vL[:, perm]
    for i in eachindex(eigs)
        scale = dot(vL[:, i], vR[:, i])
        vL[:, i] ./= scale
    end
    return eigs, vL, vR
end

"""
    densetoBlockDiag(A, bSizes)

Extract diagonal blocks from a dense block-diagonal matrix and return
as a sparse `blockdiag`.
"""
function densetoBlockDiag(A::AbstractMatrix, bSizes)
    indices = cumsum(bSizes)
    starts = vcat(1, indices[1:end-1] .+ 1)
    blocks = [sparse(A[starts[i]:indices[i], starts[i]:indices[i]])
              for i in eachindex(starts)]
    return blockdiag(blocks...)
end

"""
    gm_basis(v) → Matrix

Gram-Schmidt orthonormal basis whose first column is `v / ‖v‖`.
Returns a `(d × d)` matrix.
"""
function gm_basis(v::AbstractVector)
    d = length(v)
    Id = Matrix{eltype(v)}(I, d, d)
    X = [v / norm(v)]
    for i in 1:d-1
        # Project out all existing basis vectors
        x = Id[:, i]
        for q in X
            x .-= dot(q, x) .* q
        end
        x ./= norm(x)
        push!(X, x)
    end
    return hcat(X...)
end


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

"""
    unique_in_sorted(arr, returnCounts::Bool=false)

For a **sorted** array, return unique values (and optionally their counts)
in O(n).
"""
function unique_in_sorted(arr, pos_returnCounts::Bool=false; returnCounts::Bool=pos_returnCounts)
    isempty(arr) && return returnCounts ? (eltype(arr)[], Int[]) : eltype(arr)[]
    uniq = [arr[1]]
    counts = returnCounts ? [1] : nothing
    for i in 2:length(arr)
        if arr[i] != arr[i-1]
            push!(uniq, arr[i])
            returnCounts && push!(counts, 1)
        else
            returnCounts && (counts[end] += 1)
        end
    end
    return returnCounts ? (uniq, counts) : uniq
end

end # module TamLib
