module TamFermion

using LinearAlgebra
using SparseArrays
using LinearMaps

using ..TamLib

# ═══════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════

# Data types
export DtSite, DtMb, FGate
# Hilbert space
export int2occ, getReducedHilSpace
# Basis conversion
export uint_for_bits, combineSpinInts, splitSpinInts
# Fermion gates
export singleSpinChannelCombos, enumerate_ferm_excitations, sortGatesByIJ
# Excitation operators
export excitation_operator_sector, fgateToExpSector, fgateToExp, tau_g_operator_sector, fgateToTauSector
# Translation invariance
export translOpnD, buildOrbitsnD, translnDCOB, reorderByTotMom_orbit
# Slater momentum basis
export SlaterCOB_RtoK_nparticle, sectorTotMom, fullSlaterMomBasis
# Hamiltonians
export findForwardLatticeNeighbors, findLatticeEdges
export fermionNNHopping, fermionOnSiteSpinDensity, Hubbard, HubbardRealSpace, HubbardOrbitBasis, HubbardMomentumBasis
# Conversions to exact exponential
export get_basis_sector, conf_to_int, conjugate_canonical, key_to_canonical

# ═══════════════════════════════════════════════════════════════════════
# DATA TYPES
# ═══════════════════════════════════════════════════════════════════════

const DtSite = UInt8
const DtMb = UInt32

"""
Fermion gate: creation/annihilation operators for up and down spins.
Each field is a bit-mask with 1s at the sites being created/annihilated.
"""
struct FGate
    cre_up::DtMb
    ann_up::DtMb
    cre_dn::DtMb
    ann_dn::DtMb
end


# ═══════════════════════════════════════════════════════════════════════
# REDUCED HILBERT SPACE
# ═══════════════════════════════════════════════════════════════════════

"""
    int2occ(s, N) → Vector{Int}

Convert integer bit-string `s` to a vector of **1-based** occupied site
indices for an `N`-site system.
"""
function int2occ(s::Integer, N::Integer)
    return [b + 1 for b in 0:N-1 if (s >> b) & 1 == 1]
end

"""
    int2occ(s::AbstractVector, N) → Matrix{Int} or Vector{Vector{Int}}

Vectorised version.  Returns an `(n × k)` matrix if all states have the
same particle count, otherwise a vector of vectors.
"""
function int2occ(s::AbstractVector{<:Integer}, N::Integer)
    pops = [count_ones(x) for x in s]
    if all(pops .== pops[1])
        n = pops[1]
        occ = Matrix{Int}(undef, length(s), n)
        for (i, x) in enumerate(s)
            k = 1
            for b in 0:N-1
                if (x >> b) & 1 == 1
                    occ[i, k] = b + 1
                    k += 1
                end
            end
        end
        return occ
    else
        return [int2occ(x, N) for x in s]
    end
end

"""
    getReducedHilSpace(N, n) → (ints, occ)

Enumerate all `(N choose n)` basis states using Gosper's hack.
Returns:
- `ints`: `Vector{DtMb}` of integer bit-strings.
- `occ`:  `(dim × n)` matrix of **1-based** occupied site indices.
"""
function getReducedHilSpace(N::Integer, n::Integer, pos_returnOcc::Bool=true; returnOcc::Bool=pos_returnOcc)
    dim_H = binomial(N, n)
    ints = Vector{DtMb}(undef, dim_H)
    local occ
    if returnOcc
        occ = Matrix{Int}(undef, dim_H, n)
    end

    if n == 0
        ints[1] = DtMb(0)
        return returnOcc ? (ints, occ) : ints
    end

    # First state: n lowest bits set
    s = DtMb((1 << n) - 1)
    for idx in 1:dim_H
        ints[idx] = s
        if returnOcc
            # Extract occupied sites via trailing_zeros
            tmp = s
            for k in 1:n
                occ[idx, k] = trailing_zeros(tmp) + 1   # 1-based
                tmp &= tmp - DtMb(1)                     # clear lowest set bit
            end
        end
        # Gosper's hack: next combination
        if idx < dim_H
            c = s & (-s)                    # lowest set bit
            r = s + c                       # carry
            s = (((r ⊻ s) >> 2) ÷ c) | r   # next combination
        end
    end
    return returnOcc ? (ints, occ) : ints
end


# ═══════════════════════════════════════════════════════════════════════
# BASIS CONVERSION
# ═══════════════════════════════════════════════════════════════════════

"""Return the narrowest unsigned integer type that fits `N` bits."""
function uint_for_bits(N::Integer)
    N <= 8 && return UInt8
    N <= 16 && return UInt16
    N <= 32 && return UInt32
    N <= 64 && return UInt64
    return UInt128
end

"""
    combineSpinInts(ints_up, ints_dn, N) → Vector{UInt}

Pack two N-bit spin-channel integers into a single 2N-bit integer:
`combined = ints_up | (ints_dn << N)`.
"""
function combineSpinInts(ints_up, ints_dn, N::Integer)
    T = uint_for_bits(2N)
    return T.(ints_up) .| (T.(ints_dn) .<< T(N))
end

"""
    splitSpinInts(ints, N) → (ints_up, ints_dn)

Unpack a 2N-bit integer into two N-bit spin-channel integers.
"""
function splitSpinInts(ints, N::Integer)
    mask = (one(eltype(ints)) << N) - one(eltype(ints))
    return ints .& mask, (ints .>> N) .& mask
end


# ═══════════════════════════════════════════════════════════════════════
# FERMION GATES
# ═══════════════════════════════════════════════════════════════════════

"""
    singleSpinChannelCombos(N, nc, na; basis=nothing, returnBasis=false,
                            returnSupport=false, returnMom=false, Lvec=nothing)

Enumerate all single-spin-channel excitation operators with `nc`
creation and `na` annihilation operators on `N` sites.

Returns a `Vector{DtMb}` of creation masks and annihilation masks,
plus optional support and momentum information.
"""
function singleSpinChannelCombos(N::Integer, nc::Integer, na::Integer,
    pos_returnMomTransfer::Bool=false,
    pos_Lvec=nothing;
    allow_overlap::Bool=false,
    returnMomTransfer::Bool=pos_returnMomTransfer,
    Lvec=pos_Lvec,
    basis=nothing,
    returnBasis::Bool=false,
    returnSupport::Bool=false,
    returnMom::Bool=false,
    include_diagonal::Bool=false)

    actual_Lvec = Lvec !== nothing ? Lvec : pos_Lvec

    if nc == 0 && na == 0
        cre = DtMb[0]
        ann = DtMb[0]
        return returnMomTransfer ? (cre, ann, Int[0]) : (cre, ann)
    end
    too_many = allow_overlap ? (nc > N || na > N) : (nc + na > N)
    if too_many
        cre = DtMb[]
        ann = DtMb[]
        return returnMomTransfer ? (cre, ann, Int[]) : (cre, ann)
    end

    if allow_overlap
        cre_int, cre_occ = getReducedHilSpace(N, nc, true)
        ann_int, ann_occ = (na == nc) ? (cre_int, cre_occ) : getReducedHilSpace(N, na, true)
        cre_flat, ann_flat = cartesian_prod(cre_int, ann_int)
        Dc = length(cre_int)
        Da = length(ann_int)

        if !returnMomTransfer
            result = (cre_flat, ann_flat)
            if returnSupport
                all_support = cre_flat .| ann_flat
                result = (result..., all_support)
            end
            if returnMom && actual_Lvec !== nothing
                ndim = length(actual_Lvec)
                coords = getLatticeCoord(actual_Lvec)
                qsum_cre = [sum(coords[cre_occ[i, k], a] for k in 1:nc) for i in 1:Dc, a in 1:ndim]
                qsum_ann = (na == nc) ? qsum_cre : [sum(coords[ann_occ[j, k], a] for k in 1:na) for j in 1:Da, a in 1:ndim]
                qsum_cre_flat = repeat(qsum_cre, inner=(Da, 1))
                qsum_ann_flat = repeat(qsum_ann, Dc, 1)
                all_mom = qsum_cre_flat .- qsum_ann_flat
                result = (result..., all_mom)
            end
            if returnBasis
                if basis === nothing
                    basis = getReducedHilSpace(N, nc, returnOcc=false)
                end
                result = (result..., basis)
            end
            return result
        end

        if actual_Lvec === nothing
            error("returnMomTransfer=true requires Lvec.")
        end
        ndim = length(actual_Lvec)
        dims = Tuple(actual_Lvec)
        coords = getLatticeCoord(actual_Lvec)
        qsum_cre = [sum(coords[cre_occ[i, k], a] for k in 1:nc) for i in 1:Dc, a in 1:ndim]
        qsum_ann = (na == nc) ? qsum_cre : [sum(coords[ann_occ[j, k], a] for k in 1:na) for j in 1:Da, a in 1:ndim]
        qsum_cre_flat = repeat(qsum_cre, inner=(Da, 1))
        qsum_ann_flat = repeat(qsum_ann, Dc, 1)
        delta = qsum_cre_flat .- qsum_ann_flat
        delta_int = Vector{Int}(undef, Dc * Da)
        for idx in 1:(Dc*Da)
            coord_tuple = ntuple(a -> delta[idx, a], ndim)
            delta_int[idx] = ravel_c(coord_tuple, dims; mode=:wrap)
        end
        return cre_flat, ann_flat, delta_int

    else
        # Choose nc+na occupied sites
        supp_int, supp_occ = getReducedHilSpace(N, nc + na, returnOcc=true)
        D = length(supp_int)

        # bit_val: (D, nc+na)
        bit_val = DtMb(1) .<< DtMb.(supp_occ .- 1)

        # Split which slots are creators
        split_int = getReducedHilSpace(nc + na, nc, returnOcc=false)
        K = length(split_int)

        # cre_bits: (K, nc+na)
        cre_bits = [Int((split_int[k] >> j) & 1) for k in 1:K, j in 0:(nc+na-1)]

        # cre: D × K
        cre = bit_val * Matrix{DtMb}(cre_bits')

        # ann: D × K
        ann = supp_int .- cre

        # Flatten cre and ann in C-order
        cre_flat = Vector{DtMb}(undef, D * K)
        ann_flat = Vector{DtMb}(undef, D * K)
        idx = 1
        for d in 1:D
            for k in 1:K
                cre_flat[idx] = cre[d, k]
                ann_flat[idx] = ann[d, k]
                idx += 1
            end
        end

        local D_diag = 0
        local diag_ints
        if include_diagonal && nc == na && nc > 0
            diag_ints = getReducedHilSpace(N, nc, false)
            D_diag = length(diag_ints)
        end

        if D_diag > 0
            append!(cre_flat, diag_ints)
            append!(ann_flat, diag_ints)
        end

        if !returnMomTransfer
            result = (cre_flat, ann_flat)
            if returnSupport
                all_support = [supp_int[d] for d in 1:D for k in 1:K]
                if D_diag > 0
                    append!(all_support, diag_ints)
                end
                result = (result..., all_support)
            end
            if returnMom && actual_Lvec !== nothing
                ndim = length(actual_Lvec)
                dims = Tuple(actual_Lvec)
                occ_coords = unravel_c(supp_occ .- 1, dims)
                sign_mat = 2 .* cre_bits .- 1
                delta_a = [occ_coords[a] * sign_mat' for a in 1:ndim]
                all_mom = Matrix{Int}(undef, D * K, ndim)
                idx_mom = 1
                for d in 1:D
                    for k in 1:K
                        for a in 1:ndim
                            all_mom[idx_mom, a] = delta_a[a][d, k]
                        end
                        idx_mom += 1
                    end
                end
                if D_diag > 0
                    all_mom = vcat(all_mom, zeros(Int, D_diag, ndim))
                end
                result = (result..., all_mom)
            end
            if returnBasis
                if basis === nothing
                    basis = getReducedHilSpace(N, nc, returnOcc=false)
                end
                result = (result..., basis)
            end
            return result
        end

        # returnMomTransfer is true, so compute delta
        if actual_Lvec === nothing
            error("returnMomTransfer=true requires Lvec.")
        end
        ndim = length(actual_Lvec)
        dims = Tuple(actual_Lvec)

        occ_coords = unravel_c(supp_occ .- 1, dims)
        sign_mat = 2 .* cre_bits .- 1
        delta_a = [occ_coords[a] * sign_mat' for a in 1:ndim]

        delta_int = Vector{Int}(undef, D * K)
        idx = 1
        for d in 1:D
            for k in 1:K
                coord_tuple = ntuple(a -> delta_a[a][d, k], ndim)
                delta_int[idx] = ravel_c(coord_tuple, dims; mode=:wrap)
                idx += 1
            end
        end

        if D_diag > 0
            append!(delta_int, zeros(Int, D_diag))
        end

        return cre_flat, ann_flat, delta_int
    end
end

"""
    _match_mom(del_u, del_d, N; Lvec) → Vector{Tuple{Int,Int}}

Find pairs (iu, id) where up-momentum + down-momentum = 0 (mod Lvec).
`del_u, del_d`: `(n × ndim)` matrices of momentum transfer vectors.
"""
function _match_mom(del_u::AbstractVector{<:Integer}, del_d::AbstractVector{<:Integer}, N::Integer)
    if isempty(del_u) || isempty(del_d)
        return Int[], Int[]
    end
    order_u = sortperm(del_u)
    order_d = sortperm(del_d)

    cts_u = zeros(Int, N)
    for x in del_u
        cts_u[x+1] += 1
    end
    cts_d = zeros(Int, N)
    for x in del_d
        cts_d[x+1] += 1
    end

    start_u = fill(-1, N)
    curr = 1
    for q in 1:N
        if cts_u[q] > 0
            start_u[q] = curr
            curr += cts_u[q]
        end
    end

    start_d = fill(-1, N)
    curr = 1
    for q in 1:N
        if cts_d[q] > 0
            start_d[q] = curr
            curr += cts_d[q]
        end
    end

    iu_parts = Int[]
    id_parts = Int[]

    for q in 1:N
        if cts_u[q] == 0 || cts_d[q] == 0
            continue
        end
        su = start_u[q]
        sd = start_d[q]
        iu_q = order_u[su:su+cts_u[q]-1]
        id_q = order_d[sd:sd+cts_d[q]-1]

        for u_val in iu_q
            for _ in 1:cts_d[q]
                push!(iu_parts, u_val)
            end
        end
        for _ in 1:cts_u[q]
            append!(id_parts, id_q)
        end
    end

    return iu_parts, id_parts
end

"""
    enumerate_ferm_excitations(p, Lvec, conserve_mom=true, conserve_sz=true, conserve_su2=false)

Enumerate all fermion excitations of order `p` on a lattice with
dimensions `Lvec`.  Returns a `Vector{FGate}`.

Order `p` = number of creation + annihilation operators per spin channel,
summed over both channels.
"""
function enumerate_ferm_excitations(p::Integer, Lvec,
    pos_conserve_mom::Bool=true,
    pos_conserve_sz::Bool=true,
    pos_conserve_su2::Bool=false,
    pos_include_diagonal::Bool=true;
    allow_overlap::Bool=false,
    conserve_mom::Bool=pos_conserve_mom,
    conserve_sz::Bool=pos_conserve_sz,
    conserve_su2::Bool=pos_conserve_su2,
    include_diagonal::Bool=pos_include_diagonal)

    if conserve_su2
        throw(ErrorException("Total-spin (SU(2)) conservation is a different, non-abelian symmetry; not implemented."))
    end

    N = prod(Lvec)
    if p > N
        @warn "No p-excitation exists for p=$p > N=$N."
        return FGate[]
    end

    local INV_IDX
    if conserve_mom
        dims = Tuple(Lvec)
        INV_IDX = Vector{Int}(undef, N)
        for k in 0:N-1
            coords = unravel_c(k, dims)
            inv_coords = Tuple(-c for c in coords)
            INV_IDX[k+1] = ravel_c(inv_coords, dims; mode=:wrap)
        end
    end

    splits = Tuple{Int,Int}[]
    if conserve_sz
        for nc in 0:p
            push!(splits, (nc, nc))
        end
    else
        for nc in 0:p
            for na in 0:p
                push!(splits, (nc, na))
            end
        end
    end

    need = Set{Tuple{Int,Int}}()
    for (nc_up, na_up) in splits
        push!(need, (nc_up, na_up))
        push!(need, (p - nc_up, p - na_up))
    end

    S = Dict{Tuple{Int,Int},Tuple}()
    for (nc, na) in need
        S[(nc, na)] = singleSpinChannelCombos(N, nc, na; allow_overlap=allow_overlap, returnMomTransfer=conserve_mom, Lvec=Lvec, include_diagonal=include_diagonal)
    end

    gates = FGate[]
    for (nc_up, na_up) in splits
        if conserve_mom
            cre_up, ann_up, del_up = S[(nc_up, na_up)]
            cre_dn, ann_dn, del_dn = S[(p - nc_up, p - na_up)]
            iup, idn = _match_mom(del_up, INV_IDX[del_dn.+1], N)
            for k in eachindex(iup)
                push!(gates, FGate(cre_up[iup[k]], ann_up[iup[k]], cre_dn[idn[k]], ann_dn[idn[k]]))
            end
        else
            cre_up, ann_up = S[(nc_up, na_up)]
            cre_dn, ann_dn = S[(p - nc_up, p - na_up)]
            for iu in eachindex(cre_up)
                for id in eachindex(cre_dn)
                    push!(gates, FGate(cre_up[iu], ann_up[iu], cre_dn[id], ann_dn[id]))
                end
            end
        end
    end

    # filter out hermitian conjugate
    filter!(g -> (g.cre_up, g.cre_dn) <= (g.ann_up, g.ann_dn), gates)

    return gates
end

"""
    tau_g_operator_sector(g::FGate, N, basis; sortOrder=nothing, spec_mask=nothing)
Matrix-free representation of the excitation generator operator τ_g
restricted to a symmetry sector defined by `basis`, where τ_g = C†_I C_J + C†_J C_I.
Returns a `LinearMap{ComplexF64}` of size `(d × d)`.
"""
function tau_g_operator_sector(g::FGate, N::Integer,
    basis::AbstractVector{<:Integer};
    sortOrder=nothing,
    spec_mask=nothing,
    antihermitian::Bool=false)
    d = length(basis)
    nbits = 2N
    s_I = UInt64(g.cre_up) | (UInt64(g.cre_dn) << N)
    s_J = UInt64(g.ann_up) | (UInt64(g.ann_dn) << N)
    basis64 = UInt64.(basis)

    s_Ip = s_I & ~s_J
    s_Jp = s_J & ~s_I
    Delta = s_I ⊻ s_J
    s_IJ = s_I | s_J
    p = count_ones(s_J)

    sign0 = (div(p * (p - 1), 2) % 2 == 0) ? 1.0 : -1.0
    sgn_ref = _jw_sign_ref(s_I, s_J, nbits)
    mask = spec_mask !== nothing ? spec_mask : _odd_spectator_mask(Delta, s_IJ, nbits)

    if s_I == s_J
        if antihermitian
            return LinearMap{ComplexF64}(
                v -> fill!(similar(v, ComplexF64), 0.0),
                v -> fill!(similar(v, ComplexF64), 0.0),
                d, d
            )
        else
            val = 2.0 * sign0
            function _apply_diag!(w, v)
                for i in 1:d
                    if (basis64[i] & s_I) == s_I
                        w[i] = val * v[i]
                    else
                        w[i] = 0.0
                    end
                end
                return w
            end
            return LinearMap{ComplexF64}(
                v -> _apply_diag!(similar(v, ComplexF64), v),
                d, d; ishermitian=true
            )
        end
    end

    srcJ_mask = ((basis64 .& s_J) .== s_J) .& ((basis64 .& s_Ip) .== UInt64(0))
    isrcJ = findall(srcJ_mask)
    srcJ = basis64[isrcJ]

    srcI_mask = ((basis64 .& s_I) .== s_I) .& ((basis64 .& s_Jp) .== UInt64(0))

    if sortOrder !== nothing
        order = sortOrder
    else
        order = sortperm(basis64)
    end
    sorted_basis = basis64[order]

    itgtI = Vector{Int}(undef, length(isrcJ))
    for k in eachindex(isrcJ)
        s = srcJ[k]
        t = s ⊻ Delta
        j = searchsortedfirst(sorted_basis, t)
        if j <= d && sorted_basis[j] == t
            itgtI[k] = order[j]
        else
            throw(ArgumentError("The given gate maps states out of the provided basis. Either gate doesn't preserve sector's quantum numbers or basis is incomplete."))
        end
    end

    if sum(srcJ_mask) != sum(srcI_mask)
        throw(ArgumentError("The given gate maps states out of the provided basis. Either gate doesn't preserve sector's quantum numbers or basis is incomplete."))
    end

    signs = Vector{Float64}(undef, length(isrcJ))
    for k in eachindex(isrcJ)
        s = srcJ[k]
        spec_parity = count_ones(s & mask) & 1
        signs[k] = sgn_ref * (spec_parity == 1 ? -1.0 : 1.0)
    end

    function _apply!(w, v)
        fill!(w, 0.0)
        for k in eachindex(isrcJ)
            si, ti = isrcJ[k], itgtI[k]
            w[si] = -signs[k] * v[ti]
            w[ti] = signs[k] * v[si]
        end
        return w
    end

    function _apply_herm!(w, v)
        fill!(w, 0.0)
        for k in eachindex(isrcJ)
            si, ti = isrcJ[k], itgtI[k]
            w[si] = signs[k] * v[ti]
            w[ti] = signs[k] * v[si]
        end
        return w
    end

    if antihermitian
        function _apply_adj!(w, v)
            fill!(w, 0.0)
            for k in eachindex(isrcJ)
                si, ti = isrcJ[k], itgtI[k]
                w[si] = signs[k] * v[ti]
                w[ti] = -signs[k] * v[si]
            end
            return w
        end
        return LinearMap{ComplexF64}(
            v -> _apply!(similar(v, ComplexF64), v),
            v -> _apply_adj!(similar(v, ComplexF64), v),
            d, d
        )
    else
        return LinearMap{ComplexF64}(
            v -> _apply_herm!(similar(v, ComplexF64), v),
            d, d; ishermitian=true
        )
    end
end

"""
    fgateToTauSector(gateLabels, N, basis) → Vector{LinearMap}

Build the generator operator `τ_k` for each gate label `k`.
"""
function fgateToTauSector(gateLabels::AbstractVector{FGate},
    N::Integer,
    basis::AbstractVector{<:Integer};
    antihermitian::Bool=false)
    order = sortperm(UInt64.(basis))
    nbits = 2N
    spec_masks = Vector{UInt64}(undef, length(gateLabels))
    for k in eachindex(gateLabels)
        g = gateLabels[k]
        s_I = UInt64(g.cre_up) | (UInt64(g.cre_dn) << N)
        s_J = UInt64(g.ann_up) | (UInt64(g.ann_dn) << N)
        Delta = s_I ⊻ s_J
        s_IJ = s_I | s_J
        spec_masks[k] = _odd_spectator_mask(Delta, s_IJ, nbits)
    end
    return [tau_g_operator_sector(gateLabels[k], N, basis; sortOrder=order, spec_mask=spec_masks[k], antihermitian=antihermitian)
            for k in eachindex(gateLabels)]
end

# ═══════════════════════════════════════════════════════════════════════
# EXCITATION OPERATORS
# ═══════════════════════════════════════════════════════════════════════

"""
    _jw_sign_ref(M_I, M_J, nbits) → Int

Jordan-Wigner sign from a fixed reference ordering for the operator
C†_{M_I} C_{M_J} applied to the reference state with all bits set.
"""
function _jw_sign_ref(M_I::Integer, M_J::Integer, nbits::Integer)
    y = UInt64(M_J)
    s = 1
    # Annihilators act first in reverse/decreasing order
    for j in nbits-1:-1:0
        if ((M_J >> j) & 1) == 1
            mask = (one(UInt64) << j) - one(UInt64)
            if (count_ones(y & mask) & 1) == 1
                s = -s
            end
            y &= ~(one(UInt64) << j)
        end
    end
    # Then creators act in reverse/decreasing order
    for i in nbits-1:-1:0
        if ((M_I >> i) & 1) == 1
            mask = (one(UInt64) << i) - one(UInt64)
            if (count_ones(y & mask) & 1) == 1
                s = -s
            end
            y |= (one(UInt64) << i)
        end
    end
    return s
end


"""
    _odd_spectator_mask(Delta, s_IJ, nbits) → UInt64

Mask whose set bits indicate the odd-parity spectator positions between
the creation/annihilation sites.
"""
function _odd_spectator_mask(Delta::UInt64, s_IJ::UInt64, nbits::Integer)
    y = Delta
    sh = 1
    while sh < nbits
        y = y ⊻ (y >> sh)
        sh <<= 1
    end
    is_spectator = ((one(UInt64) << nbits) - one(UInt64)) ⊻ s_IJ
    return (y >> 1) & is_spectator
end

# Backward compatibility fallback
function _odd_spectator_mask(M_IJ::UInt64, nbits::Integer)
    return _odd_spectator_mask(M_IJ, M_IJ, nbits)
end

"""
    excitation_operator_sector(g::FGate, N, a, basis; sortOrder=nothing, spec_mask=nothing)

Matrix-free representation of exp(i·a·τ_g) restricted to a symmetry
sector defined by `basis`, where τ_g = C†_I C_J + C†_J C_I.

Returns a `LinearMap{ComplexF64}` of size `(d × d)`.
"""
function excitation_operator_sector(g::FGate, N::Integer, a::Real,
    basis::AbstractVector{<:Integer};
    sortOrder=nothing,
    spec_mask=nothing,
    antihermitian::Bool=false)
    d = length(basis)
    nbits = 2N
    s_I = UInt64(g.cre_up) | (UInt64(g.cre_dn) << N)
    s_J = UInt64(g.ann_up) | (UInt64(g.ann_dn) << N)
    basis64 = UInt64.(basis)

    s_Ip = s_I & ~s_J
    s_Jp = s_J & ~s_I
    Delta = s_I ⊻ s_J
    s_IJ = s_I | s_J
    p = count_ones(s_J)

    sign0 = (div(p * (p - 1), 2) % 2 == 0) ? 1.0 : -1.0
    sgn_ref = _jw_sign_ref(s_I, s_J, nbits)
    mask = spec_mask !== nothing ? spec_mask : _odd_spectator_mask(Delta, s_IJ, nbits)

    if s_I == s_J
        if antihermitian
            return LinearMap{ComplexF64}(
                v -> copy(v),
                v -> copy(v),
                d, d
            )
        else
            phase_val = exp(2im * a * sign0)
            function _apply_diag!(w, v, ph)
                for i in 1:d
                    if (basis64[i] & s_I) == s_I
                        w[i] = ph * v[i]
                    else
                        w[i] = v[i]
                    end
                end
                return w
            end
            return LinearMap{ComplexF64}(
                v -> _apply_diag!(similar(v, ComplexF64), v, phase_val),
                v -> _apply_diag!(similar(v, ComplexF64), v, conj(phase_val)),
                d, d
            )
        end
    end

    srcJ_mask = ((basis64 .& s_J) .== s_J) .& ((basis64 .& s_Ip) .== UInt64(0))
    isrcJ = findall(srcJ_mask)
    srcJ = basis64[isrcJ]

    srcI_mask = ((basis64 .& s_I) .== s_I) .& ((basis64 .& s_Jp) .== UInt64(0))

    if sortOrder !== nothing
        order = sortOrder
    else
        order = sortperm(basis64)
    end
    sorted_basis = basis64[order]

    itgtI = Vector{Int}(undef, length(isrcJ))
    for k in eachindex(isrcJ)
        s = srcJ[k]
        t = s ⊻ Delta
        j = searchsortedfirst(sorted_basis, t)
        if j <= d && sorted_basis[j] == t
            itgtI[k] = order[j]
        else
            throw(ArgumentError("The given gate maps states out of the provided basis. Either gate doesn't preserve sector's quantum numbers or basis is incomplete."))
        end
    end

    if sum(srcJ_mask) != sum(srcI_mask)
        throw(ArgumentError("The given gate maps states out of the provided basis. Either gate doesn't preserve sector's quantum numbers or basis is incomplete."))
    end

    signs = Vector{Float64}(undef, length(isrcJ))
    for k in eachindex(isrcJ)
        s = srcJ[k]
        spec_parity = count_ones(s & mask) & 1
        signs[k] = sgn_ref * (spec_parity == 1 ? -1.0 : 1.0)
    end

    ca = cos(a)
    if antihermitian
        coef = sin(a) .* signs
        function _apply_anti!(w, v, off_signs)
            w .= v
            for k in eachindex(isrcJ)
                si, ti = isrcJ[k], itgtI[k]
                vs, vt = v[si], v[ti]
                w[si] = ca * vs - off_signs[k] * vt
                w[ti] = off_signs[k] * vs + ca * vt
            end
            return w
        end
        return LinearMap{ComplexF64}(
            v -> _apply_anti!(similar(v, ComplexF64), v, coef),
            v -> _apply_anti!(similar(v, ComplexF64), v, -coef),
            d, d
        )
    else
        coef = 1im * sin(a) .* signs
        function _apply!(w, v, off_signs)
            w .= v
            for k in eachindex(isrcJ)
                si, ti = isrcJ[k], itgtI[k]
                vs, vt = v[si], v[ti]
                w[si] = ca * vs + off_signs[k] * vt
                w[ti] = off_signs[k] * vs + ca * vt
            end
            return w
        end
        return LinearMap{ComplexF64}(
            v -> _apply!(similar(v, ComplexF64), v, coef),
            v -> _apply!(similar(v, ComplexF64), v, -coef),
            d, d
        )
    end
end

"""
    fgateToExpSector(gateLabels, A, N, basis) → Vector{LinearMap}

Build `exp(i·A[k]·τ_k)` for each gate label `k`.
"""
function fgateToExpSector(gateLabels::AbstractVector{FGate},
    A::AbstractVector{<:Real},
    N::Integer,
    basis::AbstractVector{<:Integer};
    antihermitian::Bool=false)
    order = sortperm(UInt64.(basis))
    nbits = 2N
    spec_masks = Vector{UInt64}(undef, length(gateLabels))
    for k in eachindex(gateLabels)
        g = gateLabels[k]
        s_I = UInt64(g.cre_up) | (UInt64(g.cre_dn) << N)
        s_J = UInt64(g.ann_up) | (UInt64(g.ann_dn) << N)
        Delta = s_I ⊻ s_J
        s_IJ = s_I | s_J
        spec_masks[k] = _odd_spectator_mask(Delta, s_IJ, nbits)
    end
    return [excitation_operator_sector(gateLabels[k], N, A[k], basis; sortOrder=order, spec_mask=spec_masks[k], antihermitian=antihermitian)
            for k in eachindex(gateLabels)]
end

"""
    fgateToExp(gateLabels, A, N, basis) → Vector{LinearMap}

Alias for `fgateToExpSector`.
"""
function fgateToExp(gateLabels::AbstractVector{FGate},
    A::AbstractVector{<:Real},
    N::Integer,
    basis::AbstractVector{<:Integer};
    antihermitian::Bool=false)
    return fgateToExpSector(gateLabels, A, N, basis; antihermitian=antihermitian)
end

"""
    sortGatesByIJ(gateLabels::AbstractVector{FGate}, N::Integer) → (sorted_labels, order)

Order gates by the 2N-bit integers (s_I, s_J): s_I varies SLOWEST, s_J FASTEST.
"""
function sortGatesByIJ(gateLabels::AbstractVector{FGate}, N::Integer)
    s_I = [UInt64(g.cre_up) | (UInt64(g.cre_dn) << N) for g in gateLabels]
    s_J = [UInt64(g.ann_up) | (UInt64(g.ann_dn) << N) for g in gateLabels]
    order = sortperm(1:length(gateLabels); by=k -> (s_I[k], s_J[k]))
    return gateLabels[order], order
end


# ═══════════════════════════════════════════════════════════════════════
# TRANSLATION INVARIANCE
# ═══════════════════════════════════════════════════════════════════════

"""
    translOpnD(Lvec; dof=:fermion, basis=nothing)

Build nD translation operators on a lattice with dimensions `Lvec`.

`dof`:
- `:spinless` — bosonic / spinless, no sign.
- `:fermion`  — single-channel fermionic, JW signs included.
- `:spinhalf` — two-channel fermionic (2N-bit basis), JW signs included.

Returns `(T_list, basis)` where `T_list[a]` is the sparse translation
operator along axis `a`.
"""
function translOpnD(Lvec,
    pos_dof::Union{Symbol,String}="fermion",
    pos_basis=nothing;
    dof::Union{Symbol,String}=pos_dof,
    basis=pos_basis)
    Lvec = collect(Int, Lvec)
    N = prod(Lvec)
    ndim = length(Lvec)
    isfermionic = Symbol(dof) in (:fermion, :spinhalf)
    nbits = Symbol(dof) == :spinhalf ? 2N : N

    if basis === nothing
        basis = DtMb.(0:2^nbits-1)
    end
    d = length(basis)

    # C-order strides
    c_strides = Vector{Int}(undef, ndim)
    c_strides[ndim] = 1
    for k in ndim-1:-1:1
        c_strides[k] = c_strides[k+1] * Lvec[k+1]
    end

    # Build C-order site array
    sites = reshape_c(collect(0:N-1), Tuple(Lvec))     # 0-based site indices

    # Bit masks for each site (0-based site i → bit 2^i)
    masks = DtMb(1) .<< DtMb.(0:N-1)

    Tops = Matrix{DtMb}(undef, ndim, d)
    signs = Matrix{Int8}(undef, ndim, d)

    for a in 1:ndim
        # Single-site permutation: shift along axis a by -1
        shift = ntuple(d -> d == a ? -1 : 0, ndim)
        ss_perm = flatten_c(circshift(sites, shift))    # 0-based perm

        # For spinhalf, duplicate the perm for the down-spin channel
        if Symbol(dof) == :spinhalf
            full_perm = vcat(ss_perm, ss_perm .+ N)
            full_masks = vcat(masks, DtMb(1) .<< DtMb.(N:2N-1))
        else
            full_perm = ss_perm
            full_masks = masks
        end

        # Bit-position shifts for each site
        shifts = full_perm .- collect(0:length(full_perm)-1)

        # Compute inversion masks for fermionic sign
        inv_masks = zeros(DtMb, length(full_perm))
        if isfermionic
            s_a = c_strides[a]
            L_a = Lvec[a]
            bs_a = L_a * s_a
            nblocks = N ÷ bs_a
            flat_idcs = collect(0:N-1)
            aidcs = (flat_idcs .÷ s_a) .% L_a
            nonbdy = aidcs .< (L_a - 1)

            start_bdy = collect(DtMb, 0:nblocks-1) .* DtMb(bs_a) .+ DtMb((L_a - 1) * s_a)
            bdy_block = DtMb.((1 << s_a) - 1) .<< start_bdy

            block_of = flat_idcs .÷ bs_a

            for i in 1:N
                if !nonbdy[i]
                    inv_masks[i] = bdy_block[block_of[i]+1]
                end
            end
            if Symbol(dof) == :spinhalf
                for i in 1:N
                    inv_masks[N+i] = DtMb(inv_masks[i]) << DtMb(N)
                end
            end
        end

        new_states = zeros(DtMb, d)
        parity = zeros(UInt8, d)

        for i in eachindex(full_perm)
            ibits = basis .& full_masks[i]
            sh = Int(shifts[i])
            if sh > 0
                new_states .|= DtMb.(UInt64.(ibits) .<< sh)
            elseif sh < 0
                new_states .|= DtMb.(UInt64.(ibits) .>> (-sh))
            else
                new_states .|= ibits
            end

            if isfermionic && inv_masks[i] != DtMb(0)
                p_i = UInt8.(fastXOR.(basis .& inv_masks[i]))
                is_set = UInt8.(ibits .!= DtMb(0))
                parity .⊻= is_set .& p_i
            end
        end

        Tops[a, :] .= new_states
        signs[a, :] .= [parity[k] == 1 ? -1 : 1 for k in 1:d]
    end

    return Tops, signs
end



function buildOrbitsnD(Lvec,
    pos_dof::Union{Symbol,String}="fermion",
    pos_basis=nothing,
    pos_state2idx=nothing;
    dof::Union{Symbol,String}=pos_dof,
    basis=pos_basis,
    state2idx=pos_state2idx)
    Lvec = collect(Int, Lvec)
    ndim = length(Lvec)
    N = prod(Lvec)

    if basis === nothing
        is_spinhalf = Symbol(dof) == :spinhalf
        nbits = is_spinhalf ? 2N : N
        basis = DtMb.(0:2^nbits-1)
    else
        basis = DtMb.(basis)
    end

    if state2idx === nothing
        state2idx = Dict{DtMb,Int}(s => i for (i, s) in enumerate(basis))
    end

    Tops, signs = translOpnD(Lvec, dof, basis)

    visited = falses(length(basis))
    orbits = Dict{Int,Dict{Symbol,Any}}()

    for (i, root) in enumerate(basis)
        visited[i] && continue
        visited[i] = true

        signed_orbit = Int[Int(root)]
        orbit_idcs = Int[i]
        orbit_signs = Int8[1]
        rvecs = [zeros(UInt8, ndim)]

        queue = [(i, 1, zeros(UInt8, ndim))]
        stabilizers = Vector{Vector{UInt8}}()
        stabilizer_signs = Int8[]

        while !isempty(queue)
            i_curr, sgn_curr, r_curr = popfirst!(queue)

            for a in 1:ndim
                next_state = Tops[a, i_curr]
                next_sign = signs[a, i_curr]
                i_next = state2idx[next_state]
                r_next = copy(r_curr)
                r_next[a] = (r_next[a] + 1) % Lvec[a]

                if !visited[i_next]
                    cumul_sgn = sgn_curr * next_sign
                    visited[i_next] = true
                    push!(signed_orbit, cumul_sgn * Int(next_state))
                    push!(orbit_idcs, i_next)
                    push!(orbit_signs, cumul_sgn)
                    push!(rvecs, copy(r_next))
                    push!(queue, (i_next, cumul_sgn, r_next))
                elseif i_next == i && !all(r_next .== 0)
                    push!(stabilizers, copy(r_next))
                    push!(stabilizer_signs, sgn_curr * next_sign)
                end
            end
        end

        # convert rvecs, stabilizers to Matrix
        rvec_mat = Matrix{UInt8}(undef, length(rvecs), ndim)
        for row in eachindex(rvecs)
            rvec_mat[row, :] .= rvecs[row]
        end

        stab_mat = Matrix{UInt8}(undef, length(stabilizers), ndim)
        for row in eachindex(stabilizers)
            stab_mat[row, :] .= stabilizers[row]
        end

        orbits[Int(root)] = Dict(
            :signed_orbit => signed_orbit,
            :orbit_idcs => orbit_idcs,
            :orbit_signs => orbit_signs,
            :rvec => rvec_mat,
            :stabilizers => stab_mat,
            :stabilizer_signs => stabilizer_signs
        )
    end
    return orbits
end

function translnDCOB(Lvec,
    pos_dof::Union{Symbol,String}="fermion",
    pos_basis=nothing,
    pos_state2idx=nothing,
    pos_returnType::String="sp";
    dof::Union{Symbol,String}=pos_dof,
    basis=pos_basis,
    state2idx=pos_state2idx,
    returnType::String=pos_returnType)
    Lvec = collect(Int, Lvec)
    N = prod(Lvec)
    ndim = length(Lvec)

    if basis === nothing
        is_spinhalf = Symbol(dof) == :spinhalf
        nbits = is_spinhalf ? 2N : N
        basis = DtMb.(0:2^nbits-1)
    else
        basis = DtMb.(basis)
    end

    if state2idx === nothing
        state2idx = Dict{DtMb,Int}(s => i for (i, s) in enumerate(basis))
    end

    orbits = buildOrbitsnD(Lvec, dof, basis, state2idx)
    lcm_L = lcm(Lvec...)
    axis_scale = [div(lcm_L, Lvec[a]) for a in 1:ndim]
    num_states = length(basis)

    P = zeros(ComplexF64, num_states, num_states)
    blockSizes = Dict{Vector{Int},Int}()
    currCol = 1

    # Iterate over all possible qvecs
    qvecs = Iterators.product((0:L-1 for L in Lvec)...)
    for qvec_tuple in qvecs
        qvec = collect(Int, qvec_tuple)
        bs = 0
        qvec_scaled = qvec .* axis_scale

        for m in keys(orbits)
            stabilizers = orbits[m][:stabilizers]
            stabilizer_signs = orbits[m][:stabilizer_signs]

            compatible = true
            if size(stabilizers, 1) > 0
                phase = (stabilizers * qvec_scaled) .% lcm_L
                expected = [div(1 - s, 2) * div(lcm_L, 2) for s in stabilizer_signs]
                compatible = all(phase .== expected)
            end

            if compatible
                orbit_idcs = orbits[m][:orbit_idcs]
                orbit_signs = orbits[m][:orbit_signs]
                M = length(orbit_idcs)
                bs += 1
                rvecs = orbits[m][:rvec] # (M × ndim)

                for idx_row in 1:M
                    oi = orbit_idcs[idx_row]
                    val = (1 / sqrt(M)) * exp(-2π * 1im * sum(qvec .* rvecs[idx_row, :] ./ Lvec)) * orbit_signs[idx_row]
                    P[oi, currCol] = val
                end
                currCol += 1
            end
        end
        blockSizes[qvec] = bs
    end

    if returnType == "sp"
        return sparse(P), blockSizes
    else
        return P, blockSizes
    end
end

function reorderByTotMom_orbit(P, bSizes_up::Dict, bSizes_down::Dict, Lvec)
    Lvec = collect(Int, Lvec)
    ndim = length(Lvec)
    dims = Tuple(Lvec)

    qvecs = collect(keys(bSizes_up)) # vector of Vector{Int}

    qu = Vector{Int}[]
    for q in qvecs
        for _ in 1:bSizes_up[q]
            push!(qu, q)
        end
    end

    qd = Vector{Int}[]
    for q in qvecs
        for _ in 1:bSizes_down[q]
            push!(qd, q)
        end
    end

    # Kronecker-product like combinations (qu is outer, qd is inner/fastest)
    qt = Matrix{Int}(undef, length(qu) * length(qd), ndim)
    idx = 1
    for i in eachindex(qu)
        for j in eachindex(qd)
            qt[idx, :] .= mod.(qu[i] .+ qd[j], Lvec)
            idx += 1
        end
    end

    # Ravel to 1D sort keys
    qt_idx = [ravel_c(Tuple(qt[k, :]), dims) for k in 1:size(qt, 1)]
    perm = sortperm(qt_idx)
    qt_sorted = qt_idx[perm]
    qt_unique, counts = unique_in_sorted(qt_sorted, true)

    return P[:, perm], counts
end


# ═══════════════════════════════════════════════════════════════════════
# SLATER MOMENTUM BASIS
# ═══════════════════════════════════════════════════════════════════════

"""
    SlaterCOB_RtoK_nparticle(Lvec, n) → (F_n, basis)

Build the `(dim × dim)` change-of-basis matrix from the real-space
Slater determinant basis to the momentum-space basis for `n` particles
on a lattice with dimensions `Lvec`.
"""
function SlaterCOB_RtoK_nparticle(Lvec, n::Integer)
    N = prod(Lvec)
    dims = Tuple(Lvec)
    ndim = length(Lvec)

    # Single-particle Fourier transform matrix (N × N)
    # F[k, r] = (1/√N) exp(-2πi Σ_a k_a r_a / L_a)
    F1 = Matrix{ComplexF64}(undef, N, N)
    for r in 0:N-1
        r_coords = unravel_c(r, dims)
        for k in 0:N-1
            k_coords = unravel_c(k, dims)
            phase = sum(k_coords[a] * r_coords[a] / Lvec[a] for a in 1:ndim)
            F1[k+1, r+1] = exp(-2π * 1im * phase) / sqrt(N)
        end
    end

    # n-particle basis: all (N choose n) states
    basis, occ = getReducedHilSpace(N, n)
    dim_H = length(basis)

    # Build the Slater determinant COB
    F_n = Matrix{ComplexF64}(undef, dim_H, dim_H)
    for i in 1:dim_H
        ki = occ[i, :]        # 1-based site indices for row (k-space)
        for j in 1:dim_H
            rj = occ[j, :]    # 1-based site indices for column (r-space)
            # Slater determinant = det(F1[ki, rj])
            F_n[i, j] = det(F1[ki, rj])
        end
    end

    return F_n, basis
end

"""
    sectorTotMom(Lvec, n) → (qt_sorted, sort_order, unique_q, counts)

Sort the n-particle basis by total momentum on a lattice with dimensions `Lvec`.
"""
function sectorTotMom(Lvec, n::Integer, pos_coords=nothing, pos_sort_flag::Bool=true;
    coords=pos_coords, sort_flag::Bool=pos_sort_flag, sort::Bool=pos_sort_flag)
    actual_sort = sort_flag && sort
    N = prod(Lvec)
    dims = Tuple(Lvec)
    ndim = length(Lvec)
    if coords === nothing
        coords = getLatticeCoord(Lvec)
    end
    ints, occ = getReducedHilSpace(N, n; returnOcc=true)
    dim_H = length(ints)

    qtotvecs = Matrix{Int}(undef, dim_H, ndim)
    for i in 1:dim_H
        for a in 1:ndim
            qtotvecs[i, a] = mod(sum(coords[occ[i, k], a] for k in 1:n), Lvec[a])
        end
    end

    qtot = [ravel_c(Tuple(qtotvecs[i, :]), dims) for i in 1:dim_H]
    sortOrder = sortperm(qtot)
    qtot_sorted = qtot[sortOrder]
    qtot_unique, counts = unique_in_sorted(qtot_sorted, true)

    if actual_sort
        ints = ints[sortOrder]
        occ = occ[sortOrder, :]
        qtot = qtot_sorted
    end

    return Dict{String,Any}(
        "ints" => ints,
        "occ" => occ,
        "qtot" => qtot,
        "sortOrder" => sortOrder,
        "qtot_unique" => qtot_unique,
        "counts" => counts
    )
end

"""
    fullSlaterMomBasis(Lvec, n_up, n_dn) → Dict{String, Any}

Full two-channel Slater momentum basis for `n_up` spin-up and `n_dn`
spin-down particles on a lattice with dimensions `Lvec`.
"""
function fullSlaterMomBasis(Lvec, n_up::Integer, n_dn::Integer)
    Lvec = collect(Int, Lvec)
    N = prod(Lvec)
    ndim = length(Lvec)
    dims = Tuple(Lvec)
    coords = getLatticeCoord(Lvec)

    up = sectorTotMom(Lvec, n_up, coords, true)
    dn = sectorTotMom(Lvec, n_dn, coords, true)

    qu = Vector{Int}()
    for (q, cnt) in zip(up["qtot_unique"], up["counts"])
        append!(qu, fill(q, cnt))
    end
    qd = Vector{Int}()
    for (q, cnt) in zip(dn["qtot_unique"], dn["counts"])
        append!(qd, fill(q, cnt))
    end

    qtot_up, qtot_dn = cartesian_prod(qu, qd)
    ints_up, ints_dn = cartesian_prod(up["ints"], dn["ints"])

    ints = combineSpinInts(ints_up, ints_dn, N)

    addmom = Matrix{Int}(undef, N, N)
    for qu_idx in 0:N-1
        for qd_idx in 0:N-1
            cu = unravel_c(qu_idx, dims)
            cd = unravel_c(qd_idx, dims)
            ctot = mod.(cu .+ cd, Lvec)
            addmom[qu_idx+1, qd_idx+1] = ravel_c(ctot, dims)
        end
    end

    qtot = [addmom[qtot_up[i]+1, qtot_dn[i]+1] for i in eachindex(qtot_up)]

    order = sortperm(qtot)
    qtot_unique, counts = unique_in_sorted(qtot[order], true)

    return Dict{String,Any}(
        "ints" => ints[order],
        "qtot_up" => qtot_up[order],
        "qtot_dn" => qtot_dn[order],
        "qtot" => qtot[order],
        "qtot_unique" => qtot_unique,
        "counts" => counts,
        "sortOrder" => order
    )
end


# ═══════════════════════════════════════════════════════════════════════
# FERMIONIC HAMILTONIANS
# ═══════════════════════════════════════════════════════════════════════

"""
    findForwardLatticeNeighbors(Lvec; use_pbc=true) → Matrix{Int}

Find forward nearest neighbors on an nD lattice (C-order site numbering).
Returns `(ndim × N)` matrix of **1-based** neighbor site indices.
"""
function findForwardLatticeNeighbors(Lvec; use_pbc=true)
    Lvec = collect(Int, Lvec)
    N = prod(Lvec)
    ndim = length(Lvec)
    # C-order site array with 1-based site indices
    sites = reshape_c(collect(1:N), Tuple(Lvec))

    if use_pbc
        neighbors = Matrix{Int}(undef, ndim, N)
        for a in 1:ndim
            shift = ntuple(d -> d == a ? -1 : 0, ndim)
            neighbors[a, :] = flatten_c(circshift(sites, shift))
        end
        return neighbors
    else
        # Open boundary: mark missing neighbors with 0
        neighbors = zeros(Int, ndim, N)
        for a in 1:ndim
            shift = ntuple(d -> d == a ? -1 : 0, ndim)
            shifted = circshift(sites, shift)
            # Identify which sites are at the boundary (last position along axis a)
            for ci in CartesianIndices(size(sites))
                if ci[a] < size(sites, a)
                    flat_src = ravel_c(Tuple(ci[d] - 1 for d in 1:ndim), Tuple(Lvec)) + 1
                    flat_dst = ravel_c(Tuple(
                            d == a ? ci[d] : ci[d] - 1 for d in 1:ndim), Tuple(Lvec)) + 1
                    neighbors[a, flat_src] = flat_dst
                end
            end
        end
        return neighbors
    end
end

"""
    findLatticeEdges(Lvec; use_pbc=true) → Vector{Tuple{Int,Int}}

Find all nearest-neighbor edges on an nD lattice.
Returns pairs of **1-based** site indices.
"""
function findLatticeEdges(Lvec; use_pbc=true)
    neighbors = findForwardLatticeNeighbors(Lvec; use_pbc=use_pbc)
    N = prod(Lvec)
    ndim = length(Lvec)
    edges = Tuple{Int,Int}[]
    for a in 1:ndim
        for site in 1:N
            nb = neighbors[a, site]
            nb == 0 && continue   # OBC missing neighbor
            push!(edges, (site, nb))
        end
    end
    return edges
end

"""
    fermionNNHopping(basis, edges, t) → SparseMatrixCSC

Build the nearest-neighbor hopping Hamiltonian for a single spin channel.

`basis`: vector of integer bit-strings.
`edges`: vector of `(i, j)` pairs (1-based site indices).
`t`:     hopping amplitude.

Returns a sparse Hermitian matrix of size `(d × d)`.
"""
function fermionNNHopping(basis::AbstractVector{<:Integer}, edges, t::Real)
    d = length(basis)
    basis32 = DtMb.(basis)

    # Build index lookup
    order = sortperm(basis32)
    sorted_basis = basis32[order]

    rows = Int[]
    cols = Int[]
    vals = ComplexF64[]

    for (i, j) in edges
        # bit positions (0-based) for 1-based sites
        bp_i = i - 1
        bp_j = j - 1
        mask_i = DtMb(1) << DtMb(bp_i)
        mask_j = DtMb(1) << DtMb(bp_j)

        # Mid mask: bits strictly between bp_i and bp_j
        lo, hi = min(bp_i, bp_j), max(bp_i, bp_j)
        if hi > lo + 1
            mid_mask = ((DtMb(1) << DtMb(hi)) - DtMb(1)) ⊻
                       ((DtMb(1) << DtMb(lo + 1)) - DtMb(1))
        else
            mid_mask = DtMb(0)
        end

        for k in 1:d
            s = basis32[k]
            # c†_i c_j : annihilate j, create i
            has_j = (s & mask_j) != DtMb(0)
            empty_i = (s & mask_i) == DtMb(0)
            if has_j && empty_i
                new_s = (s ⊻ mask_j) | mask_i
                # JW sign from spectators between i and j
                n_between = count_ones(s & mid_mask)
                sgn = iseven(n_between) ? 1 : -1

                # Find new state in basis
                idx_new = searchsortedfirst(sorted_basis, new_s)
                if idx_new <= d && sorted_basis[idx_new] == new_s
                    push!(rows, order[idx_new])
                    push!(cols, k)
                    push!(vals, -t * sgn)
                end
            end

            # c†_j c_i : annihilate i, create j (Hermitian conjugate)
            has_i = (s & mask_i) != DtMb(0)
            empty_j = (s & mask_j) == DtMb(0)
            if has_i && empty_j
                new_s = (s ⊻ mask_i) | mask_j
                n_between = count_ones(s & mid_mask)
                sgn = iseven(n_between) ? 1 : -1

                idx_new = searchsortedfirst(sorted_basis, new_s)
                if idx_new <= d && sorted_basis[idx_new] == new_s
                    push!(rows, order[idx_new])
                    push!(cols, k)
                    push!(vals, -t * sgn)
                end
            end
        end
    end

    return sparse(rows, cols, vals, d, d)
end

"""
    fermionOnSiteSpinDensity(ubasis, dbasis; u=1.0) → SparseMatrixCSC

On-site Hubbard interaction: u Σ_i n_{i↑} n_{i↓}.

`ubasis`, `dbasis`: integer basis vectors for up and down channels.
Returns a sparse diagonal matrix in the Kronecker product basis
`|up⟩ ⊗ |dn⟩`.
"""
function fermionOnSiteSpinDensity(ubasis, dbasis; u=1.0)
    d_up = length(ubasis)
    d_dn = length(dbasis)
    d = d_up * d_dn

    diag_vals = Vector{Float64}(undef, d)
    for iu in 1:d_up
        for id in 1:d_dn
            overlap = count_ones(DtMb(ubasis[iu]) & DtMb(dbasis[id]))
            flat_idx = (iu - 1) * d_dn + id
            diag_vals[flat_idx] = u * overlap
        end
    end
    return spdiagm(0 => diag_vals)
end

"""
    HubbardRealSpace(t::Real, u::Real, Lvec, nvec; use_pbc::Bool=true, returnBasis::Bool=true)

Build the Hubbard Hamiltonian in the real-space (position) basis.
"""
function HubbardRealSpace(t::Real, u::Real, Lvec, nvec;
    use_pbc::Bool=true,
    returnBasis::Bool=true)
    Lvec = collect(Int, Lvec)
    N = prod(Lvec)
    n_up, n_dn = nvec

    edges = findLatticeEdges(Lvec; use_pbc=use_pbc)
    basis_up, uocc = getReducedHilSpace(N, n_up; returnOcc=true)
    basis_dn, docc = getReducedHilSpace(N, n_dn; returnOcc=true)

    hop_up = fermionNNHopping(basis_up, edges, t)
    hop_dn = fermionNNHopping(basis_dn, edges, t)

    d_up = length(basis_up)
    d_dn = length(basis_dn)

    I_up = sparse(I, d_up, d_up)
    I_dn = sparse(I, d_dn, d_dn)
    H_hop = kron(I_up, hop_dn) + kron(hop_up, I_dn)

    H_int = fermionOnSiteSpinDensity(basis_up, basis_dn; u=u)

    H = H_hop + H_int

    if returnBasis
        return H, (basis_up, basis_dn), (uocc, docc)
    else
        return H
    end
end


# ═══════════════════════════════════════════════════════════════════════
# MOMENTUM AND ORBIT BASIS HAMILTONIAN CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════

"""
    sub_mom(k_idx::Integer, q_idx::Integer, Lvec) → 1D linear index

Compute the momentum index for k - q modulo Lvec.
"""
function sub_mom(k_idx::Integer, q_idx::Integer, Lvec)
    dims = Tuple(Lvec)
    ck = unravel_c(k_idx, dims)
    cq = unravel_c(q_idx, dims)
    return ravel_c(mod.(ck .- cq, Lvec), dims)
end

"""
    add_mom(k_idx::Integer, q_idx::Integer, Lvec) → 1D linear index

Compute the momentum index for k + q modulo Lvec.
"""
function add_mom(k_idx::Integer, q_idx::Integer, Lvec)
    dims = Tuple(Lvec)
    ck = unravel_c(k_idx, dims)
    cq = unravel_c(q_idx, dims)
    return ravel_c(mod.(ck .+ cq, Lvec), dims)
end

"""
    fermionBilinear(basis::AbstractVector{<:Integer}, terms, N::Integer) → SparseMatrixCSC

Build matrix representation of quadratic fermion operator:
    ∑ val * c_i^† c_j
with correct Jordan-Wigner signs under the given `basis`.
"""
function fermionBilinear(basis::AbstractVector{<:Integer}, terms, N::Integer)
    d = length(basis)
    basis32 = DtMb.(basis)
    order = sortperm(basis32)
    sorted_basis = basis32[order]

    rows = Int[]
    cols = Int[]
    vals = ComplexF64[]

    for (i, j, val) in terms
        if abs(val) < 1e-15
            continue
        end
        bp_i = i - 1
        bp_j = j - 1
        mask_i = DtMb(1) << DtMb(bp_i)
        mask_j = DtMb(1) << DtMb(bp_j)

        if i == j
            for k in 1:d
                s = basis32[k]
                if (s & mask_i) != DtMb(0)
                    push!(rows, k)
                    push!(cols, k)
                    push!(vals, val)
                end
            end
        else
            lo, hi = min(bp_i, bp_j), max(bp_i, bp_j)
            if hi > lo + 1
                mid_mask = ((DtMb(1) << DtMb(hi)) - DtMb(1)) ⊻ ((DtMb(1) << DtMb(lo + 1)) - DtMb(1))
            else
                mid_mask = DtMb(0)
            end

            for k in 1:d
                s = basis32[k]
                if (s & mask_j) != DtMb(0) && (s & mask_i) == DtMb(0)
                    new_s = (s ⊻ mask_j) | mask_i
                    sgn = iseven(count_ones(s & mid_mask)) ? 1.0 : -1.0
                    idx_new = searchsortedfirst(sorted_basis, new_s)
                    if idx_new <= d && sorted_basis[idx_new] == new_s
                        push!(rows, order[idx_new])
                        push!(cols, k)
                        push!(vals, val * sgn)
                    end
                end
            end
        end
    end
    return sparse(rows, cols, vals, d, d)
end

"""
    HubbardOrbitBasis(t::Real, u::Real, Lvec, nvec; q_target::Union{Integer, Nothing}=nothing, returnBasis::Bool=true)

Build the Hubbard Hamiltonian in the orbit momentum basis.
"""
function HubbardOrbitBasis(t::Real, u::Real, Lvec, nvec;
    q_target::Union{Integer,Nothing}=nothing,
    returnBasis::Bool=true)
    Lvec = collect(Int, Lvec)
    n_up, n_dn = nvec
    ndim = length(Lvec)
    dims = Tuple(Lvec)

    # 1. Build the real space Hamiltonian and extract single-spin bases
    H_real, (basis_up, basis_dn), _ = HubbardRealSpace(t, u, Lvec, nvec; use_pbc=true, returnBasis=true)

    # 2. Build orbit basis change matrices
    P_up, bSizes_up = translnDCOB(Lvec, "fermion", basis_up)
    P_dn, bSizes_dn = translnDCOB(Lvec, "fermion", basis_dn)

    # 3. Combine bases using Kronecker product
    P_total = kron(P_up, P_dn)

    # 4. Reorder by total momentum
    P_reordered, counts = reorderByTotMom_orbit(P_total, bSizes_up, bSizes_dn, Lvec)

    # 5. Filter by q_target if provided
    if q_target !== nothing
        qvecs = collect(keys(bSizes_up))
        qu = Vector{Int}[]
        for q in qvecs
            for _ in 1:bSizes_up[q]
                push!(qu, q)
            end
        end
        qd = Vector{Int}[]
        for q in qvecs
            for _ in 1:bSizes_dn[q]
                push!(qd, q)
            end
        end
        qt = Matrix{Int}(undef, length(qu) * length(qd), ndim)
        idx = 1
        for i in eachindex(qu)
            for j in eachindex(qd)
                qt[idx, :] .= mod.(qu[i] .+ qd[j], Lvec)
                idx += 1
            end
        end
        qt_idx = [ravel_c(Tuple(qt[k, :]), dims) for k in 1:size(qt, 1)]
        perm = sortperm(qt_idx)
        qt_sorted = qt_idx[perm]

        sub_indices = findall(v -> v == q_target, qt_sorted)
        P_reordered = P_reordered[:, sub_indices]
        counts = [length(sub_indices)]
    end

    H_orbit = P_reordered' * H_real * P_reordered

    if returnBasis
        return H_orbit, P_reordered, counts
    else
        return H_orbit
    end
end

"""
    HubbardMomentumBasis(t::Real, u::Real, Lvec, nvec; q_target::Union{Integer, Nothing}=nothing, basis_sector::Union{AbstractVector{<:Integer}, Nothing}=nothing, indexer=nothing, returnBasis::Bool=true)

Build the Hubbard Hamiltonian in the Slater momentum basis. If `indexer` or `basis_sector` is provided,
the returned Hamiltonian matrix and basis dictionary are permuted to match the specified basis sector order.
"""
function HubbardMomentumBasis(t::Real, u::Real, Lvec, nvec;
    q_target::Union{Integer,Nothing}=nothing,
    basis_sector::Union{AbstractVector{<:Integer},Nothing}=nothing,
    indexer=nothing,
    returnBasis::Bool=true)
    Lvec = collect(Int, Lvec)
    N = prod(Lvec)
    n_up, n_dn = nvec
    dims = Tuple(Lvec)

    # if indexer is supplied use it to figure out what the momentum subspace is.
    if indexer !== nothing
        if basis_sector === nothing
            basis_sector = get_basis_sector(indexer, Lvec, N)
        end
        if q_target === nothing && !isnothing(indexer.k) && !isnothing(indexer.lattice_dims)
            q_target = ravel_c(Tuple(k - 1 for k in indexer.k), Tuple(Lvec))
        end
    end

    # 1. Get Slater momentum basis & total momentum info
    basis_dict = fullSlaterMomBasis(Lvec, n_up, n_dn)

    # 2. Single-spin bases from sectorTotMom
    coords = getLatticeCoord(Lvec)
    up = sectorTotMom(Lvec, n_up, coords, true)
    dn = sectorTotMom(Lvec, n_dn, coords, true)
    basis_up = up["ints"]
    basis_dn = dn["ints"]
    d_up = length(basis_up)
    d_dn = length(basis_dn)

    # 3. Build diagonal hopping in momentum space using analytical dispersion (PBC nearest-neighbor)
    eps = zeros(Float64, N)
    for k in 0:N-1
        k_coords = unravel_c(k, dims)
        eps[k+1] = -2t * sum(cos(2π * k_coords[a] / Lvec[a]) for a in eachindex(Lvec))
    end

    # 4. Filter the basis dictionary according to q_target (subspace vs full space)
    if q_target !== nothing
        subspace_indices = findall(v -> v == q_target, basis_dict["qtot"])
        basis_dict = Dict{String,Any}(
            "ints" => basis_dict["ints"][subspace_indices],
            "qtot_up" => basis_dict["qtot_up"][subspace_indices],
            "qtot_dn" => basis_dict["qtot_dn"][subspace_indices],
            "qtot" => basis_dict["qtot"][subspace_indices],
            "qtot_unique" => [q_target],
            "counts" => [length(subspace_indices)],
            "sortOrder" => basis_dict["sortOrder"][subspace_indices]
        )
    else
        subspace_indices = 1:length(basis_dict["qtot"])
    end

    d_sub = length(subspace_indices)
    subspace_ints = basis_dict["ints"]
    subspace_order = basis_dict["sortOrder"]
    counts = basis_dict["counts"]

    # 5. Build diagonal hopping directly on the subspace/full space
    hop_diag = zeros(ComplexF64, d_sub)
    for i in 1:d_sub
        s = subspace_ints[i]
        s_up = s & ((one(s) << N) - one(s))
        s_dn = s >> N
        for k in 0:N-1
            if ((s_up >> k) & 1) == 1
                hop_diag[i] += eps[k+1]
            end
            if ((s_dn >> k) & 1) == 1
                hop_diag[i] += eps[k+1]
            end
        end
    end
    H_hop = spdiagm(hop_diag)

    # 6. Build interaction directly on the subspace/full space
    inv_orig = zeros(Int, d_up * d_dn)
    for i in 1:d_sub
        inv_orig[subspace_order[i]] = i
    end

    sub_r_up = [((idx - 1) ÷ d_dn) + 1 for idx in subspace_order]
    sub_r_dn = [((idx - 1) % d_dn) + 1 for idx in subspace_order]

    rows = Int[]
    cols = Int[]
    vals = ComplexF64[]

    for q in 0:N-1
        # rho_up(q) = sum_k c_k^\dagger c_{k-q}
        terms_up = [(k + 1, sub_mom(k, q, Lvec) + 1, 1.0) for k in 0:N-1]
        rho_up = fermionBilinear(basis_up, terms_up, N)

        # rho_dn(-q) = sum_k c_k^\dagger c_{k+q}
        terms_dn = [(k + 1, add_mom(k, q, Lvec) + 1, 1.0) for k in 0:N-1]
        rho_dn = fermionBilinear(basis_dn, terms_dn, N)

        # Iterate over each state j in the subspace
        for j in 1:d_sub
            c_up = sub_r_up[j]
            c_dn = sub_r_dn[j]

            # Non-zeros in column c_up of rho_up and column c_dn of rho_dn
            start_up = rho_up.colptr[c_up]
            end_up = rho_up.colptr[c_up+1] - 1
            start_dn = rho_dn.colptr[c_dn]
            end_dn = rho_dn.colptr[c_dn+1] - 1

            for idx_up in start_up:end_up
                r_up = rho_up.rowval[idx_up]
                v_up = rho_up.nzval[idx_up]

                for idx_dn in start_dn:end_dn
                    r_dn = rho_dn.rowval[idx_dn]
                    v_dn = rho_dn.nzval[idx_dn]

                    row_full = (r_up - 1) * d_dn + r_dn
                    row_sub = inv_orig[row_full]

                    if row_sub > 0
                        push!(rows, row_sub)
                        push!(cols, j)
                        push!(vals, v_up * v_dn)
                    end
                end
            end
        end
    end

    H_int = sparse(rows, cols, vals, d_sub, d_sub)
    H_int *= (u / N)

    H_mom_sorted = H_hop + H_int

    if basis_sector !== nothing
        state_to_idx = Dict(val => idx for (idx, val) in enumerate(basis_dict["ints"]))
        perm = [state_to_idx[val] for val in basis_sector]
        H_mom_sorted = H_mom_sorted[perm, perm]
        basis_dict["ints"] = basis_dict["ints"][perm]
        for key in ("qtot_up", "qtot_dn", "qtot", "sortOrder")
            if haskey(basis_dict, key)
                basis_dict[key] = basis_dict[key][perm]
            end
        end
    end

    if returnBasis
        return H_mom_sorted, basis_dict, counts
    else
        return H_mom_sorted
    end
end

"""
    Hubbard(t::Real, u::Real, Lvec, nvec, basis_type::Symbol=:real; use_pbc::Bool=true, returnBasis::Bool=true, q_target::Union{Integer, Nothing}=nothing, basis_sector::Union{AbstractVector{<:Integer}, Nothing}=nothing, indexer=nothing)

Build the Hubbard Hamiltonian in the specified basis:
- `:real`: real-space (position) basis.
- `:orbit`: orbit momentum basis (using translnDCOB and reorderByTotMom_orbit).
- `:momentum`: momentum basis (using fullSlaterMomBasis and momentum conservation).

Returns:
- For `:real`: `(H, (basis_up, basis_dn), (uocc, docc))` if `returnBasis=true`, else `H`.
- For `:orbit`: `(H_orbit, P_reordered, counts)` if `returnBasis=true`, else `H_orbit`.
- For `:momentum`: `(H_mom, basis_dict, counts)` if `returnBasis=true`, else `H_mom`.
"""
function Hubbard(t::Real, u::Real, Lvec, nvec, basis_type::Symbol=:real;
    use_pbc::Bool=true,
    returnBasis::Bool=true,
    q_target::Union{Integer,Nothing}=nothing,
    basis_sector::Union{AbstractVector{<:Integer},Nothing}=nothing,
    indexer=nothing)
    if basis_type == :real
        if q_target !== nothing || basis_sector !== nothing || indexer !== nothing
            throw(ArgumentError("q_target, basis_sector, and indexer are not supported for real-space basis"))
        end
        return HubbardRealSpace(t, u, Lvec, nvec; use_pbc=use_pbc, returnBasis=returnBasis)
    elseif basis_type == :orbit
        @assert use_pbc == true
        return HubbardOrbitBasis(t, u, Lvec, nvec; q_target=q_target, returnBasis=returnBasis)
    elseif basis_type == :momentum
        @assert use_pbc == true
        return HubbardMomentumBasis(t, u, Lvec, nvec; q_target=q_target, basis_sector=basis_sector, indexer=indexer, returnBasis=returnBasis)
    else
        throw(ArgumentError("Unknown basis_type: $basis_type. Expected :real, :orbit, or :momentum"))
    end
end

# Backward compatibility method
function Hubbard(t::Real, u::Real, Lvec, nvec, use_pbc::Bool, returnBasis::Bool=true)
    return Hubbard(t, u, Lvec, nvec, :real; use_pbc=use_pbc, returnBasis=returnBasis)
end




## Exact exponential compatibility ##

# Helper to convert key to canonical
function key_to_canonical(k)
    [(get_clean_coords(c), spin, op) for (c, spin, op) in k]
end
function conjugate_canonical(ck)
    conj_ops = [(c, spin, op == :create ? :annihilate : :create) for (c, spin, op) in ck]
    cre = sort(filter(op -> op[3] == :create, conj_ops), by=op -> (op[1], op[2]))
    ann = sort(filter(op -> op[3] == :annihilate, conj_ops), by=op -> (op[1], op[2]))
    return [cre; ann]
end

"""
    get_clean_coords(c) -> Tuple

Robustly extract the flat coordinates tuple from a Lattices.Coordinate object,
supporting both in-memory and JLD2-reconstructed structures.
"""
function get_clean_coords(c)
    coords = c.coordinates
    if !isempty(coords) && coords[1] isa Tuple
        return coords[1]
    else
        return coords
    end
end

function coord_to_site_idx(coord, Lvec)
    c0 = get_clean_coords(coord) .- 1
    return ravel_c(c0, Tuple(Lvec))
end

function coord_set_to_binary(coord_set, Lvec)
    val = zero(UInt)
    for coord in coord_set
        site_idx = coord_to_site_idx(coord, Lvec)
        val |= (one(UInt) << site_idx)
    end
    return val
end

function conf_to_int(conf, Lvec, N_sites)
    u_bin = coord_set_to_binary(conf[1], Lvec)
    d_bin = coord_set_to_binary(conf[2], Lvec)
    return combineSpinInts(u_bin, d_bin, N_sites)
end
"""
    get_basis_sector(indexer, Lvec, N_sites)

Reconstruct the basis state integers.
"""
function get_basis_sector(indexer, Lvec, N_sites)
    basis_sector = Vector{UInt}(undef, length(indexer.inv_comb_dict))
    for (idx, conf) in enumerate(indexer.inv_comb_dict)
        basis_sector[idx] = conf_to_int(conf, Lvec, N_sites)
    end
    return basis_sector
end


end # module TamFermion