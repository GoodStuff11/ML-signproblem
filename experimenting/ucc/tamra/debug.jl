include("TamFermion.jl")
import .TamFermion
using LinearAlgebra
using LinearMaps

const DtMb = UInt32
const FGate = TamFermion.FGate

# Re-implement correct excitation_operator_sector with order[j]
function correct_excitation_operator_sector(g::FGate, N::Integer, a::Real,
    basis::AbstractVector{<:Integer};
    sortOrder=nothing)
    d = length(basis)
    nbits = 2N
    M_I = UInt64(g.cre_up) | (UInt64(g.cre_dn) << N)
    M_J = UInt64(g.ann_up) | (UInt64(g.ann_dn) << N)
    M_IJ = M_I | M_J

    basis64 = UInt64.(basis)

    if sortOrder !== nothing
        order = sortOrder
    else
        order = sortperm(basis64)
    end
    sorted_basis = basis64[order]

    isrcJ = Int[]
    itgtI = Int[]

    for (i, s) in enumerate(basis64)
        if (s & M_J) == M_J && (s & M_I) == UInt64(0)
            t = s ⊻ M_IJ
            j = searchsortedfirst(sorted_basis, t)
            if j <= d && sorted_basis[j] == t
                push!(isrcJ, i)
                push!(itgtI, order[j]) # FIX HERE: order[j] instead of inv_order[j]
            end
        end
    end

    sgn_ref = TamFermion._jw_sign_ref(M_I, M_J, nbits)
    mid_mask = TamFermion._odd_spectator_mask(M_IJ, nbits)

    signs = ones(Int, length(isrcJ))
    for k in eachindex(isrcJ)
        s = basis64[isrcJ[k]]
        spec_parity = count_ones(s & mid_mask) & 1
        signs[k] = sgn_ref * (spec_parity == 1 ? -1 : 1)
    end

    ca = cos(a)
    coef = 1im * sin(a) .* signs

    function _apply!(w, v, off_signs)
        w .= v
        if !isempty(isrcJ)
            for k in eachindex(isrcJ)
                si, ti = isrcJ[k], itgtI[k]
                vs, vt = v[si], v[ti]
                w[si] = ca * vs + off_signs[k] * vt
                w[ti] = off_signs[k] * vs + ca * vt
            end
        end
        return w
    end

    return LinearMap{ComplexF64}(
        v -> _apply!(similar(v, ComplexF64), v, coef),
        v -> _apply!(similar(v, ComplexF64), v, .-coef),
        d, d
    )
end

# Test it
Lvec = (2, 3)
N = Int(prod(Lvec))
n_up, n_dn = (2, 2)
mb = TamFermion.fullSlaterMomBasis(Lvec, n_up, n_dn)

basis0 = mb["ints"][mb["qtot"] .== 0]

p = 2
gates = TamFermion.enumerate_ferm_excitations(p, Lvec; conserve_mom=true, conserve_sz=true)

sig = 1
using Random
Random.seed!(42)
A = randn(length(gates)) .* sig

order = sortperm(UInt64.(basis0))
ops = [correct_excitation_operator_sector(g, N, a, basis0; sortOrder=order)
       for (g, a) in zip(gates, A)]

d0 = length(basis0)
v0 = zeros(ComplexF64, d0); v0[1] = 1.0

v = copy(v0)
norm_ok = true
for (i, op) in enumerate(ops)
    global v, norm_ok
    v = op * v
    nrm = norm(v)
    if abs(nrm - 1.0) > 1e-12
        println("Operator $i failed to preserve norm! Norm = $nrm")
        norm_ok = false
        break
    end
end

if norm_ok
    println("All operators successfully preserved the norm!")
    println("Final norm: $(norm(v))")
end
