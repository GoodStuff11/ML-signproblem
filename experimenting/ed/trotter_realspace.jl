#=
trotter_realspace.jl

Real-space basis support for the HVA ansatz.

The ED data this repo ships is stored in a *fixed-total-momentum* sector: the
basis integers returned by `get_basis_sector` label occupied **momentum modes**,
and the eigenvectors live in the `d_sector`-dimensional subspace of one total
momentum `q`. That is exactly the right home for `enumerate_ferm_excitations`
with `conserve_mom=true`, whose gates keep you inside the sector.

The HVA gate set (`enumerate_ferm_excitations_HVA`) is a *real-space* ansatz:
nearest-neighbour hopping `c†_{iσ}c_{jσ} + h.c.` and on-site `n_{i↑}n_{i↓}`.
Real-space hopping does not conserve total momentum, so those gates cannot be
restricted to a momentum sector — projecting them there would silently gut the
generators. Running the HVA therefore means moving to the *full* particle-number
sector in the real-space occupation basis, of dimension `d_up * d_dn`.

This file does that move:

- [`realspace_basis`](@ref) builds the combined real-space basis in the same
  ordering `HubbardRealSpace` uses for its matrix (up index slow, down index
  fast), which is the ordering `fgateToTauSector` and the loss functions expect.
- [`momentum_sector_to_realspace`](@ref) lifts momentum-sector state vectors into
  that basis. The change of basis factorizes over spin under the `:spin_first`
  sign convention (all up operators ordered before all down operators), so it is
  `kron(F_up, F_dn)` with `F_n = SlaterCOB_RtoK_nparticle(Lvec, n)` the
  n-particle Slater determinant Fourier matrix, applied as a matrix triple
  product rather than an explicit Kronecker product.
- [`check_realspace_transform`](@ref) verifies the result against physics rather
  than against the derivation: norms must be preserved and the Hubbard energy
  computed in the real-space basis must agree with the one computed in the
  momentum basis. A convention mismatch anywhere in the chain shows up here.
=#

"""
    realspace_basis(Lvec, nvec) → (basis, basis_up, basis_dn)

Combined real-space occupation basis for `nvec = (n_up, n_dn)` particles on a
lattice of dimensions `Lvec`, as 2N-bit integers `s_up | (s_dn << N)`.

The ordering is up-slow / down-fast, i.e. element `(iu - 1) * d_dn + id`, which
is the ordering of `HubbardRealSpace`'s Hamiltonian matrix
(`kron(I_up, hop_dn) + kron(hop_up, I_dn)`).
"""
function realspace_basis(Lvec, nvec)
    N = prod(Lvec)
    n_up, n_dn = nvec
    basis_up = TamFermion.getReducedHilSpace(N, n_up; returnOcc=false)
    basis_dn = TamFermion.getReducedHilSpace(N, n_dn; returnOcc=false)
    basis = [TamFermion.combineSpinInts(basis_up[iu], basis_dn[id], N)
             for iu in eachindex(basis_up) for id in eachindex(basis_dn)]
    return basis, basis_up, basis_dn
end

"""
    _embed_sector(v, basis_sector, index_up, index_dn, d_dn, N) → Vector{ComplexF64}

Scatter a momentum-sector vector into the full `d_up * d_dn` momentum-occupation
vector, zero outside the sector.
"""
function _embed_sector(v::AbstractVector, basis_sector::AbstractVector,
    index_up::Dict, index_dn::Dict, d_up::Int, d_dn::Int, N::Int)
    length(v) == length(basis_sector) || throw(DimensionMismatch(
        "state vector has length $(length(v)) but the momentum sector basis has " *
        "length $(length(basis_sector))"))
    full = zeros(ComplexF64, d_up * d_dn)
    mask = (one(UInt) << N) - one(UInt)
    for (k, s) in enumerate(basis_sector)
        su = UInt(s) & mask
        sd = UInt(s) >> N
        iu = get(index_up, su, 0)
        id = get(index_dn, sd, 0)
        (iu == 0 || id == 0) && throw(ArgumentError(
            "basis_sector entry $k does not decompose into a valid ($(iu), $(id)) " *
            "pair of fixed-particle-number spin configurations"))
        full[(iu-1)*d_dn+id] = v[k]
    end
    return full
end

"""
    momentum_sector_to_realspace(vecs, basis_sector, Lvec, nvec) → Matrix{ComplexF64}

Transform momentum-sector state vectors into the full real-space occupation basis
returned by [`realspace_basis`](@ref).

`vecs` is either a matrix whose **rows** are states (the layout `load_ED_data`
returns) or a vector of state vectors. The result is always a matrix whose rows
are the corresponding real-space states, of width `d_up * d_dn`.

The transform is `ψ_mom = kron(F_up, F_dn) ψ_real` with
`F_n = SlaterCOB_RtoK_nparticle(Lvec, n)[1]`, inverted using unitarity of `F`.
Writing the full vector as the matrix `M[id, iu]` (column-major, which is exactly
the up-slow/down-fast ordering), `kron(A, B) * vec(M) == vec(B * M * transpose(A))`,
so `M_real = F_dn' * M_mom * conj(F_up)`.
"""
function momentum_sector_to_realspace(vecs, basis_sector::AbstractVector, Lvec, nvec)
    N = prod(Lvec)
    n_up, n_dn = nvec

    _, basis_up, basis_dn = realspace_basis(Lvec, nvec)
    d_up = length(basis_up)
    d_dn = length(basis_dn)

    index_up = Dict(UInt(basis_up[i]) => i for i in eachindex(basis_up))
    index_dn = Dict(UInt(basis_dn[i]) => i for i in eachindex(basis_dn))

    F_up, _ = TamFermion.SlaterCOB_RtoK_nparticle(Lvec, n_up)
    F_dn, _ = TamFermion.SlaterCOB_RtoK_nparticle(Lvec, n_dn)
    # F is unitary, so F⁻¹ = F' and (transpose(F))⁻¹ = conj(F).
    Fup_inv_t = conj(F_up)
    Fdn_inv = adjoint(F_dn)

    rows = if vecs isa AbstractMatrix
        [view(vecs, i, :) for i in 1:size(vecs, 1)]
    else
        collect(vecs)
    end

    out = Matrix{ComplexF64}(undef, length(rows), d_up * d_dn)
    for (i, v) in enumerate(rows)
        full = _embed_sector(v, basis_sector, index_up, index_dn, d_up, d_dn, N)
        M_mom = reshape(full, d_dn, d_up)
        M_real = Fdn_inv * M_mom * Fup_inv_t
        out[i, :] = vec(M_real)
    end
    return out
end

"""
    apply_spin_exchange(v, basis, N, nvec) → Vector

Apply the spin-exchange operator `P` (swap the ↑ and ↓ occupation patterns) to a
state in the combined real-space basis `basis`. Under the `:spin_first` ordering
convention (all ↑ operators before all ↓ operators) reordering the two blocks
costs a global `(-1)^(n_up * n_dn)`.

Requires `n_up == n_dn`, since otherwise `P` leaves the particle-number sector.
"""
function apply_spin_exchange(v::AbstractVector, basis::AbstractVector, N::Int, nvec)
    n_up, n_dn = nvec
    n_up == n_dn || throw(ArgumentError(
        "spin exchange is only defined within a sector with n_up == n_dn; got $nvec"))
    index = Dict(UInt(basis[i]) => i for i in eachindex(basis))
    mask = (one(UInt) << N) - one(UInt)
    sign = iseven(n_up * n_dn) ? 1 : -1
    w = similar(v)
    for i in eachindex(basis)
        s = UInt(basis[i])
        su = s & mask
        sd = s >> N
        w[index[sd|(su<<N)]] = sign * v[i]
    end
    return w
end

"""
    spin_tied_fidelity_bound(ref, target, basis, N, nvec) → NamedTuple

Upper bound on the fidelity `|⟨target|U|ref⟩|²` reachable by any circuit `U` that
**commutes with spin exchange** — which is exactly what tying a bond's ↑ and ↓
gates to one coefficient produces (`tie=:full` and `tie=:spin` in
[`enumerate_ferm_excitations_HVA`](@ref)).

If `target` is an eigenstate of `P` with eigenvalue `λ = ±1`, then for any
`P`-commuting `U` the reachable fidelity is capped by the weight of `ref` in the
`λ` eigenspace, `‖(1 + λP)/2 · ref‖² = (1 + λ⟨ref|P ref⟩) / 2`.

This bites in practice: the Slater reference state this repo uses fills a
*degenerate* single-particle level, and the ↑ and ↓ channels can end up on
different orbitals, so `⟨ref|P ref⟩ = 0` and a spin-tied ansatz is capped at
fidelity 0.5 no matter how many layers it is given. `tie=:none` (independent ↑/↓
coefficients) does not commute with `P` and is not capped.

Returns `(; bound, ref_parity, target_parity, target_is_eigenstate)`. When
`target` is not a `P` eigenstate the bound is reported as `NaN`.
"""
function spin_tied_fidelity_bound(ref::AbstractVector, target::AbstractVector,
    basis::AbstractVector, N::Int, nvec)
    n_up, n_dn = nvec
    n_up == n_dn || return (bound=NaN, ref_parity=NaN, target_parity=NaN,
        target_is_eigenstate=false)

    P_ref = apply_spin_exchange(ref, basis, N, nvec)
    P_tgt = apply_spin_exchange(target, basis, N, nvec)

    ref_parity = real(dot(ref, P_ref)) / max(norm(ref)^2, eps())
    target_parity = real(dot(target, P_tgt)) / max(norm(target)^2, eps())

    is_eig = isapprox(abs(target_parity), 1.0; atol=1e-6)
    lambda = target_parity >= 0 ? 1.0 : -1.0
    bound = is_eig ? (1 + lambda * ref_parity) / 2 : NaN

    return (bound=bound, ref_parity=ref_parity, target_parity=target_parity,
        target_is_eigenstate=is_eig)
end

"""
    check_realspace_transform(v_mom, v_real, H_mom, H_real; atol=1e-8, label="")

Verify one transformed state against physics: the norm must be preserved and the
Hamiltonian expectation value must agree between the two bases. Throws
`ErrorException` on failure, since a silent convention mismatch here would
produce a plausible-looking but wrong optimization target.

Returns `(; norm_mom, norm_real, energy_mom, energy_real)`.
"""
function check_realspace_transform(v_mom::AbstractVector, v_real::AbstractVector,
    H_mom, H_real; atol::Float64=1e-8, label::String="")

    n_mom = norm(v_mom)
    n_real = norm(v_real)
    e_mom = real(dot(v_mom, H_mom * v_mom)) / max(n_mom^2, eps())
    e_real = real(dot(v_real, H_real * v_real)) / max(n_real^2, eps())

    tag = isempty(label) ? "" : " ($label)"

    if !isapprox(n_mom, n_real; atol=atol, rtol=1e-8)
        error("real-space transform$tag did not preserve the norm: " *
              "momentum basis $n_mom vs real-space basis $n_real")
    end
    if !isapprox(e_mom, e_real; atol=max(atol, 1e-6 * max(abs(e_mom), 1.0)), rtol=1e-6)
        error("real-space transform$tag disagrees on the Hubbard energy: " *
              "momentum basis $e_mom vs real-space basis $e_real. The momentum-sector " *
              "sign/ordering convention does not match the real-space Slater basis.")
    end
    return (norm_mom=n_mom, norm_real=n_real, energy_mom=e_mom, energy_real=e_real)
end
