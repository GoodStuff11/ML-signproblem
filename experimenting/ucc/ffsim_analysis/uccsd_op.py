
from __future__ import annotations

import itertools
import dataclasses
from dataclasses import dataclass
from typing import cast

import numpy as np
import scipy.sparse.linalg
from ffsim import contract, gates, linalg, protocols
import ffsim

def _get_gso_indices(dim, norb, nelec):
    # Mapping from spatial (norb, nelec) to GSO (2*norb, (sum(nelec), 0))
    # We use strings as intermediate
    # ffsim returns strings as integers. For (norb, (na, nb)), bits 0..N-1 are alpha, N..2N-1 are beta
    # This matches GSO if we map Beta spatial orbitals to N..2N-1
    indices = np.arange(dim)
    strings = ffsim.addresses_to_strings(indices, norb, nelec)
    
    gso_norb = 2 * norb
    gso_nelec = (sum(nelec), 0)
    
    # Map to GSO addresses
    # strings is a list of integers, which is what strings_to_addresses expects
    gso_indices = ffsim.strings_to_addresses(strings, gso_norb, gso_nelec)
    return gso_indices

def uccsd_linear_operator(
    t1: tuple[np.ndarray, np.ndarray],
    t2: tuple[np.ndarray, np.ndarray, np.ndarray],
    norb: int,
    nelec: tuple[int, int],
) -> scipy.sparse.linalg.LinearOperator:
    """Return a linear operator for a UCCSD operator generator."""
    gso_norb = 2 * norb
    gso_nelec = (sum(nelec), 0)
    
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    na, va = t1a.shape
    nb, vb = t1b.shape
    
    # One body tensor (2N x 2N)
    obs = np.zeros((gso_norb, gso_norb), dtype=complex)
    
    # Helper to slicing
    ao = slice(0, na)
    av = slice(na, norb)
    bo = slice(norb, norb+nb)
    bv = slice(norb+nb, 2*norb)
    
    # T1a -> Alpha Occ to Alpha Virt
    obs[av, ao] = t1a.T
    obs[ao, av] = -t1a.T.conj().T
    
    # T1b -> Beta Occ to Beta Virt
    obs[bv, bo] = t1b.T
    obs[bo, bv] = -t1b.T.conj().T
    
    # Two body tensor
    tbs = np.zeros((gso_norb, gso_norb, gso_norb, gso_norb), dtype=complex)
    
    # T2aa: (i, j, a, b) -> dest: [av, ao, av, ao]
    tbs[av, ao, av, ao] = t2aa.transpose(2, 0, 3, 1)
    tbs[ao, av, ao, av] = -t2aa.transpose(0, 2, 1, 3).conj()
    
    # T2bb: dest: [bv, bo, bv, bo]
    tbs[bv, bo, bv, bo] = t2bb.transpose(2, 0, 3, 1)
    tbs[bo, bv, bo, bv] = -t2bb.transpose(0, 2, 1, 3).conj()
    
    # T2ab: dest: [av, ao, bv, bo]
    tbs[av, ao, bv, bo] = t2ab.transpose(2, 0, 3, 1)
    tbs[ao, av, bo, bv] = -t2ab.transpose(0, 2, 1, 3).conj()
    
    return contract.two_body_linop(tbs, norb=gso_norb, nelec=gso_nelec, one_body_tensor=obs)

@dataclass(frozen=True)
class UCCSDOp(protocols.SupportsApplyUnitary, protocols.SupportsApproximateEquality):
    """Unrestricted unitary coupled cluster, singles and doubles operator."""

    # t1: (t1a, t1b)
    # t1a shape: (nocc_a, nvirt_a)
    # t1b shape: (nocc_b, nvirt_b)
    t1: tuple[np.ndarray, np.ndarray]
    
    # t2: (t2aa, t2ab, t2bb)
    # t2aa shape: (nocc_a, nocc_a, nvirt_a, nvirt_a)
    # t2ab shape: (nocc_a, nocc_b, nvirt_a, nvirt_b)
    # t2bb shape: (nocc_b, nocc_b, nvirt_b, nvirt_b)
    t2: tuple[np.ndarray, np.ndarray, np.ndarray]
    
    final_orbital_rotation: np.ndarray | None = None
    validate: dataclasses.InitVar[bool] = True
    rtol: dataclasses.InitVar[float] = 1e-5
    atol: dataclasses.InitVar[float] = 1e-8

    def __post_init__(self, validate: bool, rtol: float, atol: float):
        if validate:
            t1a, t1b = self.t1
            t2aa, t2ab, t2bb = self.t2
            na, va = t1a.shape
            nb, vb = t1b.shape
            
            # Check dimensions match
            if t1b.shape != (nb, vb):
                raise ValueError(f"t1b shape mismatch: expected {(nb, vb)}, got {t1b.shape}")
                
            if t2aa.shape != (na, na, va, va):
                 raise ValueError(f"t2aa shape mismatch: expected {(na, na, va, va)}, got {t2aa.shape}")
            if t2ab.shape != (na, nb, va, vb):
                 raise ValueError(f"t2ab shape mismatch: expected {(na, nb, va, vb)}, got {t2ab.shape}")
            if t2bb.shape != (nb, nb, vb, vb):
                 raise ValueError(f"t2bb shape mismatch: expected {(nb, nb, vb, vb)}, got {t2bb.shape}")
            
            if self.final_orbital_rotation is not None:
                norb = na + va
                if self.final_orbital_rotation.shape != (norb, norb):
                    raise ValueError(f"final_orbital_rotation shape mismatch: expected {(norb, norb)}, got {self.final_orbital_rotation.shape}")
                if not linalg.is_unitary(self.final_orbital_rotation, rtol=rtol, atol=atol):
                    raise ValueError("Final orbital rotation was not unitary.")

    @property
    def norb(self):
        nocc_a, nvirt_a = self.t1[0].shape
        return nocc_a + nvirt_a

    @staticmethod
    def n_params(norb: int, nelec: tuple[int, int], *, with_final_orbital_rotation: bool = False) -> int:
        nocc_a, nocc_b = nelec
        nvirt_a = norb - nocc_a
        nvirt_b = norb - nocc_b
        
        # T1 params
        n_t1a = nocc_a * nvirt_a
        n_t1b = nocc_b * nvirt_b
        
        # T2 params
        # T2aa: unique pairs of indices (i<j, a<b) -> (Na choose 2) * (Va choose 2)
        n_t2aa = (nocc_a * (nocc_a - 1) // 2) * (nvirt_a * (nvirt_a - 1) // 2)
        
        # T2bb: (Nb choose 2) * (Vb choose 2)
        n_t2bb = (nocc_b * (nocc_b - 1) // 2) * (nvirt_b * (nvirt_b - 1) // 2)
        
        # T2ab: Na * Nb * Va * Vb
        n_t2ab = nocc_a * nocc_b * nvirt_a * nvirt_b
        
        n_amplitudes = n_t1a + n_t1b + n_t2aa + n_t2bb + n_t2ab
        
        if with_final_orbital_rotation:
             # Spatial orbital rotation parameters (anti-hermitian generator)
             # norb * (norb - 1) // 2
             return n_amplitudes + norb * (norb - 1) // 2
        
        return n_amplitudes

    @staticmethod
    def from_parameters(params: np.ndarray, *, norb: int, nelec: tuple[int, int], with_final_orbital_rotation: bool = False) -> UCCSDOp:
        nocc_a, nocc_b = nelec
        nvirt_a = norb - nocc_a
        nvirt_b = norb - nocc_b
        
        t1a = np.zeros((nocc_a, nvirt_a))
        t1b = np.zeros((nocc_b, nvirt_b))
        t2aa = np.zeros((nocc_a, nocc_a, nvirt_a, nvirt_a))
        t2ab = np.zeros((nocc_a, nocc_b, nvirt_a, nvirt_b))
        t2bb = np.zeros((nocc_b, nocc_b, nvirt_b, nvirt_b))
        
        idx = 0
        
        # T1a
        size = nocc_a * nvirt_a
        t1a = params[idx:idx+size].reshape(nocc_a, nvirt_a)
        idx += size
        
        # T1b
        size = nocc_b * nvirt_b
        t1b = params[idx:idx+size].reshape(nocc_b, nvirt_b)
        idx += size
        
        # T2aa (i<j, a<b)
        for i, j in itertools.combinations(range(nocc_a), 2):
            for a, b in itertools.combinations(range(nvirt_a), 2):
                val = params[idx]
                idx += 1
                t2aa[i, j, a, b] = val
                t2aa[j, i, b, a] = val
                t2aa[i, j, b, a] = -val
                t2aa[j, i, a, b] = -val
        
        # T2bb (i<j, a<b)
        for i, j in itertools.combinations(range(nocc_b), 2):
            for a, b in itertools.combinations(range(nvirt_b), 2):
                val = params[idx]
                idx += 1
                t2bb[i, j, a, b] = val
                t2bb[j, i, b, a] = val
                t2bb[i, j, b, a] = -val
                t2bb[j, i, a, b] = -val
        
        # T2ab
        size = nocc_a * nocc_b * nvirt_a * nvirt_b
        t2ab = params[idx:idx+size].reshape(nocc_a, nocc_b, nvirt_a, nvirt_b)
        idx += size
        
        final_orbital_rotation = None
        if with_final_orbital_rotation:
             final_orbital_rotation = linalg.unitary_from_parameters(params[idx:], norb, real=True)
             idx += norb * (norb - 1) // 2
        
        if idx != len(params):
            raise ValueError(f"Parameters length mismatch. Expected {idx}, got {len(params)}")
            
        return UCCSDOp(t1=(t1a, t1b), t2=(t2aa, t2ab, t2bb), final_orbital_rotation=final_orbital_rotation)

    def to_parameters(self) -> np.ndarray:
        t1a, t1b = self.t1
        t2aa, t2ab, t2bb = self.t2
        na, va = t1a.shape
        nb, vb = t1b.shape
        norb = na + va
        
        # Calculate size first
        n_params = self.n_params(norb, (na, nb), with_final_orbital_rotation=self.final_orbital_rotation is not None)
        params = np.zeros(n_params)
        idx = 0
        
        # T1a
        size = na * va
        params[idx:idx+size] = t1a.ravel()
        idx += size
        
        # T1b
        size = nb * vb
        params[idx:idx+size] = t1b.ravel()
        idx += size
        
        # T2aa
        for i, j in itertools.combinations(range(na), 2):
            for a, b in itertools.combinations(range(va), 2):
                params[idx] = t2aa[i, j, a, b]
                idx += 1
                
        # T2bb
        for i, j in itertools.combinations(range(nb), 2):
            for a, b in itertools.combinations(range(vb), 2):
                params[idx] = t2bb[i, j, a, b]
                idx += 1
                
        # T2ab
        size = na * nb * va * vb
        params[idx:idx+size] = t2ab.ravel()
        idx += size
        
        if self.final_orbital_rotation is not None:
             params[idx:] = linalg.unitary_to_parameters(self.final_orbital_rotation, real=True)
             
        return params

    def _apply_unitary_(self, vec: np.ndarray, norb: int, nelec: tuple[int, int], copy: bool) -> np.ndarray:
        if copy:
            vec = vec.copy()
            
        # 1. Map to GSO basis
        dim_orig = vec.shape[0]
        gso_indices = _get_gso_indices(dim_orig, norb, nelec)
        
        gso_norb = 2 * norb
        gso_nelec = (sum(nelec), 0)
        dim_gso = ffsim.dim(gso_norb, gso_nelec)
        
        vec_gso = np.zeros(dim_gso, dtype=vec.dtype)
        vec_gso[gso_indices] = vec
        
        # 2. Get Linear Operator (GSO basis)
        linop = uccsd_linear_operator(self.t1, self.t2, norb, nelec)
        
        # 3. Apply
        vec_gso = scipy.sparse.linalg.expm_multiply(linop, vec_gso, traceA=0.0)
        
        # Apply final orbital rotation if present
        if self.final_orbital_rotation is not None:
            # Need to apply spatial rotation to both alpha and beta sectors in GSO
            # ffsim.apply_orbital_rotation expects a spatial rotation matrix (norb, norb)
            # and applies it to the state.
            # But the state is in GSO basis (2*norb spatial orbitals from ffsim perspective).
            # If we pass norb=gso_norb to apply_orbital_rotation, we need a (2N, 2N) matrix.
            # The rotation U is spatial, so it acts block-diagonally on spin:
            # U_gso = [U 0; 0 U]
            
            U = self.final_orbital_rotation
            U_gso = linalg.block_diag(U, U) # hypothetical helper? or construct manually
            # actually scipy.linalg.block_diag or similar.
            
            # Construct 2N x 2N U_gso
            U_gso = np.zeros((gso_norb, gso_norb), dtype=U.dtype)
            U_gso[:norb, :norb] = U
            U_gso[norb:, norb:] = U
            
            vec_gso = gates.apply_orbital_rotation(
                vec_gso, U_gso, norb=gso_norb, nelec=gso_nelec, copy=False
            )
        
        # 4. Map back
        vec_new = vec_gso[gso_indices]
        if not copy:
             vec[:] = vec_new
             return vec
        return vec_new

    def _approx_eq_(self, other, rtol: float, atol: float) -> bool:
        if isinstance(other, UCCSDOp):
            if not np.allclose(self.t1[0], other.t1[0], rtol=rtol, atol=atol): return False
            if not np.allclose(self.t1[1], other.t1[1], rtol=rtol, atol=atol): return False
            if not np.allclose(self.t2[0], other.t2[0], rtol=rtol, atol=atol): return False
            if not np.allclose(self.t2[1], other.t2[1], rtol=rtol, atol=atol): return False
            if not np.allclose(self.t2[2], other.t2[2], rtol=rtol, atol=atol): return False
            
            if (self.final_orbital_rotation is None) != (other.final_orbital_rotation is None):
                return False
            if self.final_orbital_rotation is not None:
                return np.allclose(
                    cast(np.ndarray, self.final_orbital_rotation),
                    cast(np.ndarray, other.final_orbital_rotation),
                    rtol=rtol,
                    atol=atol
                )
            return True
        return NotImplemented

from ffsim.protocols.linear_operator_protocol import _apply_fermion_term

def jacobian_uccsd_unrestricted(
    params: np.ndarray,
    vec: np.ndarray,
    norb: int,
    nelec: tuple[int, int]
) -> np.ndarray:
    """Compute the Jacobian of the UCCSD ansatz (tangent vectors).
    
    Returns matrix of shape (dim, n_params) where column j is T_j |psi>.
    """
    
    nocc_a, nocc_b = nelec
    nvirt_a = norb - nocc_a
    nvirt_b = norb - nocc_b
    
    # 1. Map input vector to GSO basis
    dim_orig = vec.shape[0]
    gso_indices = _get_gso_indices(dim_orig, norb, nelec)
    
    gso_norb = 2 * norb
    gso_nelec = (sum(nelec), 0)
    dim_gso = ffsim.dim(gso_norb, gso_nelec)
    
    vec_gso = np.zeros(dim_gso, dtype=vec.dtype)
    vec_gso[gso_indices] = vec.astype(complex) # Ensure complex for phase application
    
    # Jacobian matrix in GSO basis
    # We will map back to spatial at the end
    n_params = len(params)
    jac_gso = np.zeros((dim_gso, n_params), dtype=complex)
    
    # Generator logic matches from_parameters order
    idx = 0
    print("Computing Jacobian: Starting...")
    
    idx_gso_alpha = lambda x: x
    idx_gso_beta = lambda x: x + norb
    
    def apply_excitation(target_vec, creation_ops, annihilation_ops):
        # creation_ops: list of (orb_index) in GSO
        # annihilation_ops: list of (orb_index) in GSO
        term_t = []
        for o in creation_ops:
            term_t.append((True, False, o))
        for o in annihilation_ops:
            term_t.append((False, False, o))
            
        res = _apply_fermion_term(target_vec, tuple(term_t), gso_norb, gso_nelec)
        
        term_tdag = []
        for o in reversed(annihilation_ops):
            term_tdag.append((True, False, o))
        for o in reversed(creation_ops):
            term_tdag.append((False, False, o))
            
        res_dag = _apply_fermion_term(target_vec, tuple(term_tdag), gso_norb, gso_nelec)
        
        return res - res_dag

    # T1a: i (occ a) -> a (virt a)
    for i in range(nocc_a):
        for a in range(nvirt_a):
            gso_cre = [idx_gso_alpha(nocc_a + a)]
            gso_ann = [idx_gso_alpha(i)]
            jac_gso[:, idx] = apply_excitation(vec_gso, gso_cre, gso_ann)
            idx += 1
            
    # T1b: i (occ b) -> a (virt b)
    for i in range(nocc_b):
        for a in range(nvirt_b):
            gso_cre = [idx_gso_beta(nocc_b + a)]
            gso_ann = [idx_gso_beta(i)]
            jac_gso[:, idx] = apply_excitation(vec_gso, gso_cre, gso_ann)
            idx += 1
            
    if idx >= 20: print(f"Computed {idx} T1/T2 columns...")

    # T2aa: i<j (occ), a<b (virt)
    for i, j in itertools.combinations(range(nocc_a), 2):
        for a, b in itertools.combinations(range(nvirt_a), 2):
            gso_cre = [idx_gso_alpha(nocc_a + a), idx_gso_alpha(nocc_a + b)]
            gso_ann = [idx_gso_alpha(j), idx_gso_alpha(i)]
            
            # Scaling factor 2.0
            jac_gso[:, idx] = 2.0 * apply_excitation(vec_gso, gso_cre, gso_ann)
            idx += 1
    
    if idx >= 50: print(f"Computed {idx} T2aa columns...")
            
    # T2bb
    for i, j in itertools.combinations(range(nocc_b), 2):
        for a, b in itertools.combinations(range(nvirt_b), 2):
             gso_cre = [idx_gso_beta(nocc_b + a), idx_gso_beta(nocc_b + b)]
             gso_ann = [idx_gso_beta(j), idx_gso_beta(i)]
             
             # Scaling factor 2.0
             jac_gso[:, idx] = 2.0 * apply_excitation(vec_gso, gso_cre, gso_ann)
             idx += 1

    if idx >= 80: print(f"Computed {idx} T2bb columns...")
             
    # T2ab: i (occ a), j (occ b), a (virt a), b (virt b)
    for i in range(nocc_a):
        for j in range(nocc_b):
            for a in range(nvirt_a):
                for b in range(nvirt_b):
                     gso_cre = [idx_gso_alpha(nocc_a + a), idx_gso_beta(nocc_b + b)]
                     gso_ann = [idx_gso_beta(j), idx_gso_alpha(i)]
                     
                     # Scaling factor 0.5
                     jac_gso[:, idx] = 0.5 * apply_excitation(vec_gso, gso_cre, gso_ann)
                     idx += 1
                     
    print(f"Computed {idx} total columns. Mapping back...")

    # Map Jacobian back to Spatial basis
    jac_spatial = jac_gso[gso_indices, :]
    return jac_spatial
