import functools as ft
import warnings
import itertools as it
from collections import deque
import numpy as np
import math as mth
import scipy
import scipy.sparse as sp
from scipy.sparse import linalg as sla
from scipy import linalg as la
from scipy import special as ss
from scipy.interpolate import interp1d
import scipy.optimize as opt
from TamLib import isBlockDiag,isherm,isunitary,fastXOR,dec2binnD,binnD2dec,circshiftBitsnD,apply_perm_to_bitstring,unique_in_sorted,cartesian_prod

from timeit import default_timer as timer
import time

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DATA TYPES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dt_site = np.uint8   # for indexing a particular site i; since tot # sites can't be too large, use smallest int type uint8, which supports up to 256 sites
dt_mb = np.uint32    # for indexing many-body bit string s = ∑_i 2^i; supports up to 32 sites

# Data type for the index/label of a fermionic gate/excitation is an np structured array
# Example usage: fgates = np.array(n, dtype = dt_plabel), pgates[i]['cre_up'] = s_up where s_up ∈ [0,2^N] and Hamming weight |s_up| = n_up
dt_fgate = np.dtype([('cre_up', dt_mb), ('ann_up', dt_mb), ('cre_dn', dt_mb), ('ann_dn', dt_mb)])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# REDUCED HILBERT SPACE OF N PARTICLES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Converts integer or array of integers representing many-body occupational state into an array of occupied sites given N tot sites
# Inputs
    # s: int or array of ints
    # N: tot # of sites
# Returns 
    # occ: len(s) array where occ[n] = tuple of occupied indices for state s[n]
def int2occ(s,N):
    scalar_input = np.ndim(s) == 0
    arr = np.asarray(s if not scalar_input else [s], dtype=np.uint64 if N <= 63 else object)
    if N <= 63:
        shifts = np.arange(N, dtype=np.uint64)
        bits   = ((arr[:, None] >> shifts) & 1).astype(bool)  # (len(s), N)
        popcounts = bits.sum(axis=1)
        if np.all(popcounts == popcounts[0]):
            # uniform popcount -> clean 2D array
            n   = int(popcounts[0])
            occ = np.argwhere(bits)[:, 1].reshape(len(arr), n)
            return [int(x) for x in occ[0]] if scalar_input else occ
        else:
            # variable popcount -> list of tuples
            result = [tuple(int(x) for x in np.flatnonzero(row)) for row in bits]
            return result[0] if scalar_input else result
    else:
        # fallback for N > 63: Python-level bit extraction
        result = [tuple(b for b in range(N) if (int(x) >> b) & 1) for x in arr]
        return result[0] if scalar_input else result

# Obtains all possible (N choose n) occupations of N tot sites with n particles
# Uses Gosper's hack to find all N-bitstrings with Hamming weight n
# Inputs
    # N: tot # of sites
    # n: # of particles = Hamming weight in bit string representation
    # returnOcc (optional): whether to return indices of occupied sites for each state
# Returns
    # ints: int representation of each of (N choose n) states
    # occ: (N choose n) x n array of occupied site indices where each site index is in [0, N-1]
def getReducedHilSpace(N,n,returnOcc = True):
    if n > N:
        raise ValueError("# of occupied n = %d too large for lattice with N = %d sites" % (n,N))
    if n == 0:
        if returnOcc:
            return np.zeros(1, dtype=dt_mb), np.empty((1, 0), dtype=dt_site)
        else:
            return np.zeros(1, dtype=dt_mb)
    dim_H = mth.comb(int(N),n)
    ints = np.zeros(dim_H, dtype=dt_mb)
    x = (1 << int(n)) - 1 # x = 2^n - 1 = (1...1)_2 = bit representation has n 1's back to back
    limit = 1 << N
    i = 0
    while x < limit: # use Gosper's hack to find next highest bit string with same Hamming weight
        ints[i] = x
        # Gosper update
        c = x & (-x) # isolates rightmost (LSB) 1 in x
        r = x + c # replaces rightmost '01' with '10' and everything to the right of it with 0
        x = r + (((r^x)//c)>>2) # add back any 1 string after the '01' and pack the string of bits to the far right
        i+=1
    if returnOcc:
        # Vectorized LSB-peel: n iterations over all dim_H states simultaneously.
        # IEEE 754: cast lsb to float64 (value conversion), view bits as int64,
        # shift off mantissa and mask exponent => log2(lsb) exactly for any power of 2.
        occ = np.empty((dim_H, n), dtype=dt_site)
        tmp = ints.copy()
        for k in range(n):
            lsb       = tmp & (-tmp)
            occ[:, k] = ((lsb.astype(np.float64).view(np.int64) >> 52) & 0x7FF) - 1023
            tmp      ^= lsb
        return ints, occ
    else:
        return ints


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FERMIONIC BASIS CONVERSION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Computes narrowest unsigned numpy int type that holds N bits
def _uint_for_bits(N):
	for dt in (np.uint8, np.uint16, np.uint32, np.uint64):
		if N <= 8*np.dtype(dt).itemsize: return dt
	raise ValueError("N=%d exceeds 64 (out of dense-vector range anyway)" % N)

# Pack per-spin N-bit strings into one combined 2N-bit string (up -> bits [0,N-1] down -> bits [N,2N-1]
def combineSpinInts(ints_up, ints_dn, N, dtype=None):
	dtype = dtype or _uint_for_bits(2*N)
	iu = np.asarray(ints_up, dtype=dtype); idn = np.asarray(ints_dn, dtype=dtype)
	return iu | (idn << dtype(N))

# Unpack a combined 2N-bit string back into (up N-bit, down N-bit)
def splitSpinInts(ints, N):
	ints = np.asarray(ints); dN = _uint_for_bits(N); m = ints.dtype.type((1<<N)-1)
	return (ints & m).astype(dN), (ints >> ints.dtype.type(N)).astype(dN)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FERMION GATES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Enumerates (creation, annihilation) many-body ints for a fixed spin-σ channel with nc creators and na annihilators,
# and optionally the tot mom transfer ∑ q_cre - ∑ q_ann.  (nc==na reduces to the original balanced channel.)
# Inputs
	# nc: number of creation ops in the spin channel ;  na: number of annihilation ops
	# N: total number of sites
# Returns
	# (s_i, s_j,): N-bit ints for the len-nc creation and len-na annihilation strings; size: (N choose nc+na)*((nc+na) choose nc)
	# s_i = \sum_m 2^{i_m} ;  ∆ = \sum q_cre - \sum q_ann
def singleSpinChannelCombos(N,nc,na,returnMomTransfer=False,Lvec=None):
	if returnMomTransfer:
		if Lvec is None:
			raise ValueError("returnMomTransfer=True requires Lvec.")
		if N!=int(np.prod(Lvec)):
			raise ValueError("N not equal to prod_a L_a.")
	if nc == 0 and na == 0: # no spin-σ ops -> (i_σ,j_σ) = (0,0)
		cre = np.zeros(1, dtype=dt_mb); ann = np.zeros(1, dtype=dt_mb)
		return (cre, ann, np.zeros(1, dtype=dt_site)) if returnMomTransfer else (cre, ann)
	if nc + na > N: # more spin-σ ops than sites --> term vanishes
		cre = np.array([], dtype=dt_mb); ann = np.array([], dtype=dt_mb)
		return (cre, ann, np.array([], dtype=dt_site)) if returnMomTransfer else (cre, ann)

	# Choose nc+na idcs for i ∪ j for spin-σ channel: every N-bit mask of size nc+na, with its sorted site list
	supp_int, supp_occ = getReducedHilSpace(N, nc + na, returnOcc=True)  # (D,), (D, nc+na) where D = N choose (nc+na)
	bit_val = (dt_mb(1) << supp_occ.astype(dt_mb))  # (D, nc+na); 2^{i} for each occupied i

	# Split which of the nc+na slots are creators — built ONCE, broadcast over all possible supports
	split_int = getReducedHilSpace(nc + na, nc, returnOcc=False)  # (K,) where K = (nc+na) choose nc
	# can think of split_int as (nc+na)-bit integer where Hamming weight is constrained to be nc
	# convert int rep into boolean mask for which of the nc+na sites go to the creation string
	cre_bits = ((split_int[:, None] >> np.arange(nc + na)) & 1)  # (K, nc+na); shift bits by {0,...,nc+na-1} to check if each bit is 1
	cre_mask = cre_bits.astype(bool)  # (K, nc+na)

	# Convert boolean mask for which nc of the nc+na sites go to creation string into int rep of occupied sites
	# bit_val = (D,nc+na), cre_mask = (K,nc+na), apply each cre_mask to each bit_val -> matrix mult along len-(nc+na) axis
	cre = (bit_val[:, None, :] * cre_mask[None, :, :]).sum(2).astype(dt_mb)  # (D, K); D = #(i ∪ j), K = #(creator splits)
	ann = supp_int[:, None].astype(dt_mb) - cre   # int(ann) = int(whole support) − int(cre)
	if not returnMomTransfer:
		return cre.ravel(), ann.ravel()

	# Compute momentum transfer ∆ = ∑ k_cre - ∑ k_ann, turned back into index in [0,N-1]
	kvecs = np.stack(np.unravel_index(supp_occ, Lvec), axis=-1)  # (D, nc+na, d); d-dim wavevector of each occupied site
	sign = 2*cre_bits - 1  # (K, nc+na); +1 on creator slots, −1 on annihilator slots (must stay signed)
	delta = np.einsum('djc,kj->cdk', kvecs, sign)  # (d, D, K); 'c' = spatial-component axis
	delta_int = np.ravel_multi_index(delta, Lvec, mode='wrap').astype(dt_site).ravel()  # (D*K,); ∆ folded to BZ, flattened
	return cre.ravel(), ann.ravel(), delta_int

# Given two lists of momentum transfer variables {∆_u}, {∆_d} corresponding to 2 channels a and b, computes idcs of Caretsian product where ∆_u = ∆_d
# Inputs
	# del_u, del_d: 2 lists/arrays of momentum transfer values in the range [0,N-1]
	# N: number of reciprocal lattice sites -> possible integer values of mom
# Returns
	# iup, idn = idcs that produce Cartesian product of all matching ∆'s; e.g. del_u[iup[n]] = del_d[idn[n]] for all indices n
# Example
	# del_u = [0,1,1,4,3,4,3,3,5] -> size 9
	# del_d = [2,4,3,5,4,3,5,1] -> size 8
	# del_u[order_u] = [0, 1, 1, 3, 3, 3, 4, 4, 5], del_d[order_d] = [1, 2, 3, 3, 4, 4, 5, 5]
	# (cts_u, cts_d) for 0: (1,0), 1: (2,1), 2: (0,1), 3: (3,2), 4: (2,2), 5: (1,2)
	# Cartesian product of idcs: iup = [1, 2, 4, 4, 6, 6, 7, 7, 3, 3, 5, 5, 8, 8], idn = [7, 7, 2, 5, 2, 5, 2, 5, 1, 4, 1, 4, 3, 6]
	# del_u[iup] = del_d[idn] = [1, 1, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5]
def _match_mom(del_u, del_d, N):
	if del_u.size==0 or del_d.size==0:
		return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
	# First, get ordering that sorts {∆}
	order_u = np.argsort(del_u, kind='stable')
	order_d = np.argsort(del_d, kind='stable')

	# Count how many times each possible value of ∆={0,1,...,N-1} appears in each channel
	# cts[i] = # of occurences of site i in ∆
	cts_u = np.bincount(del_u, minlength=N)
	cts_d = np.bincount(del_d, minlength=N)
	if cts_u.size != N or cts_d.size != N:  # if there is a ∆ >= N, bincount will return size larger than N
		raise ValueError("Momentum transfer outside [0, N-1]; check Lvec / ravel_multi_index(mode='wrap')")

	# Get start idcs of each val in [0,N-1] in the sorted ∆, if present; if not present, start = -1
	start_u = np.where(cts_u == 0, -1, np.concatenate(([0], np.cumsum(cts_u)[:-1])))
	start_d = np.where(cts_d == 0, -1, np.concatenate(([0], np.cumsum(cts_d)[:-1])))

	iu_parts, id_parts = [], []
	for q in range(N):
		if cts_u[q] == 0 or cts_d[q] == 0: # no matching mom transfer values
			continue
		# Get idcs of elems in ∆ = q group
		iu_q = order_u[start_u[q] : start_u[q] + cts_u[q]]
		id_q = order_d[start_d[q] : start_d[q] + cts_d[q]]
		# Get Cartesian product of idcs A x B
		iu_parts.append(np.repeat(iu_q, cts_d[q]))
		id_parts.append(np.tile(id_q, cts_u[q]))
	if not iu_parts:
		return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
	return np.concatenate(iu_parts), np.concatenate(id_parts)

# Enumerates idcs for order-p fermionic gate c^†_{i_1,σ_1}...c^†_{i_p,σ_p} c_{j_1,σ'_1}...c_{j_p,σ'_p}, with (iσ)∩(jσ')=∅ per spin channel
# Inputs
	# p: excitation order; # of c^†'s = # of c's ==> total number conserving
	# conserve: list of which ops to conserve; {'sz','mom','su2'}
		# sz: conserves n_↑-n_↓; since ferm excitation is already tot number (n_↑+n_↓) conserving already, this implies (n_↑,n_↓) is conserved individually
		# mom: momentum conservaation; K(i)-K(j) = 0 <=> momentum transfer ∆_↑ + ∆_↓ = 0 where ∆_σ = K(i_σ) - K(j_σ)
		# su2: NOT IMPLEMENTED YET; conserves total-spin S²
# Returns
	# pgates: 1D array of fermionic gate labels, of dtype dt_fgate
def enumerate_ferm_excitations(p, Lvec, conserve_mom=True, conserve_sz=True, conserve_su2=False):
	N = int(np.prod(Lvec))
	if conserve_su2:
		raise NotImplementedError("Total-spin (SU(2)) conservation is a different, non-abelian symmetry; not implemented.")
	if p > N:  # need ≥1 spin channel to host p ops with room for the partner channel; impossible once p > N
		warnings.warn("No p-excitation exists for p=%d > N=%d." % (p, N))
		return np.empty(0, dtype=dt_fgate)
	if conserve_mom:
		# precompute integer INDEX of inverse -k for each d-dim momentum wavevector k
		INV_IDX  = np.ravel_multi_index([-k for k in np.unravel_index(np.arange(N),Lvec)], Lvec, mode='wrap').astype(dt_site)

	# spin-splits (nc_up, na_up): up-channel has nc_up creators / na_up annihilators, down-channel gets the remainder (p-nc_up, p-na_up)
	splits = [(nc, nc) for nc in range(p+1)] if conserve_sz else [(nc, na) for nc in range(p+1) for na in range(p+1)]
	# precompute all creation/annihilation strings for each (nc,na) channel the splits actually touch — built once
	need = {ncna for (nc_up, na_up) in splits for ncna in ((nc_up, na_up), (p - nc_up, p - na_up))}
	S = {ncna: singleSpinChannelCombos(N, ncna[0], ncna[1], returnMomTransfer=conserve_mom, Lvec=Lvec) for ncna in need}

	pgates = []
	for (nc_up, na_up) in splits:  # p ≤ N guarantees at least one split survives
		if conserve_mom:
			cre_up, ann_up, del_up = S[(nc_up, na_up)]
			cre_dn, ann_dn, del_dn = S[(p - nc_up, p - na_up)]
			iup, idn = _match_mom(del_up, INV_IDX[del_dn], N)   # keep only pairs with ∆_up + ∆_dn = 0
			pgate = np.empty(iup.size, dtype=dt_fgate)
			pgate['cre_up'] = cre_up[iup];  pgate['ann_up'] = ann_up[iup]
			pgate['cre_dn'] = cre_dn[idn];  pgate['ann_dn'] = ann_dn[idn]
		else:
			cre_up, ann_up = S[(nc_up, na_up)]
			cre_dn, ann_dn = S[(p - nc_up, p - na_up)]
			usize, dsize = cre_up.size, cre_dn.size
			pgate = np.empty(usize * dsize, dtype=dt_fgate)
			pgate['cre_up'] = np.repeat(cre_up, dsize);  pgate['ann_up'] = np.repeat(ann_up, dsize)
			pgate['cre_dn'] = np.tile(cre_dn, usize);    pgate['ann_dn'] = np.tile(ann_dn, usize)
		pgates.append(pgate)
	return np.concatenate(pgates) if pgates else np.empty(0, dtype=dt_fgate)


def _jw_sign_ref(M_I, M_J, nbits):
	# sign of T=C†_I C_J acting on the reference source |M_J> (J occupied, I empty, no spectators).
	# operator order: c†_{i increasing} c_{j increasing}; rightmost (largest mode) acts first.
	cre=[m for m in range(nbits) if (M_I>>m)&1]; ann=[m for m in range(nbits) if (M_J>>m)&1]
	y=M_J; s=1
	for j in sorted(ann, reverse=True):        # annihilators act first
		if bin(y & ((1<<j)-1)).count('1') & 1: s=-s
		y &= ~(1<<j)
	for i in sorted(cre, reverse=True):        # then creators
		if bin(y & ((1<<i)-1)).count('1') & 1: s=-s
		y |= (1<<i)
	return s

def _odd_spectator_mask(M_IJ, nbits):
	# spectator modes l (not in I∪J) lying above an ODD number of I∪J modes -> their occupation flips the JW sign
	M_odd=0
	for l in range(nbits):
		if (M_IJ>>l)&1: continue                          # not a spectator
		if bin(M_IJ >> (l+1)).count('1') & 1: M_odd |= (1<<l)
	return M_odd

# Takes in gate label, which defines τ_{IJ}, and outputs matrix-free exp(i a τ_{IJ}) restricted to a reduced Hil space defined by inputted basis
# Given gate τ_{IJ} should preserve quantum numbers associated to the reduced basis; if τ_{IJ} takes you out of that Hil space, throws error 
	# e.g. if basis is for reduced Hil space defined by (n_↑, n_↓, qtot), then τ_{IJ} ought to conserve those quantum numbers
# Inputs
	# g: gate label object of dtype dt_fgate (np structured array) of 4 integers for the fields {'cre_up', 'ann_up', 'cre_dn', 'ann_dn'}
	# N: tot # of sites
	# a: coefficient of gate
	# basis: 1D array of 2N-bit strings for occupied sites; bits [0,N-1] -> up occupation, bits [N,2N-1] -> down occupation
	# sortOrder (optional) = argsort(basis); pass it once when batching so it isn't recomputed per gate.
# Returns
	# exp(iaτ_{IJ}): scipy LinearOperator object
def _excitation_operator_sector(g, N, a, basis, sortOrder=None):
	basis=np.asarray(basis, dtype=np.uint64); d=basis.size
	M_I=int(g['cre_up'])|(int(g['cre_dn'])<<N); M_J=int(g['ann_up'])|(int(g['ann_dn'])<<N) # 2N bit integers for cre and ann strings
	if M_I==M_J:
		raise ValueError("I == J gives (C^†_I C_I + h.c.) number operator, not an excitation (C^†_I C_J  + h.c.) with I != J.")
	mI=np.uint64(M_I); mJ=np.uint64(M_J)
	Delta=np.uint64(M_I^M_J) # bits where I and J differ
	M_IJ=M_I|M_J # bits where either I or J have occupation
	p=bin(M_J).count('1') # number of ann ops

	# Nonvanishing action of τ_{IJ} = C^†_I C_J + C^†_J C_I <==> either (1) J fully occupied/I fully empty or (2) I fully occupied/J fully empty
	srcJ_mask=((basis & mJ)==mJ) & ((basis & mI)==np.uint64(0))  # bool mask for J occupied, I empty; src for C^†_I C_J
	srcI_mask=((basis & mI)==mI) & ((basis & mJ)==np.uint64(0))  # bool mask for I occupied, J empty; src for C^†_J C_I

	# Only need to define action of C^†_I C_J b/c will use h.c. for C^†_J C_I
	# Find possible J-src states with multi-mode generalization of fermionNNHopping's  (s & mask_i)!=0 & (s & mask_j)==0
	isrcJ=np.flatnonzero(srcJ_mask).astype(np.intp) # idcs of J-src states
	srcJ=basis[isrcJ] # actual ints for J-src states
	# Find corresponding tgt states in basis; sort basis first so tgt lookup is fast
	if sortOrder is None: 
		sortOrder=np.argsort(basis, kind='stable') # sorted by integer value
	bs=basis[sortOrder]
	tgtI = srcJ^Delta
	pos=np.minimum(np.searchsorted(bs, tgtI), d-1) # if tgt in basis returns its sorted index; if not, returns idx of where it _would_ be in sorted basis, or clamps to value d-1
	# Gate must act within sector: every J-src's tgt present, AND no orphan I-src state (one whose J-src partner is absent).
	# Since the tgt check guarantees each J-src tgt (an I-src state) is in basis, #I-src == #J-src <==> no orphans <==> τ closed on basis
	if (srcJ_mask.sum()!=srcI_mask.sum()) or (isrcJ.size and not np.all(bs[pos]==tgtI)):
		raise ValueError("The given gate maps states out of the provided basis. Either gate doesn't preserve sector's quantum numbers or basis is incomplete.")
	itgtI=sortOrder[pos].astype(np.intp) # idx of I-tgt state in original basis

	# JW sign of T|src> = σ0 · (−1)^popcount(src & M_odd) — identical formula to the full-space build (basis-independent)
	# popcount = Hamming weight
	sigma0=1 if (p*(p-1)//2)%2==0 else -1
	M_odd=_odd_spectator_mask(M_IJ, 2*N)
	par=fastXOR(srcJ & np.uint64(M_odd)).astype(np.int64)
	sgn=(sigma0*(1-2*par)).astype(np.float64)
	ca=float(np.cos(a)); coef=(1j*np.sin(a))*sgn
	def _apply(v, off):                                         # off=+coef -> U ; off=−coef -> U†
		v=np.asarray(v); shp=v.shape; v=v.ravel()
		w=v.astype(np.complex128, copy=True)                    # identity action on the d_sec in-sector configs
		if isrcJ.size:
			vs=v[isrcJ]; vt=v[itgtI]
			w[isrcJ]=ca*vs+off*vt; w[itgtI]=off*vs+ca*vt              # 2-level rotation on active (src,tgt) blocks
		return w.reshape(shp)
	return sla.LinearOperator((d,d), matvec=lambda v:_apply(v, coef), rmatvec=lambda v:_apply(v,-coef), dtype=np.complex128)


# Like fgateToExp, but each operator acts on the reduced `basis` (dim d_sec = len(basis)) instead of the full 2^{2N}.
def fgateToExpSector(gateLabels, A, N, basis):
	labels=np.atleast_1d(gateLabels); A=np.broadcast_to(np.atleast_1d(np.asarray(A, dtype=float)), labels.shape)
	basis=np.asarray(basis, dtype=np.uint64); sortOrder=np.argsort(basis, kind='stable')   # built once, shared over the batch
	ops=[ _excitation_operator_sector(g, N, a, basis, sortOrder)
	      for g,a in zip(labels, A) ]
	return np.array(ops, dtype=object)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TRANSLATION INVARIANCE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Gets single-site translation operator along each axis for spin/qubit or fermion lattice sys as a "signed" permutation on 2^N states
# Convention: transl op T_a along axis a increases index a by (1 mod L_a)
# Inputs
    # Lvec: lattice dimensions; e.g. (Lx,Ly)
    # dof (optional): 'qubit'/'spin' or 'fermion'
    # basis (optional): int array of states to compute transl op on; will default to all 2^N states unless otherwise specified
# Returns 
    # Tops: np array of size len(Lvec) x 2^N; Tops[a,s] = int(T_a[s]) = integer rep of T_a applied to bit string state s
    # signs: np array of size len(Lvec) x 2^N; ±1 for sign of T_a[s]
def translOpnD(Lvec,dof='fermion',basis=None,DEBUG_MODE=False):
    ndim = len(Lvec) # number of axes
    N = np.prod(Lvec) # tot number of sites
    isfermionic = dof.lower()=='fermion'
    # Choose which basis states will compute transl op on (e.g. states within certain number sector) ------------------------------------------------
    if basis is None:
        basis = np.arange(1<<N, dtype=dt_mb) # default is all 2^N occupational basis states
    else:
        basis = np.asarray(basis, dtype=dt_mb)

    num_states = len(basis) # number of states that will restrict attn to
    Tops = np.zeros((ndim,num_states),dtype=dt_mb)
    signs = np.zeros((ndim,num_states),dtype=np.int8)

    # Precomputations ------------------------------------------------------------------------------------------------
    masks = (dt_mb(1) << np.arange(N, dtype=dt_mb))  # precompute masks for checking occupancy of each site; gives [2^i, i=0,...,N-1]
    site_idcs = np.arange(N,dtype=dt_site)
    sites = np.reshape(site_idcs, Lvec)
    strides= np.ones(ndim, dtype=np.uint32)
    for k in range(ndim - 2, -1, -1):
        strides[k] = strides[k + 1] * Lvec[k + 1]
    if DEBUG_MODE:
        print("strides: %s" % strides)
        print("sites:\n%s" % sites)

    # Perform circshfit along each axis ------------------------------------------------------------------------------------------------
    for a in range(ndim): # for each axis
        if DEBUG_MODE: print("----------- T_%d -----------" % a)

        # 1) Single site transl along axis a as a perm on N sites ------------------------------------------------
        ss_perm = np.roll(sites,-1,axis=a).ravel() # ss_perm[i] = new loc of site i
        # ex: 2x3 lattice has sites [[0,1,2],[3,4,5]]; for transl along axis 0, ss_perm = [3,4,5,0,1,2]; for axis 1, [1,2,0,4,5,3]
        if DEBUG_MODE: print("ss_perm: %s " % ss_perm)

        # 2) Compute inversion masks for fermionic parity computation ------------------------------------------------
        # inv_masks[i] = \sum_j 2^j for each j that i hops over; puts a 1 in jth bit for each site i hops over
        # ex: 3x4x2 lattice, transl along axis 1 => site 0 hops over {6,7} => mask is 192 = 2^6 + 2^7
        # given state s, site j is occupied if (s & 2^j) > 0
        if isfermionic:
            s_a, L_a = int(strides[a]), int(Lvec[a]) # stride for axis a, size of axis a
            bs_a = L_a * s_a # size of block for axis a
            nblocks = N//bs_a # number of a-blocks
            block_perSite = site_idcs//bs_a # block idx for each site
            aidcs = (site_idcs//s_a) % L_a # a^{th} coordinate of each site
            nonbdy_mask = aidcs < (L_a - 1) # which sites are nonboundary wrt axis a
            # bdy idcs for each block are (blockIdx*blockSize + (L_a-1)s_a : 1 : blockIdx*blockSize + L_a s_a - 1)
            startIdcs_bdy = np.arange(nblocks,dtype=np.uint32)*bs_a + ((L_a-1)*s_a)
            bdyMask_perBlock = dt_mb((1 << s_a) - 1) << startIdcs_bdy
            inv_masks = np.where(nonbdy_mask,bdyMask_perBlock[block_perSite],dt_mb(0)) # (N,); inv_masks[i] = mask for sites that i hops over if i is nonbdy, otherwise is 0
            tot_parity = np.zeros(num_states, dtype=np.uint8)
            if DEBUG_MODE:
                print("inv masks: %s" % inv_masks)

        # 2) Compute translated occupational basis states, with sign factor if fermionic ------------------------------------------------
        # strategy: transl N-bit strings along axis a by seeing which entries are 1 and using ss_perm
        out = np.zeros_like(basis) # will contain int rep of each of the 2^N bit strings translated by T_a
        shifts = ss_perm.astype(np.int32) - site_idcs # precompute shift from old loc to new loc
        for i in range(N):
            # check if site i is 1 in all basis states at once
            ibits = basis & masks[i] # if ith bit of basis[k] is 1, then ibits[k] has 1 in ith spot; otherwise all N bits are 0.
            # if i^th bit is 1, shift the 1 to the new loc of i
            shift = int(shifts[i])
            if shift>0: out |= (ibits << shift)
            else:       out|= (ibits >> -shift)

            # Get sign factor if fermionic
            if dof.lower()=='fermion' and inv_masks[i]:
                # parity contribution due to reordering ith bit = (-1)^{# inv to put i in place}
                if hasattr(np, 'bitwise_count'):
                    p_i = np.bitwise_count(basis & inv_masks[i]).astype(np.uint8) & np.uint8(1)
                else:
                    p_i = fastXOR(basis & inv_masks[i]).astype(np.uint8)

                tot_parity ^= ibits.astype(bool) & p_i  # accumulate parity

        Tops[a] = out
        signs[a] = 1-2*tot_parity if isfermionic else np.int8(1)

    return Tops, signs

# Construct all orbits under translation group on nD lattice for bit string states
# Inputs
    # Lvec: lattice dimensions; e.g. (Lx,Ly)
    # dof (optional): 'qubit'/'spin' or 'fermion'
    # basis (optional): int array of states to compute transl op on; will default to all 2^N states unless otherwise specified
    # state2idx (optional): dict = {integer_state: idx in reduced basis}
# Returns
    # orbits: dict with key = int rep of generator, value = dict of properties of orbit
    # 'signed_orbit'
    # 'orbit_idcs'
    # 'orbit_signs'
    # 'rvec'
    # 'stabilizers'
    # 'stabilizer_signs'
def buildOrbitsnD(Lvec, dof='fermion', basis=None, state2idx = None, DEBUG_MODE=False):
    ndim = len(Lvec) # number of axes
    N = np.prod(Lvec) # tot number of sites

    # Get reduced Hil space of interest ------------------------------------------------
    if basis is None:
        basis = np.arange(1<<N, dtype=dt_mb) # default is all 2^N occupational basis states
        state2idx = dict(zip(basis,basis))
    else:
        basis = np.asarray(basis, dtype=dt_mb)
        if state2idx is None:
            state2idx = {s: i for i,s in enumerate(basis)} # for fast indexing
    if DEBUG_MODE:
        print("basis: %s" % basis)
    # Get single-site transl op along each axis ------------------------------------------------
    Tops, signs = translOpnD(Lvec, dof=dof, basis=basis)

    # Generate orbits by iterating over all states in reduced Hil space ------------------------------------------------
    visited = np.zeros(len(basis), dtype=bool) # initialize; i^{th} entry will be true once visit i^{th} basis state
    orbits = {} # dictionary with key = generator of orbit, value = dict of properties of orbit
    for i, root in enumerate(basis):
        if DEBUG_MODE:
            print("root: %d\t i: %d" % (root,i))
        if visited[i]:
        	continue

        # BFS to build orbit ------------------------------------------------
        visited[i] = True
        if DEBUG_MODE: 
            print("building tree...")
        signed_orbit = [int(root)]  # root will be a pos int
        orbit_idcs = [i] # index of root within basis/reduced Hil space
        orbit_signs = [1] # cumulative signs for each state within orbit
        rvecs = [np.zeros(ndim, dtype=dt_site)] # transl vec for each state in orbit, i.e. stores r = (r_1,...,r_n) assoc to T_1^{r_1}...T_n^{r_n} 
        queue = deque([(i, 1, np.zeros(ndim, dtype=dt_site))])  # (index, cumulative_sign, rvec) for each state
        stabilizers = []  # will store rvecs that stabilize the generator
        stabilizer_signs = []  # add alongside stabilizers list
        while queue:
            i_curr, sgn_curr, r_curr = queue.popleft()
            if DEBUG_MODE: print("curr state: %s\t r = %s" % (basis[i_curr],r_curr))

            next_states = Tops[:, i_curr]  # all states T_a[s]; shape: (ndim,)
            next_signs = signs[:, i_curr]  # shape: (ndim,)
            if DEBUG_MODE: 
                print("next states: %s" % (next_states*next_signs))

            for a in range(ndim): # add each Ts branch to orbit, if not already visited
                next_state, next_sign = next_states[a], next_signs[a]
                i_next = state2idx[next_state]
                r_next = r_curr.copy() # new rvec will be r_curr with r_a incremented
                r_next[a] = (r_next[a] + 1) % Lvec[a]
                if not visited[i_next]: # if not visited, make Ts the root of a new tree
                    cumul_sgn = sgn_curr * next_sign # cumulative sgn when reaching state from generator                
                    visited[i_next] = True
                    signed_orbit.append(cumul_sgn * next_state) # add to orbit as signed integer
                    orbit_idcs.append(i_next)
                    orbit_signs.append(cumul_sgn)
                    rvecs.append(r_next.copy())
                    if DEBUG_MODE: 
                        print("adding %d w/idx %d" % (cumul_sgn*np.intp(next_state),i_next)) 
                    queue.append((i_next, cumul_sgn, r_next))
                elif i_next == i and not np.array_equal(r_next,np.zeros(ndim,dtype=dt_site)):  # if translation maps back to ±1 times the root/generator itself
                    stabilizers.append(r_next.copy())
                    stabilizer_signs.append(sgn_curr * int(next_sign))  # cumulative sign along stabilizer path
        
        # Store orbit with generator as key
        orbits[int(root)] = {
            'signed_orbit': np.array(signed_orbit, dtype=np.int64), # SIGNED (not dt_mb): jointly encodes occupation |s| and fermion sign; sign bit limits to <=63 sites. shape: (Orb size,)
            'orbit_idcs': np.array(orbit_idcs,dtype=np.uint32), # shape: (Orb size, )
            'orbit_signs': np.array(orbit_signs,dtype=np.int8), # shape: (Orb size, )
            'rvec': np.array(rvecs,dtype=dt_site), # shape: (Orb size, ndim)
            'stabilizers': np.array(stabilizers, dtype=dt_site) if stabilizers else np.zeros((0, ndim), dtype=dt_site), # shape: (# stabilizers, ndim)
            'stabilizer_signs': np.array(stabilizer_signs, dtype=np.int8) if stabilizer_signs else np.ones(0, dtype=np.int8)
        }
    return orbits

# Gets change of basis matrix into momentum eigenbasis; cols are mom eigenstates with
# Inputs
    # Lvec: lattice dimensions; e.g. (Lx,Ly)
    # dof (optional): 'qubit'/'spin' or 'fermion'
    # basis (optional): int array of states to compute transl op on; will default to all 2^N states unless otherwise specified
    # state2idx (optional): dict = {integer_state: idx in reduced basis}
    # returnType (optional): return type of momentum eigenstates; 'sp' for sparse or 'np' = numpy dense array
def translnDCOB(Lvec,dof='fermion',basis=None,state2idx=None,returnType='sp',DEBUG_MODE=False):
    Lvec = np.array(Lvec,dtype=np.uint32)
    N = int(np.prod(Lvec)) # tot number of sites
    # Get reduced Hil space of interest ------------------------------------------------
    if basis is None:
        basis = np.arange(1<<N, dtype=dt_mb) # default is all 2^N occupational basis states
        state2idx = dict(zip(basis,basis))
    else:
        basis = np.asarray(basis, dtype=dt_mb)
        if state2idx is None:
            state2idx = {s: i for i,s in enumerate(basis)} # for fast indexing
    # Get orbits under nD transl group ------------------------------------------------
    orbits = buildOrbitsnD(Lvec, dof=dof, basis=basis, state2idx = state2idx)
    lcm_L = int(np.lcm.reduce(Lvec)) # LCM of all axis lengths; so that q_a/L_a can be put into fraction with same denom
    axis_scale = (lcm_L // Lvec).astype(np.int64) # shape (ndim,); LCM/La = scaling factor for q_a
    if DEBUG_MODE:
        print(orbits.keys())
    num_states = len(basis)
    if returnType.lower()=='np':
        P = np.zeros(shape=(num_states,num_states),dtype=complex)
    else:
        P = sp.lil_matrix((num_states,num_states),dtype=complex)
    if DEBUG_MODE:
        print("P shape: %s"% (P.shape,))
    blockSizes = {} # will store size of each mom block
    currCol = 0
    for qvec in it.product(*[range(L) for L in Lvec]): # iterate over integer wavectors where e^{ik_a} = e^{i2πq_a/L}
        if DEBUG_MODE: print("qvec: %s" % list(qvec))
        bs = 0
        qvec_scaled = np.array(qvec, dtype=np.int64) * axis_scale # (ndim,)
        # Check compatibility condition with each possible generator ------------------------------------------------
        for m in orbits.keys():
            stabilizers = orbits[m]['stabilizers'] # shape: (n_stab,ndim)
            stabilizer_signs = orbits[m]['stabilizer_signs']
            if len(stabilizers)==0:
                compatible = True # automatically compatible
            else:
                # Note: qvec * stabilizers/Lvec = (n_stab,ndim)
                    # compatible iff exp(2πi sum_a q_a s_a/L_a) = σ_s where σ_s = ±1 is the sign of the stabilizer vector s
                    # if σ_s = -1, want sum_a q_a s_a/L_a = half integer; if σ_s = +1, want sum_a q_a s_a/L_a = integer
                # Exact integer compatibility check: (sum_a s_ja * q_a * (lcm/L_a)) % lcm == 0 (integer,σ = +1) OR 
                # compatible iff k_j % lcm == 0 (sigma=+1) or lcm//2 (sigma=-1)
                phase = (stabilizers@qvec_scaled) % lcm_L
                expected = ((1 - stabilizer_signs.astype(np.int64)) // 2) * (lcm_L // 2)
                compatible = bool(np.all(phase==expected))
            if compatible:
                orbit_idcs = orbits[m]['orbit_idcs']
                orbit_signs = orbits[m]['orbit_signs']
                M = len(orbit_idcs)  # orbit size
                bs=bs+1
                # row = computational basis state that have overlap with = basis states in orbit
                # col = which mom eigenstate
                rvecs=orbits[m]['rvec'] # shape: (M,ndim) where M = orbit size
                P[orbit_idcs,currCol] = (1/np.sqrt(M))*np.exp(-1j*2*np.pi*np.sum(((qvec*rvecs)/Lvec),axis=1))*orbit_signs
                if DEBUG_MODE: print("compatible with %g\t col: %s" % (m,currCol))
                currCol = currCol+1  # increment col
        blockSizes.update({qvec: bs})
    if returnType.lower()=='np':
        return P, blockSizes
    else:
        return P.tocsr(), blockSizes

def reorderByTotMom_orbit(P, bSizes_up, bSizes_down, Lvec):
    Lvec = np.asarray(Lvec)
    qvecs = np.array(list(bSizes_up.keys()))
    # repeat each integer mom qvec based on block sizes --> labels each col of Pu or Pd by corresponding qvec
    qu = np.repeat(qvecs,list(bSizes_up.values()),axis=0) # (nu,ndim)
    qd = np.repeat(qvecs,list(bSizes_down.values()),axis=0) # (nd,ndim)
    nu, nd = sum(bSizes_up.values()), sum(bSizes_down.values()) # number of mom eigenstates in each sector; nu = (N choose n_up), nd = (N choose n_down)
    # P = Pu ⊗ Pd : tot mom vec for every column of P, via broadcasting (col_u repeated nd times, col_d tiled nu times -- done implicitly)
    qt = ((qu[:, None, :] + qd[None, :, :]) % Lvec).reshape(-1, len(Lvec)) # (nu*nd, ndim); (qx_up + qx_down, qy_up + qy_down)
    qt_idx = np.ravel_multi_index(qt.T, Lvec) # integer rep of qtot (C-ravel) -> 1D sort key
    # Sort by qtot (integer key: argsort faster than lexsort over ndim rows)
    perm  = np.argsort(qt_idx, kind='stable')
    qt_sorted = qt_idx[perm]
    # New block sizes based on tot mom: count occurrences of each qt in one O(nu*nd) pass over the sorted keys
    qt_unique, counts = unique_in_sorted(qt_sorted, returnCounts=True) # number of cols per qtot block
    return P[:, perm], counts


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SLATER DETERMINANT MOMENTUM BASIS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Gets n-particle sector COB matrix to fermionic Slater momentum/Fourier basis
# Auto-batches over the K (row) block so estimated peak memory stays under mem_budget (GB)
# Inputs
	# Lvec: list/tuple/1d array lattice dimensions; (L_1,...,L_d)
	# n: number of particles
# Returns
	# P(K<-R): COB matrix P_{K,R}|K><R|; size (N choose n) x (N choose n) where N = ∏_i L_i 
	# P_{K,R} = <K|R> = det Φ_{ab} where Φ_{ab} = (1/sqrt(N)e^{i k_a•r_b}) where k = 2πq/L for integer mom q
def SlaterCOB_RtoK_nparticle(Lvec, n, mem_budget=128.0):
    Lvec = np.asarray(Lvec); N = int(np.prod(Lvec)); dH = mth.comb(N, n)
    if n == 0:
        return np.ones((1, 1), complex)
    budget    = mem_budget * 2**30                 # GB -> bytes
    out_bytes = 16.0 * dH * dH                     # 16 bytes/complex # => mem needed to output a particular P^{K,R}
    # Batch K values/rows into groups of size b
    # Each batch involves computing d_H determinants; to get determinant, need to store n^2 complex #s -> 16 d_H n^2 bytes per batch
    per_row  = 32.0 * dH * n * n  # (16 d_H n^2) with factor of 2 for safety
    if budget < out_bytes + per_row:
        need = (out_bytes + per_row) / 2**30
        raise MemoryError(f"n={n}: Need mem_budget >= {need:.2f} GB (output {out_bytes/2**30:.2f} GB + >=1 row), "
                          f"have {mem_budget} GB")
    # max nrows that can fit = (bytes leftover for transient per batch computation)/(bytes used per row)
    batch = min(dH, int((budget - out_bytes) // per_row)) # batch_size = min(d_H, max nrows)
    coords = np.indices(Lvec).reshape(len(Lvec), -1).T
    F = np.exp(2j*np.pi * (coords @ (coords/Lvec).T)) / np.sqrt(N)
    occ  = getReducedHilSpace(N, n, returnOcc=True)[1]
    cols = occ[None, :, None, :]
    P = np.empty((dH, dH), complex)
    for s in range(0, dH, batch):
        P[s:s+batch] = np.linalg.det(F[occ[s:s+batch, None, :, None], cols])
    return P

# Get n-particle Slater momentum basis for a single spin sector, sorted by total momentum
# many-body config |Q_σ> = |q_1,...,q_{n_σ}> -> qtot = q_1 + ... + q_{n_σ}
# Inputs
	# Lvec: lattice dimensions
	# n: number of particles
	# sort (optional): bool for whether to sort by tot mom
# Returns
	# dict containing:
		# 1) ints, occs for n-particle sector sorted by qtot in C-ravel order
		# 2) the corresponding vector of their qtots
		# 3) the order that sorts them from outputs of getReducedHilSpace
		# 4) counts for each qtot block
		# 5) the unique qtot's present 
def sectorTotMom(Lvec, n, coords=None, sort=True):
    Lvec = np.asarray(Lvec); N = int(np.prod(Lvec)); d = len(Lvec)
    if coords is None:
        coords = np.indices(Lvec).reshape(d, -1).T
    ints, occ = getReducedHilSpace(N, n, returnOcc=True)         # canonical order, ints aligned with occ
    qtotvecs = coords[occ].sum(1) % Lvec                         # (d_s, d) total momentum vector per config
    qtot = np.ravel_multi_index(qtotvecs.T, Lvec)            # 1D sort key = integer rep qtot (C-ravel order)
    sortOrder = np.argsort(qtot, kind='stable')
    qtot_sorted = qtot[sortOrder]                                     # sorted list of keys/integer qtots
    # Retrieve unique elems and counts in one O(N) pass
    qtot_unique, counts = unique_in_sorted(qtot_sorted, returnCounts=True)  # sorted-order: unique qtots + sector sizes
    if sort:                                                     # materialize configs already in qtot-sorted order
        ints, occ, qtotvecs = ints[sortOrder], occ[sortOrder], qtotvecs[sortOrder]
    return dict(ints=ints, occ=occ, qtot=qtot, sortOrder=sortOrder, qtot_unique=qtot_unique, counts=counts)

# Get full Slater momentum space for (n_↑, n_↓) sector, sorted and grouped by tot mom qtot = qtot_↑ + qtot_↓ (C-ravel order) 
# Within a qtot sector, sort by qtot_↑ (slowest), then qtot_↓, then canonical config order inherited from sectorTotMom
# Canonical config order needed since for each qtot_σ there are many (q_1,...,q_{n_σ}) that give same qtot_σ
# Input
	# Lvec: lattice dimension
	# n_up, n_dn: ints for # of up/down fermions
# Return
	# dict containing data for all states in (n_↑, n_↓) sector, organized by momentum; D = (N choose n_↑)•(N choose n_↓)
		# 1) ints: 				(D,) combined 2N-bit integer Slater states where up -> bits [0,N-1], down -> bits [N, 2N-1]
		# 2) qtot_up, qtot_dn:	(D,) per-spin tot mom codes (raveled indices in [0,N-1]); sets sub-order within a qtot block
		# 3) qtot: 				(D,) combined tot mom code per state (the block label)
		# 4) qtot_unique: 		(nblocks,) unique qtot per block; nblocks ≤ N
		# 5) counts:			(nblocks,) integer sizes of each tot mom block
def fullSlaterMomBasis(Lvec, n_up, n_dn):
	Lvec = np.asarray(Lvec); N = int(np.prod(Lvec)); d = len(Lvec)
	coords = np.indices(Lvec).reshape(d, -1).T
	up = sectorTotMom(Lvec, n_up, coords, sort=True)             # each spin sector sorted by its own q_tot_σ
	dn = sectorTotMom(Lvec, n_dn, coords, sort=True)
	# per-state q_tot_σ codes, in the sorted product order (up slow, down fast)
	qu = np.repeat(up['qtot_unique'], up['counts']); qd = np.repeat(dn['qtot_unique'], dn['counts'])
	qtot_up, qtot_dn = cartesian_prod(qu, qd)
	# combined 2N-bit Slater state |I_up, I_dn>, same product order
	ints_up, ints_dn = cartesian_prod(up['ints'], dn['ints'])
	ints = combineSpinInts(ints_up, ints_dn, N)
	# combined total momentum via the N x N (q + q') addition table (integer-only)
	addmom = np.ravel_multi_index(((coords[:,None,:]+coords[None,:,:]) % Lvec).reshape(-1, d).T, Lvec).reshape(N, N)
	qtot = addmom[qtot_up, qtot_dn]
	# group by combined q_tot; (q_tot_↑, q_tot_↓) inherited from the sorted product
	order = np.argsort(qtot, kind='stable')
	qtot_unique, counts = unique_in_sorted(qtot[order], returnCounts=True)
	return dict(ints=ints[order],
	            qtot_up=qtot_up[order], qtot_dn=qtot_dn[order],
	            qtot=qtot[order], qtot_unique=qtot_unique, counts=counts)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FERMIONIC HAMILTONIANS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Given an nD lattice, finds the "forward" NN of each site, i.e. given (...,i_a,...), forward neighbor along axis a has coords (...,i_a+1,...)
# Returns d # neighbors: shape (ndim, N) where ndim = # axes, N = tot # of sites; neighbors[a,i] = idx of the neighbor in the +a direction of i
def findForwardLatticeNeighbors(Lvec, use_pbc = True):
    N = int(np.prod(Lvec))
    ndim = len(Lvec)
    sites = np.arange(N, dtype=dt_site).reshape(Lvec)
    if use_pbc:
        neighbors = np.empty((ndim, N), dtype=dt_site)
        # Vectorized: all at once
        for a in range(ndim):
            neighbors[a] = np.roll(sites, -1, axis=a).ravel()
    else:
        neighbors = -np.ones((ndim, N), dtype=np.int32) # SIGNED (not dt_site): -1 sentinel marks a missing forward neighbor under OBC
        # Vectorized boundary check
        indices = np.arange(N, dtype=dt_site)
        coords = np.indices(Lvec).reshape(len(Lvec), -1).T  # (N, ndim); nD coords for each of the N sites
        for a in range(len(Lvec)):
            # Mask: all sites that have a forward neighbor in direction a
            nonbdy = coords[:, a] < Lvec[a] - 1
            shiftedCoords = coords.copy()
            shiftedCoords[:, a] = (shiftedCoords[:, a]+1) % Lvec[a]
            neighbor_idcs = np.ravel_multi_index(tuple(shiftedCoords.T), Lvec)
            neighbors[a, nonbdy] = neighbor_idcs[nonbdy]

    return neighbors

# Finds all NN edges of a lattice 
# iterates over every site and finds all "forward" neighbors of the site
# Lvec: lattice dimensions, e.g. (Lx,Ly)
# use_pbc (optional): bool for whether have periodic BC
# Returns
# edges = list of (i,j) pairs where j and i are neighbors
def findLatticeEdges(Lvec, use_pbc=True):
    N = np.prod(Lvec)
    ndim = len(Lvec)
    sites = np.arange(N, dtype=dt_site).reshape(Lvec)
    # To count each edge only once, for each site, consider "forward" edges in + direction along each axis
    src_sites = [None]*ndim # possible "source" sites of edge
    dst_sites = [None]*ndim # possible "destination" sites of edge
    if use_pbc:
        srcs = sites.ravel()
        # Vectorized: all at once
        for a in range(ndim):
            src_sites[a] = srcs
            # Vectorized: all at once
            dst_sites[a] = np.roll(sites, -1, axis=a).ravel()
    else:
        for a in range(ndim):
            # Initialize slices --- slices(None) equivalent to :
            sl_src = [slice(None)] * ndim
            sl_dst = [slice(None)] * ndim
            sl_src[a] = slice(0, -1)  # Equivalent to :-1
            sl_dst[a] = slice(1, None) # Equivalent to 1:
            src_sites[a] = sites[tuple(sl_src)].ravel()
            dst_sites[a] = sites[tuple(sl_dst)].ravel()
    return list(zip(np.concatenate(src_sites), np.concatenate(dst_sites)))

# Computes hopping term 
# basis: bit string states spanning the reduced Hil space that one is considering
# edges: list of (i,j) pairs defining connectivity of lattice/graph; e.g. 2x3 lattice has [(0,3),(0,1),(1,4),(1,2),...]
# t: hopping strength; if scalar, then uniform; otherwise, must be same size as edges
# state2idx (optional): dict {s: i for i,s in enumerate(basis)}; key=int rep of basis state, val=index within reduced Hil space
# Returns
    # csr_matrix respresentation of hopping Hamiltonian
def fermionNNHopping(basis, edges, t, state2idx=None, DEBUG_MODE=False):
    rows = []
    cols = []
    data = []
    basis = np.array(basis, dtype=dt_mb)
    num_states = len(basis)
    basis_idcs = np.arange(num_states,dtype=np.uint32)
    if DEBUG_MODE:
        print("basis: %s" % basis)
    for i, j in edges:
        mask_i = dt_mb(1) << dt_mb(i)
        mask_j = dt_mb(1) << dt_mb(j)

        # Mask for i->j hop (i occupied, j unoccupied) ------------------------------------------------
        hop_mask = ((basis & mask_i) != 0) & ((basis & mask_j) == 0)
        if not np.any(hop_mask):
            continue

        # Extract basis states that can source i->j hop ------------------------------------------------
        src_states = basis[hop_mask]
        src_idcs = basis_idcs[hop_mask] # use int b/c sparse csr uses int for indexing so avoids conversion

        # Extract basis states that are the outputs of i->j hop ------------------------------------------------
        dst_states = (src_states ^ mask_i) | mask_j
        if state2idx is None:
            dst_idcs = np.searchsorted(basis, dst_states)
        else:
            dst_idcs = np.array([state2idx[s] for s in dst_states], dtype=np.uint32)

        # Compute fermionic sign ------------------------------------------------
        # the number of (-1) factors is based on the number of occupied sites with index between i and j
        low, high = (i, j) if i < j else (j, i)
        mid_mask = ((dt_mb(1) << dt_mb(high)) - 1) ^ ((dt_mb(1) << (dt_mb(low) + 1)) - 1) # puts a 1 for all pwrs of 2 b/t i and j
        x = (src_states & mid_mask) # vectorize: for each basis state, puts a 1 in occupied sites with indices b/t i and j
        parity = fastXOR(x) # fastXOR to count number of occupied sites between i and j
        signs = (1 - 2 * parity.astype(np.int8)) # signs for EACH of the output basis states from i->j hop

        # Append to hopping csr_matrix data ------------------------------------------------
        # h_{ab} = t c^†_a c_b = t (sign) |dst_a >< src_b|
        if DEBUG_MODE:
            print("edge (%d,%d): %s" % (i,j,list(zip(src_states.tolist(),dst_states.tolist()))))
            print("      idcs: %s" % (list(zip(src_idcs.tolist(),dst_idcs.tolist()))))
            print("     signs: %s" % signs)
        rows.append(dst_idcs)
        cols.append(src_idcs)
        data.append(-t * signs)

    # Flatten lists and create the sparse matrix
    if data:
        hop = sp.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(num_states, num_states))
    else:
        hop = sp.csr_matrix((num_states, num_states))   # no hops (vacuum, fully-filled, or no edges)
    return hop + hop.getH()  # hermitize to include edges in opposite direction

# Creates diagonal spin-density interaction term sum_i n_{i↑} n_{i↓}
# N: tot # of sites
# ubasis: np array of integers representing Fock states in the n_up sector, i.e. with fixed n_up
# dbasis: np array of integers representing Fock states in the n_down sector, i.e. with fixed n_down
# u: spin-density interaction strength
# Note: n_up + n_down <= N
def fermionOnSiteSpinDensity(N,ubasis,dbasis,u=1):
    masks = (dt_mb(1) << np.arange(N, dtype=dt_mb)) # [2^i for i in range(N)]
    Mu = ((ubasis[:, np.newaxis] & masks) != 0).astype(np.float64)  # (nu, N); (Mu)_ai = 0 or 1 for whether ustate a has site i occupied
    Md = ((dbasis[:, np.newaxis] & masks) != 0).astype(np.float64)  # (nd, N); (Md)_bi = 0 or 1 for whether dstate b has site i occupied
    # sum_i (Mu)_ai (Md)_bi = number of bits ustate a and dstate b agree on
    return sp.diags(u*(Mu@Md.T).ravel())

# Lvec: lattice dimensions; e.g. (Lx,Ly)
# nvec: (n_up, n_down); specifies sector that Hamiltonian will have 
# t: hopping strength
# u: same-site interaction/repulsion strength 
# use_pbc (optional): whether to use periodic boundary conditions (PBCs) for hopping term
# H = -t \sum_{<ij>,σ} c^†_{iσ} c_{jσ} + u\sum_{i} n_{i↑} n_{i↓}
# Returns 
    # H: sparse matrix of size (N choose n_up) x (N choose n_down)
# Note: c-style ordering of flattened site indices
# Note: canonical ordering of creation operators is highest index applied to vacuum first, i.e. |110> == c^†_0 c^†_1 |000>
def Hubbard(t,u,Lvec,nvec,use_pbc = True,returnBasis=True):
    n_up,n_down = nvec
    N = np.prod(Lvec) # total number of sites
    # get reduced Hil basis for each up/down sector
    ubasis, uocc = getReducedHilSpace(N, n_up)
    dbasis, docc = getReducedHilSpace(N, n_down)

    # Map basis states to indices for fast lookup ------------------------------------------------
    u_map = {s: i for i, s in enumerate(ubasis)}
    d_map = {s: i for i, s in enumerate(dbasis)}

    # Hopping term ------------------------------------------------
    # c^†_iσ c_jσ
    edges = findLatticeEdges(Lvec, use_pbc=use_pbc)
    hop_up = fermionNNHopping(ubasis, edges, t, u_map)
    hop_down = fermionNNHopping(dbasis, edges, t, d_map)
    H_hop = sp.kronsum(hop_down,hop_up,format='csr') # H_hop = (1 ⊗ H_down) + (H_up ⊗ 1)

    # On-site spin density term ------------------------------------------------
    H_onsite = fermionOnSiteSpinDensity(N,ubasis,dbasis,u=u)

    # Full Hamiltonian in computational basis ------------------------------------------------
    H = H_hop + H_onsite
    # H = H + H.T.conj() # force perfect Hermiticity
    if returnBasis:
        return H, (ubasis,dbasis), (uocc,docc)
    else:
        return H