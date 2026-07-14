import functools as ft
import re
import itertools as it
import numpy as np
import math as mth
import scipy
import scipy.sparse as sp
from scipy.sparse import linalg as sla
from scipy import linalg as la
from scipy import special as ss
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.colors import ListedColormap, Normalize, BoundaryNorm, LinearSegmentedColormap, ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from fractions import Fraction
from collections import Counter 
from timeit import default_timer as timer
import time
import warnings

from matplotlib.collections import PolyCollection # for plotting filled waterfall plots
from matplotlib.collections import LineCollection

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PERMUTATIONS 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Computes parity of permutation on N lements using cycle decomposition of permutation
# decompose σ into prod of disjoint cyc (c_1)•(c_2)•... and then sgn(σ) = (-1)^{N - N_cyc}
# perm: a len-N (ordered) list/array of elems [0,1,...,N-1]
# Returns: ±1 depending on whether the perm is odd or even
def perm_parity_cyc(perm, validate=False):
    if isinstance(perm, np.ndarray):
        if np.issubdtype(perm.dtype, np.integer):
            perm = perm.tolist()
        elif np.issubdtype(perm.dtype, np.floating):
            with np.errstate(invalid='ignore'):        # nan/inf % 1 -> nan, silently
                int_valued = np.all(perm % 1 == 0)     # nan/inf != 0 -> still rejected
            if not int_valued:
                raise ValueError("Permutation must have integer values.")
            perm = perm.astype(np.intp).tolist()
        else:
            raise ValueError("Permutation must be integer-valued.")
    N = len(perm)
    if validate and (set(perm) != set(range(N))):
        raise ValueError("Invalid permutation.")
    # Pure Python list is faster here than NumPy array
    visited = [False] * N
    n_cyc = 0
    for i in range(N):
        if not visited[i]:
            n_cyc += 1
            j = i
            while not visited[j]:
                visited[j] = True
                j = perm[j]
    # Bitwise AND (& 1) is slightly faster than modulo (% 2)
    return -1 if (N - n_cyc) & 1 else 1

def cycle_decomp(perm, include_fixed=False, returnParity=False, validate=False):
    if isinstance(perm, np.ndarray):
        if np.issubdtype(perm.dtype, np.integer):
            perm = perm.tolist()
        elif np.issubdtype(perm.dtype, np.floating):
            with np.errstate(invalid='ignore'):        # nan/inf % 1 -> nan, silently
                int_valued = np.all(perm % 1 == 0)     # nan/inf != 0 -> still rejected
            if not int_valued:
                raise ValueError("Permutation must have integer values.")
            perm = perm.astype(np.intp).tolist()
        else:
            raise ValueError("Permutation must be integer-valued.")
    N = len(perm)
    if validate and (set(perm) != set(range(N))):
        raise ValueError("Invalid permutation.")
    visited = [False] * N
    cycles = []
    n_cyc = 0
    for i in range(N):
        if not visited[i]:
            n_cyc += 1
            j, cyc = i, []
            while not visited[j]:
                visited[j] = True
                cyc.append(j)
                j = perm[j]
            if include_fixed or len(cyc) > 1:
                cycles.append(cyc)
    if returnParity:
        return cycles, (-1 if (N - n_cyc) & 1 else 1)
    return cycles


# computes parity of permutation required to sort the array
# equivalent to (-1)^{# inversions to bubble sort}
def parity_sort(arr):
    sortOrder = np.argsort(arr)
    # sortOrder[i] = source index of ith elem sorted array --> where sortedarr[i] comes from
    # perm = inverse of sortOrder since want perm[i] = where i goes
    perm = np.zeros_like(arr)
    perm[sortOrder] = np.arange(len(arr))
    return perm_parity_cyc(perm)

# counts number of inversions needed to bubble sort the array
# can be used to compute parity if array is a permutation using (-1)^{# inv} 
# but is less efficient if arr is not directly a perm, e.g. arr = [1,4,2] -> parity = -1 b/c sorted is [1,2,4]
def num_inversions(arr):
    arr = np.asarray(arr)
    inv = 0
    for i in range(len(arr)):
        inv += np.sum(arr[i] > arr[i+1:]) # increment inv everytime a digit is out of place and have to move it to the right
    return -1 if inv % 2 else +1


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GENERAL UTILITY 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# returns the nth row of Pascal's triangle
def pascal(n):
    return np.array([mth.comb(n,k) for k in range(n+1)])

# linearly maps values from one array x into range of y
def linmap(x,y):
    x0,xm = min(x),max(x)
    y0,ym = min(y),max(y)
    xm = ((x-x0)/(xm-x0))*(ym-y0) + y0
    return xm

# Finds index of sublist in nested list (or list of arrays) that contains a particular element
# e.g. find 4 in [[0,1,2],[3,4],[5,6,7]] --> outputs 1
def find_sublist_index(elem,nestedList):
    for i,sublist in enumerate(nestedList):
        if elem in sublist:
            return i
    return None

# Fills the diagonals of an array with a constant value
# offset: integer representing which diagonal from the main diagonal
# for N x N array, the main diagonal has offset 0 and the possible offsets range from [-(N-1),N-1]
def fill_diagonal_offset(arr,val,offset=0):
    rows, cols = arr.shape
    if offset >= 0:
        # Upper diagonal or main diagonal
        diag_len = min(rows, cols - offset)  # Length of the diagonal to fill
        arr[np.arange(diag_len), np.arange(offset, offset + diag_len)] = val
    else:
        # Lower diagonal
        diag_len = min(rows + offset, cols)  # Length of the diagonal to fill
        arr[np.arange(-offset, -offset + diag_len), np.arange(diag_len)] = val

    return arr

# normalizes last axis of X
def normalize(X):
    idcs = np.arange(len(X.shape))
    cr = circshift(idcs,1)
    cl = circshift(idcs,-1)
    return np.transpose((np.transpose(X,cr)/la.norm(X,axis=-1)),cl)

# get n geometrically spaced samples from min x_0 to max x_{n-1}
# x_k = x_0 b^k where x_{n-1} = x_0*b^{n-1} ==> b = (max/min)^{1/(n-1)} 
def logspace(n,x0=2*np.pi,xm=2*np.pi*(10**4)):
    b = (xm/x0)**(1/(n-1))
    nvec = np.arange(0,n)
    xk = x0*(b**nvec)
    return xk

# returns a piecewise linear func with arbitrary # of transition points
# y0 = y-intercept of line that intersects 0
# tlist = list of x positions of transition points
# mlist = list of slopes of each line
# nth line: m_n x + (m_{n-1} - m_n)t_{n-1} + .... + (m_1 - m_2)t_1 + y0
def pwlin(x, y0, tlist, mlist):
    if len(mlist) != len(tlist)+1:
        raise Exception("len(mlist) =/= len(tlist)+1 !")
    conds = [x < tlist[0]] + [(tlist[i] <= x)*(x < tlist[i+1]) for i in np.arange(len(tlist)-1)] + [x>=tlist[-1]]
    # NOTE: idx i is not looked up until lambda is called, so must bind the i of lambda func to the for loop i
    funcs = [lambda x, i=i: y0 + mlist[i]*x + np.sum([(mlist[j]-mlist[j+1])*tlist[j] for j in np.arange(i)]) for i in range(len(mlist))]
    return np.piecewise(x, conds, funcs)

# y0 = y-intercept of line that crosses y-axis
# t = transition point
# m1, m2 are slopes
def pwlin2(x, y0, t, m1, m2):
    return (x<t)*(y0 + m1*x) + (x>=t)*(m2*x + (m1-m2)*t + y0)


# Groups 2D array A by values in a given row
# Inputs 
    # A: np array of size (n ≥ 2,m)
# Returns
    # dict where key = unique elems in row r, i.e. A[r,...] and value = block of A associated to that key; size (n-1,m)
def group_by_row(A, r):
    row  = A[r]                          # the row that defines the grouping
    other = np.delete(A, r, axis=0)       # drop row r  -> shape (n-1, ...)
    order = np.argsort(row, kind="stable")
    sorted_row = row[order]
    change_idcs = np.flatnonzero(sorted_row[1:] != sorted_row[:-1]) + 1 # idcs where sorted_row changes value -> new unique key
    groups = np.split(other[:, order], change_idcs, axis=1) # split cols based on unique keys to get groups
    keys = sorted_row[np.concatenate(([0], change_idcs))]
    return dict(zip(keys, groups))

# O(N) unique elements (and optionally counts) for sorted array
def unique_in_sorted(arr, returnCounts=False):
    arr = np.asarray(arr)
    if arr.size == 0:
        return (np.array([], dtype=arr.dtype), np.array([], dtype=int)) if returnCounts else np.array([], dtype=arr.dtype)
    mask = np.concatenate(([True], arr[1:] != arr[:-1]))
    uniq = arr[mask]
    if returnCounts:
        indices = np.flatnonzero(mask)
        counts = np.diff(np.concatenate((indices, [arr.size])))
        return uniq, counts
    return uniq

# Computes Cartesian product of 2 arrays x and y
# Equivalent to 2 cols of np.array(list(it.prod(x,y))) but is faster
# e.g. x = [x1,x2,x3], y = [y1,y2,y3] -> output = [x1,x1,x1,x2,x2,x2,x3,x3,x3], [y1,y2,y3,y1,y2,y3,y1,y2,y3]
def cartesian_prod(x,y):
    return np.repeat(x, y.size), np.tile(y, x.size)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# COMPARISON CHECKS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def maxposdiff(a,b):
    if a.shape!=b.shape:
        raise Exception("Inputs have incompatible sizes!")
    return np.max(np.abs(a-b))

# check if two elements or arrays are equal to a certain precision
def isapproxeq(a,b,tol=1e-15):
    return maxposdiff(a,b)<=tol

def isdiag(A,tol=1e-15):
    return isapproxeq(A,np.diag(A.diagonal()),tol=tol)

# checks if real to certain precision
def isreal(a,tol=1e-15):
    return isapproxeq(a,a.real,tol=tol)

# checks if Hermitian
def isherm(a,tol=1e-15):
    return isapproxeq(a,a.T.conj(),tol=tol)

# checks if symmetric
def issymm(a,tol=1e-15):
    return isapproxeq(a,a.T,tol=tol)

# checks if unitary to certain precision
def isunitary(U,tol=1e-15):
    return isapproxeq(U@np.conj(U.T),np.eye(U.shape[0]),tol=tol)

# Checks if a matrix A is block diagonal with a given vector of block sizes
# A: 2D array/matrix
# bs: vector of block sizes, bs[i] = size of ith block
# if A is block diag with block sizes bs, then n
# Returns: bool
def isBlockDiag(A,bs,tol=1e-15):
    if sum(bs) != A.shape[0] or A.shape[0] != A.shape[1]:
        raise ValueError(f"Block sizes {bs} must sum to matrix dimension {A.shape[0]}, and A must be square.")

    # Assign a block label to every row/column index, e.g. [0,0,1,2,2] for bs=[2,1,2]
    block_labels = np.repeat(np.arange(len(bs)), bs)

    # on_block[i,j] is True iff (i,j) belongs to a diagonal block
    on_block = block_labels[:, None] == block_labels[None, :]

    return not np.any(np.abs(A[~on_block]) > tol)

# given a numpy array of sparse matrices, compares entries elementwise
# vectorizes the function for object arrays
sp_equals = np.vectorize(lambda a,b: (a!=b).nnz==0)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# COMBINATORIAL GENERATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# get all possible binary strings of size n
# returns a list of bit strings, e.g. ['000','001',...]
def getBinCombos(N,returnArray=False):
    return list(map(dec2bin,np.arange(2**N),N*np.ones((2**N))))

# given a finite alphabet (list of chars), generate all length-N strings using that alphabet
# returns list of size len(alphabet)^N
# FASTER than recursive methods for strings
def genStrings(alphabet,N):
    return [''.join(x) for x in it.product(alphabet, repeat=N)]

# given an alphabet of arrays/tokens, generate all length-N combinations of the arrays
# returns array of (len(alphabet))^N x len(alphabet) x dim(tok)
# alphabet = list or np.array of tokens, which are themselves scalars or np arrays
# recursive -- FASTER than it.product()
def genTokenCombos_r(alphabet,N):
    alphabet = np.array(alphabet) # alphabet = N x dim(tok)
    tokenCombos = np.zeros((len(alphabet)**N,)+alphabet.shape)
    k=0
    def genTokenCombos_helper(config,n): # k = idx in [1,len(alph)^N], n = # of remaining tokens in string of length N
        if n==0:
            nonlocal k
            tokenCombos[k] = config
            k+=1
            return
        for tok in alphabet:
            config[N-n,:] = tok
            genTokenCombos_helper(config,n-1)
    genTokenCombos_helper(np.zeros((N,)+tokdim),N)
    return tokenCombos

# given an alphabet of arrays/tokens, generate all length-N combinations of the arrays
# returns array of (len(alphabet))^N x len(alphabet) x dim(tok)
# alphabet = list or np.array of tokens, which are themselves lists or arrays
# recursive -- faster than list iteration for vectors
def genTokenCombos(alphabet,N):
    alphabet = np.array(alphabet) # alphabet = N x dim(tok)
    tokenCombos = np.array(list(it.product(alphabet,repeat=N)))
    return tokenCombos


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BIT MANIPULATIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Helper method for computing XOR on all bits of np integer type or for all such elements in np array
def fastXOR(x):
    # Determine the bit-width of the input array, e.g. 32-bit integer
    # .itemsize returns bytes; multiply by 8 for bits
    bit_width = x.itemsize * 8
    # Perform dynamic bit-folding --- start the shift at half the bit_width and divide by 2 each step
    shift = bit_width // 2
    while shift > 0:
        x ^= x >> shift
        shift //= 2
    return x & 1 # Mask the LSB to get the parity of the original bits

# converts a number x into its binary representation with N bits
# returnType (optional): string or np array
def dec2bin(x,N,returnType='list'):
    if x > 0 and np.log2(x)>=N:
        raise Exception("insufficient bit width N")
    if returnType=='list':
        return [int(b) for b in np.binary_repr(x,N)]
    else:
        return np.binary_repr(x,N)

# Convert decimal to nD binary pattern
# raveled index of each site in nD lattice = corresponding power of 2
# x: int in [0,2^N-1] where N = tot # sites = prod_i L_i
# Lvec: dimension of nD lattice; e.g. (Lx,Ly)
def dec2binnD(x,Lvec):
    N = np.prod(Lvec)
    return np.reshape(np.unpackbits(np.frombuffer(int(x).to_bytes((N + 7) // 8, byteorder='little'), dtype=np.uint8), bitorder='little')[:N],Lvec)

# Convert nD binary pattern into decimal 
# raveled index of each site in nD lattice = corresponding power of 2
# xarr: nD bit pattern as np array
def binnD2dec(xarr):
    return int.from_bytes(np.packbits(xarr.ravel(), bitorder='little'), byteorder='little')

# a = number; n = number of bits in representtion
def flipBits(a,n,returnType="int"):
    if returnType=="int":
        return int(bin((a ^ (2 **(n+1) - 1)))[3:],2)
    else:
        return bin((a ^ (2 **(n+1) - 1)))[3:]

# N = num bits
# x = integer representation of a binary configuration of N bits
# by default, shifts bits to the RIGHT. To shift to LEFT, shift value should be negative
# returns integer representation of shifted bit config of x
def circshiftBits(x,s,N):
    x = x%(2**N)  # ensures that x is a valid integer representation
    if not (isinstance(s, int) or s.is_integer()):
        raise Exception("shift amount not integer!")
    # s = s % N
    s = int(s)
    if s==0:
        return x
    elif s<0: # shift to left
        return ((x<<np.abs(s)) & (2**N-1)) | ((x & (2**N-1)) >> (N-np.abs(s)) )
    else: # s>=0: shift to right
        return ((x & (2**N - 1)) >> s) | ((x << (N-s)) & (2**N-1))

# Shifts nD bit pattern by t places along axis a
# x: integer rep of bit pattern
# Lvec: dimensions of nD bit pattern
# e.g. x = 13, Lvec = (2,3) -> bitrep(x) = [[0,0,1],[1,0,1]] 
# t: # of shifts
# axis: axis along which to shift; 0,1,...,len(Lvec)-1
# Note: np.roll rolls rows down and cols right; in general, rolls axis in direction of increasing index
def circshiftBitsnD(x,t,a,Lvec,returnBitRep=False):
    # get nD bit array rep
    bitrep = np.reshape(np.unpackbits(np.frombuffer(int(s).to_bytes((N + 7) // 8, byteorder='little'), dtype=np.uint8), bitorder='little')[:N],Lvec)
    print(bitrep)
    # apply transl along certain axis
    shiftedBitRep = np.roll(bitrep,t,axis=a)
    print(shiftedBitRep)
    # convert back to int
    xnew = int.from_bytes(np.packbits(shiftedBitRep.ravel(),bitorder='little'))
    
    if returnBitRep:
        return xnew, shiftedBitRep
    else:
        return xnew

# helper method: applies a perm on N sites to arbitrary N-bit string state
# s: int in [0,2^N-1]; represents bit string state
# ss_perm: perm on N elements
def apply_perm_to_bitstring(s,ss_perm):
    out = 0
    for i in range(len(ss_perm)):
        if (s >> i) & 1: # checks if bit i is 1 by shifting it all the way to the right and comparing it to 1
            out |= (1 << ss_perm[i])  # adds 2^[new loc of i]
    return out

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PERIODIC RING METHODS 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# circularly shifts lists by n elements to the RIGHT
# for LEFT shift, make shift amount n negative
def circshift(arr,n=1,direction="right"):
    arr = list(arr)
    if n<0:
        return arr[np.abs(n)::] + arr[:np.abs(n):] 
    else:
        return arr[-n:] + arr[0:-n] 

# Same as np.roll but 3x faster
# x: input arr
# n: shift amount; if neg, rolls to left
def roll(x, n, axis=-1):
    # Get the length of the target axis
    axis_len = x.shape[axis]
    n = n % axis_len
    if n == 0:
        return x.copy()
    xs = np.empty_like(x) # np.empty_like is faster than np.zeros_like since we overwrite it anyway
    
    # Create lists of slice(None) - this is the equivalent of ":" in array indexing
    idx_out_1 = [slice(None)] * x.ndim
    idx_in_1 = [slice(None)] * x.ndim
    idx_out_2 = [slice(None)] * x.ndim
    idx_in_2 = [slice(None)] * x.ndim
    
    # Shifted part
    idx_out_1[axis] = slice(n, None)
    idx_in_1[axis]  = slice(None, -n)
    
    # Wrapped around part
    idx_out_2[axis] = slice(None, n)
    idx_in_2[axis]  = slice(-n, None)
    
    # Apply the slices as tuples
    xs[tuple(idx_out_1)] = x[tuple(idx_in_1)]
    xs[tuple(idx_out_2)] = x[tuple(idx_in_2)]
    
    return xs

# Shifts an array x by n places, and pads with 0's or other given fill value
def shiftarr(x, n, axis=-1, fill=0):
    axis_len = x.shape[axis]
    if n == 0:
        return x.copy()
    if abs(n) >= axis_len:            # shifted entirely off the end
        return np.full_like(x, fill)

    xs = np.empty_like(x)
    idx_out = [slice(None)] * x.ndim
    idx_in  = [slice(None)] * x.ndim
    idx_pad = [slice(None)] * x.ndim

    if n > 0:                         # toward higher indices
        idx_out[axis] = slice(n, None);  idx_in[axis] = slice(None, -n);  idx_pad[axis] = slice(None, n)
    else:                             # toward lower indices
        idx_out[axis] = slice(None, n);  idx_in[axis] = slice(-n, None);  idx_pad[axis] = slice(n, None)

    xs[tuple(idx_out)] = x[tuple(idx_in)]    # main block — same copy your roll does
    xs[tuple(idx_pad)] = fill                # was the wrap-copy in roll; now a fill
    return xs

def levicivita(idx):
    p = 1
    for a, b in combinations(idx, 2):
        if   b > a: continue        # sgn(+) -> *1
        elif b < a: p = -p          # sgn(-) -> *(-1)
        else:       return 0        # equal pair -> repeated index -> 0
    return p

def levicivita_tensor(n):
    g = np.indices((n,) * n)                 # g[k] = k-th coordinate
    eps = np.ones((n,) * n, dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            eps *= np.sign(g[j] - g[i])
    return eps


# finds the relative angle between two angular positions
def getThetaRel(th1,th2):
    return abs((th1%(2*np.pi))-th2%(2*np.pi))

# Given a cwise interval defined by [iL,iR] on a ring of sites [0,L-1], enumerates all the sites contained in the interval
# Returns list ([iL,iL+1,...,iR] % L)
def enumerateCwiseInterval(iL,iR,L):
    if not (0 <= iL < L and 0 <= iR < L):
        raise ValueError("iL and iR must be in the range [0, L-1]")
    if iL <= iR:
        return list(range(iL, iR + 1))
    else:
        return list(range(iL, L)) + list(range(0, iR + 1))

# Given a list/array/set of sites that presumably form one contiguous interval, returns the cwise interval bounds as (start,end)
def repAsCwiseInterval(sites,L):
    sites = {x%L for x in sites} # ensures sites are given mod L and there are no duplicates by forcing it to be a set
    size = len(sites)
    for start in sites: # try each element in sites to see if is the start of the interval
        candidate_sites = np.arange(start,start+size) % L
        if set(sites)==set(candidate_sites):
            return (start,candidate_sites[-1])
    raise ValueError("Sites don't form one contiguous interval.")

# Sorts a list/array of sites on a ring of size L by clockwise distance from a reference site (default is 0)
# Note: ref site does NOT have to be in the sites given!
# sites are labeled from 0,1,...,L-1
def sortCwiseFrom(sites, L, ref=0):
    sites = [s % L for s in sites]
    return sorted(sites, key=lambda s: (s - ref) % L)

# Returns the size of the clockwise interval from l to r on a ring of size L
def cwise_dist(iL, iR, L):
    # return (iR - iL + 1) % L if iL <= iR else (L - iL + iR + 1)
    return (iR - iL) % L

# Checks whether a particular site x is contained in a cwise interval on a ring of size L
# x: site; in {0,1,...,L-1}
# interval: of the form (iL,iR)
# L: tot # sites
def checkInIntervalRing(x,interval,L):
    iL,iR = interval
    # if iR>= iL:
    #     return iL <= x <= iR
    # else: # interval passes through 0
    #     return x>=iL or x<=iR # Note: b/c iR<iL, (x>=iL and x>=iR) reduces to x>=iL and (x<=iR and x<=iL) reduces to x<=iR
    return cwise_dist(iL,x,L) <= cwise_dist(iL,iR,L)

# Computes the intersection between two cwise intervals on a ring of size L
# Intersection is either None, one contiguous cwise interval, or two disjoint cwise intervals
def intersectionOnRing(interval1, interval2, L):
    if not isinstance(interval1,tuple):
        interval1 = tuple(interval1)
    if not isinstance(interval2,tuple):
        interval2 = tuple(interval2)
    iL1, iR1 = interval1
    iL2, iR2 = interval2
    
    # Trivial/special cases
    if interval1 == interval2:
        return tuple(interval1)
    # full‑ring intervals
    if cwise_dist(iL1, iR1, L) == L-1:
        return tuple(interval2)
    if cwise_dist(iL2, iR2, L) == L-1:
        return tuple(interval1)
    
    # Check whether each interval’s start lies inside the other
    L1in2 = checkInIntervalRing(iL1, interval2, L)
    L2in1 = checkInIntervalRing(iL2, interval1, L)
    
    # no overlap
    if not (L1in2 or L2in1):
        return None
    
    R1in2 = checkInIntervalRing(iR1,interval2,L)
    R2in1 = checkInIntervalRing(iR2,interval1,L)
    in2 = L1in2 and R1in2
    in1 = L2in1 and R2in1
    
    if in2 and in1: # each interval contains the endpoints of the other, but are not the same interval
        # Two disjoint intervals
        return ((iL1,iR2),(iL2,iR1))
    elif in2 and not in1: # 1 is subinterval of 2
        return tuple(interval1)
    elif in1 and not in2: # 2 is subinterval of 1
        return tuple(interval2)
    elif (R1in2 and L2in1 and not L1in2 and not R2in1):
        return (iL2,iR1) # 1 shifted to the left of 2
    else: # (R2in1 and L1in2 and not L2in1 and not R1in2)
        return (iL1,iR2) # 2 shifted to the left of 1

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LATTICE COORD METHODS 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Inputs
    # Lvec: lattice dimension; size-d array/list/tuple where d = # dimensions
    # a (optional): lattice spacing along each axis (if size = Lvec); if integer, assumes uniform spacing
# Returns
    # coords: np array of coords of size (prod_i L_i) x d
def getLatticeCoord(Lvec,a=1):
    return a*np.indices(Lvec).reshape(len(Lvec),-1).T

# returns matrix of cosine angles between pairs of vectors
# coord = N x d where N = # vecs, d = dim of each vec
def pairwiseAngle(coord):
    dp = coord@coord.T  # matrix of dot products
    return np.arccos(dp)

# returns matrix of distances between each pair of vecs in coords
# coord = N x d where N = # vecs, d = dim of each vec
def pairwiseDist(coord):
    sqdist = np.sum([(xi[np.newaxis]-xi[np.newaxis].T)**2 for xi in coord.T],axis=0)
    r_ij = np.sqrt(sqdist)
    return r_ij

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SAMPLING 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Sample vectors from Haar measure on U(d) if complex or O(d) if real (i.e. uniformly on S^(d-1))
# d = dim of vectors returned
# n = num of vectors to sample
# returns 2d array of size n x d where the ROWS are the vectors, i.e. X[0] = 0th vec, X[1] = 1st vec, and so on
def Haarvec(d,n=1,real=False,seed=None):
    if seed is not None:
        np.random.seed(seed)
    if real:
        X = np.random.normal(0,1,(n,d))
    else:
        X = np.random.randn(n,d) + 1j * np.random.randn(n,d)

    return X/la.norm(X,axis=-1,keepdims=True)

# Sample n orthogonal matrices from Haar measure on O(d)
# returns array of size n x d x d if n=#samples>1, else returns d x d orthogonal matrix
def COE(d,n=1,seed=None):
    if seed is not None:
        np.random.seed(seed)
    M = np.random.normal(0,1,(n,d,d))
    Qarr,Rarr= np.linalg.qr(M)
    # to make decomp unique, choose convention that R has positive diagonal entries
    sgnR = np.array([np.sign(np.diag(R)) for R in Rarr])
    Qparr = Qarr*np.expand_dims(sgnR,axis=1)
    return Qparr if n>1 else np.squeeze(Qparr,axis=0)

# Sample n unitary matrices from Haar measure on U(d)
# returns array of size n x d x d if n=#samples>1, else returns d x d unitary matrix
def CUE(d,n=1,seed=None):
    if seed is not None:
        np.random.seed(seed)
    M = np.random.normal(0,1,(n,d,d)) + 1j*np.random.normal(0,1,(n,d,d))
    Qarr,Rarr = np.linalg.qr(M)
    # to make decomp unique, choose convention that R has all positive diagonals
    sgnR = np.array([np.sign(np.diag(R)) for R in Rarr])
    Qparr = Qarr*np.expand_dims(sgnR,axis=1)
    return Qparr if n>1 else np.squeeze(Qparr,axis=0)

# Sample n GOE real random Hermitian matrices of size N x N
# sigsq = N*(variance of off diagonal elements) --> should be O(1) => var ~ 1/N
# var(off diag) = σ^2/2N
# var(diag) = σ^2/N
def GOE(N,sigsq,n=1,seed=None):
    if seed is not None:
        np.random.seed(seed)
    M = np.random.normal(0,np.sqrt(sigsq/(4*N)),size=(n,N,N))
    if n==1:
        return (M + np.transpose(M,axes=[0,2,1]))[0]
    else:
        return M + np.transpose(M,axes=[0,2,1])

# Sample n GUE complex random Hermitian matrices of size N x N
# sigsq = N*(variance of off diagonal elements) --> should be O(1) => var ~ 1/N
# var(off diag) = σ^2/2N
# var(diag) = σ^2/N
def GUE(N,sigsq,n=1,seed=None):
    if seed is not None:
        np.random.seed(seed)
    Mr = np.random.normal(0,np.sqrt(sigsq/(8*N)),size=(n,N,N))
    Mi = 1j*np.random.normal(0,np.sqrt(sigsq/(8*N)),size=(n,N,N))
    M = Mr + Mi
    if n==1:
        return (M + np.transpose(M,axes=[0,2,1]).conj())[0]
    else:
        return M + np.transpose(M,axes=[0,2,1]).conj()

def gaussianFunc(x, mean, A, var):
    return A*np.exp(- (x - mean)**2 / (2*var))

# CARTESIAN: params = (mean real, var real, mean imag, var imag)
# POLAR: params = (mean R, std R), angle is chosen uniformly
def getRandomGaussComplex(params,shape,seed=None):
    if seed is not None:
        np.random.seed(seed)
    if len(params)==2:  # POLAR
        return np.random.normal(loc=params[0],scale=np.sqrt(params[1]),size=shape)*np.exp(1j*np.random.uniform(low=-np.pi,high=np.pi,size=shape))
    else:   # CARTESIAN
        return np.random.normal(loc=params[0],scale=np.sqrt(params[1]),size=shape) + 1j*np.random.normal(loc=params[2],scale=np.sqrt(params[3]),size=shape)

# CARTESIAN: params = (lowBnd_real, highBnd_real, lowBnd_imag, highBnd_imag) OR (lowBnd R, highBnd R)
# POLAR: params = (mean R, std R), angle is chosen uniformly
def getUniform(params,shape,seed=None):
    if seed is not None:
        np.random.seed(seed)
    if len(params)==2: # POLAR
        return np.random.uniform(low=params[0],high=params[1],size=shape)*np.exp(1j*np.random.uniform(low=-np.pi,high=np.pi,size=shape))
    else:
        return np.random.uniform(low=params[0],high=params[1],size=shape)+ 1j*np.random.uniform(low=params[2],high=params[3],size=shape)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LINALG 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Finds decomposition of A = R Λ R^{-1} = R Λ L^T where L^T = R^{-1}, i.e.  vL[:,i]^† Λ vR[:,j] = δ_{ij}
# A: 2d np array
def biorthogonal_eig(A):
    eigs, vL, vR = la.eig(A, left=True, right=True)
    # Normalize each pair so that 
    for i in range(len(eigs)):
        scale = np.vdot(vL[:, i], vR[:, i])  # = vL[:, i].conj().T @ vR[:, i]
        vL[:, i] /= scale
    return eigs, vL, vR

# Converts block diagonal matrix into sparse sp.block_diag object
# A: 2D np array or matrix
# bSizes: list or vector of block sizes of A
def densetoBlockDiag(A, bSizes):
    # 1. Calculate the split points (exclude the last index)
    indices = np.cumsum(bSizes)[:-1]

    # 2. Split the matrix horizontally (into columns blocks)
    # Then split each of those results vertically (into row blocks)
    # Using a list comprehension here is faster than a manual loop
    blocks = [A[i:j, i:j] for i, j in zip(np.insert(indices, 0, 0), np.append(indices, matrix.shape[0])) ]

    return sp.block_diag(blocks)

def isBlockDiag(A,bs,tol=1e-15):
    if sum(bs) != A.shape[0] or A.shape[0] != A.shape[1]:
        raise ValueError(f"Block sizes {bs} must sum to matrix dimension {A.shape[0]}, and A must be square.")

    # Assign a block label to every row/column index, e.g. [0,0,1,2,2] for bs=[2,1,2]
    block_labels = np.repeat(np.arange(len(bs)), bs)

    # on_block[i,j] is True iff (i,j) belongs to a diagonal block
    on_block = block_labels[:, None] == block_labels[None, :]

    return not np.any(np.abs(A[~on_block]) > tol)

# Gram-Schmidt basis -- returns a matrix of col vecs, where each col vector is orthogonal to v
# if v is d-dim, then matrix is d x (d-1) dim
def gm_basis(v):
    d = len(v)
    Id = np.eye(d)
    X = [v/la.norm(v)]
    for i in range(d-1):
        coeffs = X[0].conj().T@np.array(X).T
        x = Id[:,i]-np.sum(np.einsum('i,ij->ij',np.array(X)[:,i],np.array(X)),axis=0)
        x = x/la.norm(x)
        X = X + [x]

    return np.array(X).T


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CONVERSIONS 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# converts ndarray into df
# diff axes will be the indices of the df, the value of the elements of the array will be the last col of the df
# x = ndarray
# clabels = DICT that contains the names of the axes, e.g. {0: 'i', 1:'j', 2='k'}
# construct dict where each of the axes yields a column for the idx, plus a last column for the data
# Note: the name of the data col is referred to by axis idx len(x.shape), e.g. x = 2x3x5 ==> data col idxed by 3
# irange = DICT that contains the ORDERED values that each idx value maps to, e.g. can have axis0 idcs 1,2,3 --> axis0 vals -1,0,-1
def arr2df(x, clabels = {}, irange={},DEBUG_MODE=True):
    numIdcs = len(x.shape)
    clabels = clabels.copy()
    irange = irange.copy()
    for n in set(range(numIdcs)).difference(set(clabels.keys())):
        clabels.update({n: 'i%d' % n})
    if numIdcs+1 not in clabels.keys():
        clabels.update({numIdcs+1: 'data'})

    for n in set(range(numIdcs)).difference(set(irange.keys())):
        irange.update({n: range(x.shape[n])})
    # convert dictionaries into an ordered list of axes values and col labels
    clabels = [i[1] for i in sorted(clabels.items())]
    axesvals = [i[1] for i in sorted(irange.items())]

    cols = list(zip(*it.product(*axesvals)))
    cols.append(tuple(x.ravel()))
    
    dataDict = dict(zip(clabels,cols))
    if DEBUG_MODE: print(dataDict)
    df = pd.DataFrame.from_dict(dataDict)
    
    return df

# convert nD array with n dimensions into df
# the first n-1 axes index the cols, the last axes of size L_n will yield L_n data cols in the df
# clabels is a dictionary of the col labels
def arr2dfn(x,irange={},clabels={},ilabels={}):
    numIdcs = len(x.shape)-1
    numCols = x.shape[-1]
    ilabels = ilabels.copy()
    irange = irange.copy()
    clabels = clabels.copy()
    
    # give names to idx cols that don't have names already
    for i in set(range(numIdcs)).difference(set(ilabels.keys())):
        ilabels.update({i: "i%d"%i})
        
    # give names to data cols that don't have names already
    for j in set(range(numCols)).difference(set(clabels.keys())):
        clabels.update({j: "x%d"%j})
    
    # get range of idx values for axes that don't have assigned values already
    for n in set(range(numIdcs)).difference(set(irange.keys())):
        irange.update({n: range(x.shape[n])})
    
    # create multi-Index from dictionaries by first converting into an ordered list of idx labels and idx ranges
    iLabels = [i[1] for i in sorted(ilabels.items())]
    iVals = [i[1] for i in sorted(irange.items())]
    mi = pd.MultiIndex.from_product(iVals,names=iLabels)
    
    # create df with multiindex
    data = np.array([x[:,:,i].ravel() for i in clabels.keys()]).T
    df = pd.DataFrame(data,mi,[clabels[i] for i in sorted(clabels)])
    # print(df)
    
    return df

# Input: df of n cols where first n-1 cols are index cols and last col is data col
# Convert to multidimensional array
def df2arr(df):
    header = list(df.columns)
    grouped = df.groupby(header[:-1])[header[-1]].mean()
    idcs = [grouped.index.unique(level=k).to_numpy() for k in np.arange(grouped.index.nlevels)]
    return grouped2arr(grouped),idcs

# grouped df to array
def grouped2arr(grouped):
    arr = np.full(tuple(map(len,grouped.index.levels)),np.nan) # create empty NaN array of appropriate size
    arr[tuple(grouped.index.codes)] = grouped.values.flat
    return arr

# turns decimals into string with 'p' instead of decimal place, useful for filenames
# sci_thresh: number of trailing zeros or zeros after decimal before switch to scientific notation
# 1.000 -> "1"
# e.g. 0.5 --> "0p5"
# 1.00e-5 --> "1e5"
def dec2str(x,sci_thresh=5):
    mantissa, exp = f"{x:e}".split("e")
    if abs(int(exp))>=sci_thresh:
        mantissa = mantissa.rstrip("0").replace('.','p')
        exp = exp.replace("-0", "-").replace("+0","+")
        return "%se%s" % (mantissa,exp)
    else:
        if float(x).is_integer():
            return str(int(x))
        else:
            return str(x).replace('.','p')

def str2dec(s):
    return float(s.replace('p','.'))

def dec2latexfrac(x):
    # rational approximation
    ratio = float(x).as_integer_ratio()
    if ratio[1]==1:
        return str(ratio[0])
    else:
        return r"$\frac{%d}{%d}$" % ratio
        # return "%d/%d" % ratio


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PLOTTING 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Imitate Matlab's parula colormap ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905],  [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143],  [0.1959047619, 0.2644571429, 0.7279], 
[0.1707285714, 0.2919380952, 0.779247619], [0.1252714286, 0.3242428571, 0.8302714286],  [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238,  0.8819571429], 
[0.0059571429, 0.4086142857, 0.8828428571], [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571,   0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429],  
[0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 0.8467], [0.0779428571, 0.5039857143, 0.8383714286],  [0.079347619, 0.5200238095, 0.8311809524], 
[0.0749428571, 0.5375428571, 0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 0.819852381], 
[0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 0.7607190476], 
[0.0383714286, 0.6742714286, 0.743552381], [0.0589714286, 0.6837571429, 0.7253857143], [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
[0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], [0.2586428571, 0.7317142857, 0.5954285714], 
[0.3021714286, 0.7376047619, 0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 0.5033142857], 
[0.4871238095, 0.7490619048, 0.4839761905], [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], [0.6473, 0.7456, 0.4188], 
[0.6834190476, 0.7434761905, 0.4044333333], [0.7184095238, 0.7411333333, 0.3904761905], [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 0.3632714286], 
[0.8185047619, 0.7327333333, 0.3497904762], [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], [0.9139333333, 0.7257857143, 0.3062761905],
[0.9449571429, 0.7261142857, 0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 0.2164142857], 
[0.9955333333, 0.7860571429, 0.196652381], [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
[0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 0.0948380952], [0.9661, 0.9514428571, 0.0755333333], [0.9763, 0.9831, 0.0538]]
parula = LinearSegmentedColormap.from_list('parula', cm_data)

def getLinearColormap(from_rgb,to_rgb):
    # from color r,g,b
    r1,g1,b1 = from_rgb

    # to color r,g,b
    r2,g2,b2 = to_rgb

    cdict = {'red': ((0, r1, r1),
                   (1, r2, r2)),
           'green': ((0, g1, g1),
                    (1, g2, g2)),
           'blue': ((0, b1, b1),
                   (1, b2, b2))}

    cmap = LinearSegmentedColormap('custom_cmap', cdict)
    return cmap

# get subset of colormap as new colormap
# minval: lower bound in [0,1] from old cmap
# maxval: upper bound in [0,1] from old cmap
# n (opt): number of colors; default is 256
def subcmap(cmap,minval=0,maxval=1,n=256):
    new_cmap = LinearSegmentedColormap.from_list('{n}_trunc({a:g},{b:g})'.format(n=cmap.name, a=minval, b=maxval),cmap(np.linspace(minval, maxval, n)),n)
    return new_cmap

# vals: values to assign to discretized colorbar
# minval: lower bound in [0,1] from old cmap
# maxval: upper bound in [0,1] from old cmap
def sm_cmap(vals,cmapname,minval=0,maxval=1):
    n = len(vals)
    cmap = subcmap(plt.get_cmap('plasma'),minval,maxval,n)
    sm = plt.cm.ScalarMappable(cmap=cmap);
    dval = np.min(vals[1:]-vals[:-1])/2 # minimum difference in values = discretization in value
    sm.set_array(np.linspace(min(vals)-dval,max(vals)+dval,len(vals)));
    return sm,cmap

# return discrete colorbar object
# vals: values associated to each discrete color; ASSUMES EQUALLY SPACED
# cmap: colormap
# places value labels at center of each discrete pixel
def addDiscreteColorbar(vals,cmap,labels=None,labelsize=None,fig=None,ax=None,**kwargs):
    sm = plt.cm.ScalarMappable(cmap=cmap);
    dval = np.min(vals[1:]-vals[:-1])/2 # minimum difference in values = discretization in value
    sm.set_array(np.linspace(min(vals)-dval,max(vals)+dval,len(vals)));
    if not fig:
        fig = plt.gcf()
    if not ax:
        ax = plt.gca()

    if labels is None:
        cbar = fig.colorbar(sm,ax=ax);
    else:
        cbar = fig.colorbar(sm,ax=ax,ticks=labels,**kwargs);

    return cbar,sm,dval

# returns image of matrix with nonzero elements colored black
def getIm(H):
    Im = np.logical_not(abs(H - 0) < 1e-13);
    return Im

# Construct the vertex list which defines the polygon filling the space under the (x, y) line graph
# This assumes x is in ascending order.
def polygon_under_graph(x, y):
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]


def add_headers(fig,col_arrow=True,*,row_headers=None,col_headers=None,row_pad=1,col_pad=5,rotate_row_headers=False,**text_kwargs):
    # Based on https://stackoverflow.com/a/25814386
    axes = [ax for ax in fig.get_axes() if ax._label!='<colorbar>']
    for ax in axes:
        sbs = ax.get_subplotspec()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )
            if col_arrow:
                ax.annotate("", xy=(0.75, 0.9), xycoords='figure fraction',
                                xytext=(0.125,0.9), textcoords='figure fraction',
                                arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - row_pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=rotate_row_headers * 90,
                **text_kwargs,
            )

# Below two functions are for labelling the axes in multiples of a rational number; default is π/12
# numerator = numerator of rational number
# maxdenom = denominator of rational number
# Usage example:
    # plt.yticks(np.pi*np.linspace(0,1,5)) # 0,π/4,π/2,3π/4,π
    # plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter(np.pi,4)))
def multiple_formatter(numerator=np.pi, maxdenom=2, latex=r'\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = maxdenom
        num = np.int64(np.rint(den*x/numerator))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

class Multiple:
    def __init__(self, maxdenom=12, number=np.pi, latex=r'\pi'):
        self.denominator = maxdenom
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.number, self.denominator,self.latex))

def set_ticks_multiples(ax,axis_name='x',num=np.pi, d=4,latex=r'\pi'):
    """
    Set yticks to rational multiples of (num/d) on the given axis, with reduced fractions.
    
    Parameters:
    - ax: The Matplotlib Axes object, e.g. plt.gca()
    - axis_name: 'x' or 'y'
    - num: The numerator for the custom unit (e.g., π or another constant).
    - d: Integer denominator for the unit num/d.
    """
    # Get the axis limits
    if axis_name=='x':
        lim = ax.get_xlim()
        axis = ax.xaxis
    elif axis_name=='y':
        lim = ax.get_ylim()
        axis = ax.yaxis
    else:
        raise ValueError("Invalid axis_name. Use 'x' or 'y'.")
    
    # Generate tick positions as multiples of num/d within the axis limits
    step = num / d
    ticks = np.arange(np.ceil(lim[0] / step), np.floor(lim[1] / step) + 1) * step  # Generate tick positions
    
    # Generate tick labels
    labels = []
    for tick in ticks:
        k = int(round(tick / step))  # Rational multiplier k
        fraction = Fraction(k, d).limit_denominator()  # Simplify k/d
        den,num = fraction.denominator, fraction.numerator

        if den == 1:
            if num == 0:
                labels.append(r'$0$')
            elif num == 1:
                labels.append(r'$%s$' % latex)
            elif num == -1:
                labels.append(r'$-%s$' % latex)
            else:
                labels.append(r'$%s%s$' % (num, latex))
        else:
            if num == 1:
                labels.append(r'$\frac{%s}{%s}$' % (latex, den))
            elif num == -1:
                labels.append(r'$\frac{-%s}{%s}$' % (latex, den))
            else:
                labels.append(r'$\frac{%s%s}{%s}$' % (num, latex, den))
    
    ## Apply ticks and labels
    axis.set_ticks(ticks)
    axis.set_ticklabels(labels)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FILE I/O 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# like np.loadtxt but saves commented lines
# comment_chars: list of first characters that mark commented lines
# strip_chars: list of chars to ignore/skip at the beginning and end of commented lines
# dtype: type of the data that is in file, which determines how to parse the (non-commented) data
# sep: delimiter b/t values in data rows
def readtxt(filename,dtype=float,comment_chars = '#',strip_chars =['\n',' '],sep='\t',**kwargs):
    arr,comments = [],[]
    with open(filename, 'r') as f:
        for line in f.readlines():
            if line.startswith(comment_chars):
                # strip the comment character and any escape characters
                strip_str = "".join(list(comment_chars)+list(strip_chars))
                comments.append(line.strip(strip_str)) # removes # and any trailing whitespace
            else:
                arr.append(np.fromstring(line, dtype=dtype,sep=sep))
    arr = np.array(arr)
    return arr,comments

# reads multidimensional array with column labels from txt file into data frame
# assumes have header line prefixed by # which is a list of col names
# returns df and header = list of col names, and footer
def load2df(filename,footer=False,**kwargs):
    with open(filename, 'r') as f:
        # Load into dataframe ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if footer:  # if also have footer that starts with #, need to get col names separateley
            with open(filename) as f:
                headerline = f.readline()  # read first line/header
            header = headerline.strip().lstrip("#").split() # get col names
            df = pd.read_csv(filename,names=header,comment='#',index_col=False) # ignores 
        else: # if there are no other lines that start with # besides the header, then can just treat # as an escape character
            df = pd.read_csv(filename,escapechar="#",index_col=False)
            df.columns = df.columns.str.strip()
            header = list(df.columns)
        df[header[-1]] = df[header[-1]].apply(lambda x: round(x,16)) # round last col = data col to 16 decimal places, the maximum precision given when outputting to txt file as str

    return df, header


# parses multidimensional arr data from file
# assumes data in txt file is unraveled multidim nD array where last col is the array and first n-1 ceols are indices/params
# returns reshaped array and list of unique indices labelleling the multidim arr
def load2arr(f):
    data = np.loadtxt(f)
    idcs = [None]*(data.shape[-1]-1)
    shape = [None]*(data.shape[-1]-1)
    for ax in range(data.shape[-1]-1):
        idcs[ax] = np.unique(data[:,ax])
        shape[ax] = len(idcs[ax]) 
    A = np.reshape(data[:,-1],shape)
    return A,idcs

# same as above but loads to df, then converts into nD array and retrieves idx list
def load2arr2(f):
    data = np.loadtxt(f)
    arr, idcs = df2arr(pd.DataFrame(data)) # convert to df so can organize indices, then convert back into arr
    return arr,idcs

# export data (x,f(x)) as 2d array
def exportxy(x,y,filestr,delim='\t',fmt=['%g','%s'],header="",footer=""):
    data = np.stack((x,y)).T
    np.savetxt(filestr,data,fmt=fmt,delimiter=delim,header=header,footer=footer)

# given some 2D matrix Z with rows indexed by values r and cols by values c
def export2DArr(r,c,Z,filestr, header="",footer=""):
    R,C = zip(*it.product(r,c))
    df = pd.DataFrame().assign(x=R,y=C,z=Z.ravel())
    np.savetxt(filestr, df.values, delimiter='\t',fmt='%s',header=header,footer=footer)

# A = nD array with first n-1 axes indexing the vectors and the last axis indexing the vector components
# output: df with multiidx for vec
def exportMultivec(A,idxcols=[], veccols = [],saveBool=False, filestr="",header="",footer=""):
    if not idxcols:
        idxcols = [chr(x) for x in np.arange(97,97+len(A.shape[:-1]))]
    if not veccols:
        veccols = ["x%d" % i for i in np.arange(A.shape[-1])]

    axesIdxVals = [np.arange(a) for a in A.shape[:-1]]
    multiIdx = pd.MultiIndex.from_product(axesIdxVals, names=idxcols)
    df = pd.DataFrame(np.reshape(A,(np.prod(A.shape[:-1]),len(veccols))), columns=veccols,index=multiIdx)
    if saveBool and filestr:
        print(filestr)
        with open(filestr, 'a') as file:
            file.write(header+"\n")
            df.to_csv(file, sep='\t',header=True, index=True)
            file.write(footer)
        # np.savetxt(filestr, df, fmt='%f',header=header,footer=footer)
    return df

# exports array M produced by a (#L)-parameter sweep to text file using np.savetxt
# data will be (params_shape)x(data shape) where for each of the L-param configs, we have an nD arary
# e.g. if kth param takes on n_k values, M.shape = (n_1,n_2,...,n_L,)+(data shape,)
# M will be exported to text file and footer will be ORDERED list of param values for each axis
# params: list of param values, e.g. [p0_vals,p1_vals,...]
# delim: delimiter for data
# sep: separator when print params in footer
def exportSweepTxt(M, filestr, params=[], sep=' ',delim='\t',**kwargs):
    if M.shape[:len(params)]!=tuple([len(p) for p in params]):
        raise Exception("array shape does not match number of params")
        
    # Footer for param configs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    paramList = [np.array2string(p,max_line_width=np.inf,separator=sep) for p in params] # uses same delimiter for params as for data
    footer = ('\n').join(paramList)
    
    # Reshape data into 2d array for saving ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Each row will be the raveled data associated to the L-param config
    Mp = M.reshape(M.shape[:len(params)]+(np.prod(M.shape[len(params):]),)) # shape = (param_shape,)+(len(raveled data))
    data = Mp.reshape((np.prod(Mp.shape[:-1]),)+(Mp.shape[-1],)) # shape = (# param configs) x (len(raveled data))

    # Save data to file ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    np.savetxt(filestr,data,delimiter=delim,footer=footer,**kwargs)
    # print("Saved %s" % filestr)

# exports nD array produced by a parameter sweep to text file with n+1 columns
# first n cols correspond to param configs, last col is data col
# e.g. M is nD array <==> n param sweep; M.shape[k] gives the number of values swept for param k
# params (optional): dictionary of param values; key = param/axis #; value = range of param values (e.g. params = {0: p0_vals, 1: p1_vals}
# fmt (optional): dictionary of fmt for each col; key = col #; value = fmt string, e.g. 0: '%s'
# clabels (optional): dictionary of column labels associated to different params; axis=-1 or n (since start at 0) corresponds to (n+1)th col = data col
# header (optional): any additional header string before col names; does not include newline charac '\n'
def exportNdArray(M, filestr, params={},fmt={}, clabels={},header="",footer="",delim='\t'):
    pvals = [None]*len(M.shape)
    fmt_arr = [None]*(len(M.shape)+1)
    clabel_arr = [None]*(len(M.shape)+1)

    # make list of param vals and their formats
    for i in range(len(M.shape)+1): # i labels param col; if M is nD have n+1 total cols
        if i < len(M.shape):
            pvals[i] = params[i] if i in params.keys() else range(M.shape[i])
        fmt_arr[i] = fmt[i] if i in fmt.keys() else '%s'
        clabel_arr[i] = clabels[i] if i in clabels.keys() else ("i%d" % i if i < len(M.shape) else 'data')

    header = (delim).join(clabel_arr) if not header else header+"\n"+(delim).join(clabel_arr)

    I = list(zip(*it.product(*pvals))) # each element in the list is a column vector containing the column data
    I.append(tuple(M.ravel()))
    dataDict = dict(zip(clabel_arr,I))
    df = pd.DataFrame.from_dict(dataDict)
    
    np.savetxt(filestr, df.values, fmt=fmt_arr, delimiter=delim,header=header,footer=footer)

# assumes for each key have a value which can generically be a multidimensional array A with n axes
# indexing the values of the data by (i_1,i_2,...i_n, k) where k is the key value
# cols are "i_1, i_2, .... i_n, k, dict[k][0,1,...n]"
def exportDict(dict, filestr, clabels=[],header="",footer="",delim='\t'):
    arrShape = list(dict.values())[0].shape
    if not clabels:
        clabels = [chr(x) for x in np.arange(97,97+len(arrShape)+1)] + ['data']
    if not header:
        header = (delim).join(clabels)

    M = np.array(list(dict.values())) # np array where 0th axis is the keys
    M = M.transpose(list(range(1,len(M.shape)))+[0]) # make keys LAST axis

    idcs = [range(x) for x in M.shape[:-1]]
    idcs = idcs + [list(dict.keys())]

    exportNdArray(M, filestr, idcs=idcs, clabels=clabels,header=header,footer=footer,delim=delim)

# for parsing filename with params in format [name]_val, e.g. N_10
def parse(paramname,filename):
    val = re.search('%s_([^_.]+)' % paramname,filename)
    if val is None:
        return val
    else:
        return val[1]