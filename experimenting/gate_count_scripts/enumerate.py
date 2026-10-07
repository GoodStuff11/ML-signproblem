"""Enumerate momentum- and spin-conserving two-body UCC generators, count params."""
import itertools, sys
from collections import defaultdict

def momenta(Lx,Ly):
    return [(x,y) for x in range(Lx) for y in range(Ly)]

def add(k1,k2,Lx,Ly):
    return ((k1[0]+k2[0])%Lx,(k1[1]+k2[1])%Ly)

def generators(Lx,Ly):
    """Return list of generators. Each generator = (creation_set, annihilation_set)
    of (k,spin) spin-orbitals, canonical, deduplicated by h.c."""
    ks = momenta(Lx,Ly); N=len(ks)
    gens=[]
    chan=defaultdict(int)
    # same-spin channels
    for s in (0,1):
        buckets=defaultdict(list)
        for a,b in itertools.combinations(range(N),2):
            buckets[add(ks[a],ks[b],Lx,Ly)].append((a,b))
        for Q,prs in buckets.items():
            for i in range(len(prs)):
                for j in range(i+1,len(prs)):
                    I=tuple(sorted(((prs[j][0],s),(prs[j][1],s))))
                    J=tuple(sorted(((prs[i][0],s),(prs[i][1],s))))
                    gens.append((I,J)); chan['same']+=1
    # opposite spin: I=(ku,0),(kd,1)
    buckets=defaultdict(list)
    for a in range(N):
        for b in range(N):
            buckets[add(ks[a],ks[b],Lx,Ly)].append(((a,0),(b,1)))
    for Q,prs in buckets.items():
        for i in range(len(prs)):
            for j in range(i+1,len(prs)):
                gens.append((prs[j],prs[i])); chan['opp']+=1
    return gens, chan, N

def g_factor(Lx,Ly):
    return (2 if Lx%2==0 else 1)*(2 if Ly%2==0 else 1)

print(f"{'lattice':>10} {'N':>4} {'same':>8} {'opp':>8} {'P':>8} {'formula':>9} {'paper':>9}")
for (Lx,Ly) in [(2,1),(3,1),(4,1),(5,1),(6,1),(2,2),(3,2),(4,2),(3,3),(4,3),(4,4),(5,2),(6,2),(5,3),(8,1),(9,1),(12,1)]:
    gens,chan,N = generators(Lx,Ly)
    g=g_factor(Lx,Ly)
    same_f = (N**3-4*N**2+(g+2)*N)//4   # both spins
    opp_f  = N**2*(N-1)//2
    P_f = same_f+opp_f
    paper = (3*N**3+2*N**2)//4 if N%2==0 else (3*N**3+2*N**2-N)//4
    ok = "OK" if len(gens)==P_f else "MISMATCH"
    print(f"{Lx}x{Ly:<8} {N:>4} {chan['same']:>8} {chan['opp']:>8} {len(gens):>8} {P_f:>9} {paper:>9}  {ok}")
