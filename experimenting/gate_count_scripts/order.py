import sys, itertools, random, math
from fractions import Fraction
sys.path.insert(0,'.')
from enumerate import generators

def stats(Lx,Ly,perm=None,interleave=False):
    gens,chan,N=generators(Lx,Ly)
    pi = perm if perm else list(range(N))
    def oi(k,s): return (2*pi[k]+s) if interleave else (s*N+pi[k])
    tot=0; tos=0; nos=0; tss=0; nss=0
    for I,J in gens:
        o=sorted([oi(k,s) for k,s in I]+[oi(k,s) for k,s in J])
        w=2+(o[1]-o[0])+(o[3]-o[2]); tot+=w
        if I[0][1]==I[1][1]: tss+=w; nss+=1
        else: tos+=w; nos+=1
    return len(gens), Fraction(tot,len(gens)), Fraction(tos,nos), (Fraction(tss,nss) if nss else 0)

print("1D chains, blocked ordering:")
for N in [4,6,8,9,12,16,20]:
    P,wb,wo,ws=stats(N,1)
    print(f"  N={N:>3} P={P:>5} wbar={float(wb):6.3f} w_os={wo} (2N+8)/3={Fraction(2*N+8,3)}  w_ss={float(ws):6.3f} (N+8)/3={float(Fraction(N+8,3)):.3f}")
print("2D, blocked:")
for (Lx,Ly) in [(4,3),(4,4),(6,4),(3,3)]:
    P,wb,wo,ws=stats(Lx,Ly)
    print(f"  {Lx}x{Ly} N={Lx*Ly} wbar={float(wb):6.3f} w_os={wo}={float(wo):.3f} (2N+8)/3={float(Fraction(2*Lx*Ly+8,3)):.3f}")
print("\n4x3: momentum-ordering search (blocked vs interleaved, random perms):")
N=12
base=stats(4,3)
print("  identity blocked   ", float(base[1]))
print("  identity interleave", float(stats(4,3,interleave=True)[1]))
best=(1e9,None); random.seed(0)
for t in range(4000):
    p=list(range(N)); random.shuffle(p)
    v=float(stats(4,3,p)[1])
    if v<best[0]: best=(v,p)
print("  best random perm   ", best[0], best[1])
worst=(0,None)
for t in range(2000):
    p=list(range(N)); random.shuffle(p)
    v=float(stats(4,3,p)[1])
    if v>worst[0]: worst=(v,p)
print("  worst random perm  ", worst[0])
