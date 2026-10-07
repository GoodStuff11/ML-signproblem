import sys, math
from fractions import Fraction
sys.path.insert(0,'.')
from gates2 import build, sched

def gfac(Lx,Ly): return (2 if Lx%2==0 else 1)*(2 if Ly%2==0 else 1)

LATS=[(2,2),(3,2),(4,2),(3,3),(4,3),(4,4),(5,4),(6,4),(6,6)]
print("%% TABLE 1: parameter counts")
print("lat  N  M  n_ss  n_os  P   P_formula  P_paper")
for Lx,Ly in LATS:
    b,N,M=build(Lx,Ly); P=len(b); g=gfac(Lx,Ly)
    nss=sum(1 for x in b if x[2]); nos=P-nss
    Pf=(N**3-6*N**2+(g+2)*N)//4 + 0
    Pf=(3*N**3-6*N**2+(g+2)*N)//4
    pap=(3*N**3+2*N**2)//4 if N%2==0 else (3*N**3+2*N**2-N)//4
    print(f"{Lx}x{Ly} {N} {M} {nss} {nos} {P} {Pf} {pap}  diag={pap-P}")

print("\n%% TABLE 2: staircase circuit")
print("lat N M P wbar CNOT Rz 1q Dseq Dpar ratio  CNOT_asym")
for Lx,Ly in LATS:
    b,N,M=build(Lx,Ly); P=len(b); ws=[x[1] for x in b]; g=gfac(Lx,Ly)
    wbar=sum(ws)/P
    cn=sum(16*(w-1) for w in ws)
    d=lambda w:16*(w-1)
    Dseq=sum(d(w) for w in ws); Dpar=sched(b,M,d)
    asym=(20/3)*N**4+(28/3)*N**3-40*N**2-4*(g+2)*N
    print(f"{Lx}x{Ly} {N} {M} {P} {wbar:.3f} {cn} {8*P} {64*P} {Dseq} {Dpar} {Dseq/Dpar:.3f}  {asym:.0f} ({100*(asym/cn-1):+.2f}%)")

print("\n%% TABLE 3: improved circuits")
print("lat N P | Yord CNOT Dseq Dpar | Yord+tree Dpar | LUCJ k=N gates depth")
for Lx,Ly in LATS:
    b,N,M=build(Lx,Ly); P=len(b); ws=[x[1] for x in b]
    cy=sum(2*w+5 for w in ws)
    dy=lambda w: max(13,2*w-1); Dsy=sum(dy(w) for w in ws); Dpy=sched(b,M,dy)
    dt=lambda w: 13+2*math.ceil(math.log2(max(w-4,2))); Dpt=sched(b,M,dt)
    # LUCJ, k layers, M=2N spin-orbitals, spin-blocked orbital rotations
    k=N
    givens=2*(N*(N-1)//2)         # per layer, two spin blocks
    g2q=2*2*givens + N*(2*N+1)    # two rotations (fwd+inv) x 2 gates per Givens + diagonal Coulomb
    dep=2*(2*N)+2*N               # ~ depth per layer
    print(f"{Lx}x{Ly} {N} {P} | {cy} {Dsy} {Dpy} | {Dpt} | {k*g2q} {k*dep}")

print("\n%% max concurrency check: 2N/wbar and Dseq/Dpar")
for Lx,Ly in LATS+[(7,6),(8,6)]:
    b,N,M=build(Lx,Ly); P=len(b); ws=[x[1] for x in b]; wbar=sum(ws)/P
    d=lambda w:16*(w-1)
    print(f"  {Lx}x{Ly} N={N} wbar/2N={wbar/(2*N):.4f} maxconc={2*N/wbar:.3f} greedy={sum(d(w) for w in ws)/sched(b,M,d):.3f}  [asym 5/18={5/18:.4f}, 18/5={18/5}]")
