import sys, math
sys.path.insert(0,'.')
from gates2 import build, sched

def ehv(Lx,Ly,mult=1.0):
    N=Lx*Ly; m=min(Lx,Ly)
    dl = 2*m+1 if m%2==0 else 2*m+2      # fully-connected, Cade Table I (+ footnote: min(nx,ny))
    nl = int(round(mult*N))
    return dl, nl, (N-1) + nl*dl + 1      # Givens state prep + layers + 1 (their 5x5 formula)

print("sanity 5x5:", ehv(5,5,1.0), "(paper says 24+25*12+1=325)")
print()
hdr=f"{'lat':>5} {'N':>3} {'EHVd/l':>7} {'L=N':>5} {'D_EHV':>7} {'D_EHV(1.5N)':>12} | {'UCC stair':>10} {'UCC Yord':>9} {'UCC UCJ':>8} | {'stair/EHV':>10} {'Yord/EHV':>9} {'UCJ/EHV':>8}"
print(hdr)
for (Lx,Ly) in [(2,2),(3,2),(4,2),(3,3),(4,3),(4,4),(5,4),(6,4),(6,6)]:
    b,N,M=build(Lx,Ly); P=len(b); ws=[x[1] for x in b]
    Dst=sched(b,M,lambda w:16*(w-1))
    Dyo=sched(b,M,lambda w:max(13,2*w-1))
    Ducj=N*(2*(2*N)+2*N)
    dl,nl,D=ehv(Lx,Ly,1.0); _,_,D15=ehv(Lx,Ly,1.5)
    print(f"{Lx}x{Ly:<3} {N:>3} {dl:>7} {nl:>5} {D:>7} {D15:>12} | {Dst:>10} {Dyo:>9} {Ducj:>8} | "
          f"{Dst/D:>10.0f} {Dyo/D:>9.0f} {Ducj/D:>8.1f}")
