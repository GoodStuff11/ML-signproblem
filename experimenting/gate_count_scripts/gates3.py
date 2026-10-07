import sys, math
from fractions import Fraction
sys.path.insert(0,'.')
from gates2 import build, sched, load
print()
print(f"{'lat':>5} {'N':>3} {'wbar_ss':>18} {'wbar_os':>12} {'(N+8)/3':>9} | {'vol/M':>10} {'Dseq/vol':>8} {'2N/wbar':>7} {'Dseq/Dpar':>9} {'Dseq/Dlb':>8}")
for (Lx,Ly) in [(2,2),(3,2),(4,2),(3,3),(4,3),(4,4),(5,4),(6,4),(6,5),(6,6),(7,6),(8,6)]:
    blocks,N,M=build(Lx,Ly); P=len(blocks)
    ws=[b[1] for b in blocks]
    wss=[b[1] for b in blocks if b[2]]; wos=[b[1] for b in blocks if not b[2]]
    dlin=lambda w: 8*(2*(w-1)+1)
    Dseq=sum(dlin(w) for w in ws); Dpar=sched(blocks,M,dlin); Dlb=load(blocks,M,dlin)
    vol=sum(dlin(w)*w for w in ws)/M
    fss=Fraction(sum(wss),len(wss)) if wss else Fraction(0)
    fos=Fraction(sum(wos),len(wos))
    print(f"{Lx}x{Ly:<3} {N:>3} {str(fss)+f' ({float(fss):.3f})':>18} {str(fos):>12} {Fraction(N+8,3)!s:>9} | "
          f"{vol:10.0f} {Dseq/vol:8.2f} {2*N/(sum(ws)/P):7.2f} {Dseq/Dpar:9.2f} {Dseq/Dlb:8.2f}")
# qubit load profile for 4x4
blocks,N,M=build(4,4)
dlin=lambda w: 8*(2*(w-1)+1)
L=[0]*M
for supp,w,s in blocks:
    for q in supp: L[q]+=dlin(w)
print("\n4x4 per-qubit CNOT-layer load (blocked JW ordering, qubits 0..31):")
print(" ".join(f"{x//1000}k" for x in L))
