"""Cost of compiling exp(A^(2)) by double factorisation, momentum-space aware."""
import sys; sys.path.insert(0,'.')
from gates2 import build, sched

def gfac(Lx,Ly): return (2 if Lx%2==0 else 1)*(2 if Ly%2==0 else 1)

def df_costs(N, k):
    """One DF factor: pair-rotations (O(N) gates, O(1) depth, momentum-space)
       + rank-1 diagonal Coulomb on M=2N qubits (complete graph: M(M-1)/2 gates, M-1 depth)."""
    M=2*N
    rot_g, rot_d = 2*(2*M), 8          # fwd+inv, 2 two-qubit gates per 2-mode Givens
    cou_g, cou_d = M*(M-1)//2, M-1
    return k*(rot_g+cou_g), k*(rot_d+cou_d)

def ehv(Lx,Ly):
    N=Lx*Ly; m=min(Lx,Ly); dl=2*m+1 if m%2==0 else 2*m+2
    return (N-1)+N*dl+1

def ehv_dof(Lx,Ly):
    m=min(Lx,Ly); p = 3 if (Lx,Ly)==(2,2) else (3 if m==1 else (4 if m==2 else 5))
    return p*Lx*Ly
def np_dof(Lx,Ly): return (10*Lx*Ly-4*Lx-4*Ly)*Lx*Ly

print(f"{'lat':>5} {'N':>3} {'P':>6} | {'R=2N^2':>7} {'exact g':>9} {'exact d':>8} | "
      f"{'k=N g':>7} {'k=N d':>6} {'/Cade':>6} | {'stair d':>9} {'Yord d':>8} {'Cade d':>7}")
for (Lx,Ly) in [(2,2),(3,2),(4,2),(3,3),(4,3),(4,4),(5,4),(6,4),(6,6)]:
    b,N,M=build(Lx,Ly); P=len(b)
    Dst=sched(b,M,lambda w:16*(w-1)); Dyo=sched(b,M,lambda w:max(13,2*w-1))
    R=2*N*N
    eg,ed = df_costs(N,R); kg,kd = df_costs(N,N)
    C=ehv(Lx,Ly)
    print(f"{Lx}x{Ly:<3} {N:>3} {P:>6} | {R:>7} {eg:>9} {ed:>8} | {kg:>7} {kd:>6} {kd/C:>6.1f} | {Dst:>9} {Dyo:>8} {C:>7}")

print("\nDOF per unit depth (higher is better):")
print(f"{'lat':>5} {'N':>3} | {'EHV':>7} {'NP':>7} | {'UCC stair':>10} {'UCC Yord':>9} {'UCC UCJ(k=N)':>13}")
for (Lx,Ly) in [(2,2),(4,3),(4,4),(6,4),(6,6),(10,10),(14,14),(20,20)]:
    N=Lx*Ly
    if N<=36:
        b,_,M=build(Lx,Ly); P=len(b); Dst=sched(b,M,lambda w:16*(w-1)); Dyo=sched(b,M,lambda w:max(13,2*w-1))
    else:
        g=gfac(Lx,Ly); P=(3*N**3-6*N**2+(g+2)*N)//4; Dst=int(6.67*N**4/1.23); Dyo=int(0.83*N**4/1.25)
    C=ehv(Lx,Ly); _,kd=df_costs(N,N)
    print(f"{Lx}x{Ly:<3} {N:>3} | {ehv_dof(Lx,Ly)/C:>7.1f} {np_dof(Lx,Ly)/C:>7.1f} | {P/Dst:>10.3f} {P/Dyo:>9.3f} {P/kd:>13.1f}")
