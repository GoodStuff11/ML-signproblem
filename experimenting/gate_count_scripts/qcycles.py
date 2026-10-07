"""Cycle structure of the transferred-momentum shift k -> k (-) Q on an Lx x Ly BZ."""
from math import gcd, lcm
def order_Q(Q,Lx,Ly):
    ox = Lx//gcd(Q[0],Lx) if Q[0]%Lx else 1
    oy = Ly//gcd(Q[1],Ly) if Q[1]%Ly else 1
    return lcm(ox,oy)
print(f"{'lat':>6} {'N':>3} | {'m=1':>4} {'m=2':>4} {'m>2':>4} | {'mean m':>7} {'max m':>6} | "
      f"{'rot gates':>10} {'coul gates':>11} {'rot/coul':>9}")
for (Lx,Ly) in [(2,2),(3,2),(4,2),(3,3),(4,3),(4,4),(5,4),(6,4),(6,6)]:
    N=Lx*Ly; ms=[order_Q((qx,qy),Lx,Ly) for qx in range(Lx) for qy in range(Ly)]
    n1=sum(1 for m in ms if m==1); n2=sum(1 for m in ms if m==2); n3=len(ms)-n1-n2
    # per Q there are <=2N ranks n; rotation for one factor: sum over cycles of Clements cost
    # (N/m) blocks of size m per spin, 2 spins: 2*(N/m)*m(m-1)/2 = N(m-1) Givens ~ 2 gates each
    rot = sum(2*N*(2*N*(m-1)) for m in ms)          # 2N factors per Q, N(m-1) Givens x2 gates
    coul= sum(2*N*(2*N*(2*N-1)//2) for m in ms)     # 2N factors per Q, CPhase on complete graph
    print(f"{Lx}x{Ly:<4} {N:>3} | {n1:>4} {n2:>4} {n3:>4} | {sum(ms)/len(ms):>7.2f} {max(ms):>6} | "
          f"{rot:>10} {coul:>11} {rot/coul:>9.2f}")
