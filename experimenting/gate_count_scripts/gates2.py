import sys, math, json
from collections import defaultdict
sys.path.insert(0,'.')
from enumerate import generators

def idx(k,s,N): return s*N+k          # blocked JW ordering

def weight(four):
    i=sorted(four); return 2+(i[1]-i[0])+(i[3]-i[2])

def build(Lx,Ly):
    gens,chan,N=generators(Lx,Ly); M=2*N
    out=[]
    for I,J in gens:
        orbs=[idx(k,s,N) for k,s in I]+[idx(k,s,N) for k,s in J]
        assert len(set(orbs))==4
        i=sorted(orbs)
        w=2+(i[1]-i[0])+(i[3]-i[2])
        supp=set(i)|set(range(i[0],i[1]))|set(range(i[2],i[3]))
        assert len(supp)==w
        same = (I[0][1]==I[1][1])
        out.append((frozenset(supp),w,same))
    return out,N,M

def sched(blocks,M,depth_of):
    free=[0]*M; mk=0
    for b in sorted(blocks,key=lambda b:-b[1]):
        d=depth_of(b[1]); t=max(free[q] for q in b[0])
        for q in b[0]: free[q]=t+d
        mk=max(mk,t+d)
    return mk

def load(blocks,M,depth_of):
    L=[0]*M
    for b in blocks:
        d=depth_of(b[1])
        for q in b[0]: L[q]+=d
    return max(L)

print(f"{'lat':>5} {'N':>3} {'P':>6} {'wbar':>6} {'wbar_ss':>8} {'wbar_os':>8} {'wmax':>5} "
      f"{'CNOT':>10} {'Dseq':>10} {'Dpar':>10} {'Dlb':>10} {'sp':>5} | {'Dseq_tree':>10} {'Dpar_tree':>10} {'sp':>5}")
res=[]
for (Lx,Ly) in [(2,2),(3,2),(4,2),(3,3),(4,3),(4,4),(5,3),(5,4),(6,4),(5,5),(6,5),(6,6)]:
    blocks,N,M=build(Lx,Ly); P=len(blocks)
    ws=[b[1] for b in blocks]
    wss=[b[1] for b in blocks if b[2]]; wos=[b[1] for b in blocks if not b[2]]
    cnot=sum(16*(w-1) for w in ws)
    dlin=lambda w: 8*(2*(w-1)+1)
    dtree=lambda w: 8*(2*math.ceil(math.log2(w))+1)
    Dseq=sum(dlin(w) for w in ws); Dpar=sched(blocks,M,dlin); Dlb=load(blocks,M,dlin)
    Dseq_t=sum(dtree(w) for w in ws); Dpar_t=sched(blocks,M,dtree)
    r=dict(lat=f"{Lx}x{Ly}",N=N,M=M,P=P,wbar=sum(ws)/P,wss=(sum(wss)/len(wss) if wss else 0),
           wos=sum(wos)/len(wos),wmax=max(ws),cnot=cnot,Dseq=Dseq,Dpar=Dpar,Dlb=Dlb,
           Dseq_t=Dseq_t,Dpar_t=Dpar_t,rz=8*P,onq=64*P)
    res.append(r)
    print(f"{r['lat']:>5} {N:>3} {P:>6} {r['wbar']:6.2f} {r['wss']:8.2f} {r['wos']:8.2f} {max(ws):>5} "
          f"{cnot:>10} {Dseq:>10} {Dpar:>10} {Dlb:>10} {Dseq/Dpar:5.2f} | {Dseq_t:>10} {Dpar_t:>10} {Dseq_t/Dpar_t:5.2f}")
json.dump(res,open('./res.json','w'),indent=1)
