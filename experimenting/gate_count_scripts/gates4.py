import sys, math, json
sys.path.insert(0,'.')
from gates2 import build, sched

def go(Lx,Ly):
    blocks,N,M=build(Lx,Ly); P=len(blocks); ws=[b[1] for b in blocks]
    models={
      'stair'      : (lambda w: 16*(w-1),          lambda w: 16*(w-1)),
      'stair_tree' : (lambda w: 16*(w-1),          lambda w: 8*2*math.ceil(math.log2(w))),
      'yordanov'   : (lambda w: 2*w+5,             lambda w: max(13,2*w-1)),
      'yord_tree'  : (lambda w: 2*w+5,             lambda w: 13+2*math.ceil(math.log2(max(w-4,2)))),
      'swapnet'    : (lambda w: 13+0*w,            lambda w: 11+0*w),   # all-local, w=4 (SWAP cost accounted separately)
    }
    out={'lat':f"{Lx}x{Ly}",'N':N,'M':M,'P':P,'wbar':sum(ws)/P}
    for name,(c,d) in models.items():
        cn=sum(c(w) for w in ws); Dseq=sum(d(w) for w in ws)
        Dpar=sched([(b[0], b[1], b[2]) for b in blocks], M, d)
        out[name]={'cnot':cn,'Dseq':Dseq,'Dpar':Dpar,'sp':Dseq/Dpar}
    return out

print(f"{'lat':>5} {'N':>3} {'P':>6} {'wbar':>6} | "
      + " ".join(f"{m:>26}" for m in ['staircase (CNOT/Dseq/Dpar)','Yordanov (CNOT/Dseq/Dpar)']))
rows=[]
for lat in [(2,2),(3,2),(4,2),(3,3),(4,3),(4,4),(5,4),(6,4),(6,6)]:
    r=go(*lat); rows.append(r)
    s=r['stair']; y=r['yordanov']; yt=r['yord_tree']
    print(f"{r['lat']:>5} {r['N']:>3} {r['P']:>6} {r['wbar']:6.2f} | "
          f"{s['cnot']:>9}/{s['Dseq']:>9}/{s['Dpar']:>9}({s['sp']:.2f})  "
          f"{y['cnot']:>8}/{y['Dseq']:>8}/{y['Dpar']:>8}({y['sp']:.2f})  "
          f"tree Dpar={yt['Dpar']:>7}({yt['sp']:.2f})  CNOTratio={s['cnot']/y['cnot']:5.2f} Dratio={s['Dpar']/y['Dpar']:5.2f}")
json.dump(rows,open('./rows4.json','w'),indent=1)
