"""Minimal Jordan-Wigner transform + Pauli algebra."""
MUL = {}
for a in 'IXYZ': MUL[('I',a)]=(1,a); MUL[(a,'I')]=(1,a)
for a in 'XYZ': MUL[(a,a)]=(1,'I')
MUL[('X','Y')]=(1j,'Z'); MUL[('Y','X')]=(-1j,'Z')
MUL[('Y','Z')]=(1j,'X'); MUL[('Z','Y')]=(-1j,'X')
MUL[('Z','X')]=(1j,'Y'); MUL[('X','Z')]=(-1j,'Y')

def pmul(t1,t2):
    """t=(coeff, frozenset-free dict qubit->char). Returns product."""
    c1,d1=t1; c2,d2=t2
    c=c1*c2; d=dict(d1)
    for q,p2 in d2.items():
        p1=d.get(q,'I')
        f,p=MUL[(p1,p2)]
        c*=f
        if p=='I': d.pop(q,None)
        else: d[q]=p
    return (c,d)

def smul(A,B):
    """multiply two Pauli sums (list of (coeff,dict))"""
    out={}
    for t1 in A:
        for t2 in B:
            c,d=pmul(t1,t2)
            key=tuple(sorted(d.items()))
            out[key]=out.get(key,0)+c
    return [(c,dict(k)) for k,c in out.items() if abs(c)>1e-12]

def jw_op(j, dag):
    """JW image of c_j or c_j^dagger, as Pauli sum."""
    Zs={q:'Z' for q in range(j)}
    s = -1j if dag else 1j
    t1=(0.5, {**Zs, j:'X'})
    t2=(0.5*s, {**Zs, j:'Y'})
    return [t1,t2]

def generator_paulis(creations, annihilations):
    """tau = c^d_p c^d_q c_r c_s - h.c. ; creations=(p,q), annihilations=(r,s)
    returns list of (real_coeff_of_iP, pauli_dict) i.e. tau = sum_j coeff_j * (i P_j)"""
    A=[(1.0,{})]
    for p in creations: A=smul(A,jw_op(p,True))
    for r in annihilations: A=smul(A,jw_op(r,False))
    # subtract h.c.: (sum c P)^dag = sum c* P  (Pauli strings hermitian)
    out={}
    for c,d in A:
        key=tuple(sorted(d.items()))
        out[key]=out.get(key,0)+c
    for c,d in A:
        key=tuple(sorted(d.items()))
        out[key]=out.get(key,0)-c.conjugate()
    res=[]
    for k,c in out.items():
        if abs(c)>1e-12:
            assert abs(c.real)<1e-12, (k,c)
            res.append((c.imag, dict(k)))
    return res
