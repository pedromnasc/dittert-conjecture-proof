#!/usr/bin/env python3
"""Primary exact verifier for the n=4 Dittert certificate.

All proof decisions use Python arbitrary-precision integers. NumPy is used only
for reading the certificate arrays.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

if not __debug__:
    raise RuntimeError("Run without Python -O; the verifier uses assertions for proof checks.")

EXPECTED_MIN = 72694203872
EXPECTED_MIN_ALPHA = (0,0,0,0,0,0,0,0,1,1,0,0,1,1)

def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def comps(t:int,k:int,p=()):
    if k==1:
        yield p+(t,); return
    for i in range(t+1): yield from comps(t-i,k-1,p+(i,))

def act(a,q):
    b=[0]*len(a)
    for i,v in enumerate(a):
        if v: b[q[i]]+=int(v)
    return tuple(b)

def gram(L):
    rows,cols=L.shape
    x=[[int(L[i,j]) for j in range(cols)] for i in range(rows)]
    return [[sum(x[i][k]*x[j][k] for k in range(cols)) for j in range(rows)] for i in range(rows)]

def mult4(a):
    m=24
    for v in a:m//=math.factorial(v)
    return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('certificate',nargs='?',type=Path,default=Path(__file__).with_name('dittert_n4_twozero_certificate.npz'))
    ap.add_argument('--json',type=Path)
    a=ap.parse_args(); path=a.certificate.resolve()
    z=np.load(path,allow_pickle=False)
    assert set(z.files)=={'denominator_exponent','allowed','group','pair_reps','z2','L0_num','Lpair_num','pattern'}
    e=int(z['denominator_exponent']); assert e==30
    assert str(z['pattern'])=='1000/0100/0000/0000'
    allowed=[tuple(map(int,r)) for r in z['allowed']]
    G=[tuple(map(int,r)) for r in z['group']]
    preps=[tuple(map(int,r)) for r in z['pair_reps']]
    z2=[tuple(map(int,r)) for r in z['z2']]
    L0=np.asarray(z['L0_num']); Lp=np.asarray(z['Lpair_num'])
    n=14
    assert allowed==[(i,j) for i in range(4) for j in range(4) if (i,j) not in {(0,0),(1,1)}]
    assert L0.shape==(105,105) and Lp.shape==(16,14,14)
    assert np.all(np.triu(L0,1)==0) and np.all(np.triu(Lp,1)==0)
    assert np.all(np.diag(L0)>0) and np.all(np.diagonal(Lp,axis1=1,axis2=2)>0)
    ident=tuple(range(n)); GS=set(G)
    assert len(G)==len(GS)==16 and ident in GS
    for q in G:
        assert tuple(sorted(q))==ident
        for r in G: assert tuple(r[q[i]] for i in range(n)) in GS
    H=set()
    for s in itertools.combinations(range(n),4):
        if len({allowed[i][0] for i in s})==4 or len({allowed[i][1] for i in s})==4:H.add(s)
    assert len(H)==274
    for q in G: assert {tuple(sorted(q[i] for i in s)) for s in H}==H
    ez2=set()
    for i in range(n):
        for j in range(i,n):
            x=[0]*n;x[i]+=1;x[j]+=1;ez2.add(tuple(x))
    assert len(z2)==len(set(z2))==105 and set(z2)==ez2
    unseen={(i,j) for i in range(n) for j in range(i,n)}; orbits=[]
    while unseen:
        p=next(iter(unseen)); o={tuple(sorted((q[p[0]],q[p[1]]))) for q in G};orbits.append(o);unseen-=o
    assert len(orbits)==16 and len(preps)==16
    assert len({next(i for i,o in enumerate(orbits) if p in o) for p in preps})==16

    Q0=gram(L0); Qp=[gram(Lp[p]) for p in range(16)]
    sos=defaultdict(int)
    for q in G:
        az=[act(x,q) for x in z2]
        for i in range(105):
            for j in range(i,105):
                c=Q0[i][j]*(1 if i==j else 2)
                if c:sos[tuple(az[i][k]+az[j][k] for k in range(n))]+=c
    for p,(u,v) in enumerate(preps):
        Q=Qp[p]
        for q in G:
            for i in range(n):
                for j in range(i,n):
                    c=Q[i][j]*(1 if i==j else 2)
                    if c:
                        x=[0]*n
                        for k in (q[u],q[v],q[i],q[j]):x[k]+=1
                        sos[tuple(x)]+=c
    D=1<<(2*e); scale=1<<(2*e-13)
    residual={}
    for alpha in comps(4,n):
        t=61*mult4(alpha)*scale
        supp=tuple(i for i,v in enumerate(alpha) if v)
        if len(supp)==4 and all(alpha[i]==1 for i in supp) and supp in H:t-=D
        residual[alpha]=t-sos.get(alpha,0)
    assert len(sos)==len(residual)==2380
    amin=min(residual,key=residual.get); amax=max(residual,key=residual.get)
    mn=residual[amin];mx=residual[amax]
    assert mn>0 and mn==EXPECTED_MIN and amin==EXPECTED_MIN_ALPHA

    # Independent exact point evaluations of the expanded identity.
    def mon(a,x): return math.prod(x[i]**v for i,v in enumerate(a) if v)
    points=[]
    for i in range(n):
        x=[0]*n;x[i]=1;points.append(tuple(x))
    points += [tuple([1]*n),tuple((3*i+1)%5 for i in range(n)),tuple((i*i+2*i+3)%7 for i in range(n))]
    for x in points:
        mat=[[0]*4 for _ in range(4)]
        for val,(r,c) in zip(x,allowed):mat[r][c]=val
        row=math.prod(sum(r) for r in mat)
        col=math.prod(sum(mat[r][c] for r in range(4)) for c in range(4))
        per=sum(math.prod(mat[r][p[r]] for r in range(4)) for p in itertools.permutations(range(4)))
        f=sum(math.prod(x[i] for i in s) for s in H)
        assert f==row+col-per
        target=61*(sum(x)**4)*scale-f*D
        expanded=sum((sos.get(alpha,0)+residual[alpha])*mon(alpha,x) for alpha in residual)
        assert target==expanded

    report={'status':'CERTIFIED','certificate_sha256':sha256(path),'variables':14,'symmetry_group_order':16,
            'hyperedges':274,'quartic_monomials':2380,'minimum_residual_numerator':mn,
            'minimum_residual_denominator':D,'minimum_residual_decimal':mn/D,
            'minimum_residual_exponent':list(amin),'maximum_residual_numerator':mx,'maximum_residual_decimal':mx/D,
            'exact_evaluation_tests':len(points)}
    if a.json:a.json.write_text(json.dumps(report,indent=2)+'\n')
    print('Dittert n=4 exact certificate verifier')
    print('certificate SHA-256:',report['certificate_sha256'])
    print('quartic hyperedges in F:',len(H))
    print('quartic monomials checked:',len(residual))
    print(f'minimum residual: {mn}/{D} = {mn/D:.17g}')
    print(f'maximum residual: {mx}/{D} = {mx/D:.17g}')
    print('CERTIFIED')
if __name__=='__main__':main()
