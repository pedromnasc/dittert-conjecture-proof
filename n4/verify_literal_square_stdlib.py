#!/usr/bin/env python3
"""Third exact verifier for the n=4 Dittert certificate.

Uses only the Python standard library. It parses .npz/.npy itself, expands every
saved factor as literal polynomial squares, constructs the target directly from
row products, column products and the permanent, and checks all degree-4
coefficients with arbitrary-precision integers.
"""
from __future__ import annotations
import ast, hashlib, itertools, math, struct, sys, zipfile
from collections import defaultdict
from pathlib import Path
if not __debug__:
    raise RuntimeError("Run without Python -O; the verifier uses assertions for proof checks.")
EXPECTED_SHA="d76533bd1c5566ea8d96aa3b58a0b6a8bf3310eb438776ebd99a6bd88d5a11f6"
EXPECTED_MIN=72694203872
EXPECTED_MIN_ALPHA=(0,0,0,0,0,0,0,0,1,1,0,0,1,1)
def prod(xs):
 r=1
 for x in xs:r*=x
 return r
def parse_npy(data):
 assert data[:6]==b'\x93NUMPY';ver=tuple(data[6:8]);pos=8
 if ver==(1,0):hlen=struct.unpack_from('<H',data,pos)[0];pos+=2
 elif ver in ((2,0),(3,0)):hlen=struct.unpack_from('<I',data,pos)[0];pos+=4
 else:raise ValueError(ver)
 enc='utf-8' if ver==(3,0) else 'latin1';h=ast.literal_eval(data[pos:pos+hlen].decode(enc).strip());pos+=hlen
 assert not h['fortran_order'];shape=tuple(h['shape']);cnt=prod(shape) if shape else 1;d=h['descr'];raw=data[pos:]
 if d=='|i1':assert len(raw)==cnt;vals=tuple(struct.unpack(f'{cnt}b',raw))
 elif d=='<i8':assert len(raw)==8*cnt;vals=tuple(struct.unpack(f'<{cnt}q',raw))
 elif d.startswith('<U'):
  nc=int(d[2:]);assert cnt==1 and len(raw)==4*nc;vals=(raw.decode('utf-32-le').rstrip('\x00'),)
 else:raise ValueError(d)
 return shape,vals
def reshape(vals,shape):
 if not shape:return vals[0]
 if len(shape)==1:return tuple(vals)
 st=prod(shape[1:]);return tuple(reshape(vals[i*st:(i+1)*st],shape[1:]) for i in range(shape[0]))
def load_npz(path):
 out={};expected={x+'.npy' for x in ('denominator_exponent','allowed','group','pair_reps','z2','L0_num','Lpair_num','pattern')}
 with zipfile.ZipFile(path) as z:
  assert set(z.namelist())==expected
  for fn in sorted(expected):
   sh,v=parse_npy(z.read(fn));out[fn[:-4]]=reshape(v,sh)
 return out
def add(a,b):return tuple(x+y for x,y in zip(a,b))
def unit(i,n):
 a=[0]*n;a[i]=1;return tuple(a)
def act(a,q):
 b=[0]*len(a)
 for i,e in enumerate(a):b[q[i]]+=e
 return tuple(b)
def square(out,terms,prefix=None):
 terms=[t for t in terms if t[1]]
 for i,(a,ca) in enumerate(terms):
  for j in range(i,len(terms)):
   b,cb=terms[j];e=add(a,b);e=add(e,prefix) if prefix is not None else e;out[e]+=ca*cb*(1 if i==j else 2)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 path=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).with_name('dittert_n4_exact_certificate.npz');digest=sha(path);assert digest==EXPECTED_SHA
 z=load_npz(path);assert z['denominator_exponent']==30 and z['pattern']=='1000/0100/0000/0000'
 allowed=tuple(map(tuple,z['allowed']));G=tuple(map(tuple,z['group']));preps=tuple(map(tuple,z['pair_reps']));z2=tuple(map(tuple,z['z2']));L0=z['L0_num'];Lp=z['Lpair_num'];n=14
 assert allowed==tuple((i,j) for i in range(4) for j in range(4) if (i,j) not in {(0,0),(1,1)})
 assert len(G)==len(set(G))==16 and all(sorted(q)==list(range(n)) for q in G)
 ez=set()
 for i in range(n):
  for j in range(i,n):
   a=[0]*n;a[i]+=1;a[j]+=1;ez.add(tuple(a))
 assert len(z2)==105 and set(z2)==ez and len(preps)==16
 assert len(L0)==105 and all(len(r)==105 for r in L0);assert len(Lp)==16 and all(len(M)==14 and all(len(r)==14 for r in M) for M in Lp)
 sos=defaultdict(int)
 for q in G:
  az=[act(a,q) for a in z2]
  for c in range(105):square(sos,[(az[r],L0[r][c]) for r in range(105)])
 u=[unit(i,n) for i in range(n)]
 for k,(a,b) in enumerate(preps):
  for q in G:
   pref=add(u[q[a]],u[q[b]]);au=[u[q[i]] for i in range(n)]
   for c in range(n):square(sos,[(au[r],Lp[k][r][c]) for r in range(n)],pref)
 D=1<<60;scale=1<<47;target=defaultdict(int)
 for t in itertools.product(range(n),repeat=4):
  a=[0]*n
  for i in t:a[i]+=1
  target[tuple(a)]+=61*scale
 idx={c:i for i,c in enumerate(allowed)};rows=[[idx[(r,c)] for c in range(4) if (r,c) in idx] for r in range(4)];cols=[[idx[(r,c)] for r in range(4) if (r,c) in idx] for c in range(4)]
 for ch in itertools.product(*rows):
  a=[0]*n
  for i in ch:a[i]+=1
  target[tuple(a)]-=D
 for ch in itertools.product(*cols):
  a=[0]*n
  for i in ch:a[i]+=1
  target[tuple(a)]-=D
 for p in itertools.permutations(range(4)):
  cells=[(r,p[r]) for r in range(4)]
  if any(c not in idx for c in cells):continue
  a=[0]*n
  for c in cells:a[idx[c]]+=1
  target[tuple(a)]+=D
 assert len(target)==math.comb(n+3,4)==2380 and set(sos)==set(target)
 residual={a:target[a]-sos[a] for a in target};amin=min(residual,key=residual.get);amax=max(residual,key=residual.get);mn=residual[amin];mx=residual[amax]
 assert amin==EXPECTED_MIN_ALPHA and mn==EXPECTED_MIN>0
 print('Standard-library literal-square verifier');print('certificate SHA-256:',digest);print('quartic monomials checked:',len(residual));print(f'minimum residual: {mn}/{D} = {mn/D:.17g}');print(f'maximum residual: {mx}/{D} = {mx/D:.17g}');print('CERTIFIED')
if __name__=='__main__':main()
