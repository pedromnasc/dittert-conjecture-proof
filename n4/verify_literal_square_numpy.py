#!/usr/bin/env python3
"""Independent literal-square audit of the exact n=4 Dittert certificate."""
from __future__ import annotations
import argparse,hashlib,itertools
from collections import defaultdict
from pathlib import Path
import numpy as np

if not __debug__:
    raise RuntimeError("Run without Python -O; the verifier uses assertions for proof checks.")
EXPECTED_MIN=72694203872
EXPECTED_ALPHA=(0,0,0,0,0,0,0,0,1,1,0,0,1,1)
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def add(a,b):return tuple(x+y for x,y in zip(a,b))
def unit(i,n):
 a=[0]*n;a[i]=1;return tuple(a)
def act(a,q):
 b=[0]*len(a)
 for i,v in enumerate(a):b[q[i]]+=v
 return tuple(b)
def square(out,terms,pref=None):
 for i,(a,ca) in enumerate(terms):
  if not ca:continue
  for j in range(i,len(terms)):
   b,cb=terms[j]
   if not cb:continue
   e=add(a,b);e=add(e,pref) if pref is not None else e
   out[e]+=ca*cb*(1 if i==j else 2)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('certificate',nargs='?',type=Path,default=Path(__file__).with_name('dittert_n4_twozero_certificate.npz'));args=ap.parse_args()
 p=args.certificate.resolve();z=np.load(p,allow_pickle=False);e=int(z['denominator_exponent']);assert e==30
 allowed=[tuple(map(int,r)) for r in z['allowed']];G=[tuple(map(int,r)) for r in z['group']];preps=[tuple(map(int,r)) for r in z['pair_reps']];z2=[tuple(map(int,r)) for r in z['z2']]
 L0=np.asarray(z['L0_num']);Lp=np.asarray(z['Lpair_num']);n=14;u=[unit(i,n) for i in range(n)]
 sos=defaultdict(int)
 for q in G:
  az=[act(a,q) for a in z2]
  for c in range(105):square(sos,[(az[r],int(L0[r,c])) for r in range(105)])
 for k,(a,b) in enumerate(preps):
  for q in G:
   pref=add(u[q[a]],u[q[b]]);au=[u[q[i]] for i in range(n)]
   for c in range(n):square(sos,[(au[r],int(Lp[k,r,c])) for r in range(n)],pref)
 D=1<<(2*e);scale=1<<(2*e-13);target=defaultdict(int)
 for t in itertools.product(range(n),repeat=4):
  a=[0]*n
  for i in t:a[i]+=1
  target[tuple(a)]+=61*scale
 idx={c:i for i,c in enumerate(allowed)}
 rows=[[idx[(r,c)] for c in range(4) if (r,c) in idx] for r in range(4)]
 cols=[[idx[(r,c)] for r in range(4) if (r,c) in idx] for c in range(4)]
 for ch in itertools.product(*rows):
  a=[0]*n
  for i in ch:a[i]+=1
  target[tuple(a)]-=D
 for ch in itertools.product(*cols):
  a=[0]*n
  for i in ch:a[i]+=1
  target[tuple(a)]-=D
 for perm in itertools.permutations(range(4)):
  cells=[(r,perm[r]) for r in range(4)]
  if any(c not in idx for c in cells):continue
  a=[0]*n
  for c in cells:a[idx[c]]+=1
  target[tuple(a)]+=D
 assert len(target)==len(sos)==2380
 residual={a:target[a]-sos[a] for a in target};amin=min(residual,key=residual.get);amax=max(residual,key=residual.get)
 mn=residual[amin];mx=residual[amax]
 assert mn>0 and mn==EXPECTED_MIN and amin==EXPECTED_ALPHA
 print('Independent literal-square audit')
 print('certificate SHA-256:',sha(p))
 print('quartic monomials checked:',len(residual))
 print(f'minimum residual: {mn}/{D} = {mn/D:.17g}')
 print(f'maximum residual: {mx}/{D} = {mx/D:.17g}')
 print('CERTIFIED')
if __name__=='__main__':main()
