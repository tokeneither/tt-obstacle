import itertools,json
import numpy as np
n=2; d=3; h=1/(n+1)
K=np.array([[2.,-1.],[-1.,2.]])/h
M=h*np.array([[4.,1.],[1.,4.]])/6
L=np.linalg.cholesky(M)
V=np.linalg.inv(L)@K@np.linalg.inv(L.T)
w,Q=np.linalg.eigh(V)
E=np.linalg.solve(L.T,Q)
for j in range(n):
 if E[0,j]<0:E[:,j]*=-1
B=E[:,0]; bD=np.kron(B,B); MD=np.kron(M,M)
A=np.kron(K,MD)+np.kron(M,np.kron(K,M))+np.kron(M,np.kron(M,K))
H0=K+2*w[0]*M
vJ=np.array([0.,1.]); ellJ=np.array([1.,0.]); gJ=H0@vJ-ellJ
f0=np.kron(gJ,MD@bD);u0=np.kron(vJ,bD)
indices=[(0,0),(1,0),(0,1)]
rng=np.random.default_rng(876)
coeff={nu:rng.normal(size=n) for nu in indices}
ws={};ts={};phis={};chis={}
for nu in indices:
 phi=np.kron(E[:,nu[0]],E[:,nu[1]]);phis[nu]=phi
 H=K+(w[nu[0]]+w[nu[1]])*M
 wi=np.array([0.,coeff[nu][1]/H[1,1]])
 ws[nu]=wi;ts[nu]=H@wi-coeff[nu]
 chis[nu]=np.max(np.abs(phi/bD))
sv=sum(chis[nu]*abs(ws[nu][1]) for nu in indices)
sl=sum(chis[nu]*abs(ts[nu][0]) for nu in indices)
eps=.2*min(1/sv,1/sl)
fm=f0.copy(); um=u0.copy()
for nu in indices:
 fm+=eps*np.kron(coeff[nu],MD@phis[nu]);um+=eps*np.kron(ws[nu],phis[nu])
lam=A@um-fm
assert um.min()>-1e-12 and lam.min()>-1e-12 and abs(um@lam)<1e-10
S=np.column_stack([np.kron(np.eye(n)[:,j],phis[nu]) for nu in indices for j in range(n)])

def qp(S,f):
 H=S.T@A@S;g=S.T@f;r=H.shape[0];N=S.shape[0]
 best=None
 for kk in range(min(r,N)+1):
  for rows in itertools.combinations(range(N),kk):
   C=S[list(rows)]
   T=np.block([[H,-C.T],[C,np.zeros((kk,kk))]])
   if np.linalg.matrix_rank(T)<r+kk:continue
   sol=np.linalg.solve(T,np.r_[g,np.zeros(kk)])
   u=S@sol[:r]
   if u.min() < -1e-8 or sol[r:].min(initial=0) < -1e-8:continue
   obj=.5*u@A@u-f@u
   if best is None or obj<best[0]:best=(obj,u)
 assert best is not None
 return best[1]
phi=np.kron(E[:,1],E[:,1]); r=np.array([1.,-.7]); results=[]
for amp in [.01,.1,1.,10.]:
 tail=amp*np.kron(r,MD@phi); f=fm+tail
 u=qp(np.eye(8),f);us=qp(S,f)
 err=np.sqrt((u-us)@A@(u-us));delta=np.sqrt(tail@np.linalg.solve(A,tail))
 assert err<=delta*(1+1e-8)+1e-9
 results.append(dict(amplitude=amp,error=err,bound=delta,ratio=err/delta,contact=np.where(u<1e-8)[0].tolist()))
print(json.dumps(dict(mask_KKT_residual=float(np.linalg.norm(um*lam)),eps=eps,cases=results),indent=2))
