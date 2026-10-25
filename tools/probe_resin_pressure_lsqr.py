"""JAX rectangular mass-whitened LSQR on one saved physical pressure input."""
import json,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j
from mpm_jax.resin_mac import divergence,transpose_divergence
p=np.load('/input/rejected-pressure-input.npz');h=float(p['h']);dt=float(p['dt']);fluid=j.asarray(p['fluid']);velocity=tuple(j.asarray(p['velocity'+str(i)]) for i in range(3));mass=tuple(j.asarray(p['mass'+str(i)]) for i in range(3));fixed=tuple(j.asarray(p['fixed'+str(i)]) for i in range(3));inv=tuple(j.where((m>0)&~f,1/j.maximum(m,1e-300),0.) for m,f in zip(mass,fixed));root=tuple(j.sqrt(v) for v in inv);shape=tuple(v.shape for v in velocity);sizes=[int(np.prod(v)) for v in shape];ends=np.cumsum([0]+sizes)
def split(x):return tuple(x[ends[i]:ends[i+1]].reshape(shape[i]) for i in range(3))
def pack(v):return j.concatenate(tuple(a.ravel() for a in v))
def B(x):return j.where(fluid,divergence(tuple(r*v for r,v in zip(root,split(x))),h),0.)
def BT(q):return pack(tuple(r*g for r,g in zip(root,transpose_divergence(j.where(fluid,q,0.),h))))
a,b,c=inv;diag=h**4*(a[1:]+a[:-1]+b[:,1:]+b[:,:-1]+c[:,:,1:]+c[:,:,:-1]);scale=j.where(fluid,1/j.sqrt(j.maximum(diag,1e-300)),0.);rhs=j.where(fluid,divergence(velocity,h),0.);norm=j.linalg.norm(rhs)
def A(x):return scale*B(x)
def AT(q):return BT(scale*q)
@jax.jit
def solve():
 beta=j.linalg.norm(scale*rhs);u=scale*rhs/beta;v=AT(u);alpha=j.linalg.norm(v);v=v/alpha;t=u/alpha
 initial=(j.asarray(0,j.int32),u,v,t,v,t,j.zeros_like(v),j.zeros_like(u),alpha,beta,alpha,j.asarray(j.inf),j.asarray(j.inf))
 def cond(z):return (z[0]<1600)&((z[-2]>1e-8)|(z[-1]>1e-7))
 def body(z):
  k,u,v,t,w,tw,x,q,alpha,phibar,rhobar,primal,dual=z
  u=A(v)-alpha*u;beta=j.linalg.norm(u);u=u/beta;vn=AT(u)-beta*v;alpha=j.linalg.norm(vn);v=vn/alpha;t=u/alpha-(beta/alpha)*t;rho=j.hypot(rhobar,beta);c=rhobar/rho;sn=beta/rho;theta=sn*alpha;rhobar=-c*alpha;phi=c*phibar;phibar=sn*phibar;x=x+(phi/rho)*w;q=q+(phi/rho)*tw;w=v-(theta/rho)*w;tw=t-(theta/rho)*tw
  primal,dual=jax.lax.cond((k+1)%20==0,lambda _:(j.linalg.norm(B(x)-rhs)/norm,j.linalg.norm(B(BT(scale*q))-rhs)/norm),lambda _:(primal,dual),None)
  return k+1,u,v,t,w,tw,x,q,alpha,phibar,rhobar,primal,dual
 z=jax.lax.while_loop(cond,body,initial);return z[0],z[6],scale*z[7]
started=time.monotonic();k,x,lam=solve();jax.block_until_ready(lam);elapsed=time.monotonic()-started;gradient=transpose_divergence(lam,h);corrected=tuple(v-i*g for v,i,g in zip(velocity,inv,gradient));fixture=tuple(j.where(f,g,0.) for f,g in zip(fixed,gradient));work=sum(j.sum(I*v) for I,v in zip(fixture,velocity));change=.5*sum(j.sum(m*(b*b-a*a)) for m,a,b in zip(mass,velocity,corrected));loss=.5*sum(j.sum(m*(b-a)**2) for m,a,b in zip(mass,velocity,corrected));momentum=j.stack([j.sum(m*(b-a))-j.sum(I) for m,a,b,I in zip(mass,velocity,corrected,fixture)]);out=dict(step=int(p['step']),physical_steps_executed=0,seconds=elapsed,iterations=int(k),primal_relative=float(j.linalg.norm(B(x)-rhs)/norm),pressure_flux_relative=float(j.linalg.norm(B(BT(lam))-rhs)/norm),whitened_adjoint_gap=float(j.linalg.norm(x-BT(lam))/j.linalg.norm(x)),maximum_divergence_per_s=float(j.max(j.abs(j.where(fluid,divergence(corrected,h),0.)))/h**3),pressure_work_J=float(work),projection_loss_J=float(loss),pressure_work_identity_J=float(work-change-loss),momentum_residual_Ns=float(j.linalg.norm(momentum)),fixed_face_change=float(max(j.max(j.abs(j.where(f,b-a,0.))) for f,a,b in zip(fixed,velocity,corrected))),scope='All LSQR/operator/velocity/pressure tensors JAX. Host reads authenticated input only; zero physical advance. Velocity correction and pressure multiplier use the same recovered solution.');primal_velocity=tuple(v-r*z for v,r,z in zip(velocity,root,split(x)))
primal_change=.5*sum(j.sum(m*(b*b-a*a)) for m,a,b in zip(mass,velocity,primal_velocity));primal_loss=.5*sum(j.sum(m*(b-a)**2) for m,a,b in zip(mass,velocity,primal_velocity));primal_momentum=j.stack([j.sum(m*(b-a))-j.sum(I) for m,a,b,I in zip(mass,velocity,primal_velocity,fixture)]);out.update(primal_maximum_divergence_per_s=float(j.max(j.abs(j.where(fluid,divergence(primal_velocity,h),0.)))/h**3),primal_pressure_work_identity_J=float(work-primal_change-primal_loss),primal_momentum_residual_Ns=float(j.linalg.norm(primal_momentum)),primal_projection_loss_J=float(primal_loss),primal_fixed_face_change=float(max(j.max(j.abs(j.where(f,b-a,0.))) for f,a,b in zip(fixed,velocity,primal_velocity))),mass_metric_stationarity_absolute=float(j.linalg.norm(x-BT(lam))),primal_velocity_vs_pressure_velocity_mass_norm=float(j.linalg.norm(pack(tuple(j.sqrt(m)*(b-a) for m,a,b in zip(mass,primal_velocity,corrected))))))
Path('pressure-lsqr-review.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
