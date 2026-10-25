"""Finite-precision mixed pressure projection in the actual face-mass metric.

LSQR acts on B=D M^(-1/2), retaining primal correction and same-iterate dual.
Actual velocity divergence and stationarity are separate admission conditions.
No incompatible right-hand side is projected away or physical mass changed.
"""
import jax
import jax.numpy as j
from .resin_mac import divergence,transpose_divergence,PressureResult

def project(velocity,mass,fixed,fluid,h,dt,inverse,coverage,positive,tolerance,max_iterations):
    root=tuple(j.sqrt(v) for v in inverse);shapes=tuple(v.shape for v in velocity);sizes=tuple(v.size for v in velocity);ends=(0,sizes[0],sizes[0]+sizes[1],sum(sizes))
    def split(x):return tuple(x[ends[i]:ends[i+1]].reshape(shapes[i]) for i in range(3))
    def pack(v):return j.concatenate(tuple(a.ravel() for a in v))
    def B(x):return j.where(fluid,divergence(tuple(r*v for r,v in zip(root,split(x))),h),0.)
    def BT(q):return pack(tuple(r*g for r,g in zip(root,transpose_divergence(j.where(fluid,q,0.),h))))
    a,b,c=inverse;diagonal=h**4*(a[1:]+a[:-1]+b[:,1:]+b[:,:-1]+c[:,:,1:]+c[:,:,:-1]);scale=j.where(fluid,1/j.sqrt(j.maximum(diagonal,1e-300)),0.);rhs=j.where(fluid,divergence(velocity,h),0.);norm=j.linalg.norm(rhs)
    def A(x):return scale*B(x)
    def AT(q):return BT(scale*q)
    def solve(_):
        beta=j.linalg.norm(scale*rhs);u=scale*rhs/beta;v=AT(u);alpha=j.linalg.norm(v);v=v/alpha;t=u/alpha
        initial=(j.asarray(0,j.int32),u,v,t,v,t,j.zeros_like(v),j.zeros_like(u),alpha,beta,alpha,j.asarray(j.inf),j.asarray(j.inf))
        def cond(z):return (z[0]<max_iterations-128)&((z[-2]>tolerance)|(z[-1]>1e-10))
        def body(z):
            k,u,v,t,w,tw,x,q,alpha,phibar,rhobar,primal,stationarity=z
            u=A(v)-alpha*u;beta=j.linalg.norm(u);u=u/j.maximum(beta,1e-300);vn=AT(u)-beta*v;alpha=j.linalg.norm(vn);v=vn/j.maximum(alpha,1e-300);t=u/j.maximum(alpha,1e-300)-(beta/j.maximum(alpha,1e-300))*t;rho=j.hypot(rhobar,beta);c=rhobar/rho;sn=beta/rho;theta=sn*alpha;rhobar=-c*alpha;phi=c*phibar;phibar=sn*phibar;x=x+(phi/rho)*w;q=q+(phi/rho)*tw;w=v-(theta/rho)*w;tw=t-(theta/rho)*tw
            primal,stationarity=jax.lax.cond((k+1)%20==0,lambda _:(j.linalg.norm(B(x)-rhs)/norm,j.linalg.norm(x-BT(scale*q))/j.maximum(j.linalg.norm(x),1e-300)),lambda _:(primal,stationarity),None)
            return k+1,u,v,t,w,tw,x,q,alpha,phibar,rhobar,primal,stationarity
        z=jax.lax.while_loop(cond,body,initial);return z[0],z[6],scale*z[7]
    iterations,x,multiplier=jax.lax.cond(coverage&positive&(norm>0),solve,lambda _:(j.asarray(0,j.int32),j.zeros(sum(sizes)),j.zeros_like(rhs)),None)
    # Recover the dual of this same primal with a bounded rectangular correction.
    # This minimizes stationarity error, not pressure work or a ledger residual.
    def refine(_):
        error=x-BT(multiplier);beta=j.linalg.norm(error);u=error/beta;v=A(u);alpha=j.linalg.norm(v);v=v/j.maximum(alpha,1e-300)
        initial=(j.asarray(0,j.int32),u,v,v,j.zeros_like(multiplier),alpha,beta,alpha,j.zeros_like(multiplier),j.linalg.norm(error))
        def condition(z):return (z[0]<128)&(z[-1]>1e-10*j.linalg.norm(x))
        def body(z):
            k,u,v,w,q,alpha,phibar,rhobar,best,best_error=z
            u=AT(v)-alpha*u;beta=j.linalg.norm(u);u=u/j.maximum(beta,1e-300);vn=A(u)-beta*v;alpha=j.linalg.norm(vn);v=vn/j.maximum(alpha,1e-300);rho=j.hypot(rhobar,beta);c=rhobar/rho;sn=beta/rho;theta=sn*alpha;rhobar=-c*alpha;phi=c*phibar;phibar=sn*phibar;q=q+(phi/rho)*w;w=v-(theta/rho)*w
            candidate=scale*q;error=j.linalg.norm(x-BT(multiplier)-BT(candidate));better=error<best_error;best=j.where(better,candidate,best);best_error=j.minimum(best_error,error)
            return k+1,u,v,w,q,alpha,phibar,rhobar,best,best_error
        z=jax.lax.while_loop(condition,body,initial);return z[-2],z[0]
    need=(j.linalg.norm(x-BT(multiplier))>1e-10*j.maximum(j.linalg.norm(x),1e-300))&j.all(j.isfinite(multiplier))
    correction,refinement_iterations=jax.lax.cond(need,refine,lambda _:(j.zeros_like(multiplier),j.asarray(0,j.int32)),None);iterations=iterations+refinement_iterations
    gradient=tuple(a+b for a,b in zip(transpose_divergence(multiplier,h),transpose_divergence(correction,h)))
    primal_velocity=tuple(v-r*z for v,r,z in zip(velocity,root,split(x)));dual_velocity=tuple(v-i*g for v,i,g in zip(velocity,inverse,gradient));dual_residual=j.where(fluid,divergence(dual_velocity,h),0.);use_dual=(j.linalg.norm(dual_residual)<=10*tolerance*j.maximum(norm,1e-300))&j.all(j.stack([j.all(j.isfinite(v)) for v in dual_velocity]));corrected=jax.lax.cond(use_dual,lambda _:dual_velocity,lambda _:primal_velocity,None)
    x=pack(tuple(j.sqrt(m)*(a-b) for m,a,b in zip(mass,velocity,corrected)))
    fixture=tuple(j.where(f,g,0.) for f,g in zip(fixed,gradient));work=sum(j.sum(I*v) for I,v in zip(fixture,velocity));change=.5*sum(j.sum(m*(b*b-a*a)) for m,a,b in zip(mass,velocity,corrected));loss=.5*sum(j.sum(m*(b-a)**2) for m,a,b in zip(mass,velocity,corrected));identity=work-change-loss
    residual=j.where(fluid,divergence(corrected,h),0.);relative=j.linalg.norm(residual)/j.maximum(norm,1e-300);absolute=j.linalg.norm(x-BT(multiplier)-BT(correction));stationarity=absolute/j.maximum(j.linalg.norm(x),1e-300);reconstruction=j.linalg.norm(B(BT(multiplier+correction))-rhs)/j.maximum(norm,1e-300)
    momentum=j.stack([j.sum(m*(b-a))-j.sum(I) for m,a,b,I in zip(mass,velocity,corrected,fixture)]);momentum_scale=sum(j.sum(j.abs(m*(b-a)))+j.sum(j.abs(I)) for m,a,b,I in zip(mass,velocity,corrected,fixture));work_scale=j.maximum(j.maximum(j.abs(work),j.abs(change)),j.maximum(loss,1e-300));finite=j.all(j.stack([j.all(j.isfinite(v)) for v in corrected]));fixed_change=j.max(j.stack([j.max(j.abs(j.where(f,b-a,0.))) for f,a,b in zip(fixed,velocity,corrected)]))
    valid=coverage&positive&finite&(relative<=10*tolerance)&(stationarity<=1e-10)&(j.abs(identity)<=10*tolerance*work_scale)&(j.linalg.norm(momentum)<=10*tolerance*j.maximum(momentum_scale,1e-300))&(fixed_change==0)
    return PressureResult(corrected,-(multiplier+correction)/dt,fixture,relative,j.max(j.abs(residual))/h**3,work,loss,identity,coverage,valid,iterations,stationarity,absolute,reconstruction,multiplier,correction,use_dual)
