"""One explicitly separate implicit-midpoint-inertiaAPIC diagnostic; not a production solver.

Uses original MLS gradient4*w*d/h^2, stretch-corotated constitutive law,
physical end-guide time law, and a certified contact-free actualCP window.
"""
import numpy as np
from functools import partial


def make_solver(material,dt,dx,times,cross_count,steps=4,active_capacity=65536):
    import jax
    import jax.numpy as jnp
    from mpm_jax.fibre_mpm import FibreConfig,stencil,particle_to_grid,grid_to_particle,grid_coordinates
    from mpm_jax.materials.isotropic_brittle import _spectrum,update_brittle
    from examples.triaxial_weaving_geometry import fixture_position
    config=FibreConfig(dt=dt);mu=material.young/(2*(1+material.poisson));lam=material.young*material.poisson/((1+material.poisson)*(1-2*material.poisson))
    def skew(w):
        zero=jnp.zeros_like(w[...,0]);a,b,c=w[...,0],w[...,1],w[...,2]
        return jnp.stack((zero,-c,b,c,zero,-a,-b,a,zero),-1).reshape(w.shape[:-1]+(3,3))
    def response(F):
        clean,stretch,Q,valid=_spectrum(F);R=clean@((Q*(1/stretch)[...,None,:])@jnp.swapaxes(Q,-1,-2));trace=jnp.sum(stretch-1,axis=-1)
        P=2*mu*clean+(lam*trace-2*mu)[...,None,None]*R
        U=jnp.swapaxes(R,-1,-2)@clean;T=jnp.trace(U,axis1=-2,axis2=-1)[...,None,None]*jnp.eye(3)-U
        inverse=jnp.linalg.inv(T);energy=.5*lam*trace**2+mu*jnp.sum((stretch-1)**2,axis=-1)
        return P,R,inverse,trace,energy,valid
    @jax.jit
    def block(state,layout,positions):
        node_coordinates,_=grid_coordinates(layout)
        cap=3*cross_count;end_ids=jnp.asarray(np.r_[np.arange(cap),np.arange(len(state.x)-cap,len(state.x))]);labels=jnp.repeat(jnp.arange(2),cap*27)
        def single(carry,_):
            old,path=carry;support=stencil(old.x,old.fibre_id,layout)
            mass,predictor=particle_to_grid(old._replace(tau=jnp.zeros_like(old.tau)),layout,config,support)
            occupied=mass>0;positive=jnp.nonzero(occupied,size=active_capacity,fill_value=layout.node_count)[0]
            remap=jnp.full(layout.node_count+1,active_capacity,jnp.int32).at[positive].set(jnp.arange(active_capacity),mode='drop')
            indices=remap[support.indices];m=jnp.take(mass,positive,mode='fill',fill_value=0.);p=jnp.take(predictor,positive,axis=0,mode='fill',fill_value=0.)
            nodes=indices[end_ids].reshape(-1);low=jnp.full(active_capacity,2,jnp.int32).at[nodes].min(labels,mode='drop');high=jnp.full(active_capacity,-1,jnp.int32).at[nodes].max(labels,mode='drop')
            clamped=(high>=0)&(m>0);free=(m>0)&~clamped;owners=jnp.maximum(high,0)
            targets=(fixture_position(old.time+dt,times,positions,jnp)-fixture_position(old.time,times,positions,jnp))/dt
            vpred=jnp.where(m[:,None]>0,p/jnp.where(m>0,m,1.)[:,None],0.);v=jnp.where(clamped[:,None],2*targets[owners]-vpred,vpred)
            B=4/dx**2*support.weights[...,None]*support.dpos
            def gradient(field):
                gathered=jnp.take(field,indices,axis=0,mode='fill',fill_value=0.)
                return jnp.einsum('pki,pkj->pij',gathered,B)
            def scatter(P):
                tensor=P@jnp.swapaxes(old.F,-1,-2)
                values=old.volume[:,None,None]*jnp.einsum('pij,pkj->pki',tensor,B)
                return jnp.zeros((active_capacity,3)).at[indices].add(values,mode='drop')
            def evaluate(velocity):
                middle_velocity=.5*(vpred+velocity)
                F=(jnp.eye(3)+dt*gradient(middle_velocity))@old.F;Fmid=.5*(old.F+F);P,R,Tinv,trace,energy,valid=response(Fmid)
                force_gradient=scatter(P);residual=m[:,None]*velocity-p+dt*force_gradient
                residual=jnp.where(free[:,None],residual,0.)
                merit=.5*jnp.sum(m[:,None]*(velocity-vpred)**2)+4*jnp.sum(old.volume*energy)
                return F,P,R,Tinv,trace,force_gradient,residual,merit,jnp.all(valid)
            initial=evaluate(v);momentum_characteristic=jnp.sum(old.mass)*jnp.sqrt((lam+2*mu)/material.density)
            scale=jnp.maximum(jnp.linalg.norm(p)+jnp.linalg.norm(m[:,None]*v),1e-6*momentum_characteristic)
            def norm(r):return jnp.linalg.norm(r)/scale
            def outer_condition(c):iteration,_,data,_,_,_=c;return (iteration<12)&(norm(data[6])>1e-7)&data[8]
            def outer(c):
                iteration,velocity,data,total_cg,negative,backtracks=c;F,P,R,Tinv,trace,force_gradient,residual,merit,valid=data
                def tangent(delta):
                    dF=gradient(jnp.where(free[:,None],delta,0.))@old.F
                    A=jnp.swapaxes(R,-1,-2)@dF;S=A-jnp.swapaxes(A,-1,-2);axial=jnp.stack((S[:,2,1],S[:,0,2],S[:,1,0]),-1)
                    omega=jnp.einsum('pij,pj->pi',Tinv,axial);dR=R@skew(omega)
                    dP=2*mu*dF+lam*jnp.trace(A,axis1=-2,axis2=-1)[:,None,None]*R+(lam*trace-2*mu)[:,None,None]*dR
                    value=m[:,None]*delta+.25*dt**2*scatter(dP)
                    return jnp.where(free[:,None],value,delta)
                a=jnp.einsum('pji,pkj->pki',old.F,B);Ra=jnp.einsum('pij,pkj->pki',R,a)
                diag_values=old.volume[:,None,None]*(2*mu*jnp.sum(a*a,axis=-1)[...,None]+lam*Ra*Ra)
                diagonal=m[:,None]+.25*dt**2*jnp.zeros((active_capacity,3)).at[indices].add(diag_values,mode='drop');diagonal=jnp.where(free[:,None],jnp.maximum(diagonal,1e-30),1.)
                rhs=-residual;z=rhs/diagonal;rz=jnp.sum(rhs*z);zero=jnp.zeros_like(rhs);rhsnorm=jnp.maximum(jnp.linalg.norm(rhs),1e-30)
                def cg_condition(q):k,x,r,direction,rz,bad=q;return (k<40)&(jnp.linalg.norm(r)>1e-8*rhsnorm)&~bad
                def cg_body(q):
                    k,x,r,direction,rz,bad=q;Ad=tangent(direction);denominator=jnp.sum(direction*Ad);bad=denominator<=0
                    alpha=rz/jnp.maximum(denominator,1e-300);alpha=jnp.where(bad,0.,alpha);x=x+alpha*direction;r=r-alpha*Ad;z=r/diagonal;newrz=jnp.sum(r*z);direction=z+(newrz/jnp.maximum(rz,1e-300))*direction
                    return k+1,x,r,direction,newrz,bad
                k,delta,linear_r,_,_,bad=jax.lax.while_loop(cg_condition,cg_body,(jnp.asarray(0),zero,rhs,z,rz,jnp.asarray(False)))
                delta=jnp.where(bad,-residual/diagonal,delta);slope=jnp.sum(residual*delta)
                def line_condition(q):count,alpha,trial=q;return (count<10)&((trial[7]>merit+1e-4*alpha*slope)|~trial[8])
                def line_body(q):count,alpha,trial=q;alpha=alpha*.5;return count+1,alpha,evaluate(velocity+alpha*delta)
                count,alpha,trial=jax.lax.while_loop(line_condition,line_body,(jnp.asarray(0),jnp.asarray(1.),evaluate(velocity+delta)))
                return iteration+1,velocity+alpha*delta,trial,total_cg+k,negative+bad.astype(jnp.int32),backtracks+count
            iteration,v,data,cg,negative,backtracks=jax.lax.while_loop(outer_condition,outer,(jnp.asarray(0),v,initial,jnp.asarray(0),jnp.asarray(0),jnp.asarray(0)))
            F,P,R,Tinv,trace,force_gradient,residual,merit,valid=data
            full_velocity=jnp.zeros((layout.node_count+1,3)).at[positive].set(v,mode='drop')[:layout.node_count]
            newv,C=grid_to_particle(full_velocity,support,layout)
            middle_full=jnp.zeros((layout.node_count+1,3)).at[positive].set(.5*(vpred+v),mode='drop')[:layout.node_count]
            middle_particle,_=grid_to_particle(middle_full,support,layout);x=old.x+dt*middle_particle
            middle_history,_=update_brittle(old.history,.5*(old.F+F),material);history,tau=update_brittle(middle_history,F,material)
            reaction=m[:,None]*v-p+dt*force_gradient;reaction=jnp.where(clamped[:,None],reaction,0.);groups=jnp.zeros((2,3)).at[owners].add(reaction)
            impulse=jnp.sum(groups,axis=0);work=jnp.sum(groups*targets);absolute=jnp.sum(jnp.abs(jnp.sum(groups*targets,axis=1)))
            bare=jnp.where(m[:,None]>0,(p-dt*force_gradient)/jnp.where(m>0,m,1.)[:,None],0.);projection=.5*jnp.sum(m[:,None]*(v*v-bare*bare))
            path=path+jnp.linalg.norm(x-old.x,axis=1);maximum=jnp.max(abs(jnp.log(jnp.linalg.svd(F,compute_uv=False))))
            converged=(norm(residual)<=1e-7)&valid&(jnp.sum(occupied)<=active_capacity)&support.valid&stencil(x,old.fibre_id,layout).valid&jnp.all(history.valid)&jnp.all(jnp.linalg.det(F)>0)&(maximum<=.12)&jnp.all(history.damage<=1e-10)&(jnp.max(path)<.00015)&jnp.all((low==high)|~clamped)
            new=old._replace(x=x,v=newv,F=F,C=C,history=history,tau=tau,time=old.time+dt,valid=old.valid&converged,boundary_impulse=old.boundary_impulse+impulse,boundary_work=old.boundary_work+projection)
            node_position=jnp.take(node_coordinates,positive,axis=0,mode='fill',fill_value=0.)
            midpoint_position=node_position+.25*dt*(vpred+v)
            angular=jnp.sum(jnp.cross(midpoint_position,reaction),axis=0);old_angular=jnp.sum(jnp.cross(node_position,reaction),axis=0)
            kinetic_kick=.5*jnp.sum(m[:,None]*(v*v-vpred*vpred));internal_work=dt*jnp.sum(force_gradient*.5*(vpred+v));kick_closure=kinetic_kick+internal_work-work
            _,_,_,_,Uold,_=response(old.F);_,_,_,_,Unew,_=response(F);material_defect=jnp.sum(old.volume*(Unew-Uold))-internal_work
            old_APIC=.5*jnp.sum(old.mass*jnp.sum(old.v*old.v,axis=1))+.5*dx**2/4*jnp.sum(old.mass*jnp.sum(old.C*old.C,axis=(1,2)))
            new_APIC=.5*jnp.sum(old.mass*jnp.sum(newv*newv,axis=1))+.5*dx**2/4*jnp.sum(old.mass*jnp.sum(C*C,axis=(1,2)))
            transfer_defect=.5*jnp.sum(m[:,None]*(vpred*vpred-v*v))-old_APIC+new_APIC
            metrics=jnp.array([norm(residual),iteration,cg,negative,backtracks,work,absolute,maximum,jnp.sum(occupied),jnp.linalg.norm(residual),scale,momentum_characteristic,*angular,*old_angular,kick_closure,material_defect,transfer_defect])
            return (new,path),metrics
        (final,path),metrics=jax.lax.scan(single,(state,jnp.zeros(len(state.x))),None,length=steps)
        return final,path,metrics
    return block
