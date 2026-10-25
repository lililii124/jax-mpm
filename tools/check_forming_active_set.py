"""One actual16-node failed-contact fixture plus rank/fixture certificates."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from mpm_jax.fibre_forming_active_set import solve_nodes,project
from mpm_jax.fibre_forming_contact import PAIRS


def certificate(m,v,n,clamp):
    active=np.stack([(m[:,a]>0)&(m[:,b]>0) for a,b in PAIRS],axis=1)
    out,lam,found,res,rank=jax.jit(solve_nodes)(jnp.array(m),jnp.array(v),jnp.array(n),jnp.array(active),jnp.array(clamp))
    out,lam,found,res,rank=map(np.asarray,(out,lam,found,res,rank))
    impulse=np.zeros_like(v);closing=[]
    for k,(a,b) in enumerate(PAIRS):
        pair=lam[:,k,None]*n[:,k];impulse[:,a]-=pair;impulse[:,b]+=pair
        closing.append(np.sum((out[:,a]-out[:,b])*n[:,k],axis=1))
    closing=np.stack(closing,axis=1);fixture=np.where(clamp[:,:,None],-impulse,0.)
    momentum=np.sum(m[:,:,None]*(out-v),axis=1)-fixture.sum(axis=1)
    deltaK=.5*np.sum(m*np.sum(out*out-v*v,axis=2),axis=1)
    metric=.5*np.sum(m*np.sum((out-v)**2,axis=2),axis=1)
    work=np.sum(fixture*v,axis=(1,2));identity=work-deltaK-metric
    assert found.all() and np.all(lam>=0) and np.max(np.where(active,closing,0))<=1e-8
    assert np.max(np.where(active,np.abs(lam*closing),0))<1e-21
    momentum_scale=np.sum(m*(np.linalg.norm(out,axis=2)+np.linalg.norm(v,axis=2)),axis=1)+np.linalg.norm(fixture.sum(axis=1),axis=1)
    work_scale=.5*np.sum(m*np.sum(out*out+v*v,axis=2),axis=1)+np.abs(work)+metric
    relative_momentum=np.linalg.norm(momentum,axis=1)/np.maximum(momentum_scale,1e-300)
    relative_work=np.abs(identity)/np.maximum(work_scale,1e-300)
    assert relative_momentum.max()<256*np.finfo(float).eps
    assert relative_work.max()<256*np.finfo(float).eps
    return out,lam,dict(found=found.tolist(),max_primal_violation=float(np.maximum(np.where(active,closing,0),0).max()),max_complementarity_J=float(np.abs(lam*closing).max()),max_momentum_residual_Ns=float(np.linalg.norm(momentum,axis=1).max()),max_work_identity_J=float(np.abs(identity).max()),max_relative_momentum=float(relative_momentum.max()),max_relative_work=float(relative_work.max()),roundoff_bound=256*np.finfo(float).eps,rank_rejections=rank.tolist())


def main():
    with np.load('/input/contact-nodes.npz') as data:
        m=data['mass'].T;v=data['velocity_before'].transpose(1,0,2);n=data['normals'].transpose(1,0,2);clamp=data['clamped'].T
    out,lam,report=certificate(m,v,n,clamp)
    # Reconstruct one equivalent gradient triangle from the actual normals.
    # This exercises the full PGS->exact wrapper, node0 padding and capacity.
    edge=np.column_stack((n[0,0],n[0,2],-n[0,1]));_,_,vh=np.linalg.svd(edge);length=vh[-1]
    if length[0]<0:length=-length
    assert np.all(length>0)
    gradient=np.zeros((3,1,3));gradient[1,0]=length[0]*n[0,0];gradient[2,0]=length[2]*n[0,1];gradient*=m[0,:,None,None]
    centres=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    wrapper=jax.jit(project)(jnp.array(m[:1].T),jnp.array(v[:1].transpose(1,0,2)),jnp.array(gradient),jnp.array(centres),jnp.array(clamp[:1].T))
    assert bool(wrapper.valid) and int(wrapper.exact_nodes)==1 and int(wrapper.exact_successes)==1 and int(wrapper.sweeps)==64
    assert np.max(np.abs(np.asarray(wrapper.velocity)[:,0]-out[0]))<1e-12
    overflow=jax.jit(lambda mm,vv,gg,cc,cl:project(mm,vv,gg,cc,cl,fallback_capacity=1))(jnp.array(np.repeat(m[:1].T,2,axis=1)),jnp.array(np.repeat(v[:1].transpose(1,0,2),2,axis=1)),jnp.array(np.repeat(gradient,2,axis=1)),jnp.array(centres),jnp.array(np.repeat(clamp[:1].T,2,axis=1)))
    assert not bool(overflow.valid) and not bool(overflow.capacity_valid) and int(overflow.exact_nodes)==2
    # Independent primal SLSQP reference at the actually failed node.
    i=0;B=np.zeros((3,3,3))
    for k,(a,b) in enumerate(PAIRS):B[k,a]=n[i,k];B[k,b]=-n[i,k]
    B=B.reshape(3,9);weights=np.repeat(m[i]/m[i].max(),3);u=v[i].ravel()
    root_weights=np.sqrt(weights);white_u=root_weights*u;white_B=B/root_weights[None,:]
    result=minimize(lambda x:.5*np.sum((x-white_u)**2),white_u,jac=lambda x:x-white_u,constraints=[dict(type='ineq',fun=lambda x:-white_B@x,jac=lambda x:-white_B)],method='SLSQP',options=dict(ftol=1e-20,maxiter=1000))
    reference_velocity=result.x/root_weights
    objective=.5*np.sum(weights*(out[i].ravel()-u)**2)
    print(json.dumps(dict(reference_status=int(result.status),reference_message=str(result.message),reference_objective=float(result.fun),exact_objective=float(objective),max_velocity_difference=float(np.abs(reference_velocity-out[i].ravel()).max()))),flush=True)
    assert result.success and abs(result.fun-objective)<1e-14
    assert np.max(np.abs(reference_velocity-out[i].ravel()))<1e-7
    # Real fixture data with one prescribed velocity field exercises counter-work.
    clamp_case=clamp[:1].copy();clamp_case[:,1]=True
    _,_,fixture_report=certificate(m[:1],v[:1],n[:1],clamp_case)
    # Redundant all-collinear constraints: full active set singular; rank2 subset
    # has the analytic mass-weighted common velocity (no pseudoinverse).
    mr=np.array([[2.,3.,5.]])*1e-12;vr=np.zeros((1,3,3));vr[0,:,0]=[1.,0.,-1.];nr=np.zeros((1,3,3));nr[:,:,0]=1.
    rank_out,_,rank_report=certificate(mr,vr,nr,np.zeros((1,3),bool))
    assert np.max(np.abs(rank_out[0,:,0]+.3))<1e-12 and sum(rank_report['rank_rejections'])>0
    _,_,infeasible,_,_=jax.jit(solve_nodes)(jnp.array(mr),jnp.array(vr),jnp.array(nr),jnp.ones((1,3),bool),jnp.ones((1,3),bool))
    assert not bool(infeasible[0])
    angle=.71;Q=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
    transformed,_,_=certificate(m,v@Q.T+[.6,-.1,.3],n@Q.T,clamp)
    assert np.max(np.abs(transformed-(out@Q.T+[.6,-.1,.3])))<1e-12
    review=dict(passed=True,actual16=report,wrapper=dict(exact_nodes=int(wrapper.exact_nodes),successes=int(wrapper.exact_successes),residual=float(wrapper.closing_residual),original_pgs_sweeps=int(wrapper.sweeps),padding_preserves_node0=True,capacity_overflow_rejected=True),independent_slsqp=dict(success=bool(result.success),objective_difference=float(result.fun-objective),max_velocity_difference=float(np.abs(reference_velocity-out[i].ravel()).max())),fixture=fixture_report,redundant_rank=rank_report,incompatible_prescribed_fields_rejected=True,objectivity_max_error=float(np.abs(transformed-(out@Q.T+[.6,-.1,.3])).max()),actual_failed_node_velocity=out[0].tolist(),actual_failed_node_multipliers_Ns=lam[0].tolist())
    Path('active-set-review.json').write_text(json.dumps(review,indent=2));print(json.dumps(review,indent=2),flush=True)

if __name__=='__main__':main()
