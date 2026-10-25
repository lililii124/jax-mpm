"""Check rigid acceleration, prescribed nodal motion and a fixed support."""
import argparse,json
from pathlib import Path
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from mpm_jax.core import Config
from mpm_jax.materials.elastic import neo_hookean
from mpm_jax.total_lagrangian import reference_grid,initial_state,step,NodalBoundary,momentum,sample_particles


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg=Config(n_grid=8,dim=3,dt=1e-3,gravity=(0,0,0),boundary_cells=0,material='neo-hookean')
    axes=np.arange(.25+1/32,.75,1/16);X=np.stack(np.meshgrid(axes,axes,axes,indexing='ij'),axis=-1).reshape(-1,3)
    ref=reference_grid(X,jnp.full(len(X),(1/16)**3),1000.,cfg);assert ref.valid
    mat=neo_hookean(2e4,.2);acceleration=jnp.array([.1,-.05,.02]);body=jnp.broadcast_to(acceleration,X.shape)
    @jax.jit
    def advance_free(state):return step(state,ref,mat,cfg,body,body)
    state=initial_state(ref,mat,cfg,body)
    for _ in range(20):state,_=advance_free(state)
    sampled=sample_particles(state,ref);T=20*cfg.dt
    free_position=float(jnp.max(abs(sampled.x-jnp.asarray(X)-.5*acceleration*T*T)))
    free_velocity=float(jnp.max(abs(sampled.v-acceleration*T)))
    free_F=float(jnp.max(abs(state.F-jnp.eye(3))))
    assert state.valid and max(free_position,free_velocity,free_F)<1e-12
    zero=jnp.zeros_like(body)
    def moving_boundary(t):
        shape=ref.nodes.shape
        return NodalBoundary(jnp.ones(shape,dtype=bool),jnp.broadcast_to(.5*acceleration*t*t,shape),jnp.broadcast_to(acceleration*t,shape),jnp.broadcast_to(acceleration,shape))
    @jax.jit
    def advance_moving(s):return step(s,ref,mat,cfg,zero,zero,moving_boundary(s.time),moving_boundary(s.time+cfg.dt))
    moving=initial_state(ref,mat,cfg,zero);rows=[]
    for _ in range(20):moving,row=advance_moving(moving);rows.append(row)
    K=.5*float(jnp.sum(ref.nodal_mass[:,None]*moving.velocity**2))
    moving_work_error=abs(K-float(moving.boundary_work))/max(K,1e-30)
    assert moving.valid and moving_work_error<1e-10
    fixed=ref.nodes[:,0]<=.25
    mask=jnp.broadcast_to(fixed[:,None],ref.nodes.shape)
    boundary=NodalBoundary(mask,jnp.zeros_like(ref.nodes),jnp.zeros_like(ref.nodes),jnp.zeros_like(ref.nodes))
    @jax.jit
    def advance_fixed(s):return step(s,ref,mat,cfg,body,body,boundary,boundary)
    held=initial_state(ref,mat,cfg,body);max_linear=max_angular=0.
    for _ in range(20):
        held,row=advance_fixed(held)
        max_linear=max(max_linear,float(row['momentum_error']));max_angular=max(max_angular,float(row['angular_momentum_error']))
    clamp_error=float(jnp.max(abs(held.displacement[fixed])))
    assert held.valid and clamp_error==0 and max(max_linear,max_angular)<1e-12
    report=dict(passed=True,free=dict(position_error_m=free_position,velocity_error_m_per_s=free_velocity,F_error=free_F),
        prescribed_motion=dict(kinetic_J=K,boundary_work_J=float(moving.boundary_work),relative_work_error=moving_work_error),
        fixed_support=dict(displacement_error_m=clamp_error,momentum_error_Ns=max_linear,angular_momentum_error_kg_m2_per_s=max_angular),
        scope='Small independent exact-solution/constraint controls; not a structural or contact validation')
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
