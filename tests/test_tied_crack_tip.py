import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import Config,block,neo_hookean
from mpm_jax.cohesive_crack import CohesiveLaw,elastic_energy,kinetic_energy
from mpm_jax.extrinsic_crack import initialize,segment_positions,step,surface_response

jax.config.update('jax_enable_x64',True)


def test_tied_finite_tip_preserves_internal_momentum_and_discrete_work():
    config=Config(dim=2,n_grid=16,dt=1e-5,gravity=(0.,0.),boundary_cells=0)
    material=neo_hookean(1e4,.2)
    state=block(config,(.25,.375),(.75,.625),dtype=jnp.float64)
    state=state._replace(F=state.F.at[:,1,1].set(1.01))
    path=initialize([[.25,.5],[.45,.5],[.6,.5]],capacity=6)
    law=CohesiveLaw(100.,1.,1e4)
    path=path._replace(birth_traction=path.birth_traction.at[:2,1].set(100.),
                      maximum=path.maximum.at[:2].set(.01),bonded=path.bonded.at[:2].set(True))

    def energy(s,p):
        response=surface_response(segment_positions(p.plus,p.minus),p.maximum,p.birth_traction,p.bonded,law)
        return kinetic_energy(s,config)+elastic_energy(s,material,config)+jnp.sum(p.area*(response.stored+response.dissipated))

    following,newpath,work=step(state,path,material,config,law,consistent_gradients=True,couple_tip_nodes=True)
    assert bool(following.valid & newpath.valid)
    np.testing.assert_allclose(energy(following,newpath)-energy(state,path)+work.transfer_loss+work.boundary_projection_loss,
                               work.boundary+work.gravity,atol=1e-10)
    np.testing.assert_allclose(jnp.sum(following.mass[:,None]*following.v,axis=0),0.,atol=1e-10)
    np.testing.assert_allclose(newpath.plus[newpath.count-1],newpath.minus[newpath.count-1],atol=1e-14)
