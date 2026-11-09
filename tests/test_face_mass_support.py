import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import Config
from mpm_jax.cohesive_crack import face_velocity,scatter_face_force
from mpm_jax.extrinsic_crack import transfers

jax.config.update('jax_enable_x64',True)


def test_tiny_mass_face_node_is_excluded_without_losing_affine_motion_force_or_power():
    config=Config(dim=2,n_grid=16,dt=1e-5,gravity=(0.,0.),boundary_cells=0)
    size=17**2;index=jnp.arange(size)
    nodes=config.dx*jnp.stack((index//17,index%17),axis=-1)
    mass=jnp.tile(jnp.where(nodes[:,0]>=.5,1.,0.),2)
    tiny=8*17+8
    mass=mass.at[tiny].set(1e-20).at[tiny+size].set(1e-20)
    positions=jnp.array([[.49,.49],[.51,.51]])
    normals=jnp.array([[0.,1.],[0.,1.]])
    path=jnp.array([[.1,.1],[.2,.1]])
    transfer=transfers(positions,normals,1.,mass,config,path,jnp.asarray(2),nodes,
                       mass_weighted=False,minimum_mass_fraction=1e-8)
    assert bool(transfer.valid)
    np.testing.assert_array_equal(jnp.where((transfer.indices==tiny)|(transfer.indices==tiny+size),transfer.weights,0.),0.)
    A=jnp.array([[.3,-.4],[.2,.5]]);b=jnp.array([.8,-.6])
    velocity=jnp.tile(nodes@A.T+b,(2,1))
    traced=face_velocity(velocity,transfer)
    np.testing.assert_allclose(traced,positions@A.T+b,atol=1e-13)
    force=jnp.array([[3.,-2.],[-1.,4.]])
    nodal=scatter_face_force(force,transfer,velocity.shape)
    np.testing.assert_allclose(nodal.sum(axis=0),force.sum(axis=0),atol=1e-13)
    repeated=jnp.tile(nodes,(2,1))
    np.testing.assert_allclose(jnp.sum(repeated[:,0]*nodal[:,1]-repeated[:,1]*nodal[:,0]),
                               jnp.sum(positions[:,0]*force[:,1]-positions[:,1]*force[:,0]),atol=1e-13)
    np.testing.assert_allclose(jnp.sum(nodal*velocity),jnp.sum(force*traced),atol=1e-13)
