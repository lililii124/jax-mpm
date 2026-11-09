import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import Config
from mpm_jax.cut_geometry import polygon
from mpm_jax.cut_fragments import fragment,split_fragment,invariants,kinetic_moment
from mpm_jax.domain_transfer import domain_stencil,particle_to_grid,grid_to_fragment,internal_force,velocity_gradient

jax.config.update('jax_enable_x64',True)


def test_integrated_split_transfer_preserves_nodal_mass_momentum_and_affine_fit():
    config=Config(dim=2,n_grid=16,dt=1e-5,gravity=(0.,0.),boundary_cells=0);size=17**2
    parent=fragment(polygon([[-.02,-.015],[.02,-.015],[.02,.015],[-.02,.015]]),1200.,
                    x=[.501,.487],v=[.6,-.4],F=[[1.07,.21],[-.11,.95]],C=[[.7,-.3],[1.1,.2]])
    children=split_fragment(parent,jnp.array([.6,.8]),.6*.501+.8*.487+.004)
    parent_batch=jax.tree.map(lambda a:a[None,...],parent)
    average=jax.jit(jax.vmap(lambda p:domain_stencil(p,config)))
    one=average(parent_batch);two=average(children)
    assert bool(jnp.all(one.valid)&jnp.all(two.valid))
    mass0,momentum0=particle_to_grid(parent_batch,one,size)
    mass1,momentum1=particle_to_grid(children,two,size)
    np.testing.assert_allclose(mass1,mass0,atol=2e-15,rtol=2e-13)
    np.testing.assert_allclose(momentum1,momentum0,atol=2e-15,rtol=2e-13)
    np.testing.assert_allclose(two.moment,jax.vmap(lambda p:kinetic_moment(p,config.dx))(children),atol=1e-17,rtol=2e-13)
    index=jnp.arange(size);nodes=config.dx*jnp.stack((index//17,index%17),axis=1)
    angular=jnp.sum(nodes[:,0]*momentum1[:,1]-nodes[:,1]*momentum1[:,0])
    np.testing.assert_allclose(angular,invariants(parent,config.dx).angular_momentum,atol=2e-15)
    affine=jnp.array([[.3,-.4],[.2,.5]]);shift=jnp.array([.8,-.6])
    velocity=nodes@affine.T+shift
    v,C=grid_to_fragment(velocity,two)
    np.testing.assert_allclose(v,children.x@affine.T+shift,atol=2e-14)
    np.testing.assert_allclose(C,jnp.broadcast_to(affine,C.shape),atol=1e-13)
    np.testing.assert_allclose(velocity_gradient(velocity,two),jnp.broadcast_to(affine,C.shape),atol=2e-13)
    # Internal work uses the same integrated gradient as the deformation rate.
    tau=jnp.array([[[2.,-.4],[-.4,1.1]],[[1.7,.3],[.3,.9]]])
    random=jnp.asarray(np.random.default_rng(912).normal(size=(size,2)))
    rv,rC=grid_to_fragment(random,two)
    particle_energy=.5*jnp.sum(children.mass*(jnp.sum(rv**2,axis=1)+jnp.trace(rC@two.moment@rC.swapaxes(-1,-2),axis1=-2,axis2=-1)))
    grid_energy=.5*jnp.sum(mass1[:,None]*random**2)
    assert float(particle_energy)<=float(grid_energy)
    force=internal_force(tau,children.reference_area,two,size)
    power=jnp.sum(children.reference_area*jnp.sum(tau*velocity_gradient(random,two),axis=(1,2)))
    np.testing.assert_allclose(jnp.sum(force*random)+power,0.,atol=1e-15)
    separated=two._replace(indices=two.indices+jnp.arange(2)[:,None]*size)
    opposites=jnp.concatenate((jnp.tile(jnp.array([1.,0.]),(size,1)),jnp.tile(jnp.array([-1.,0.]),(size,1))))
    sv,sC=grid_to_fragment(opposites,separated)
    np.testing.assert_allclose(sv,[[1.,0.],[-1.,0.]],atol=2e-14)
    np.testing.assert_allclose(sC,0.,atol=2e-13)


def test_domain_outside_quadrature_patch_is_rejected():
    config=Config(dim=2,n_grid=16,dt=1e-5,gravity=(0.,0.),boundary_cells=0)
    parent=fragment(polygon([[.3,.3],[.7,.3],[.7,.7],[.3,.7]]),1000.)
    support=domain_stencil(parent,config,tile_radius=1)
    assert not bool(support.valid)
