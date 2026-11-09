import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import Config,neo_hookean
from mpm_jax.cut_geometry import polygon
from mpm_jax.cut_fragments import fragment
from mpm_jax.domain_mpm import step,kinetic_energy,elastic_energy

jax.config.update('jax_enable_x64',True)


def domains(affine=False):
    reference=polygon(.02*jnp.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]))
    points=jnp.array([[.43,.43],[.47,.43],[.43,.47],[.47,.47]])
    C=jnp.array([[.4,-.2],[.3,.7]]) if affine else jnp.zeros((2,2))
    return jax.vmap(lambda x:fragment(reference,1000.,x=x,v=C@x+jnp.array([.1,-.05]),C=C))(points)


def test_free_translation_and_separated_fields_preserve_rigid_motion():
    config=Config(dim=2,n_grid=16,dt=1e-3,material='neo-hookean',gravity=(0.,0.),boundary_cells=0)
    material=neo_hookean(1e3,.2);state=domains()
    fields=jnp.array([0,1,0,1]);velocity=jnp.where(fields[:,None]==0,state.v,-state.v)
    state=state._replace(v=velocity)
    following,work=step(state,material,config,jnp.asarray(0.),fields=fields,n_fields=2)
    assert bool(jnp.all(following.valid))
    np.testing.assert_allclose(following.x,state.x+config.dt*velocity,atol=1e-15)
    np.testing.assert_allclose(following.v,velocity,atol=2e-15)
    np.testing.assert_allclose(following.F,state.F,atol=2e-15)
    np.testing.assert_allclose(following.C,0.,atol=2e-14)
    assert abs(float(work.ledger_error))<1e-13
    assert abs(float(work.geometry_energy))<1e-13


def test_deforming_domains_expose_the_changing_moment_in_energy_ledger():
    config=Config(dim=2,n_grid=16,dt=1e-3,material='neo-hookean',gravity=(0.,0.),boundary_cells=0)
    material=neo_hookean(1e3,.2);state=domains(affine=True)
    following,work=step(state,material,config,jnp.asarray(0.))
    assert bool(jnp.all(following.valid))
    old=kinetic_energy(state,config)+elastic_energy(state,material)
    new=kinetic_energy(following,config)+elastic_energy(following,material)
    uncorrected=new-old+work.transfer_loss+work.boundary_projection_loss-work.boundary-work.gravity-work.external
    assert abs(float(work.geometry_energy))>1e-10
    np.testing.assert_allclose(uncorrected-work.geometry_energy,work.ledger_error,atol=2e-16)
    assert abs(float(work.ledger_error))<1e-10
    assert abs(float(work.bulk_error))<1e-10
