import numpy as np
import pytest

jax=pytest.importorskip('jax')
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp

from examples.blast_sandwich_design import Settings
from mpm_jax.reduced_tl import clamped_basis,normalize_basis,newmark,ks_update


def test_clamped_basis_contains_independent_physical_families_and_mass_projection():
    x=np.stack(np.meshgrid(np.linspace(0,.3,9),np.linspace(0,.3,9),np.linspace(0,.024,7),indexing='ij'),axis=-1).reshape(-1,3)
    basis,labels,fixed=clamped_basis(x,Settings(),3)
    assert basis.shape==(len(x),3,54)
    assert {label.rsplit('-',2)[0] for label in labels}=={'bending','core-shear-x','core-shear-y','core-compression','membrane-x','membrane-y'}
    np.testing.assert_array_equal(basis[fixed],0.)
    mass=np.linspace(.1,1,len(x))
    normalized,_,_=normalize_basis(basis,mass)
    flat=normalized.reshape(-1,54)
    np.testing.assert_allclose(flat.T@(np.repeat(mass,3)[:,None]*flat),np.eye(54),atol=2e-13)


def test_newmark_work_matches_linear_kinetic_plus_elastic_energy():
    settings=Settings()
    mass=jnp.asarray([[2.,.1],[.1,1.]])
    stiffness=jnp.asarray([[4e7,-1e7],[-1e7,3e7]])
    times,u,v,energy=jax.jit(lambda:newmark(mass,stiffness,jnp.asarray([.01,.02]),settings,2e-6))()
    assert abs(float(times[-1])-.002)<1e-16
    assert np.max(np.abs(u))>0 and np.isfinite(v).all()
    error=np.asarray(energy)[:,0]+np.asarray(energy)[:,1]-np.asarray(energy)[:,2]
    assert np.max(np.abs(error))/np.max(np.abs(energy))<1e-10


def test_streamed_ks_controls_overestimate_without_exponential_overflow():
    values=jnp.asarray([.2,.98,1.03,.4,.91])
    error=.005;rho=np.log(len(values))/error
    def result(x):
        maximum,total=ks_update(jnp.asarray(0.),jnp.asarray(0.),x[:2],rho,jnp.ones(2))
        maximum,total=ks_update(maximum,total,x[2:],rho,jnp.ones(3))
        return maximum+jnp.log(total)/rho
    value=float(result(values))
    assert float(values.max())<=value<=float(values.max())+error
    derivative=np.asarray(jax.grad(result)(values))
    assert np.isfinite(derivative).all() and np.all(derivative>=0)
    np.testing.assert_allclose(derivative.sum(),1,atol=2e-13)
