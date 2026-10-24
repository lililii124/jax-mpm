import jax
import jax.numpy as jnp
import numpy as np
from mpm_jax.materials.linear_metal import (linear_metal, pressure, volume_energy,
    temperature, energy_from_temperature, flow_stress, deviatoric_return)
from mpm_jax.linear_impact_mpm import (build_grid, make_state, step, thermodynamics,
    ImpactConfig, stable_timestep, energy_ledger, validate_state)
jax.config.update('jax_enable_x64', True)

def test_linear_pressure_recoverable_work_and_no_compression_heat():
    p=linear_metal()
    J=jnp.array([.5,.8,1.,1.2])
    np.testing.assert_allclose(pressure(J,0.,p),p.bulk_modulus*(1-J))
    np.testing.assert_allclose(jax.vmap(jax.grad(lambda x:volume_energy(x,p)))(J),
                               -pressure(J,0.,p)/p.density)
    np.testing.assert_allclose(temperature(J,volume_energy(J,p),p),300.,atol=1e-12)
    np.testing.assert_allclose(temperature(J,energy_from_temperature(J,1000.,p),p),1000.)

def test_plastic_heat_fraction_and_saturated_melt_softening():
    p=linear_metal()
    strain=jnp.diag(jnp.array([.02,-.01,-.01]))
    r=deviatoric_return(jnp.zeros((3,3)),0.,strain,jnp.eye(3),1.,1e-7,300.,p)
    assert r.valid and r.specific_plastic_work>0
    np.testing.assert_allclose(r.specific_plastic_heat,.9*r.specific_plastic_work)
    np.testing.assert_allclose(temperature(1.,r.specific_plastic_heat,p),
                               300.+.9*r.specific_plastic_work/900.)
    assert float(flow_stress(.1,100.,p.melt_temperature,p))==0.
    assert float(flow_stress(.1,100.,2*p.melt_temperature,p))==0.
    assert float(flow_stress(.1,100.,p.melt_temperature-1.,p))>0.

def tiny_state(*, contact=False):
    dx=1e-4
    cube=np.array(np.meshgrid([.25,.75],[.25,.75],[.25,.75],indexing='ij')).reshape(3,-1).T*dx
    x=np.concatenate((cube,cube+[dx if contact else 10*dx,0.,0.]))
    ids=np.repeat([0,1],len(cube))
    grid=build_grid(x,ids,dx=dx)
    p=linear_metal()
    v=np.zeros_like(x);v[:8,0]=5000. if contact else 10.
    state=make_state(x,np.full(len(x),(dx/2)**3),ids,p,grid,v=v)
    return state,p,grid

def test_tiny_5kms_contact_and_transfer_are_not_temperature_sources():
    state,p,grid=tiny_state(contact=True)
    config=ImpactConfig(strength=False)
    old_energy=float(energy_ledger(state,grid).sum())
    new,d=step(state,p,grid,config,stable_timestep(state,p,grid,config)*.1)
    assert new.valid and validate_state(new,p,grid)
    np.testing.assert_allclose(thermodynamics(new,p).temperature,300.,atol=1e-10)
    assert float(d.contact_projection_loss.sum())>0
    assert float(new.numerical_reservoir)>0
    np.testing.assert_allclose(energy_ledger(new,grid).sum(),old_energy,rtol=1e-13)
    np.testing.assert_allclose(d.physical_energy_residual,-d.numerical_energy_increment,atol=1e-14)
    assert abs(float(d.physical_energy_residual))>0

def test_tiny_plastic_temperature_exactly_tracks_plastic_work():
    state,p,grid=tiny_state()
    state=state._replace(C=jnp.broadcast_to(jnp.diag(jnp.array([2e5,-1e5,-1e5])),state.C.shape))
    config=ImpactConfig()
    dt=stable_timestep(state,p,grid,config)*.1
    new,d=step(state,p,grid,config,dt)
    assert new.valid
    # Cold work stores (1-beta)Wp, so it independently identifies plastic heat.
    expected=300.+p.heat_fraction/(1-p.heat_fraction)*(new.cold_work-state.cold_work)/p.heat_capacity
    np.testing.assert_allclose(thermodynamics(new,p).temperature,expected,atol=1e-10)
