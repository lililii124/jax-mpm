import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mpm_jax.materials.shock_metal import (
    artificial_viscosity, deviatoric_return, energy_from_temperature, eos_state,
    flow_stress, hugoniot, pressure, shock_metal, temperature, volume_energy_step,
)

jax.config.update('jax_enable_x64', True)


@pytest.mark.parametrize('up',[0.,100.,1000.,2500.,4000.])
def test_planar_hugoniot_mass_momentum_energy_jump(up):
    p=shock_metal()
    J,ph,eh,us,valid=hugoniot(up,p)
    assert valid
    rho=p.density/J
    np.testing.assert_allclose(rho*(us-up),p.density*us,rtol=2e-15)
    np.testing.assert_allclose(ph,p.density*us*up,rtol=2e-15)
    np.testing.assert_allclose(eh,.5*ph*(1/p.density-1/rho),rtol=2e-14,atol=1e-10)
    state=eos_state(J,eh,p)
    assert state.valid
    np.testing.assert_allclose(state.pressure,ph,rtol=3e-15,atol=1e-5)
    assert float(state.entropy)>=-1e-10


def test_reference_units_and_energy_temperature_inverse():
    p=shock_metal()
    state=eos_state(1,0.,p)
    np.testing.assert_allclose([state.density,state.pressure,state.temperature,state.entropy],
                               [2700.,0.,300.,0.],atol=1e-12)
    np.testing.assert_allclose(state.bulk_modulus,p.density*p.c0**2,rtol=1e-15)
    np.testing.assert_allclose(state.sound_speed,p.c0,rtol=1e-15)
    J=jnp.array([.65,.8,1.,1.1,1.5])
    T=jnp.array([800.,500.,300.,280.,250.])
    energy=energy_from_temperature(J,T,p)
    np.testing.assert_allclose(temperature(J,energy,p),T,rtol=2e-14)


@pytest.mark.parametrize('J',[.65,.9,1.1,1.5])
def test_temperature_is_thermodynamically_consistent_with_pressure(J):
    p=shock_metal()
    T=700.
    energy=energy_from_temperature(J,T,p)
    denergy_dJ=jax.grad(lambda x:energy_from_temperature(x,T,p))(J)
    expected=-pressure(J,energy,p)/p.density+p.gamma0*p.heat_capacity*T
    np.testing.assert_allclose(denergy_dJ,expected,rtol=2e-11,atol=1e-6)


def test_sound_speed_uses_isentropic_not_fixed_energy_derivative():
    p=shock_metal()
    J=.8; e=1.1e6
    state=eos_state(J,e,p)
    dpdJ=jax.grad(lambda x:pressure(x,e,p))(J)
    dpde=jax.grad(lambda x:pressure(J,x,p))(e)
    independent=-dpdJ*J*J/p.density+dpde*state.pressure/state.density**2
    np.testing.assert_allclose(state.sound_speed**2,independent,rtol=2e-15)
    assert state.longitudinal_speed>state.sound_speed
    assert state.longitudinal_speed**2>=state.sound_speed**2+4*p.shear_modulus/(3*p.density)-1e-7


def test_midpoint_pressure_and_viscosity_work_are_counted_once():
    p=shock_metal()
    q=2e8; heat=1234.
    step=volume_energy_step(1.,.99,0.,q,heat,p)
    assert step.valid
    dv=(.99-1.)/p.density
    np.testing.assert_allclose(step.specific_energy,-(step.midpoint_pressure+q)*dv+heat,rtol=2e-15)
    np.testing.assert_allclose(step.viscosity_heat,-q*dv,rtol=1e-15)
    assert abs(float(step.residual))<1e-10
    assert eos_state(.99,step.specific_energy,p).entropy>0
    heating=volume_energy_step(1.,1.,0.,0.,900.,p)
    np.testing.assert_allclose(heating.specific_energy,900.)
    np.testing.assert_allclose(temperature(1.,heating.specific_energy,p),301.)


def test_homogeneous_release_converges_to_the_same_isentrope():
    p=shock_metal()
    J0=.9
    T0=500.
    e0=energy_from_temperature(J0,T0,p)
    S0=eos_state(J0,e0,p).entropy

    def release(n):
        grid=jnp.linspace(J0,1.05,n+1)
        def body(energy,pair):
            step=volume_energy_step(pair[0],pair[1],energy,0.,0.,p)
            return step.specific_energy,step.valid
        final,valid=jax.lax.scan(body,e0,jnp.stack((grid[:-1],grid[1:]),axis=-1))
        assert jnp.all(valid)
        state=eos_state(grid[-1],final,p)
        assert state.temperature<T0
        return abs(float(state.entropy-S0))

    coarse=release(30);fine=release(60)
    assert fine<.3*coarse
    assert fine<.1


def test_artificial_viscosity_is_dissipative_and_absent_in_release():
    p=shock_metal()
    div=jnp.array([-2e5,0.,2e5])
    q=artificial_viscosity(2700.,5350.,div,.0001,p)
    assert q[0]>0
    np.testing.assert_array_equal(q[1:],np.zeros(2))
    assert -q[0]*div[0]/2700.>0
    invalid=volume_energy_step(1.,1.01,0.,1.,0.,p)
    assert not invalid.valid
    assert jnp.isnan(artificial_viscosity(2700.,5350.,-1.,-.1,p))


def test_proper_rigid_rotation_preserves_strength_history_and_energy():
    p=shock_metal(A=1e12,B=0.)
    tau=jnp.diag(jnp.array([2e8,-1e8,-1e8]))
    angle=.7
    R=jnp.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
    step=deviatoric_return(tau,0.,jnp.zeros((3,3)),R,1.,1e-8,300.,p)
    assert step.valid
    np.testing.assert_allclose(step.kirchhoff_stress,R@tau@R.T,rtol=2e-15,atol=1e-7)
    np.testing.assert_allclose(step.specific_elastic_energy,jnp.sum(tau*tau)/(4*p.shear_modulus*p.density),rtol=2e-15)
    assert step.plastic_increment==step.specific_return_dissipation==0


@pytest.mark.parametrize('T',[300.,800.,1000.])
def test_radial_return_yield_and_dissipation_with_melt_softening(T):
    p=shock_metal()
    strain=jnp.diag(jnp.array([.02,-.01,-.01]))
    step=jax.jit(deviatoric_return)(jnp.zeros((3,3)),0.,strain,jnp.eye(3),.9,1e-6,T,p)
    assert step.valid and step.plastic_increment>0
    q=float(jnp.sqrt(1.5*jnp.sum(step.kirchhoff_stress**2)))
    np.testing.assert_allclose(q/.9,flow_stress(step.plastic_strain,step.plastic_increment/1e-6,T,p),atol=5e-6,rtol=2e-12)
    np.testing.assert_allclose(step.specific_plastic_heat+step.specific_cold_work,step.specific_plastic_work,rtol=2e-15,atol=1e-12)
    np.testing.assert_allclose(step.specific_return_dissipation-step.specific_plastic_work,
                               1.5*p.shear_modulus*step.plastic_increment**2/p.density,rtol=2e-12)
    if T>=p.melt_temperature:
        assert step.phase_extrapolated
        np.testing.assert_allclose(step.kirchhoff_stress,0.,atol=1e-5)
        assert pressure(.9,1e6,p)>0


def test_invalid_domains_are_reported_without_pressure_clipping():
    p=shock_metal()
    assert not eos_state(.2,0.,p).valid
    assert not eos_state(2.1,0.,p).valid
    assert not eos_state(1.,-1e9,p).valid
    assert pressure(1.01,0.,p)<0
    assert not deviatoric_return(jnp.zeros((3,3)),0.,jnp.zeros((3,3)),2*jnp.eye(3),1.,1e-6,300.,p).valid


@pytest.mark.parametrize('options',[{'density':0.},{'heat_fraction':1.1},{'minimum_J':.1},
                                    {'melt_temperature':200.},{'gamma0':-1.},{'c0':np.nan}])
def test_parameter_domain_is_explicit(options):
    with pytest.raises(ValueError):shock_metal(**options)
