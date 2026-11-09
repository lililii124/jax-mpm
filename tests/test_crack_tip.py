import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax.crack_tip import fit_tip_fields,mts_angle

jax.config.update('jax_enable_x64',True)


def test_tip_fit_recovers_modes_and_T_from_cardinal_ray_fields_after_rotation():
    KI,KII,T=3.,-.8,7.
    r=np.repeat([.6,.8],3)
    theta=np.tile([-np.pi/2,0.,np.pi/2],2)
    position=np.column_stack((r*np.cos(theta),r*np.sin(theta)))
    h=1/(2*np.sqrt(2))
    # Exact Westergaard values on the -90, 0 and +90 degree rays.
    first=np.array([[h,3*h,h],[1.,1.,0.],[h,3*h,-h]])
    second=np.array([[3*h,h,h],[0.,0.,1.],[-3*h,-h,h]])
    vector=(KI*np.tile(first,(2,1))+KII*np.tile(second,(2,1)))/np.sqrt(2*np.pi*r[:,None])
    stress=np.zeros((6,2,2));stress[:,0,0]=vector[:,0]+T;stress[:,1,1]=vector[:,1]
    stress[:,0,1]=stress[:,1,0]=vector[:,2]
    angle=.73
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    origin=np.array([.3,-.2])
    fit=jax.jit(fit_tip_fields)(jnp.asarray(position@rotation.T+origin),jnp.asarray(rotation@stress@rotation.T),
                               jnp.ones(6),jnp.asarray(origin),jnp.asarray(rotation[:,0]),1.)
    assert bool(fit.valid)
    np.testing.assert_allclose([fit.mode_I,fit.mode_II,fit.T],[KI,KII,T],rtol=1e-12)
    assert float(fit.relative_residual)<1e-12


def test_mts_mode_I_symmetry_and_mode_II_sign():
    np.testing.assert_allclose(mts_angle(1.,0.),0.)
    np.testing.assert_allclose(np.rad2deg(float(mts_angle(0.,1.))),-70.52877936550931,atol=1e-12)
    np.testing.assert_allclose(np.rad2deg(float(mts_angle(1.,.5))),-40.2078187220342,atol=1e-11)
    np.testing.assert_allclose(mts_angle(1.,.5),-mts_angle(1.,-.5),atol=1e-15)


def test_no_sample_is_reported_instead_of_inventing_a_tip_direction():
    fit=fit_tip_fields(jnp.array([[10.,10.]]),jnp.zeros((1,2,2)),jnp.ones(1),jnp.zeros(2),jnp.array([1.,0.]),1.)
    assert not bool(fit.valid) and float(fit.sample_volume)==0.
