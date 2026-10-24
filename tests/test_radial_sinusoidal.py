import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mpm_jax.adhesion import (DugdaleLaw, RadialSinusoid, sphere_interaction,
    radial_sinusoidal_interaction, radial_sinusoidal_gap_gradient)

jax.config.update("jax_enable_x64", True)


def test_explicit_mesh_zero_set_and_axis_normal():
    profile = RadialSinusoid()
    rho = jnp.array([0., .02, .04, .08, .12])
    y = -jnp.sqrt(.16**2-rho**2) + profile.height(rho)
    points = jnp.stack([rho, y, jnp.zeros_like(rho)], axis=1)
    gap, gradient = radial_sinusoidal_gap_gradient(points, jnp.zeros(3), .16, profile)
    np.testing.assert_allclose(gap, 0, atol=1e-15)
    np.testing.assert_allclose(gradient[0], [0,-1,0], atol=1e-15)
    np.testing.assert_allclose(profile.height(jnp.array([0,.04,.08])), [0,-.016,0], atol=1e-15)
    axis_point = jnp.array([[0., -.15, 0.]])
    axis_gradient = jax.grad(lambda x: radial_sinusoidal_interaction(
        x, .001, jnp.zeros(3), .16, DugdaleLaw(144,.025,4.8e6), profile).potential)(axis_point)
    assert np.isfinite(axis_gradient).all()
    # Analytic lower graph derivative and implicit normal agree.
    slope = rho/jnp.sqrt(.16**2-rho**2) - .008*(2*jnp.pi/.08)*jnp.sin(2*jnp.pi*rho/.08)
    np.testing.assert_allclose(gradient[:,0]/(-gradient[:,1]), slope, atol=1e-14)


def test_force_energy_translation_work_and_moment():
    law, profile = DugdaleLaw(144,.025,4.8e6), RadialSinusoid()
    points = jnp.array([[.021,-.15,.013],[.06,-.14,.012],[.1,-.09,.018]])
    areas = jnp.array([.001,.002,.0015])
    center = jnp.zeros(3)
    def interaction(x,c):
        return radial_sinusoidal_interaction(x,areas,c,.16,law,profile)
    result = interaction(points,center)
    np.testing.assert_allclose(result.forces, -jax.grad(lambda x: interaction(x,center).potential)(points), atol=1e-10)
    np.testing.assert_allclose(result.reaction, -jax.grad(lambda c: interaction(points,c).potential)(center), atol=1e-10)
    np.testing.assert_allclose(jnp.sum(jnp.cross(points,result.forces),axis=0)+result.reaction_moment, 0, atol=1e-12)
    velocity = jnp.array([.01,-.02,.03])
    eps=1e-6
    energy_rate=(interaction(points,center+eps*velocity).potential-interaction(points,center-eps*velocity).potential)/(2*eps)
    np.testing.assert_allclose(energy_rate,-jnp.dot(result.reaction,velocity),rtol=1e-7,atol=1e-7)


def test_zero_amplitude_exact_sphere_parity():
    points=jnp.array([[0,-.15,0],[.02,-.17,.01],[.2,-.3,.1]])
    args=(points,.001,jnp.zeros(3),.16,DugdaleLaw(144,.025,4.8e6))
    for actual, expected in zip(radial_sinusoidal_interaction(*args,RadialSinusoid(0,.08)),sphere_interaction(*args)):
        np.testing.assert_array_equal(actual,expected)


def test_example_matches_loading_and_small_step_smoke():
    from examples.radial_sinusoidal_pull_off import make_case
    from examples.adhesion_contrast import make_case as baseline
    from mpm_jax import step
    case,n=make_case(grid=8,ppc=1,frames=2)
    plain,m=baseline("adhesive-pull-off",grid=8,ppc=1,frames=2,profile="v4")
    assert n==m and case.configs==plain.configs
    assert case.description["rigid_spheres"]==plain.description["rigid_spheres"]
    assert case.description["adhesion"]==plain.description["adhesion"]
    # Three explicit steps at maximum indentation exercise nonzero corrugated loads.
    state=case.states[0]
    for i in range(3):
        force=case.forces((state,),.28+i*case.configs[0].dt)[0]
        assert np.isfinite(force).all() and np.linalg.norm(force)>0
        state=step(state,case.materials[0],case.configs[0],force=force)
    assert np.isfinite(state.x).all()


@pytest.mark.parametrize("a,w", [(-1,.08),(.01,0),(float("nan"),.08)])
def test_profile_validation(a,w):
    with pytest.raises(ValueError):
        RadialSinusoid(a,w)
