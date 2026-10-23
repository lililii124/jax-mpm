import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.adhesion_contrast import make_case
from examples.adhesive_array import make_case as array_case
from mpm_jax.adhesion import SpherePath

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def pair():
    return tuple(make_case(name, grid=16, ppc=1, frames=50) for name in
                 ("nonadhesive-pull-off", "adhesive-pull-off"))


def test_contrast_pair_changes_only_work_of_adhesion(pair):
    (plain, plain_substeps), (sticky, sticky_substeps) = pair
    assert plain_substeps == sticky_substeps
    assert plain.configs == sticky.configs
    assert plain.materials == sticky.materials
    for left, right in zip(jax.tree.leaves(plain.states), jax.tree.leaves(sticky.states)):
        np.testing.assert_array_equal(left, right)
    left, right = dict(plain.description), dict(sticky.description)
    for name in ("purpose", "suite_case", "comparison_role", "cohesive_nominal_stress_Pa"):
        left.pop(name)
        right.pop(name)
    left_law, right_law = dict(left.pop("adhesion")), dict(right.pop("adhesion"))
    assert left_law.pop("work_of_adhesion") == 0
    assert right_law.pop("work_of_adhesion") == 72
    assert left_law == right_law
    assert left_law["penalty_stiffness"] == 4.8e6
    assert left == right
    assert left["young_modulus_Pa"] == 8000
    assert left["geometry_scale"] == 1
    assert left["comparison_pair"] == "single-sphere-lift-hold-v1"


def test_contrast_withdrawal_hold_uses_actual_sphere_motion(pair):
    case, substeps = pair[0]
    description = case.description
    path = SpherePath(**description["rigid_spheres"][0])
    sample_y = description["contact_sampling_plane_y_m"]
    centers, velocities = jax.vmap(path.motion)(jnp.asarray(path.times))
    np.testing.assert_allclose(
        centers[:, 1] - path.radius - sample_y,
        [0.045, 0.045, -0.035, -0.035, 0.055, 0.055, 0.295, 0.295],
        atol=1e-15,
    )
    np.testing.assert_allclose(velocities, 0, atol=1e-14)
    times = np.arange(51) * substeps * case.configs[0].dt
    held = times[(times >= 0.72) & (times <= 0.94)]
    assert held[-1] - held[0] >= 0.10
    centers, velocities = jax.vmap(path.motion)(jnp.asarray(held))
    np.testing.assert_allclose(centers[:, 1], sample_y + path.radius + 0.055, atol=1e-15)
    np.testing.assert_allclose(velocities, 0, atol=1e-14)
    assert description["penalty_step_number_max"] <= 0.09
    assert times[-1] == pytest.approx(1.30)
    assert float(np.asarray(path.centers)[:, 1].max()) + path.radius < 1.0
    intervals = np.diff(path.times)
    increments = np.diff(np.asarray(path.centers), axis=0)
    peak_speed = np.linalg.norm(increments, axis=1) * np.pi / (2 * intervals)
    peak_acceleration = np.linalg.norm(increments, axis=1) * np.pi**2 / (2 * intervals**2)
    assert peak_speed.max() == pytest.approx(2.0943951023931953)
    assert peak_acceleration.max() == pytest.approx(36.554090374405)


def test_contrast_attraction_requires_the_current_surface_to_follow(pair):
    plain, sticky = pair[0][0], pair[1][0]
    time = 0.80
    for case in (plain, sticky):
        np.testing.assert_array_equal(case.forces(case.states, time)[0], 0)
    initial = sticky.states[0]
    top = initial.x[:, 1] == jnp.max(initial.x[:, 1])
    moved = initial._replace(x=initial.x.at[:, 1].add(jnp.where(top, 0.04, 0)))
    nonadhesive_force = plain.forces((moved,), time)[0]
    adhesive_force = sticky.forces((moved,), time)[0]
    np.testing.assert_array_equal(nonadhesive_force, 0)
    assert float(jnp.sum(adhesive_force[:, 1])) > 0
    np.testing.assert_allclose(jnp.sum(adhesive_force[:, (0, 2)], axis=0), 0, atol=1e-12)
    np.testing.assert_array_equal(adhesive_force[~top], 0)


def test_contrast_does_not_replace_the_standard_definition():
    case, _ = array_case(grid=16, ppc=1, suite_case="adhesive-pull-off")
    definition = case.description
    assert case.duration == 0.85
    assert definition["young_modulus_Pa"] == 12000
    assert definition["adhesion"]["work_of_adhesion"] == 12
    assert definition["adhesion"]["penalty_stiffness"] == 1.2e6
    assert definition["comparison_pair"] == "single-sphere-normal-pull-off-v1"
    assert "preset" not in definition and "comparison_targets" not in definition
    assert definition["rigid_spheres"][0]["times"] == (0.0, 0.06, 0.25, 0.42, 0.64, 0.85)
    standard_path = definition["rigid_spheres"][0]
    assert standard_path["centers"][-1][1] - standard_path["radius"] - definition["contact_sampling_plane_y_m"] == pytest.approx(0.145)
    assert case.configs[0].boundaries[0].upper[1] == 0.15


def test_contrast_rejects_unpaired_cases():
    with pytest.raises(ValueError, match="pull-off"):
        make_case("lateral-release")
    with pytest.raises(ValueError, match="pair"):
        array_case(suite_case="three-sphere-array", preset="lift-hold")
    with pytest.raises(ValueError, match="preset"):
        array_case(preset="unknown")
