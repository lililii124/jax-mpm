import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.adhesion_contrast import make_case
from examples.adhesive_array import make_case as array_case
from mpm_jax.adhesion import DugdaleLaw, SpherePath

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def pair():
    return tuple(make_case(name, grid=16, ppc=1, frames=50) for name in
                 ("nonadhesive-pull-off", "adhesive-pull-off"))


@pytest.fixture(scope="module")
def pair_v3():
    return tuple(make_case(name, grid=16, ppc=1, frames=50, profile="v3") for name in
                 ("nonadhesive-pull-off", "adhesive-pull-off"))


@pytest.fixture(scope="module")
def pair_v4():
    return tuple(make_case(name, grid=16, ppc=1, frames=50, profile="v4") for name in
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
    with pytest.raises(ValueError, match="profile"):
        make_case("adhesive-pull-off", profile="unknown")


def test_v3_changes_only_two_hold_waypoints_and_keeps_material_and_time_step(pair, pair_v3):
    for (old, old_steps), (new, new_steps) in zip(pair, pair_v3):
        assert old_steps == new_steps
        assert old.configs == new.configs and old.materials == new.materials
        for before, after in zip(jax.tree.leaves(old.states), jax.tree.leaves(new.states)):
            np.testing.assert_array_equal(before, after)
        a, b = old.description, new.description
        assert a["comparison_targets"] == b["comparison_targets"]
        assert a["adhesion"] == b["adhesion"]
        assert a["loading_stages_s"] == b["loading_stages_s"]
        assert a["preset"] == "lift-hold" and "loading_revision" not in a
        assert b["preset"] == "lift-hold-v3" and b["loading_revision"] == 3
        assert b["comparison_pair"] == "single-sphere-lift-hold-v3"
        assert b["comparison_hold_bottom_clearance_m"] == .035
        first, second = a["rigid_spheres"][0], b["rigid_spheres"][0]
        assert first["radius"] == second["radius"] and first["times"] == second["times"]
        expected = np.asarray(first["centers"]).copy()
        expected[[4, 5], 1] -= .020
        np.testing.assert_allclose(second["centers"], expected, atol=1e-15)
        path = SpherePath(**second)
        for instant in (.728, .832, .936):
            center, velocity = path.motion(instant)
            assert float(center[1] - path.radius - b["contact_sampling_plane_y_m"]) == pytest.approx(.035)
            np.testing.assert_array_equal(velocity, np.zeros(3))
        assert second["centers"][-1][1] - second["radius"] - b["contact_sampling_plane_y_m"] == pytest.approx(.295)


def test_v3_matched_pair_differs_only_in_work_and_needs_computed_surface_uplift(pair_v3):
    (plain, plain_steps), (sticky, sticky_steps) = pair_v3
    assert plain_steps == sticky_steps and plain.configs == sticky.configs
    left, right = dict(plain.description), dict(sticky.description)
    for name in ("purpose", "suite_case", "comparison_role", "cohesive_nominal_stress_Pa"):
        left.pop(name)
        right.pop(name)
    a, b = dict(left.pop("adhesion")), dict(right.pop("adhesion"))
    assert a.pop("work_of_adhesion") == 0 and b.pop("work_of_adhesion") == 72
    assert a == b and left == right
    for case in (plain, sticky):
        np.testing.assert_array_equal(case.forces(case.states, .8)[0], 0)
    initial = sticky.states[0]
    top = initial.x[:, 1] == jnp.max(initial.x[:, 1])
    moved = initial._replace(x=initial.x.at[:, 1].add(jnp.where(top, .02, 0)))
    np.testing.assert_array_equal(plain.forces((moved,), .8)[0], 0)
    force = sticky.forces((moved,), .8)[0]
    assert float(jnp.sum(force[:, 1])) > 0
    np.testing.assert_array_equal(force[~top], 0)


def test_v4_preserves_v3_path_initial_state_and_numerical_controls(pair_v3, pair_v4):
    for (old, old_steps), (new, new_steps) in zip(pair_v3, pair_v4):
        assert old_steps == new_steps
        assert old.configs == new.configs and old.materials == new.materials
        for before, after in zip(jax.tree.leaves(old.states), jax.tree.leaves(new.states)):
            np.testing.assert_array_equal(before, after)
        a, b = dict(old.description), dict(new.description)
        assert a.pop("preset") == "lift-hold-v3" and b.pop("preset") == "lift-hold-v4"
        assert a.pop("loading_revision") == 3 and b.pop("loading_revision") == 4
        assert a.pop("comparison_pair") == "single-sphere-lift-hold-v3"
        assert b.pop("comparison_pair") == "single-sphere-lift-hold-v4"
        assert b["comparison_hold_bottom_clearance_m"] == .035
        old_law, new_law = dict(a.pop("adhesion")), dict(b.pop("adhesion"))
        assert new_law.pop("work_of_adhesion") == 2 * old_law.pop("work_of_adhesion")
        assert old_law == new_law
        assert b.pop("cohesive_nominal_stress_Pa") == 2 * a.pop("cohesive_nominal_stress_Pa")
        assert a == b


def test_v4_pair_only_changes_work_and_doubles_attraction_at_the_same_deformation(pair_v3, pair_v4):
    (plain, steps), (sticky, sticky_steps) = pair_v4
    assert steps == sticky_steps and plain.configs == sticky.configs
    left, right = dict(plain.description), dict(sticky.description)
    for name in ("purpose", "suite_case", "comparison_role", "cohesive_nominal_stress_Pa"):
        left.pop(name)
        right.pop(name)
    a, b = dict(left.pop("adhesion")), dict(right.pop("adhesion"))
    assert a.pop("work_of_adhesion") == 0 and b.pop("work_of_adhesion") == 144
    assert a == b and left == right
    assert sticky.description["cohesive_nominal_stress_Pa"] == 5760
    for case in (plain, sticky):
        np.testing.assert_array_equal(case.forces(case.states, .8)[0], 0)
    initial = sticky.states[0]
    top = initial.x[:, 1] == jnp.max(initial.x[:, 1])
    moved = initial._replace(x=initial.x.at[:, 1].add(jnp.where(top, .02, 0)))
    np.testing.assert_array_equal(plain.forces((moved,), .8)[0], 0)
    force = sticky.forces((moved,), .8)[0]
    old_force = pair_v3[1][0].forces((moved,), .8)[0]
    assert float(jnp.sum(force[:, 1])) > 0
    np.testing.assert_allclose(force, 2 * old_force, rtol=1e-14, atol=1e-12)
    np.testing.assert_array_equal(force[~top], 0)


def test_v4_potential_traction_and_compressive_branch(pair_v3, pair_v4):
    old = DugdaleLaw(**pair_v3[1][0].description["adhesion"])
    law = DugdaleLaw(**pair_v4[1][0].description["adhesion"])
    gaps = jnp.array([-.002, -.0001, .005, .0125, .024, .03])
    derivative = jax.vmap(jax.grad(law.potential))(gaps)
    np.testing.assert_allclose(derivative, -law.traction(gaps), rtol=1e-14)
    np.testing.assert_array_equal(law.traction(gaps[:2]), old.traction(gaps[:2]))
    np.testing.assert_array_equal(law.traction(gaps[2:5]), 2 * old.traction(gaps[2:5]))
    assert float(law.potential(law.interaction_range) - law.potential(0.)) == 144
    np.testing.assert_array_equal(law.traction(gaps[-1:]), 0)
