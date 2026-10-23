"""NumPy checks of the actual boundary code, without importing the JAX solver."""

import ast
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def source_functions():
    environment = {"np": np, "jnp": np, "dataclass": dataclass, "replace": replace}
    boundary_tree = ast.parse((ROOT / "mpm_jax/boundaries.py").read_text(encoding="utf-8"))
    beam_tree = ast.parse((ROOT / "examples/brittle_engineering.py").read_text(encoding="utf-8"))
    wanted = {"VelocityBox", "MovingVelocityBox", "fixed_box"}
    nodes = [node for node in boundary_tree.body if getattr(node, "name", None) in wanted]
    wanted = {"LoadingBox", "_symmetric_concrete_boundaries", "_concrete_reference_reflections", "_concrete_symmetry_observables"}
    nodes += [node for node in beam_tree.body if getattr(node, "name", None) in wanted]
    beam = next(node for node in beam_tree.body if getattr(node, "name", None) == "_concrete_beam")
    end = next(index for index, node in enumerate(beam.body) if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "boundaries" for target in node.targets))
    beam.body = beam.body[:end + 1] + [ast.Return(value=ast.Name(id="boundaries", ctx=ast.Load()))]
    nodes.append(beam)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "<actual-beam-and-boundaries>", "exec"), environment)
    return environment


def boundaries(functions, grid):
    original = functions["_concrete_beam"](
        grid=grid, ppc=1, model="hjc", frames=60, displacement=.012,
        loading_duration=.0012, hold_duration=.0003, ramp_time=.0003, profile="strain-floor",
    )
    return original, functions["_symmetric_concrete_boundaries"](original, grid, .4)


def selected_nodes(boundary, axis, grid):
    positions = np.arange(grid + 1) * (.4 / grid)
    return np.flatnonzero((positions >= boundary.lower[axis]) & (positions <= boundary.upper[axis]))


@pytest.mark.parametrize("grid", [12, 16, 24, 32, 40, 48, 64, 80, 96, 128])
def test_actual_node_masks_are_mirror_pairs_and_single_center_planes(source_functions, grid):
    original, actual = boundaries(source_functions, grid)
    left, right, x_plane, z_plane, loading = actual
    assert loading is original[-1]
    for support, retained in zip((left, right), original[:2]):
        assert support.lower[1:] == retained.lower[1:]
        assert support.upper[1:] == retained.upper[1:]
        assert support.lower[0] == np.nextafter(retained.lower[0], -np.inf)
        assert support.upper[0] == np.nextafter(retained.upper[0], np.inf)
    assert left.components == right.components == (False, True, False)
    assert x_plane.components == (True, False, False)
    assert z_plane.components == (False, False, True)
    np.testing.assert_array_equal(selected_nodes(x_plane, 0, grid), [grid // 2])
    np.testing.assert_array_equal(selected_nodes(z_plane, 2, grid), [grid // 2])
    for axis in (0, 1, 2):
        support_left = selected_nodes(left, axis, grid)
        support_right = selected_nodes(right, axis, grid)
        assert support_left.size > 0
        np.testing.assert_array_equal(support_left, grid - support_right[::-1] if axis == 0 else support_right)
    for boundary in (x_plane, z_plane, loading):
        for axis in (0, 2):
            selected = selected_nodes(boundary, axis, grid)
            np.testing.assert_array_equal(selected, grid - selected[::-1])
    if grid == 64:
        np.testing.assert_array_equal(selected_nodes(loading, 0, grid), [30, 31, 32, 33, 34])
        np.testing.assert_array_equal(selected_nodes(left, 0, grid), [11, 12, 13])
        np.testing.assert_array_equal(selected_nodes(right, 0, grid), [51, 52, 53])


def apply_all(boundaries, velocity, positions, time):
    for boundary in boundaries:
        velocity = boundary.apply(velocity, positions, time)
    return velocity


@pytest.mark.parametrize("axis", [0, 2])
@pytest.mark.parametrize("time", [0., .00015, .0006, .00105, .0012, .0015])
def test_full_boundary_operator_commutes_with_vector_reflection(source_functions, axis, time):
    grid = 32
    original, actual = boundaries(source_functions, grid)
    coordinates = np.arange(grid + 1) * (.4 / grid)
    positions = np.stack(np.meshgrid(coordinates, coordinates, coordinates, indexing="ij"), axis=-1).reshape(-1, 3)
    reflected_ids = np.flip(np.arange(len(positions)).reshape((grid + 1,) * 3), axis=axis).ravel()
    sign = np.ones(3)
    sign[axis] = -1
    velocity = np.random.default_rng(195).normal(size=positions.shape)
    result = apply_all(actual, velocity, positions, time)
    mirrored_input = velocity[reflected_ids] * sign
    mirrored_result = apply_all(actual, mirrored_input, positions, time)
    np.testing.assert_array_equal(mirrored_result, result[reflected_ids] * sign)
    if axis == 0:
        legacy = apply_all(original, velocity, positions, time)
        legacy_mirrored = apply_all(original, mirrored_input, positions, time)
        assert np.max(np.abs(legacy_mirrored - legacy[reflected_ids] * sign)) > .1


def test_rollers_and_center_planes_leave_tangential_components_free(source_functions):
    _, actual = boundaries(source_functions, 64)
    positions = np.array([[.075, .15, .175], [.325, .15, .175], [.2, .18, .175], [.1, .18, .2]])
    velocity = np.tile([1., 2., 3.], (4, 1))
    observed = apply_all(actual, velocity, positions, 0.)
    np.testing.assert_array_equal(observed, [[1, 0, 3], [1, 0, 3], [0, 2, 3], [1, 2, 0]])


def test_symmetry_case_rejects_a_grid_without_the_center_node(source_functions):
    original, _ = boundaries(source_functions, 64)
    with pytest.raises(ValueError, match="even grid"):
        source_functions["_symmetric_concrete_boundaries"](original, 63, .4)


def test_reference_pairing_and_observers_detect_asymmetry_after_particle_reordering(source_functions):
    reference = np.stack(np.meshgrid([.1, .15, .2, .25, .3], [.15, .2], [.18, .2, .22], indexing="ij"), axis=-1).reshape(-1, 3)
    reference = reference[np.random.default_rng(82).permutation(len(reference))]
    pairs = source_functions["_concrete_reference_reflections"](reference, .2)
    displacement = np.column_stack((.02 * (reference[:, 0] - .2), -.01 * np.ones(len(reference)), -.03 * (reference[:, 2] - .2)))
    damage = np.where(reference[:, 0] < .2 - 1e-12, .75, .25)
    damage[np.abs(reference[:, 0] - .2) < 1e-12] = 1.
    state = SimpleNamespace(x=reference + displacement, volume=np.full(len(reference), 2.), mass=np.full(len(reference), 3.), history=SimpleNamespace(damage=damage))
    observed = source_functions["_concrete_symmetry_observables"](state, reference, pairs, .2)
    assert observed["left_mean_damage_reference_volume"] == .75
    assert observed["right_mean_damage_reference_volume"] == .25
    assert observed["left_damaged_reference_volume_fraction"] == 1.
    assert observed["right_damaged_reference_volume_fraction"] == 0.
    assert observed["damage_mirror_x_max"] == .5
    np.testing.assert_allclose(observed["damage_mirror_x_mae"], .4)
    np.testing.assert_allclose(observed["damage_mirror_x_rms"], np.sqrt(.2))
    assert observed["damage_mirror_z_max"] == 0.
    assert observed["displacement_mirror_x_rms_m"] < 1e-15
    assert observed["displacement_mirror_z_rms_m"] < 1e-15
    np.testing.assert_allclose(observed["center_of_mass_displacement_y_m"], -.01)
    for paired in pairs.values():
        np.testing.assert_array_equal(paired[paired], np.arange(len(reference)))
    with pytest.raises(ValueError, match="complete Cartesian lattice"):
        source_functions["_concrete_reference_reflections"](reference[:-1], .2)
