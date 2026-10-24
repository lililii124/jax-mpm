import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mpm_jax.contact_patches import contact_patches, patch_neighbours

jax.config.update('jax_enable_x64', True)


def test_graph_never_connects_adjacent_segment_ids_across_fibre_boundaries():
    pairs = np.array([[0, 4], [1, 4], [2, 4], [1, 5]], dtype=np.int32)
    owners = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    graph = patch_neighbours(pairs, owners)
    assert set(graph[0][graph[0] >= 0]) == {1, 3}
    assert np.all(graph[2] == -1)
    for a, row in enumerate(graph):
        for b in row[row >= 0]:
            assert a in graph[b]


def test_two_separated_contacts_on_one_fibre_pair_each_get_a_representative():
    pairs = np.array([[0, 8], [0, 9], [1, 9], [2, 10], [3, 11], [4, 12]], dtype=np.int32)
    graph = patch_neighbours(pairs, np.repeat([0, 1], 8))
    gap = jnp.array([-.2, -.5, -.1, 1., -.01, 1.])
    selected = contact_patches(gap, gap < 0, graph, capacity=16, max_patches=4)
    assert selected.valid and selected.patch_count == 2 and selected.active_count == 4
    np.testing.assert_array_equal(selected.representatives, [1, 4, 6, 6])


def test_connected_flat_patch_is_counted_once_with_a_stable_lowest_index_tie_break():
    pairs = np.array([[0, 4], [0, 5], [1, 4], [1, 5]], dtype=np.int32)
    graph = patch_neighbours(pairs, np.repeat([0, 1], 4))
    selected = contact_patches(jnp.full(4, -.1), jnp.ones(4, dtype=bool), graph, capacity=8, max_patches=4)
    assert selected.valid and selected.patch_count == 1
    np.testing.assert_array_equal(selected.representatives, [0, 4, 4, 4])


@pytest.mark.parametrize('defect', ['active_overflow', 'patch_overflow', 'iteration_limit', 'invalid_graph', 'nan_gap'])
def test_no_contact_or_unconverged_graph_is_silently_dropped(defect):
    pairs = np.column_stack((np.arange(6), np.arange(6)+8)).astype(np.int32)
    graph = patch_neighbours(pairs, np.repeat([0, 1], 8))
    gap = np.full(6, -.1)
    capacity, patches, iterations = 8, 8, 16
    if defect == 'active_overflow':
        capacity = 3
    elif defect == 'patch_overflow':
        graph[:] = -1
        patches = 2
    elif defect == 'iteration_limit':
        iterations = 1
    elif defect == 'invalid_graph':
        graph[0, 0] = 6
    else:
        gap[2] = np.nan
    result = contact_patches(gap, np.ones(6, dtype=bool), graph,
        capacity=capacity, max_patches=patches, max_iterations=iterations)
    assert not result.valid


def test_empty_coverage_graph_and_no_active_contact_have_no_sentinel_patch():
    result = contact_patches(jnp.empty(0), jnp.empty(0, dtype=bool), np.empty((0, 8), dtype=np.int32),
                             capacity=8, max_patches=4)
    assert result.valid and result.patch_count == result.active_count == 0
    np.testing.assert_array_equal(result.representatives, 0)


def test_nonempty_graph_without_active_cells_has_no_component():
    pairs = np.array([[0, 4], [1, 5]], dtype=np.int32)
    graph = patch_neighbours(pairs, np.repeat([0, 1], 4))
    result = contact_patches(jnp.ones(2), jnp.zeros(2, dtype=bool), graph, capacity=8, max_patches=4)
    assert result.valid and result.patch_count == result.active_count == 0
    np.testing.assert_array_equal(result.representatives, 2)


def test_scalar_gap_rejected_before_component_shape_access():
    with pytest.raises(ValueError, match='one-dimensional'):
        contact_patches(jnp.asarray(0.), jnp.asarray(False), np.empty((0, 8), dtype=np.int32))


def test_duplicate_coverage_cells_are_rejected_instead_of_double_counted():
    with pytest.raises(ValueError, match='unique'):
        patch_neighbours(np.array([[0, 2], [0, 2]], dtype=np.int32), np.array([0, 0, 1, 1]))
