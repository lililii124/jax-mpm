"""Host-only checks of the actual sparse layout builder, without importing JAX."""

import ast
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope='module')
def builders():
    path = Path(__file__).resolve().parents[1]/'mpm_jax/fibre_mpm.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    layout = next(node for node in tree.body if getattr(node, 'name', None) == 'PackedLayout')
    layout.decorator_list = layout.decorator_list[1:]
    functions = [node for node in tree.body if getattr(node, 'name', None) in ('build_layout', 'build_tube_layout')]
    env = {'np': np, 'jnp': np, 'jax': SimpleNamespace(Array=np.ndarray), 'dataclass': dataclass, 'product': product}
    exec(compile(ast.Module(body=[layout, *functions], type_ignores=[]), '<actual-tube-layout>', 'exec'), env)
    return env


def test_dilation_retains_exact_reference_stencils_and_full_padding(builders):
    x, ids = np.array([[.105, .105, .105]]), np.array([0], dtype=np.int32)
    layout = builders['build_tube_layout'](x, ids, dx=.01, padding_cells=6,
        reference_bounds=[[0., .3, 0., .3, 0., .3]])
    base = np.floor(x[0]/.01-.5).astype(int)
    expected = np.array(list(product(*(range(int(b)-6, int(b)+9) for b in base))))
    linear = layout.offsets[0]+np.sum((expected-layout.lower[0])*layout.strides[0],axis=1)
    np.testing.assert_array_equal(layout.dense_indices, np.sort(linear))
    assert layout.node_count == 15**3
    assert layout.node_map.dtype == np.int32 and layout.dense_indices.dtype == np.int32
    np.testing.assert_array_equal(layout.node_map[layout.dense_indices], np.arange(layout.node_count))
    assert np.count_nonzero(layout.node_map == -1) == len(layout.node_map)-layout.node_count


def test_same_location_fibres_have_distinct_active_grid_nodes(builders):
    x = np.array([[.105, .105, .105], [.105, .105, .105]])
    layout = builders['build_tube_layout'](x, np.array([0,1]), dx=.01, padding_cells=2)
    base = np.floor(x/.01-.5).astype(int)
    dense_ids = layout.offsets+np.sum((base-layout.lower)*layout.strides,axis=1)
    assert layout.node_map[dense_ids[0]] != layout.node_map[dense_ids[1]]
    assert np.unique(layout.dense_indices).size == layout.node_count


def test_compression_keeps_identical_aabbs_phase_and_dense_index_budget(builders):
    t = np.linspace(.03, .3, 50)
    x = np.column_stack((t, .05+.005*np.sin(t*30), .8*t))
    ids = np.zeros(len(x), dtype=np.int32)
    dense = builders['build_layout'](x, ids, dx=.01)
    tube = builders['build_tube_layout'](x, ids, dx=.01)
    for name in ('lower','shape','strides','offsets','origin'):
        np.testing.assert_array_equal(getattr(tube,name),getattr(dense,name))
    assert tube.node_count < .75*dense.node_count
    assert tube.node_map.size == dense.node_count
    with pytest.raises(ValueError, match='node budget'):
        builders['build_tube_layout'](x, ids, dx=.01, max_nodes=dense.node_count-1)


def test_old_positional_layout_construction_has_no_mapping(builders):
    layout = builders['PackedLayout'](np.zeros((1,3)),np.ones((1,3)),np.ones((1,3)),np.array([0]),np.zeros(3),.01,1,1)
    assert layout.node_map is None and layout.dense_indices is None
