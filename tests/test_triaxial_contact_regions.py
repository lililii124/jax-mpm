import numpy as np
import pytest

from examples.triaxial_failure import output_directory, plan_case, run_case


def test_connected_hertz_plan_records_source_and_fixed_component_budgets():
    _, plan = plan_case(smoke=True, law='hertz', contact_regions='connected', probe_steps=2,
                        contact_capacity=256, max_contact_patches=64, patch_iterations=32)
    assert plan['contact_regions']['capacity'] == 256
    assert plan['contact_regions']['max_patches'] == 64
    assert plan['contact_regions']['max_iterations'] == 32
    assert plan['work_of_adhesion_J_per_m2'] == 0.
    assert {'mpm_jax/contact_patches.py', 'mpm_jax/fibre_patch_contact.py'} <= plan['source']['files'].keys()


@pytest.mark.parametrize('kwargs', [dict(law='jkr'), dict(law='md'), dict(law='self-consistent'),
    dict(law='hertz', contact_capacity=0), dict(law='hertz', max_contact_patches=False),
    dict(law='hertz', patch_iterations=-1)])
def test_unsupported_adhesive_branch_merging_and_invalid_region_budgets_refused(kwargs):
    with pytest.raises(ValueError):
        plan_case(contact_regions='connected', **kwargs)


def test_single_region_default_retains_its_original_plan_and_source_scope():
    _, plan = plan_case(smoke=True, law='hertz', probe_steps=2)
    assert 'contact_regions' not in plan
    assert 'mpm_jax/contact_patches.py' not in plan['source']['files']


def test_actual_connected_hertz_driver_checkpoint_preserves_its_full_protocol(tmp_path):
    import jax
    jax.config.update('jax_enable_x64', True)
    geometry, plan = plan_case(smoke=True, law='hertz', contact_regions='connected',
        grid_storage='tube', load_transits=.03, max_strain=1e-7, frames=4,
        contact_capacity=256, max_contact_patches=64, patch_iterations=32)
    full, split = tmp_path/'full', tmp_path/'split'
    uninterrupted = run_case(geometry, plan, full)
    first = run_case(geometry, plan, split, stop_after_frame=2)
    assert first['segment_complete'] and not first['completed']
    resumed = run_case(geometry, plan, split, resume=True)
    assert uninterrupted['completed'] and resumed['completed']
    assert uninterrupted['steps'] == resumed['steps'] == plan['steps'] == 13
    with (np.load(output_directory(full, plan)/'checkpoint.npz', allow_pickle=False) as a,
          np.load(output_directory(split, plan)/'checkpoint.npz', allow_pickle=False) as b):
        fields = [name for name in a.files if name.startswith(('state_', 'history_', 'contact_'))]
        for name in fields:
            np.testing.assert_array_equal(a[name], b[name], err_msg=name)
        assert b['contact_active'].shape == (64,)
    jax.clear_caches()
