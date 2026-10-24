from copy import deepcopy
import os

import numpy as np
import pytest

from tools.review_woven_large import accepted_clock_matches, accepted_output_steps


def records():
    plan = dict(output_steps=list(range(0, 14461, 241)), full_steps=14460, probe_steps=0,
                duration_s=200e-6)
    summary = dict(completed=False, probe_complete=False, phase='checkpoint-ready',
        segment_complete=True, segment_start_frame=0, segment_end_frame=20,
        runtime_stop_after_frame=20, steps=4820, frame=20, full_steps=14460,
        target_simulation_time=200e-6)
    return plan, summary, deepcopy(summary)


def test_partial_review_preserves_full_protocol_and_does_not_scan_uncomputed_future_slots():
    plan, summary, status = records()
    before = deepcopy(plan)
    steps, segment = accepted_output_steps(plan, summary, status, 21)
    assert segment and steps == list(range(0, 4821, 241))
    assert plan == before and len(plan['output_steps']) == 61


@pytest.mark.parametrize('field,value', [('segment_complete', False), ('completed', True),
    ('probe_complete', True), ('phase', 'running'), ('segment_start_frame', 21),
    ('segment_end_frame', 19), ('runtime_stop_after_frame', 19), ('steps', 4819), ('frame', 19),
    ('full_steps', 4820)])
def test_mismatched_segment_endpoints_or_status_cannot_be_admitted(field, value):
    plan, summary, status = records()
    status[field] = value
    with pytest.raises(ValueError):
        accepted_output_steps(plan, summary, status, 21)


@pytest.mark.parametrize('field,value', [('segment_end_frame', True), ('segment_start_frame', -1),
                                      ('runtime_stop_after_frame', None), ('phase', 'failed')])
def test_summary_cannot_rename_an_unplanned_or_failed_prefix(field, value):
    plan, summary, status = records()
    summary[field] = value
    with pytest.raises(ValueError):
        accepted_output_steps(plan, summary, status, 21)


def test_probe_cannot_be_promoted_to_a_complete_plan_segment():
    plan, summary, status = records()
    plan['probe_steps'] = 128
    with pytest.raises(ValueError):
        accepted_output_steps(plan, summary, status, 21)


def test_full_run_and_original_probe_keep_the_original_output_steps():
    plan = dict(output_steps=[0, 128], probe_steps=128)
    assert accepted_output_steps(plan, {'segment_complete': False}, {}, 2) == ([0, 128], False)


@pytest.mark.parametrize('field', [0, 1, 2, 3])
def test_each_actual_state_or_metadata_clock_is_bound_to_the_same_saved_instant(field):
    dt, steps = 200e-6/14460, 4820
    clock = [steps*dt]*4
    assert accepted_clock_matches(*clock, steps, dt)
    clock[field] += dt
    assert not accepted_clock_matches(*clock, steps, dt)


def test_even_consistently_modified_clocks_must_still_match_the_full_protocol_steps():
    dt, steps = 200e-6/14460, 4820
    assert not accepted_clock_matches(*([steps*dt+dt]*4), steps, dt)
    assert not accepted_clock_matches(np.nan, 0., 0., 0., 0, dt)


@pytest.mark.skipif(os.environ.get('GITHUB_ACTIONS') != 'true', reason='Actual JAX integration runs in cloud CI')
def test_actual_full_plan_segment_checks_all_accepted_frames_and_ignores_future_allocation(tmp_path, monkeypatch):
    from examples import woven_large_plate as driver
    from examples.woven_large_geometry import sample_large_weave
    from tools.review_woven_large import review_folder
    monkeypatch.setattr(driver, 'sample_large_weave', lambda architecture:
                        sample_large_weave(architecture, size=(.02, .02)))
    sample, plan = driver.make_plan('plain', 'pressure-bending')
    result = driver.run_case(sample, plan, tmp_path, stop_after_frame=1)
    assert result['segment_complete'] and not result['completed']
    folder = tmp_path/'plain'/'pressure-bending'
    report = review_folder(folder)
    assert report['allocated_states'] == 61 and report['accepted_saved_states'] == 2
    # The footprint is deliberately small; retain those independent rejections.
    assert {k for k,v in report['checks'].items() if not v} == {
        'reference_geometry', 'rectangular_grid_has_no_cubic_air_extension'}, report
    assert not report['passed'] and not report['probe_passed'] and not report['segment_passed']
    history = np.load(folder/'trajectory_x.npy', mmap_mode='r+')
    history[2] = np.nan
    history.flush()
    history._mmap.close()
    uncomputed = review_folder(folder)
    assert uncomputed['checks'] == report['checks']
    history = np.load(folder/'trajectory_x.npy', mmap_mode='r+')
    history[1, 0, 0] += 1e-5
    history.flush()
    history._mmap.close()
    changed = review_folder(folder)
    assert not changed['checks']['all_saved_frames_match_checkpoint_descriptors']
    assert not changed['checks']['first_and_last_coordinates_match_state']
