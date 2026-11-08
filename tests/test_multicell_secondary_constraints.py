import pytest

from tools.select_multicell_binary_step import evaluate_secondary_constraints


def test_rejects_a_candidate_with_worse_broad_surface_response():
    baseline = [{'bottom_plane_displacement_max_m': 1., 'back_displacement_p90_m': 2.}]
    candidate = [{'bottom_plane_displacement_max_m': .9, 'back_displacement_p90_m': 2.2}]
    result = evaluate_secondary_constraints(
        {'bottom_plane_sample_fraction': .8}, baseline,
        {'bottom_plane_sample_fraction': .8}, candidate,
        bottom_ratio=1., p90_ratio=1., min_bottom_coverage=.7)
    assert not result['passed']
    assert result['bottom_peak']['ratio'] == .9
    assert result['p90_peak']['ratio'] == 1.1


def test_constraints_fail_closed_for_missing_surface_or_insufficient_coverage():
    baseline = [{'back_displacement_p90_m': 2.}]
    candidate = [{'back_displacement_p90_m': 1.}]
    result = evaluate_secondary_constraints({}, baseline, {'bottom_plane_sample_fraction': .3},
                                            candidate, p90_ratio=1., min_bottom_coverage=.5)
    assert not result['passed']
    with pytest.raises(ValueError, match='lacks surface statistics'):
        evaluate_secondary_constraints({}, baseline, {}, [{}], p90_ratio=1.)
