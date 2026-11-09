import pytest

from tools.select_multicell_binary_step import rank_fixed_bottom


def test_fixed_bottom_ranking_uses_same_observation_points():
    def candidate(name, underside, bottom, coverage=0.5):
        return {'name': name, 'comparison': {'candidate_sampled_peak_m': underside},
                'secondary_constraints': {'common_bottom_response': {
                    'candidate_peak_m': bottom, 'common_bottom_coverage': coverage}}}
    rows = [candidate('lower-underside', 0.2, 1.2), candidate('lower-fixed-bottom', 0.8, 0.7)]
    best, accepted = rank_fixed_bottom(rows, current_peak=1.0, fixed_coverage=0.5, minimum_reduction=0.01)
    assert best['name'] == 'lower-fixed-bottom'
    assert accepted and best['objective_reduction_fraction'] == pytest.approx(0.3)
    rows[1]['secondary_constraints']['common_bottom_response']['common_bottom_coverage'] = 0.49
    with pytest.raises(ValueError, match='retain all'):
        rank_fixed_bottom(rows, current_peak=1.0, fixed_coverage=0.5, minimum_reduction=0.01)
