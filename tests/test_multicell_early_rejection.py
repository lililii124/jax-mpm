import pytest

from tools.select_multicell_binary_step import verify_early_rejection


def test_partial_replay_can_only_prove_rejection():
    meta = {'rejection_peak_m': .9}
    history = [{'step': 8}]
    outcome = {'reason': 'peak-threshold', 'threshold_m': .9, 'observed_peak_m': .95, 'step': 8}
    result = {'baseline_sampled_peak_m': 1., 'candidate_sampled_peak_m': .95}
    verify_early_rejection(meta, history, result, outcome, minimum_reduction=.1)
    with pytest.raises(ValueError):
        verify_early_rejection(meta, history, result, outcome, minimum_reduction=0.)
