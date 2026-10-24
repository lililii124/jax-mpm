"""Rejected physical gates must survive a later diagnostic failure."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def controller(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / 'tools'))
    spec = importlib.util.spec_from_file_location('ceramic_review_probe_test',
                                                root / 'tools/probe_ceramic_full_review.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rejection_is_saved_before_spatial_failure(tmp_path, monkeypatch):
    module = controller(monkeypatch)
    physical = SimpleNamespace(review=lambda folder, allow_prefix: {
        'passed': np.bool_(False), 'completed': True, 'accepted_saved_frame': 60,
        'checks': {'finite': np.bool_(True), 'physical_gate': np.bool_(False)}})

    def failing_spatial(folder):
        assert json.loads((tmp_path / 'physical-gates.json').read_text())['failed_checks'] == ['physical_gate']
        raise ValueError('independent spatial diagnostic failed')

    with pytest.raises(ValueError, match='spatial diagnostic'):
        module.collect_reports(tmp_path / 'raw', tmp_path,
                               physical, SimpleNamespace(review=failing_spatial))
    saved = json.loads((tmp_path / 'original-full-physical-review.json').read_text())
    assert saved['passed'] is False and saved['checks']['physical_gate'] is False
    assert not (tmp_path / 'raw').exists()


def test_successful_extraction_does_not_admit_rejected_raw(tmp_path, monkeypatch):
    module = controller(monkeypatch)
    physical = SimpleNamespace(review=lambda folder, allow_prefix: {
        'passed': False, 'completed': True, 'accepted_saved_frame': 60,
        'checks': {'physical_gate': False}})
    spatial = SimpleNamespace(review=lambda folder: {'physical_admission': False, 'frame': 60})
    result = module.collect_reports(tmp_path / 'raw', tmp_path, physical, spatial)
    assert result['passed'] is False
    gates = json.loads((tmp_path / 'physical-gates.json').read_text())
    assert gates['production_eligible'] is False and gates['physical_admission'] is False
    assert (tmp_path / 'spatial-failure-frame-60.json').exists()
    with pytest.raises(FileExistsError):
        module.collect_reports(tmp_path / 'raw', tmp_path, physical, spatial)


def test_raw_directory_cannot_be_used_for_diagnostic_output(tmp_path, monkeypatch):
    module = controller(monkeypatch)
    with pytest.raises(ValueError, match='separate'):
        module.probe(tmp_path, tmp_path / 'reports')
