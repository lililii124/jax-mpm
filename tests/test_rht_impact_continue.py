"""Admission regressions: incomplete evidence must never launch a solver."""
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from tools import rht_impact_continue as continuation


def test_missing_parent_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        continuation.admit(tmp_path)


def test_changed_parent_identity_is_rejected(tmp_path):
    (tmp_path / 'transport.json').write_text('{}')
    with pytest.raises(RuntimeError, match='immutable'):
        continuation.admit(tmp_path)


def test_historical_receipt_does_not_admit_missing_parts(tmp_path):
    manifest = tmp_path / 'transport.json'
    manifest.write_text(json.dumps({'status': 'complete-raw-packaged',
                                  'parts': {'missing.tarpart': {'bytes': 1, 'sha256': 'a'}}}))
    with patch.object(continuation, 'TRANSPORT', continuation.transport.digest(manifest)):
        (tmp_path / 'retrieval-receipt.json').write_text(json.dumps({
            'status': 'all-parts-and-whole-raw-files-verified',
            'transport_sha256': continuation.TRANSPORT}))
        with pytest.raises(FileNotFoundError):
            continuation.admit(tmp_path)


def test_run_rejects_before_creating_or_launching(tmp_path):
    parent = tmp_path / 'parent'
    parent.mkdir()
    destination = tmp_path / 'new'
    with patch.object(continuation.subprocess, 'Popen') as launch:
        with pytest.raises(FileNotFoundError):
            continuation.run(parent, destination)
        launch.assert_not_called()
    assert not destination.exists()


def test_frozen_source_validation_is_syntactically_valid():
    compile(continuation.VALIDATE, '<frozen-source-validation>', 'exec')


@pytest.mark.parametrize('cpus,ram', [(2,32),(8,11),(None,32)])
def test_inadequate_resource_is_rejected(cpus, ram):
    with pytest.raises(RuntimeError,match='qualification requires'):
        continuation.resource_admission(cpus,ram*continuation.GIB)


def test_resource_gate_and_segment_are_bounded():
    assert continuation.resource_admission(8,12*continuation.GIB)['logical_cpus']==8
    assert continuation.END_FRAME==4
    assert (continuation.END_FRAME-3)*35==35
    assert continuation.GUARD_SECONDS==10800


def test_tip_crossing_and_failure_are_not_perforation_or_fragments(tmp_path):
    p=tmp_path/'observables.csv'
    p.write_text('projectile_tip_y_m,projectile_center_vy_m_per_s,tip_past_rear_m,target_failed_reference_volume_fraction,target_failed_count,target_maximum_damage,corridor_intact_count,target_minimum_J\n.13,-100,.01,.1,10,1,0,.9\n')
    result=continuation.prefix_engineering_audit(p)
    assert result['tip_has_crossed_original_rear_plane'] is True
    assert result['failed_points_present'] is True
    assert result['perforation_validated'] is False
    assert result['fragments_validated'] is False
    p.write_text(p.read_text().replace('0,.9','0,-.1'))
    with pytest.raises(RuntimeError,match='Physical engineering'):
        continuation.prefix_engineering_audit(p)
