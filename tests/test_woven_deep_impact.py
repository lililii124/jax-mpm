import json

import numpy as np
import pytest

from examples import woven_deep_impact as deep


def test_separate_bounded_higher_energy_plan():
    sample, plan = deep.make_plan()
    assert plan["particles"] == 819200
    assert sample.metadata["thickness_m"] == .002
    assert plan["clamp"]["profile"] == "four-edge-full-clamp"
    assert plan["clamp"]["backing"] is False
    assert plan["deep_impact"]["initial_energy_J"] == 40
    assert plan["deep_impact"]["ballistic_travel_m"] == .008
    assert plan["grid"]["estimated_nodes"] < 8_000_000
    assert plan["grid"]["y_bounds_m"] == [-.016, .004]
    assert plan["case_path"] == ["plain", "deep-sphere-impact"]
    assert plan["source"] == deep.base.source_record()
    assert not plan["production_eligible"]


@pytest.mark.parametrize("mass,speed", [(0, 40), (.05, 41), (.05, np.nan)])
def test_reject_unbounded_or_invalid_inputs(mass, speed):
    with pytest.raises(ValueError):
        deep.make_plan(mass=mass, speed=speed)


def test_seal_retains_exact_failed_raw_and_rejects_overwrite(tmp_path):
    raw = tmp_path/"raw"
    raw.mkdir()
    (raw/"checkpoint.npz").write_bytes(b"exact raw bytes")
    (raw/"exception.json").write_text('{"message":"step rejected"}')
    target = tmp_path/"segments/f02"
    deep.seal(raw, target)
    assert (target/"checkpoint.npz").read_bytes() == b"exact raw bytes"
    receipt = json.loads((target/"receipt.json").read_text())
    assert set(receipt["files"]) == {"checkpoint.npz", "exception.json"}
    with pytest.raises(FileExistsError):
        deep.seal(raw, target)


def test_large_receipt_hash_uses_streaming_reads(tmp_path, monkeypatch):
    from pathlib import Path
    import hashlib
    raw = tmp_path/"large.bin"
    content = b"bounded stream"*(200_000)
    raw.write_bytes(content)
    def forbidden(*args, **kwargs):
        raise AssertionError("whole-file reads are forbidden")
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    assert deep.stream_sha256(raw) == hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize("failed", [False, True])
def test_controller_persistence_keeps_single_raw_tree(tmp_path, monkeypatch, failed):
    folder = tmp_path/"plain/deep-sphere-impact"
    monkeypatch.setattr(deep, "make_plan", lambda **kwargs: (None, {}))
    monkeypatch.setattr(deep.base, "runtime_stop_frame", lambda *args: None)
    def run(*args, **kwargs):
        folder.mkdir(parents=True)
        (folder/"raw.bin").write_bytes(b"preserve exact raw")
        if failed:
            (folder/"exception.json").write_text('{"message":"rejected"}')
            raise FloatingPointError("rejected")
    monkeypatch.setattr(deep.base, "run_case", run)
    monkeypatch.setattr(deep, "review", lambda *args: {"accepted": False})
    def forbidden(*args):
        raise AssertionError("controller mode must not duplicate raw files")
    monkeypatch.setattr(deep, "seal", forbidden)
    arguments = ["--output", str(tmp_path), "--persistence", "controller"]
    if failed:
        with pytest.raises(FloatingPointError):
            deep.main(arguments)
        assert (folder/"exception.json").exists()
    else:
        deep.main(arguments)
        assert (folder/"deep-review.json").exists()
    assert (folder/"raw.bin").read_bytes() == b"preserve exact raw"
    assert not (tmp_path/"segments").exists()
