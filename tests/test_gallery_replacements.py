import pytest

from tools import collect_gallery


OLD = "material-engineering--concrete-bending--hjc"
LARGE = "material-engineering--concrete-bending-large--hjc"
SYMMETRIC = "material-engineering--concrete-bending-symmetric--hjc"


def records(*identifiers):
    return [
        {"id": identifier,
         "source_directory": "outputs/material-engineering-cloud/" + "/".join(identifier.split("--")[1:])}
        for identifier in identifiers
    ]


def beam_entries(catalogue):
    return [(name, source) for name, source in collect_gallery.gallery_groups(catalogue)["Material engineering"]
            if name.startswith("concrete-bending")]


def test_two_replacements_reach_symmetric_cloud_source():
    assert beam_entries(records(OLD, LARGE, SYMMETRIC)) == [
        ("concrete-bending-symmetric", "material-engineering-cloud/concrete-bending-symmetric/hjc")
    ]


def test_hidden_intermediate_is_not_a_fallback_without_completed_successor():
    assert beam_entries(records(OLD, LARGE)) == []


def test_hidden_final_target_stays_hidden(monkeypatch):
    monkeypatch.setattr(collect_gallery, "HIDDEN_HOME_IDS", {LARGE, SYMMETRIC})
    assert beam_entries(records(OLD, LARGE, SYMMETRIC)) == []


@pytest.mark.parametrize("mapping", [{OLD: OLD}, {OLD: LARGE, LARGE: OLD}])
def test_replacement_cycle_is_rejected(monkeypatch, mapping):
    monkeypatch.setattr(collect_gallery, "HOME_REPLACEMENTS", mapping)
    with pytest.raises(ValueError, match="Cyclic gallery replacement"):
        collect_gallery.gallery_groups(records(OLD, LARGE))


def test_existing_single_replacement_keeps_its_cloud_source():
    old = "material-engineering--ceramic-indentation--jh2"
    new = "material-engineering--ceramic-indentation-wide--jh2"
    entries = collect_gallery.gallery_groups(records(old, new))["Material engineering"]
    assert ("ceramic-indentation-wide", "material-engineering-cloud/ceramic-indentation-wide/jh2") in entries
    assert all(name != "ceramic-indentation" for name, _ in entries)


def test_symmetric_animation_preview_uses_actual_peak_deflection(tmp_path):
    folder = tmp_path / "concrete-bending-symmetric" / "hjc"
    frames = folder / "frames"
    frames.mkdir(parents=True)
    (folder / "observables.csv").write_text(
        "time,midspan_bottom_deflection_m\n0,0\n0.001,-0.012\n0.0015,-0.008\n",
        encoding="utf-8",
    )
    peak = frames / "frame_00001.png"
    peak.write_bytes(b"preview selection fixture")
    assert "concrete-bending-symmetric" in collect_gallery.ANIMATIONS
    assert collect_gallery.preview(folder, "concrete-bending-symmetric", {"frames_rendered": 3}) == (
        peak, 1, 0.001
    )
