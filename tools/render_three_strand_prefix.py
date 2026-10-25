"""Render the saved, valid three-strand loading segment."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = "4152797ee8f85ab2b551bf721efeaabd48402775ee74bd08d5c1fea3dc8d316e"


def descriptor(values):
    values = np.ascontiguousarray(values)
    return {"shape": list(values.shape), "dtype": str(values.dtype),
            "sha256": hashlib.sha256(values.tobytes()).hexdigest()}


def render(source, output, pvpython):
    checkpoint = source / "checkpoint.npz"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != CHECKPOINT:
        raise ValueError("Unexpected saved segment")
    plan = json.loads((source / "plan.json").read_text())
    old_summary = json.loads((source / "summary.json").read_text())
    with np.load(checkpoint, allow_pickle=False) as saved:
        assert int(saved["frame"]) == 12 and bool(saved["state_valid"])
        assert saved["history_valid"].all()
        assert str(saved["signature"]) == hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
        tables = json.loads(str(saved["tables"]))
        endpoint = np.array(saved["state_x"])
        solver_seconds = float(saved["solver_seconds"])
    rows, records, times = (tables[key] for key in ("rows", "records", "times"))
    assert len(rows) == len(records) == len(times) == 13
    assert rows[-1]["accepted_steps"] == 3500
    assert all(row["contact_valid"] and row["min_J"] > 0 for row in rows)
    assert all(abs(row["mass_relative_drift"]) < 1e-12 for row in rows)
    assert all(row["max_damage"] == 0 for row in rows)
    assert max(abs(row["energy_ledger_residual_J"]) for row in rows) < .15 * max(abs(row["fixture_work_J"]) for row in rows)
    arrays = {}
    for field in ("x", "speed", "damage", "temperature", "failed"):
        values = np.load(source / f"trajectory_{field}.npy", mmap_mode="r", allow_pickle=False)
        arrays[field] = np.array(values[:13])
        for index in range(13):
            assert descriptor(arrays[field][index]) == records[index][field]
    assert np.array_equal(arrays["x"][-1], endpoint)
    with np.load(source / "particle_fields.npz", allow_pickle=False) as saved:
        fields = {key: np.array(saved[key]) for key in saved.files}
    for key, values in fields.items():
        assert descriptor(values) == old_summary["particle_fields"]["fields"][key]
    fields.update({key: arrays[key] for key in ("speed", "damage", "temperature", "failed")})
    fields["body"] = np.zeros(plan["particles"], dtype=np.int32)
    output.mkdir(parents=True, exist_ok=False)
    summary = {"case": "three-strand-tension", "completed": False,
        "displayed_segment_complete": True, "last_saved_frame": 12, "frames": 13,
        "steps": 3500, "simulation_time": times[-1], "particles": [plan["particles"]],
        "solver_seconds": solver_seconds, "min_J": min(row["min_J"] for row in rows), "max_damage": 0.,
        "geometry_scale": 1, "definition": plan, "configs": old_summary["configs"],
        "materials": old_summary["materials"], "source": plan["source"],
        "initial": rows[0], "final": rows[-1],
        "scope": "Early axial loading of three stress-free manufactured helices. The displayed segment precedes the later extended-contact rejection; it is not a twisting process, completed full loading protocol, or failure-strength result.",
        "source_run": 40116768349, "source_artifact": 12569841297,
        "source_checkpoint_sha256": CHECKPOINT, "displayed_prefix_checks_passed": True}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    module = importlib.util.spec_from_file_location("vtk_export", ROOT / "mpm_jax/export.py")
    exporter = importlib.util.module_from_spec(module)
    module.loader.exec_module(exporter)
    exporter.export_series(arrays["x"], fields, np.asarray(times), output,
                           spacing=plan["geometry"]["spacing_m"], dim=3)
    command = [str(pvpython), "--force-offscreen-rendering",
               str(ROOT / "tools/paraview_three_strand_prefix.py"), str(output)]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    with (output / "render.log").open("w") as log:
        subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0, check=True)
    encode(source, output)


def encode(source, output):
    if hashlib.sha256((source / "checkpoint.npz").read_bytes()).hexdigest() != CHECKPOINT:
        raise ValueError("Unexpected saved segment")
    with np.load(source / "checkpoint.npz", allow_pickle=False) as saved:
        records = json.loads(str(saved["tables"]))["records"]
    positions = np.load(source / "trajectory_x.npy", mmap_mode="r", allow_pickle=False)
    for i in range(13):
        assert descriptor(positions[i]) == records[i]["x"]
    rendered = json.loads((output / "render.json").read_text())
    assert rendered["renderer"] == "ParaView" and rendered["frames_rendered"] == 13
    assert rendered["resolution"] == [400, 400] and rendered["geometry_scale"] == 1
    assert not rendered["axes"] and not rendered["annotations"]
    assert rendered["source_checkpoint_sha256"] == CHECKPOINT
    for name, expected in rendered["source_vtk_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    from PIL import Image
    files = [output / "frames" / f"frame_{i:05d}.png" for i in range(13)]
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in files] == rendered["png_sha256"]
    pictures = [Image.open(path).convert("RGB") for path in files]
    palette_source = Image.new("RGB", (400, 400 * 13))
    for i, picture in enumerate(pictures):
        palette_source.paste(picture, (0, i * 400))
    palette = palette_source.quantize(colors=256, dither=Image.Dither.NONE)
    indexed = [picture.quantize(palette=palette, dither=Image.Dither.NONE) for picture in pictures]
    gif = output / "three-strand-tension.gif"
    indexed[0].save(gif, save_all=True, append_images=indexed[1:],
                    duration=[150] * 12 + [600], loop=0, optimize=False, disposal=2)
    hashes = [hashlib.sha256(picture.tobytes()).hexdigest() for picture in pictures]
    with Image.open(gif) as animation:
        assert animation.size == (400, 400) and animation.n_frames == 13
        for i, frame in enumerate(indexed):
            animation.seek(i)
            assert np.array_equal(np.asarray(animation.convert("RGB")), np.asarray(frame.convert("RGB")))
    cover = output / "three-strand-tension.png"
    cover.write_bytes(files[-1].read_bytes())
    review = {"passed": True, "frames": 13, "resolution": [400, 400],
        "unique_png_states": len(set(hashes)), "axes": False, "annotations": False,
        "geometry_scale": 1, "actual_source_frames": list(range(13)),
        "no_interpolation": True, "source_checkpoint_sha256": CHECKPOINT,
        "gif_sha256": hashlib.sha256(gif.read_bytes()).hexdigest(), "gif_bytes": gif.stat().st_size,
        "cover_sha256": hashlib.sha256(cover.read_bytes()).hexdigest(),
        "decoded_frames_match_quantized_pngs": True,
        "last_minus_first_max_displacement_m": float(np.linalg.norm(positions[12] - positions[0], axis=1).max()),
        "visual_review_passed": False}
    (output / "media-review.json").write_text(json.dumps(review, indent=2) + "\n")
    for picture in pictures + indexed + [palette, palette_source]:
        picture.close()
    print(json.dumps(review, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pvpython", type=Path)
    parser.add_argument("--assemble-only", action="store_true")
    args = parser.parse_args()
    if args.assemble_only:
        encode(args.source.resolve(), args.output.resolve())
    elif args.pvpython:
        render(args.source.resolve(), args.output.resolve(), args.pvpython.resolve())
    else:
        parser.error("Provide --pvpython or --assemble-only")
