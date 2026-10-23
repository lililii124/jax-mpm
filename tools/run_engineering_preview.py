"""Run fixed engineering cases and preserve failed diagnostics on Linux CPU runners."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_woven_preview as shared

LOGS = ROOT / ".local/engineering-ci"
DELIVERY = ROOT / ".local/engineering-delivery"
GIB = 1024**3
LIMIT = 100 * 1024**2
ADHESION_PRESETS = {
    "adhesion-v2-pair": ("outputs/adhesion-visible", "lift-hold"),
    "adhesion-v3-pair": ("outputs/adhesion-visible-v3", "lift-hold-v3"),
}
PAIR_FILES = ("adhesion_comparison.json", "adhesion_comparison.csv", "paired-render.json")
PRESETS = ("ceramic-wide", "hjc-large", "hjc-symmetric", *ADHESION_PRESETS, "jc-density-diagnostic", "jc-necking-formal")
digest, save, identity = shared.digest, shared.save, shared.identity


def plan(preset):
    if preset not in PRESETS:
        raise ValueError("Unknown engineering preset")
    if preset in ADHESION_PRESETS:
        root, expected_preset = ADHESION_PRESETS[preset]
        profile_args = ["--profile", "v3"] if preset == "adhesion-v3-pair" else []
        return [{"name": name, "module": "examples.adhesion_contrast", "grid": 48, "ppc": 3,
                 "frames": 50, "particles": 367744, "duration": 1.3, "field": "displacement",
                 "root": root, "folder": f"{root}/{name}", "expected_preset": expected_preset,
                 "arguments": ["--case", name, *profile_args], "review": "tools/review_adhesion.py"}
                for name in ("nonadhesive-pull-off", "adhesive-pull-off")]
    scene, model, grid, ppc, frames, particles, duration = {
        "ceramic-wide": ("ceramic-indentation-wide", "jh2", 96, 2, 40, 386420, 3e-4),
        "hjc-large": ("concrete-bending-large", "hjc", 64, 3, 60, 121104, 1.5e-3),
        "hjc-symmetric": ("concrete-bending-symmetric", "hjc", 64, 3, 60, 121104, 1.5e-3),
        "jc-density-diagnostic": ("jc-necking", "johnson-cook", 64, 2, 60, 33224, 150e-6),
        "jc-necking-formal": ("jc-necking", "johnson-cook", 100, 2, 60, 102320, 150e-6),
    }[preset]
    root = "outputs/jc-necking-full-smoke-ppc2" if preset == "jc-density-diagnostic" else "outputs/material-engineering"
    jc = preset in ("jc-density-diagnostic", "jc-necking-formal")
    return [{"name": scene, "module": "examples.jc_necking" if jc else "examples.material_engineering",
             "grid": grid, "ppc": ppc, "frames": frames, "particles": particles, "duration": duration,
             "field": "damage", "root": root, "folder": f"{root}/{scene}/{model}",
             "arguments": ["--end-time", "150e-6", "--cfl", ".15"] if jc else ["--case", scene, "--model", model],
             "review": "tools/review_material_engineering.py",
             **({"expected_supports_profile": "mirror-rollers-with-centre-planes", "expected_steps": 10500} if preset == "hjc-symmetric" else {})}]


def command(name, arguments, *, monitor=False, accepted_codes=(0,)):
    import psutil

    LOGS.mkdir(parents=True, exist_ok=True)
    total = psutil.virtual_memory().total
    memory = {"scope": "command process tree; one-second RSS/USS samples; shared RSS can be counted more than once",
              "peak_tree_rss_bytes": 0, "peak_tree_uss_bytes": 0, "samples": 0,
              "minimum_system_available_bytes": None, "system_total_bytes": total,
              "start_system_available_bytes": 6 * GIB, "stop_system_available_bytes": 2 * GIB,
              "stop_tree_uss_bytes": min(12 * GIB, max(0, total - 3 * GIB)),
              "stopped_for_memory": False, "stop_reason": None}
    if monitor and (total < 8 * GIB or psutil.virtual_memory().available < 6 * GIB):
        save(LOGS / f"{name}.json", {"completed": False, "phase": "admission-rejected", "memory": memory})
        raise RuntimeError("Insufficient available runner memory before command startup")
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="2")
    if name.startswith("paraview") or name.startswith("vtk-check"):
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
    started = time.monotonic()
    record = {"command": [str(arg) for arg in arguments], "completed": False}
    with subprocess.Popen(arguments, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, errors="replace") as process, (LOGS / f"{name}.log").open("w", encoding="utf-8") as log:
        finished = threading.Event()
        parent = psutil.Process(process.pid)
        created = parent.create_time()

        def sample():
            while not finished.is_set():
                try:
                    if parent.create_time() != created:
                        break
                    tree = [parent, *parent.children(recursive=True)]
                    counters = []
                    for child in tree:
                        try:
                            counters.append(child.memory_full_info())
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    available = psutil.virtual_memory().available
                    uss = sum(getattr(info, "uss", 0) for info in counters)
                    memory["samples"] += 1
                    memory["peak_tree_rss_bytes"] = max(memory["peak_tree_rss_bytes"], sum(info.rss for info in counters))
                    memory["peak_tree_uss_bytes"] = max(memory["peak_tree_uss_bytes"], uss)
                    memory["minimum_system_available_bytes"] = min(memory["minimum_system_available_bytes"] or available, available)
                    if memory["samples"] == 1 or memory["samples"] % 30 == 0:
                        save(LOGS / f"{name}-memory.json", memory)
                    reason = "system available memory" if available < 2 * GIB else "process-tree USS" if uss > memory["stop_tree_uss_bytes"] else None
                    if reason:
                        memory.update(stopped_for_memory=True, stop_reason=reason)
                        for child in reversed(tree):
                            try:
                                child.kill()
                            except psutil.NoSuchProcess:
                                pass
                        break
                except psutil.NoSuchProcess:
                    break
                finished.wait(1)

        watcher = threading.Thread(target=sample, daemon=True) if monitor else None
        if watcher:
            watcher.start()
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        finally:
            finished.set()
            if watcher:
                watcher.join(timeout=3)
            record.update(returncode=process.poll(), wall_seconds=time.monotonic() - started)
            record["completed"] = record["returncode"] == 0 and not memory["stopped_for_memory"]
            if monitor:
                record["memory"] = memory
                save(LOGS / f"{name}-memory.json", memory)
            save(LOGS / f"{name}.json", record)
    if code not in accepted_codes or memory["stopped_for_memory"]:
        raise subprocess.CalledProcessError(code or 75, arguments)
    return record


def source_record():
    result = shared.source_record()
    result["review_source_sha256"] = result["files"]["tools/review_material_engineering.py"]
    result["review_sources"] = {name: result["files"][name] for name in (
        "tools/review_material_engineering.py", "tools/review_adhesion.py", "tools/review_adhesion_comparison.py")}
    if "tools/run_engineering_preview.py" not in result["files"]:
        raise ValueError("Engineering helper must belong to the checked-out commit")
    return result


def strict_passed(review):
    return review.get("passed") is True and bool(review.get("checks")) and all(value is True for value in review["checks"].values())


def axial_support_intervals(sorted_x, dx, n_grid=None):
    import numpy as np

    q = np.asarray(sorted_x, dtype=np.float64) / dx
    bases = np.floor(q - .5).astype(np.int64)
    f = q - bases
    weights = np.column_stack((.5 * (1.5 - f)**2, .75 - (f - 1)**2, .5 * (f - .5)**2))
    nodes = bases[:, None] + np.arange(3)
    positive = weights > 0
    if n_grid is not None:
        positive &= (nodes >= 0) & (nodes <= n_grid)
    if not positive.any(axis=1).all():
        raise ValueError("An active particle has no positive axial grid weight")
    lower = np.where(positive, nodes, np.iinfo(np.int64).max).min(axis=1)
    upper = np.where(positive, nodes, np.iinfo(np.int64).min).max(axis=1)
    cuts = np.flatnonzero(lower[1:] > np.maximum.accumulate(upper)[:-1])
    return bases, lower, upper, cuts


def axial_support_cuts(sorted_x, dx):
    bases, _, _, cuts = axial_support_intervals(sorted_x, dx)
    return bases, cuts


def jc_diagnostic(folder, *, expected_grid=64, expected_particles=33224, diagnostic=True):
    import numpy as np

    summary = json.loads((folder / "summary.json").read_text())
    review_name = "interpretation.json" if diagnostic else "constitutive-review.json"
    review = json.loads((folder / review_name).read_text())
    config = summary["configs"][0]
    dx = config["domain_size"] / config["n_grid"]
    with np.load(folder / "trajectory.npz") as archive:
        times = np.asarray(archive["time"]).copy()
    mapped = [np.load(folder / f"trajectory_{name}.npy", mmap_mode="r") for name in ("x", "damage", "failed")]
    positions, damage, failed = mapped
    records, support_history = [], []
    first_gap, first_failure, first_central_full_failure = None, None, None
    try:
        if (positions.shape != (61, expected_particles, 3)
                or damage.shape != positions.shape[:2] or failed.shape != damage.shape
                or times.shape != (61,) or not np.isfinite(times).all()
                or times[0] != 0 or not np.all(np.diff(times) > 0)):
            raise ValueError("Incomplete or inconsistent JC saved arrays")
        initial = np.array(positions[0])
        levels, layers = np.unique(initial[:, 0], return_inverse=True)
        count = np.bincount(layers)
        if len(levels) < 2:
            raise ValueError("JC support audit requires multiple reference layers")
        spacing = float(np.diff(levels)[0])
        center_layers = np.abs(levels - .005) <= .51 * spacing
        central_particles = center_layers[layers]
        if not central_particles.any():
            raise ValueError("No central reference-layer particles")

        def radius_squared(points):
            mean_y = np.bincount(layers, weights=points[:, 1]) / count
            mean_z = np.bincount(layers, weights=points[:, 2]) / count
            squared = (points[:, 1] - mean_y[layers])**2 + (points[:, 2] - mean_z[layers])**2
            return np.bincount(layers, weights=squared) / count

        initial_radius = radius_squared(initial)
        if np.any(initial_radius <= 0):
            raise ValueError("A reference layer has no resolved radial extent")
        for frame, instant in enumerate(times):
            if (not np.isfinite(positions[frame]).all() or not np.isfinite(damage[frame]).all()
                    or not np.isin(failed[frame], (0, 1)).all()):
                raise ValueError("Non-finite coordinates/damage or non-binary failure flags")
            active = np.flatnonzero(~np.asarray(failed[frame], dtype=bool))
            order = active[np.argsort(positions[frame, active, 0])]
            x = positions[frame, order, 0]
            bases, lower, upper, cuts = axial_support_intervals(x, dx, config["n_grid"])
            gaps = np.diff(x)
            failed_count = int(np.count_nonzero(failed[frame]))
            details = [{"gap_m": float(gaps[k]),
                        "current_edge_x_m": [float(x[k]), float(x[k + 1])],
                        "initial_edge_x_m": [float(initial[order[k], 0]), float(initial[order[k + 1], 0])],
                        "base_nodes_x": [int(bases[k]), int(bases[k + 1])],
                        "positive_x_node_intervals": [[int(lower[k]), int(upper[k])], [int(lower[k + 1]), int(upper[k + 1])]],
                        "edge_damage": [float(damage[frame, order[k]]), float(damage[frame, order[k + 1]])]}
                       for k in cuts]
            row = {"frame": frame, "time_s": float(instant), "failed_particles": failed_count,
                   "maximum_damage": float(np.max(damage[frame])), "maximum_axial_gap_m": float(gaps.max(initial=0)),
                   "axial_support_cuts": len(cuts), "positive_axial_support_components": len(cuts) + int(active.size > 0),
                   "central_active_particles": int(np.count_nonzero(~np.asarray(failed[frame, central_particles], dtype=bool))),
                   "central_geometric_rms_radius_ratio": float(np.sqrt(radius_squared(positions[frame]) / initial_radius)[center_layers].mean())}
            records.append(row)
            support_history.append({"frame": frame, "time_s": float(instant), "cuts": details})
            if cuts.size and first_gap is None:
                first_gap = {**row, "cuts": details}
            if failed_count and first_failure is None:
                first_failure = row
            if row["central_active_particles"] == 0 and first_central_full_failure is None:
                first_central_full_failure = row
        complete = bool(summary.get("completed") is True and positions.shape == (61, expected_particles, 3)
                        and config["n_grid"] == expected_grid and np.isclose(times[-1], 150e-6, rtol=1e-12, atol=0))
        if not complete:
            raise ValueError(f"JC support audit did not finish the fixed {expected_grid}/ppc2 full-duration input")
    finally:
        for array in mapped:
            array._mmap.close()
    if first_gap is None:
        ordering = "no_saved_support_disconnection"
    elif first_failure is None or first_gap["time_s"] < first_failure["time_s"]:
        ordering = "support_disconnection_observed_before_any_material_failure"
    elif first_gap["time_s"] == first_failure["time_s"]:
        ordering = "same_saved_interval_unresolved"
    else:
        ordering = "material_failure_observed_at_an_earlier_saved_state"
    result = {"diagnostic_only": bool(diagnostic), "production_eligible": False, "completed": complete,
              "strict_review_passed": strict_passed(review), "failed_checks": [key for key, value in review["checks"].items() if value is not True],
              "grid": expected_grid, "particles": expected_particles, "dx_m": dx, "initial_layer_spacing_m": spacing,
              "effective_particles_per_cell_per_axis": dx / spacing, "diameter_grid_cells": .0016 / dx,
              "first_axial_stencil_disconnection": first_gap, "first_saved_material_failure": first_failure,
              "first_saved_central_full_failure": first_central_full_failure,
              "failure_disconnection_ordering": ordering,
              "support_disconnection_before_material_failure": first_gap is not None and (first_failure is None or first_gap["time_s"] < first_failure["time_s"]),
              "support_disconnection_before_central_full_failure": first_gap is not None and (first_central_full_failure is None or first_gap["time_s"] < first_central_full_failure["time_s"]),
              "support_disconnection_with_no_failed_material_observed": any(row["axial_support_cuts"] > 0 and row["failed_particles"] == 0 for row in records),
              "saved_history": records,
              "saved_positive_axial_support_cuts": support_history,
              "support_reconstruction": "Evaluate the three quadratic x weights from saved float64 positions, retain only strictly positive weights at in-domain nodes, then split the union of active-particle integer node intervals when no node is shared. No 3dx distance shortcut or numerical weight cutoff is used.",
              "interpretation": "An active-particle x-stencil cut is sufficient to prove no shared quadratic grid node across that plane. Its absence does not prove full 3D connectivity. A gap before any material failure is not rupture, even if central material later fails. D is not a crack-opening measurement; ballistic failed mass has no crack-face contact. First times refer to saved states; ordering within one saved interval is unresolved.",
              "comparison_plan": ("Same grid, CFL, duration, grips, geometry and JC parameters as the earlier ppc1 diagnostic; only the initial particle lattice is denser." if diagnostic else
                                  "The grid100/ppc2 production run retains the JC constitutive parameters, geometry, grips, CFL and 150 microsecond loading duration; resolution differs from the grid64 diagnostic.") + " No forced-fracture parameter adjustment.",
              "runner_source_sha256": digest(Path(__file__)),
              "sources": {name: digest(folder / name) for name in ("summary.json", "observables.csv", "diagnostics.csv", review_name, "final_state_0.npz", "trajectory.npz", "trajectory_x.npy", "trajectory_damage.npy", "trajectory_failed.npy")}}
    history_name = "jc-density-history.csv" if diagnostic else "jc-support-history.csv"
    with (folder / history_name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    result["history_csv"] = {"filename": history_name, "sha256": digest(folder / history_name), "rows": len(records)}
    save(folder / ("jc-density-diagnostic.json" if diagnostic else "jc-support-audit.json"), result)
    return result


def accept_jc_support(folder, audit, *, review_source_sha256=None):
    """Append the support audit after preserving the original constitutive review."""
    original = json.loads((folder / "constitutive-review.json").read_text())
    checks = {
        "constitutive_review_passed": strict_passed(original),
        "support_audit_complete": audit.get("completed") is True,
        "no_saved_support_disconnection_before_material_failure": (
            audit.get("support_disconnection_before_material_failure") is False
            and audit.get("support_disconnection_with_no_failed_material_observed") is False),
    }
    acceptance = {"passed": all(checks.values()), "checks": checks,
        "scope": "Saved-frame support ordering gate in addition to the independent constitutive review. Coincident first saved failure and disconnection remains unresolved within that interval; it is not evidence that failure preceded disconnection.",
        "failure_disconnection_ordering": audit["failure_disconnection_ordering"],
        "runner_source_sha256": digest(Path(__file__)),
        "sources": {name: digest(folder / name) for name in ("constitutive-review.json", "jc-support-audit.json", "jc-support-history.csv")}}
    save(folder / "runner-acceptance.json", acceptance)
    augmented = {**original, "checks": {**original["checks"], **{"runner_"+name: value for name, value in checks.items()}}}
    augmented["passed"] = all(value is True for value in augmented["checks"].values())
    if review_source_sha256 is not None:
        augmented["review_source_sha256"] = review_source_sha256
    augmented["runner_source_sha256"] = acceptance["runner_source_sha256"]
    augmented["supplemental_reviews"] = {name: digest(folder / name) for name in ("constitutive-review.json", "jc-support-audit.json", "runner-acceptance.json")}
    save(folder / "interpretation.json", augmented)
    return acceptance


def pair_focus_check(definition):
    import numpy as np

    bounds = np.array([[.22, .78], [.13, .82], [.22, .78]])
    lower, upper = np.asarray(definition["lower"]), np.asarray(definition["upper"])
    radius_of_interaction = definition["adhesion"]["interaction_range"]
    records = []
    for sphere in definition["rigid_spheres"]:
        times, centers = np.asarray(sphere["times"]), np.asarray(sphere["centers"])
        start = np.flatnonzero(np.isclose(times, .72, rtol=0, atol=1e-12))
        end = np.flatnonzero(np.isclose(times, .94, rtol=0, atol=1e-12))
        if len(start) != 1 or len(end) != 1 or end[0] <= start[0] or not np.all(centers[start[0]:end[0] + 1] == centers[start[0]]):
            raise ValueError("The fixed contact view requires the stationary .72-.94 s hold")
        holding = centers[start[0]:end[0] + 1]
        approach_and_hold = centers[times <= .94 + 1e-12]
        radius = sphere["radius"]
        hold_extent = np.column_stack((holding.min(axis=0) - radius, holding.max(axis=0) + radius))
        contact_extent = np.column_stack((approach_and_hold.min(axis=0) - radius - radius_of_interaction,
                                          approach_and_hold.max(axis=0) + radius + radius_of_interaction))
        contained = lambda extent: bool(np.all(extent[:, 0] >= bounds[:, 0]) and np.all(extent[:, 1] <= bounds[:, 1]))
        if not contained(hold_extent) or not contained(contact_extent) or not (lower[1] >= bounds[1, 0] and upper[1] <= bounds[1, 1]):
            raise ValueError("The sphere hold or contact region is outside the approved fixed detail view")
        records.append({"hold_sphere_bounds": hold_extent.ravel().tolist(),
                        "approach_through_hold_interaction_envelope": contact_extent.ravel().tolist(),
                        "hold_sphere_contained": True, "contact_region_contained": True})
    if len(records) != 1:
        raise ValueError("Matched contact detail view requires one sphere per run")
    return {"scene_bounds": bounds.ravel().tolist(), "checks": records,
            "basis": "SpherePath cosine segments remain within their endpoint coordinate bounds. During approach through the hold, all points inside radius plus interaction range lie within this envelope; the full nominal layer thickness is also included."}


def export_pair(folders):
    import numpy as np
    from mpm_jax.visualization import trajectory_arrays
    from mpm_jax.export import export_series

    bounds, maximum, focus_checks = None, 0., []
    for folder in folders:
        summary = json.loads((folder / "summary.json").read_text())
        config, definition = summary["configs"][0], summary["definition"]
        focus_checks.append(pair_focus_check(definition))
        spacing = definition.get("particle_spacing_m", config["domain_size"] / config["n_grid"] / definition["particles_per_cell_per_axis"])
        with trajectory_arrays(folder) as data:
            fields = {name: data[name] for name in ("speed", "damage", "temperature", "body", "failed") if name in data}
            export_series(data["x"], fields, data["time"], folder, spacing=spacing, dim=3, displacement=True)
        metadata = json.loads((folder / "vtk_metadata.json").read_text())
        current = np.asarray(metadata["bounds"], dtype=float).reshape(3, 2)
        for sphere in summary["definition"]["rigid_spheres"]:
            centers = np.asarray(sphere["centers"], dtype=float)
            current[:, 0] = np.minimum(current[:, 0], centers.min(axis=0) - sphere["radius"])
            current[:, 1] = np.maximum(current[:, 1], centers.max(axis=0) + sphere["radius"])
        if bounds is None:
            bounds = current.copy()
        else:
            bounds[:, 0] = np.minimum(bounds[:, 0], current[:, 0])
            bounds[:, 1] = np.maximum(bounds[:, 1], current[:, 1])
        maximum = max(maximum, metadata["field_ranges"]["displacement"][1])
    return {"scene_bounds": focus_checks[0]["scene_bounds"], "full_scene_bounds": bounds.ravel().tolist(),
            "color_range": [0., maximum], "sphere_opacity": .45,
            "camera_scope": "Matched contact detail view; fixed bounds include the contact region and complete sphere throughout the hold. Outer layer edges and the upper part of the fully withdrawn sphere may be outside the view. This is display cropping only; particle positions, tool motion and the full-history color range are unchanged.",
            "renderer_argument": "--focus-bounds", "geometry_scale": 1,
            "focus_geometry_checks": focus_checks}


def gif_timing(path, original_frames, expected_size=(1200, 1200)):
    from PIL import Image

    with Image.open(path) as image:
        count, loop, size = image.n_frames, image.info.get("loop"), image.size
        durations = []
        for frame in range(count):
            image.seek(frame)
            durations.append(int(image.info.get("duration", 0)))
    if size != expected_size or loop != 0 or not 0 < count <= original_frames or sum(durations) != 60 * original_frames:
        raise ValueError("GIF size, loop or complete saved-frame display duration differs")
    return {"encoded_frames": count, "original_png_frames": original_frames,
            "total_duration_ms": sum(durations), "expected_duration_ms": 60 * original_frames,
            "identical_frames_may_be_coalesced": True}


def render(case, pair_options=None):
    import numpy as np
    from PIL import Image

    folder = ROOT / case["folder"]
    options = ["--sphere-opacity", ".45"] if case["name"] == "ceramic-indentation-wide" else []
    if pair_options:
        options = ["--sphere-opacity", ".45", "--focus-bounds", *map(str, pair_options["scene_bounds"]), "--color-range", *map(str, pair_options["color_range"])]
    command("paraview-" + case["name"], [sys.executable, "-m", "mpm_jax.visualization", str(folder), "--gif", "--field", case["field"], "--resolution", "1200", *options], monitor=True)
    command("video-" + case["name"], [sys.executable, "tools/encode_video.py", str(folder)], monitor=True)
    command("vtk-check-" + case["name"], [os.environ["PARAVIEW_PYTHON"], "--force-offscreen-rendering", "tools/check_vtk.py", str(folder)], monitor=True)
    metadata = json.loads((folder / "vtk_metadata.json").read_text())
    scene = json.loads((folder / "render.json").read_text())
    video = json.loads((folder / "video.json").read_text())
    n = case["frames"] + 1
    checks = [metadata["frames"] == scene["frames_rendered"] == video["frames"] == n,
              metadata["particles"] == case["particles"], scene["renderer"] == "ParaView",
              scene["field"] == case["field"], scene["geometry_scale"] == 1,
              scene["axes"] is False, scene["annotations"] is False,
              scene["animation"] == "simulation.gif", video["render_sha256"] == digest(folder / "render.json"),
              video["sha256"] == digest(folder / "simulation.mp4")]
    if not all(checks):
        raise ValueError("Media provenance failed")
    if (pair_options or case["name"] == "ceramic-indentation-wide") and scene.get("render_overrides", {}).get("sphere_opacity") != .45:
        raise ValueError("Rigid-sphere opacity differs from the approved rendering")
    if pair_options and (scene.get("camera_bounds") != pair_options["scene_bounds"] or scene.get("render_overrides", {}).get("focus_bounds") != pair_options["scene_bounds"]):
        raise ValueError("Adhesion detail camera differs from the fixed common bounds")
    with np.load(folder / "trajectory.npz") as archive:
        times = np.array(archive["time"])
    pvd_times = [float(item.attrib["timestep"]) for item in ET.parse(folder / "particles.pvd").getroot().findall(".//DataSet")]
    np.testing.assert_array_equal(times, pvd_times)
    for index in range(n):
        with Image.open(folder / "frames" / f"frame_{index:05d}.png") as image:
            if image.size != (1200, 1200):
                raise ValueError("Unexpected ParaView image size")
            image.verify()
    animation = gif_timing(folder / "simulation.gif", n)
    save(folder / "media-check.json", {"passed": True, "original_png_frames": n,
        "png_sha256": {f"frame_{i:05d}.png": digest(folder / "frames" / f"frame_{i:05d}.png") for i in range(n)},
        "pvd_times_match_saved_float64_times_exactly": True, "gif": animation,
        "video_sha256": video["sha256"], "render_sha256": video["render_sha256"],
        "geometry_scale": 1, "axes": False, "annotations": False})
    save(folder / "vtk_check.json", {"passed": True, "sampled_frames": [0, n - 1],
        "particles_per_frame": metadata["particles"], "rtol": 1e-6, "atol": 1e-7,
        "checker_sha256": digest(ROOT / "tools/check_vtk.py"),
        "scope": "Actual VTK first/last-frame coordinate/scalar readback; complete PVD time sequence matched to the saved trajectory"})


def package(cases, provenance, diagnostic=False):
    import numpy as np

    main = DELIVERY / "main"
    payloads = {"main": 0, "final-state": 0, "final-state-2": 0, "media": 0, "media-2": 0}
    payload_limit = LIMIT - 262144

    def copy_payload(origin, relative, group):
        length = origin.stat().st_size
        if length > payload_limit:
            raise ValueError(f"A single file exceeds the 100 MiB artifact budget including provenance: {relative} ({length} bytes)")
        candidates = (group,) if group == "main" else (group, group + "-2")
        selected = next((name for name in candidates if payloads[name] + length <= payload_limit), None)
        if selected is None:
            raise ValueError(f"Both {group} artifact budgets are full; original files remain on the runner, with no omitted frames")
        target = DELIVERY / selected / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, target)
        payloads[selected] += length

    for case in cases:
        folder = ROOT / case["folder"]
        destination = main / case["folder"]
        destination.mkdir(parents=True, exist_ok=True)
        names = ["summary.json", "status.json", "observables.csv", "diagnostics.csv", "interpretation.json", "memory.json", "numerical-source.json"]
        copy_payload(folder / "final_state_0.npz", Path(case["folder"]) / "final_state_0.npz", "final-state")
        if diagnostic:
            names += ["jc-density-diagnostic.json", "jc-density-history.csv"]
            copy_payload(folder / "trajectory.npz", Path(case["folder"]) / "trajectory.npz", "final-state")
        else:
            names += ["particles.pvd", "vtk_metadata.json", "render.json", "video.json", "vtk_check.json", "media-check.json", "scene.pvsm"]
            for name in ("final.png", "simulation.gif", "simulation.mp4"):
                copy_payload(folder / name, Path(case["folder"]) / name, "media")
            for index in range(case["frames"] + 1):
                relative = Path(case["folder"]) / "frames" / f"frame_{index:05d}.png"
                copy_payload(ROOT / relative, relative, "media")
            with np.load(folder / "trajectory.npz") as archive:
                np.savez_compressed(destination / "trajectory_index.npz", time=archive["time"], body=archive["body"])
            payloads["main"] += (destination / "trajectory_index.npz").stat().st_size
        if provenance["preset"] == "jc-necking-formal":
            names += ["constitutive-review.json", "jc-support-audit.json", "jc-support-history.csv", "runner-acceptance.json"]
        if provenance["preset"] == "hjc-symmetric":
            names += ["runtime-symmetry.json"]
        for name in names:
            copy_payload(folder / name, Path(case["folder"]) / name, "main")
    if provenance["preset"] in ADHESION_PRESETS:
        roots = {case["root"] for case in cases}
        if len(roots) != 1:
            raise ValueError("The matched adhesion pair must share one output root")
        pair_root = Path(roots.pop())
        for name in PAIR_FILES:
            copy_payload(ROOT / pair_root / name, pair_root / name, "main")
    shutil.copy2(LOGS / "numerical-source.json", main / "numerical-source.json")
    if not diagnostic:
        shutil.copy2(LOGS / "paraview-install.json", main / "paraview-install.json")
    provenance["omitted_large_data"] = ("The full compressed trajectory is included for the diagnostic. Memory-mapped duplicates and checkpoints are not included." if diagnostic else
        "Full trajectories, VTK frame files and checkpoints remain on the runner. Strict numerical review and actual first/last VTK readback completed before packaging; the saved PVD/scene references require the omitted VTK series to reopen.")
    parts = sorted(path for path in DELIVERY.iterdir() if path.is_dir())
    provenance["artifact_parts"] = [part.name for part in parts]
    provenance["restoration"] = "Verify every part manifest, then merge the recorded relative paths from every named part. Parts are complementary, not alternate results; media may span two parts without dropping or changing a frame."
    save(main / "provenance.json", provenance)
    save(main / "delivery-index.json", {"parts": provenance["artifact_parts"], "payload_bytes": payloads,
        "cases": [case["folder"] for case in cases], "diagnostic_only": diagnostic,
        "strict_review_passed": provenance["strict_review_passed"], "production_eligible": provenance["production_eligible"]})
    for part in parts:
        if part != main:
            shutil.copy2(LOGS / "numerical-source.json", part / "numerical-source.json")
            save(part / "provenance.json", {**provenance, "scope": f"{part.name} supplement; restore files by their recorded paths"})
        shared.manifest(part)
        manifest = json.loads((part / "delivery-manifest.json").read_text())
        manifest.update(diagnostic_only=diagnostic, production_eligible=provenance["production_eligible"],
                        completed=provenance["completed"], artifact_parts=provenance["artifact_parts"])
        save(part / "delivery-manifest.json", manifest)


def run(preset):
    import importlib.metadata
    import numpy as np

    cases = plan(preset)
    source = source_record()
    save(LOGS / "numerical-source.json", source)
    provenance = {**identity(), "preset": preset, "parameters": cases, "completed": False,
                  "diagnostic_only": preset == "jc-density-diagnostic", "strict_review_passed": False,
                  "runtime": {"python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(),
                              "packages": {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "numpy", "pillow", "psutil")}}}
    save(LOGS / "provenance.json", provenance)
    if shutil.disk_usage(ROOT).free < 8 * GIB:
        raise RuntimeError("Less than 8 GiB disk space before solving")
    reports = []
    for case in cases:
        folder = ROOT / case["folder"]
        if folder.exists():
            raise ValueError(f"Output already exists; fresh cloud runs must not mix sources: {folder}")
        arguments = [sys.executable, "-m", case["module"], *case["arguments"], "--grid", str(case["grid"]),
                     "--ppc", str(case["ppc"]), "--frames", str(case["frames"]), "--output", case["root"], "--no-render"]
        solve = command("solve-" + case["name"], arguments, monitor=True)
        summary = json.loads((folder / "summary.json").read_text())
        config, definition = summary["configs"][0], summary["definition"]
        status = json.loads((folder / "status.json").read_text())
        actual_ppc = definition.get("requested_particles_per_cell_per_axis", definition.get("particles_per_cell_per_axis"))
        if (summary.get("completed") is not True or summary["particles"] != [case["particles"]]
                or not np.isclose(summary["simulation_time"], case["duration"], rtol=1e-12, atol=0)
                or config["n_grid"] != case["grid"] or actual_ppc != case["ppc"]
                or status.get("completed") is not True or status["frame"] != status["frames"] or status["frames"] != case["frames"]
                or not np.isclose(config["dt"] * summary["steps"], case["duration"], rtol=1e-12, atol=0)):
            raise ValueError("Solver did not complete the fixed preset")
        if preset in ("jc-density-diagnostic", "jc-necking-formal"):
            expected_steps = 37080 if preset == "jc-density-diagnostic" else 57900
            if summary["steps"] != expected_steps or definition["acoustic_CFL_requested"] != .15:
                raise ValueError("JC preset changed its grid time step or CFL")
        if preset in ("hjc-large", "hjc-symmetric") and (definition.get("material_profile") != "strain-floor" or summary["materials"][0].get("legacy_v3") is not False):
            raise ValueError("HJC large preset did not use the strain-floor profile")
        if preset == "hjc-symmetric" and (definition.get("supports_profile") != case["expected_supports_profile"] or summary["steps"] != case["expected_steps"]):
            raise ValueError("HJC symmetric preset changed its support profile or step count")
        if preset in ADHESION_PRESETS:
            law = definition["adhesion"]
            expected_work = 0. if case["name"] == "nonadhesive-pull-off" else 72.
            if definition.get("preset") != case["expected_preset"] or summary["steps"] != 3700 or definition["young_modulus_Pa"] != 8000 or law != {"work_of_adhesion": expected_work, "interaction_range": .025, "penalty_stiffness": 4.8e6}:
                raise ValueError("Adhesion preset differs from the fixed profile or law")
        save(folder / "memory.json", {**solve["memory"], "completed": True, "solver_seconds": summary["solver_seconds"], "steps": summary["steps"], "particles": case["particles"]})
        receipt = {**source, "review_source_sha256": source["review_sources"][case["review"]]}
        save(folder / "numerical-source.json", receipt)
        review_args = [str(folder)] if case["review"] == "tools/review_adhesion.py" else ["--root", case["root"], "--case", case["name"]]
        command("review-" + case["name"], [sys.executable, case["review"], *review_args], monitor=True, accepted_codes=(0, 1))
        if preset == "jc-necking-formal":
            shutil.copy2(folder / "interpretation.json", folder / "constitutive-review.json")
        report = json.loads((folder / "interpretation.json").read_text())
        report["review_source_sha256"] = receipt["review_source_sha256"]
        save(folder / "interpretation.json", report)
        reports.append(report)
        if preset == "hjc-symmetric":
            audit_path = folder / "runtime-symmetry.json"
            audit = json.loads(audit_path.read_text())
            if (not strict_passed(audit) or audit != report.get("runtime_symmetry")
                    or report.get("runtime_symmetry_report") != {"file": audit_path.name, "sha256": digest(audit_path)}
                    or audit.get("review_source_sha256") != receipt["review_source_sha256"]):
                raise ValueError("Independent HJC runtime symmetry audit failed or is not bound to the review source")
        if preset not in ("jc-density-diagnostic", "jc-necking-formal") and not strict_passed(report):
            raise ValueError(f"Strict numerical review failed for {case['name']}; dependent work and media are not started")
    provenance["strict_review_passed"] = all(strict_passed(report) for report in reports)
    if preset == "jc-necking-formal":
        folder = ROOT / cases[0]["folder"]
        audit = jc_diagnostic(folder, expected_grid=100, expected_particles=102320, diagnostic=False)
        acceptance = accept_jc_support(folder, audit, review_source_sha256=source["review_sources"][cases[0]["review"]])
        provenance["strict_review_passed"] = acceptance["passed"]
        provenance["support_audit"] = {"filename": "jc-support-audit.json", "sha256": digest(folder / "jc-support-audit.json"),
            "failure_disconnection_ordering": audit["failure_disconnection_ordering"]}
    if preset == "jc-density-diagnostic":
        result = jc_diagnostic(ROOT / cases[0]["folder"])
        provenance.update(completed=True, production_eligible=False, diagnostic_summary={
            "completed": result["completed"], "failed_checks": result["failed_checks"],
            "first_axial_stencil_disconnection": result["first_axial_stencil_disconnection"],
            "first_saved_material_failure": result["first_saved_material_failure"],
            "first_saved_central_full_failure": result["first_saved_central_full_failure"],
            "support_disconnection_before_material_failure": result["support_disconnection_before_material_failure"]})
    else:
        if not provenance["strict_review_passed"]:
            raise ValueError("A strict numerical review failed; no media will be rendered")
        pair_options = None
        if preset in ADHESION_PRESETS:
            command("review-pair", [sys.executable, "tools/review_adhesion_comparison.py", cases[0]["root"]], monitor=True)
            pair = json.loads((ROOT / cases[0]["root"] / "adhesion_comparison.json").read_text())
            if not strict_passed(pair):
                raise ValueError("The matched adhesion pair did not pass its strict comparison")
            pair_options = export_pair([ROOT / case["folder"] for case in cases])
            save(ROOT / cases[0]["root"] / "paired-render.json", pair_options)
        for case in cases:
            render(case, pair_options)
        if pair_options:
            rendered = [json.loads((ROOT / case["folder"] / "render.json").read_text()) for case in cases]
            for name in ("camera_position", "camera_focal_point", "camera_parallel_scale", "color_range"):
                if rendered[0][name] != rendered[1][name]:
                    raise ValueError(f"Adhesion pair rendering differs: {name}")
        provenance.update(completed=True, production_eligible=True, vtk_readback_passed=True)
    if source_record() != source:
        raise ValueError("Source changed during the run")
    save(LOGS / "provenance.json", provenance)
    package(cases, provenance, diagnostic=preset == "jc-density-diagnostic")


def preserve_failure(preset, error):
    destination_root = LOGS
    if (LOGS / "failure.json").exists():
        destination_root = LOGS / "failure-history" / str(time.time_ns())
    save(destination_root / "failure.json", {"error": repr(error), "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "completed": False, "preset": preset})
    cases = plan(preset)
    for case in cases:
        folder = ROOT / case["folder"]
        destination = destination_root / "failed-cases" / case["name"]
        for name in ("summary.json", "status.json", "interpretation.json", "observables.csv", "diagnostics.csv", "memory.json", "numerical-source.json", "jc-density-diagnostic.json", "jc-density-history.csv", "constitutive-review.json", "jc-support-audit.json", "jc-support-history.csv", "runner-acceptance.json", "runtime-symmetry.json"):
            path = folder / name
            if path.is_file():
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination / name)
    for relative in sorted({case["root"] for case in cases}):
        destination = destination_root / "failed-comparisons" / Path(relative).name
        for name in PAIR_FILES:
            path = ROOT / relative / name
            if path.is_file():
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination / name)
    preserve_failure_arrays(cases, destination_root)


def preserve_failure_arrays(cases, destination_root):
    """Keep bounded raw numerical evidence even when report serialization failed."""
    part_limit = 32 * 1024**2
    existing_bytes = sum(path.stat().st_size for path in LOGS.rglob('*') if path.is_file())
    remaining = max(0, LIMIT-262144-existing_bytes)
    initial_budget = remaining
    records, part_number, part_bytes = [], 1, 0
    for name in ("final_state_0.npz", "trajectory.npz"):
        for case in cases:
            origin = ROOT / case['folder'] / name
            if not origin.is_file():
                continue
            record = {"source": (Path(case['folder']) / name).as_posix(), "preserved": False}
            try:
                length = origin.stat().st_size
                record.update(bytes=length, sha256=digest(origin))
                if remaining == 0:
                    record['reason'] = 'No remaining space in the total failure-evidence budget.'
                    records.append(record)
                    continue
                chunks = []
                copied = 0
                with origin.open('rb') as stream:
                    chunk_index = 0
                    while copied < length and remaining > 0:
                        if part_bytes == part_limit:
                            part_number, part_bytes = part_number+1, 0
                        block = stream.read(min(part_limit-part_bytes, 8*1024**2, remaining))
                        if not block:
                            break
                        relative = Path('failed-data') / f'part-{part_number:03d}' / case['name'] / f'{name}.chunk-{chunk_index:04d}'
                        target = destination_root / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(block)
                        chunks.append({"path": relative.as_posix(), "bytes": len(block), "sha256": hashlib.sha256(block).hexdigest()})
                        part_bytes += len(block)
                        remaining -= len(block)
                        copied += len(block)
                        chunk_index += 1
                record.update(preserved=copied == length, copied_bytes=copied, chunks=chunks)
                if copied != length:
                    record['reason'] = 'Only the leading bytes fit the total failure-evidence budget. These chunks are an incomplete file prefix, not a usable NPZ or a complete trajectory.'
            except OSError as failure:
                record['reason'] = repr(failure)
            records.append(record)
    save(destination_root / 'failed-data-index.json', {
        "completed": False, "production_eligible": False, "total_budget_bytes": LIMIT,
        "preexisting_log_bytes": existing_bytes, "available_numerical_payload_bytes": initial_budget,
        "budget_already_exceeded_by_existing_records": existing_bytes > LIMIT,
        "copied_numerical_payload_bytes": initial_budget-remaining, "part_limit_bytes": part_limit,
        "files": records,
        "restoration": "Verify every chunk hash and byte count, concatenate chunks in their recorded order, then verify the complete source hash and size. Raw evidence may be incomplete after a solver interruption; a preserved file is not a passed numerical review. All existing failure records are retained.",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "run"))
    parser.add_argument("--preset", choices=PRESETS, required=True)
    args = parser.parse_args()
    if os.environ.get("GITHUB_ACTIONS") != "true" or platform.system() != "Linux":
        parser.error("This helper only runs on the Linux GitHub Actions runner")
    shared.LOGS, shared.DELIVERY, shared.command = LOGS, DELIVERY, command
    LOGS.mkdir(parents=True, exist_ok=True)
    try:
        if args.action == "install":
            if args.preset != "jc-density-diagnostic":
                shared.install()
        else:
            run(args.preset)
    except Exception as error:
        preserve_failure(args.preset, error)
        raise


if __name__ == "__main__":
    main()
