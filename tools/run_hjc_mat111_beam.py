"""Run a bounded MAT111 beam prefix and preserve its restart data."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_phase_field_visible_crack import owned_session_uss, terminate_owned_group

OUT = ROOT / "outputs/hjc-mat111-qualification"
RAW = OUT / "raw"


def save(name, value):
    path = OUT / name
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def record_source():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if os.environ.get("GITHUB_SHA", head) != head:
        raise RuntimeError("Dispatched source differs from checkout")
    names = subprocess.check_output(["git", "ls-files", "mpm_jax", "examples", "tools", ".github/workflows"],
                                    cwd=ROOT, text=True).splitlines()
    hashes = {}
    for name in names:
        if not name.endswith((".py", ".yml")):
            continue
        committed = subprocess.check_output(["git", "show", f"{head}:{name}"], cwd=ROOT)
        current = (ROOT / name).read_bytes()
        if current.replace(b"\r\n", b"\n") != committed.replace(b"\r\n", b"\n"):
            raise RuntimeError("Modified source: " + name)
        hashes[name] = hashlib.sha256(current).hexdigest()
    save("source.json", {"commit": head, "files": hashes})


def worker(stop_frame, resume=False):
    import jax
    from examples.brittle_engineering import setup
    from examples.common import run, _state_arrays

    jax.config.update("jax_enable_x64", True)
    print("Constructing MAT111 beam", flush=True)
    case = setup("concrete-bending-mat111", grid=64, ppc=3, frames=60)
    substeps = round(case.duration / (60 * case.configs[0].dt))
    save("plan.json", {"definition": case.description, "particles": len(case.states[0].x),
                      "frames": 60, "substeps": substeps, "stop_frame": stop_frame,
                      "full_duration_s": case.duration})
    if not resume:
        np.savez_compressed(OUT / "initial_state.npz", **_state_arrays(case.states[0]))
    print(f"Solving {stop_frame * substeps} prefix steps of {60 * substeps}", flush=True)
    run(case.states, case.materials, case.configs, frames=60, substeps=substeps,
        output=RAW, label="concrete-bending-mat111 / hjc", observer=case.observer,
        metadata=case.description, render_output=False, stop_after_frame=stop_frame, resume=resume)


def admit_parent(expected_review, stop_frame, *, check_only=False):
    review_bytes = (OUT / "prefix-review.json").read_bytes()
    if hashlib.sha256(review_bytes).hexdigest() != expected_review:
        raise RuntimeError("Parent review differs from the independently admitted prefix")
    review = json.loads(review_bytes)
    if not review["passed"]:
        raise RuntimeError("Parent prefix failed")
    for name, expected in review["files"].items():
        path = (OUT / name).resolve()
        if not path.is_relative_to(OUT.resolve()):
            raise RuntimeError("Parent path escapes result directory")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise RuntimeError("Parent file differs: " + name)
    source = json.loads((OUT / "source.json").read_bytes())
    for name, expected in source["files"].items():
        if name.startswith(("mpm_jax/", "examples/")):
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError("Numerical source changed across continuation: " + name)
    checkpoint = json.loads((RAW / "checkpoint.json").read_bytes())
    if not checkpoint["frame"] < stop_frame:
        raise RuntimeError("Continuation must advance beyond its parent")
    if check_only:
        return
    parent = OUT / "parents" / f"frame-{checkpoint['frame']:05d}"
    parent.mkdir(parents=True, exist_ok=False)
    for name in ("source.json", "plan.json", "prefix-review.json", "resource-guard.json"):
        (parent / name).write_bytes((OUT / name).read_bytes())
    (parent / "checkpoint.json").write_bytes((RAW / "checkpoint.json").read_bytes())
    for name in checkpoint["states"]:
        (parent / name).write_bytes((RAW / name).read_bytes())


def audit():
    plan = json.loads((OUT / "plan.json").read_text())
    checkpoint = json.loads((RAW / "checkpoint.json").read_text())
    status = json.loads((RAW / "status.json").read_text())
    initial = dict(np.load(OUT / "initial_state.npz", allow_pickle=False))
    final = dict(np.load(RAW / checkpoint["states"][0], allow_pickle=False))
    count = len(initial["x"])
    dt = plan["definition"]["time_step_s"]
    dx = plan["definition"]["grid_spacing_m"]
    checks = {
        "full_resolution": count == plan["particles"] == 121104,
        "explicit_mat111_profile": plan["definition"]["material_profile"] == "mat111-compatible",
        "requested_prefix": checkpoint["frame"] == status["frame"] == plan["stop_frame"],
        "valid_restart": bool(final["valid"]) and bool(final["history_valid"].all()),
        "finite_state": all(np.isfinite(value).all() for value in final.values()),
        "positive_deformation": bool(np.all(np.linalg.det(final["F"]) > 0)),
        "mass_and_volume_preserved": all(np.array_equal(final[key], initial[key]) for key in ("mass", "volume")),
        "complete_support": bool(np.all((final["x"] >= .5 * dx) & (final["x"] < .4 - .5 * dx))),
        "whole_prefix_tables": len(checkpoint["observations"]) == len(checkpoint["rows"]) == plan["stop_frame"] + 1,
    }
    previous = np.zeros(count)
    for index in range(plan["stop_frame"] + 1):
        with np.load(RAW / "frames_data" / f"frame_{index:05d}.npz", allow_pickle=False) as frame:
            if frame["x"].shape != (count, 3) or not all(np.isfinite(frame[name]).all() for name in frame.files):
                raise RuntimeError("Invalid saved frame")
            if not np.all((frame["damage"] >= previous) & (frame["damage"] <= 1)):
                raise RuntimeError("Nonmonotone or invalid saved damage")
            if not np.isclose(frame["time"], index * plan["substeps"] * dt, rtol=0, atol=1e-15):
                raise RuntimeError("Frame time differs from physical plan")
            previous = frame["damage"].copy()
            if index == plan["stop_frame"]:
                checks["last_frame_matches_restart"] = all(np.array_equal(frame[name], final[field])
                    for name, field in (("x", "x"), ("damage", "history_damage"), ("temperature", "history_temperature")))
    observation = checkpoint["observations"][-1]
    sources = {str(path.relative_to(OUT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(OUT.rglob("*")) if path.is_file() and path.name != "prefix-review.json"}
    report = {"passed": bool(all(checks.values())), "checks": {key: bool(value) for key, value in checks.items()},
              "scope": "Saved-prefix integrity only; structural energy, temporal convergence and full loading remain unverified",
              "time_s": float(final["time"]), "steps": status["steps_completed"],
              "max_damage": float(final["history_damage"].max()),
              "min_J": float(np.linalg.det(final["F"]).min()),
              "damage_mirror_rms": {axis: observation[f"damage_mirror_{axis}_rms"] for axis in ("x", "z")},
              "observation": observation, "files": sources}
    save("prefix-review.json", report)
    if not report["passed"]:
        raise RuntimeError("Prefix integrity check failed")
    print(json.dumps({key: report[key] for key in ("passed", "scope", "time_s", "steps", "max_damage", "min_J", "damage_mirror_rms")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--parent-review-sha256")
    parser.add_argument("--stop-frame", type=int, default=4, choices=range(1, 61))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.worker:
        return worker(args.stop_frame, args.resume)
    if args.audit_only:
        return audit()
    if args.resume:
        if args.parent_review_sha256 is None:
            parser.error("--resume requires the independently admitted parent review SHA-256")
        admit_parent(args.parent_review_sha256, args.stop_frame)
    elif RAW.exists():
        raise RuntimeError("Existing prefix must be reviewed before another run")
    record_source()
    import ctypes
    if sys.platform != "linux" or ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise RuntimeError("Linux owned-session guard required")
    record = {"seconds_limit": 1800, "uss_limit_bytes": int(5.5 * 1024**3),
              "minimum_available_bytes": 2 * 1024**3, "peak_uss_bytes": 0, "guard_reason": None}
    started = time.monotonic()
    command = [sys.executable, "tools/run_hjc_mat111_beam.py", "--worker", "--stop-frame", str(args.stop_frame)]
    if args.resume:
        command.append("--resume")
    process = None
    with (OUT / "solver.log").open("wb") as log:
        try:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                uss = owned_session_uss(process.pid, psutil)
                record["peak_uss_bytes"] = max(record["peak_uss_bytes"], uss)
                reason = ("wall time" if time.monotonic() - started > record["seconds_limit"] else
                          "USS" if uss > record["uss_limit_bytes"] else
                          "available memory" if psutil.virtual_memory().available < record["minimum_available_bytes"] else None)
                if reason:
                    record["guard_reason"] = reason
                    break
                save("resource-guard.json", record)
                time.sleep(1)
            code = process.poll()
        except BaseException as error:
            record["guard_reason"] = repr(error)
            raise
        finally:
            try:
                terminate_owned_group(process)
            finally:
                record.update(returncode=process.returncode if process else None,
                              wall_seconds=time.monotonic() - started)
                save("resource-guard.json", record)
    if code != 0 or record["guard_reason"]:
        raise RuntimeError("Prefix solver stopped; raw diagnostics retained")
    audit()


if __name__ == "__main__":
    main()
