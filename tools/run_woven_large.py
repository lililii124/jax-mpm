"""Resource probe and bounded evidence for new large woven plate cases."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_engineering_preview as engine

LOGS = ROOT/".local/woven-large-ci"
DELIVERY = ROOT/".local/woven-large-delivery"
LIMIT = 192*1024**2
ARCHITECTURES = ("plain", "triaxial", "twill-2x2")
LOADS = ("sphere-impact", "hemisphere-indentation", "pressure-bending")
REVIEW_INPUTS = (
    "plan.json", "summary.json", "status.json", "layout.json", "reference.npz", "final_state_0.npz",
    "checkpoint.npz", "layout_arrays.npz", "particle_fields.npz", "trajectory.npz", "observables.csv",
    "diagnostics.csv", "trajectory_x.npy", "trajectory_speed.npy", "trajectory_damage.npy", "trajectory_temperature.npy",
)


def case_record(architecture, loading):
    if architecture not in ARCHITECTURES or loading not in LOADS:
        raise ValueError("unknown architecture or loading")
    return {"architecture": architecture, "loading": loading,
            "folder": f"outputs/woven-large-probes/{architecture}/{loading}",
            "particles": 819200, "probe_steps": 128, "full_time_s": 200e-6,
            "physical_dimensions_m": [.16, .002, .16], "full_frames": 60,
            "scope": "early resource and integration probe, not a completed impact or a material validation"}


def command_line(case):
    return [sys.executable, "-m", "examples.woven_large_plate", "--architecture", case["architecture"],
            "--case", case["loading"], "--frames", "60", "--dx", ".0005",
            "--probe-steps", "128", "--output", "outputs/woven-large-probes"]


def source_record():
    from examples.woven_large_plate import source_record as numerical_source
    source = numerical_source()
    return {**engine.identity(), "numerical_source": source,
            "controller_source_sha256": engine.digest(__file__),
            "review_source_sha256": engine.digest(ROOT/"tools/review_woven_large.py")}


def verify_review_inputs(folder, summary, review):
    """Check fresh bytes, the reviewer identity and its producer-source binding."""
    folder = Path(folder)
    records = review.get("input_sha256")
    if (not isinstance(records, dict) or set(records) != set(REVIEW_INPUTS)
            or review.get("producer_source") != summary.get("source")
            or not isinstance(summary.get("source"), dict) or not summary["source"]
            or review.get("review_source_sha256") != engine.digest(ROOT/"tools/review_woven_large.py")):
        return False
    for name, expected in records.items():
        path = folder/name
        if (not isinstance(expected, str) or len(expected) != 64 or not path.is_file()
                or path.is_symlink() or path.resolve().parent != folder.resolve() or engine.digest(path) != expected):
            return False
    return True


def probe_assessment(case, summary, review, command, *, verified_inputs=False):
    memory = command.get("memory", {})
    checks = review.get("checks")
    steps = summary.get("steps", 0)
    valid = bool(summary.get("completed") is False and summary.get("probe_complete") is True
        and steps == case["probe_steps"] and summary.get("particles") == [case["particles"]]
        and review.get("probe_passed") is True and review.get("passed") is False
        and review.get("full_saved_history_checked") is True
        and isinstance(checks, dict) and bool(checks) and all(value is True for value in checks.values())
        and isinstance(summary.get("source"), dict) and bool(summary["source"])
        and review.get("producer_source") == summary["source"] and verified_inputs is True
        and command.get("completed") is True and memory.get("samples", 0) > 0
        and not memory.get("stopped_for_memory", True)
        and memory.get("peak_tree_uss_bytes", 13*1024**3) <= 12*1024**3
        and memory.get("minimum_system_available_bytes", 0) >= 2*1024**3)
    seconds = float(summary.get("solver_seconds", 0.))
    if not seconds > 0 or not steps > 0:
        valid = False
    per_step = seconds/steps if steps else None
    estimated = per_step*summary["full_steps"] if per_step is not None else None
    return {"probe_accepted": valid, "full_protocol_completed": False, "production_eligible": False,
            "media_created": False, "input_chain_verified": verified_inputs is True,
            "measured_solver_seconds": seconds, "measured_steps": steps,
            "measured_seconds_per_step": per_step, "early_linear_full_solver_estimate_seconds": estimated,
            "fits_265_minute_solver_budget_by_early_estimate": bool(valid and estimated is not None and estimated <= 265*60),
            "contact_exercised": bool(summary.get("observables", {}).get("final", {}).get("max_contact_nodes", 0) > 0),
            "scope": "An early resource estimate only. Active contact, later deformation, output I/O and compilation can add cost; no full-run permission or physical acceptance is inferred.",
            "memory": memory}


def run_probe(case):
    engine.LOGS = LOGS
    LOGS.mkdir(parents=True, exist_ok=True)
    before = source_record()
    engine.save(LOGS/"source-before.json", before)
    engine.save(LOGS/"requested-case.json", case)
    folder = ROOT/case["folder"]
    if folder.exists():
        raise FileExistsError("each probe requires a fresh output; existing evidence is not overwritten")
    try:
        record = engine.command("solve-probe", ["timeout", "--signal=INT", "--kill-after=60s", "60m", *command_line(case)], monitor=True)
        engine.command("review-probe", ["timeout", "--signal=INT", "--kill-after=60s", "15m", sys.executable,
                       "tools/review_woven_large.py", str(folder), "--allow-probe"], monitor=True, accepted_codes=(0, 1))
        summary = json.loads((folder/"summary.json").read_text())
        review = json.loads((folder/"interpretation.json").read_text())
        after = source_record()
        verified = verify_review_inputs(folder, summary, review)
        assessment = probe_assessment(case, summary, review, record, verified_inputs=verified)
        assessment["source_unchanged"] = after == before
        assessment["probe_accepted"] &= after == before
        engine.save(LOGS/"source-after.json", {"unchanged": after == before, "source": after})
        engine.save(LOGS/"probe-assessment.json", assessment)
        engine.save(folder/"memory.json", record["memory"])
        engine.save(folder/"numerical-source.json", before)
        if not assessment["probe_accepted"]:
            raise RuntimeError("the large-plate probe did not pass resource and independent prefix checks")
        return assessment
    finally:
        after = source_record()
        engine.save(LOGS/"source-after.json", {"unchanged": after == before, "source": after})


def package(case, *, root=ROOT, logs=LOGS, delivery=DELIVERY, limit=LIMIT):
    root, logs, delivery = map(Path, (root, logs, delivery))
    reserve = 1024**2
    if type(limit) is not int or limit <= reserve:
        raise ValueError("package budget must leave one MiB for its complete manifest")
    if delivery.exists():
        raise FileExistsError("delivery already exists; a second packaging must not replace prior evidence")
    delivery.mkdir(parents=True)
    folder = root/case["folder"]
    candidates = []
    if logs.exists():
        candidates.extend((path, Path("records")/path.name) for path in sorted(logs.iterdir()) if path.is_file() and path.suffix in (".json", ".log"))
    priorities = ("plan.json", "summary.json", "status.json", "exception.json", "interpretation.json",
                  "observables.csv", "diagnostics.csv", "layout.json", "memory.json", "numerical-source.json",
                  "checkpoint.npz", "final_state_0.npz", "reference.npz", "particle_fields.npz",
                  "layout_arrays.npz", "trajectory.npz", "trajectory_x.npy", "trajectory_speed.npy",
                  "trajectory_damage.npy", "trajectory_temperature.npy")
    candidates.extend((folder/name, Path(case["folder"])/name) for name in priorities if (folder/name).is_file())
    files, omitted, used = {}, {}, 0
    for path, relative in candidates:
        if path.is_symlink():
            raise ValueError("evidence files cannot be symlinks")
        size = path.stat().st_size
        record = {"bytes": size, "sha256": engine.digest(path)}
        if used+size > limit-reserve:
            omitted[relative.as_posix()] = {**record, "reason": "complete file exceeds remaining 192 MiB package budget"}
            continue
        target = delivery/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if target.stat().st_size != size or engine.digest(target) != record["sha256"] or engine.digest(path) != record["sha256"]:
            raise ValueError("source or copied evidence changed during packaging")
        files[relative.as_posix()] = record
        used += size
    manifest = {"schema": 1, "case": case, "files": files, "omitted": omitted,
                "copied_bytes": used, "budget_bytes": limit, "reserved_manifest_bytes": reserve, "media_created": False,
                "production_eligible": False, "scope": "complete probe evidence files only; omitted arrays are not claimed locally verified after import"}
    encoded = (json.dumps(manifest, indent=2, allow_nan=False)+"\n").encode("utf-8")
    if len(encoded) > reserve or used+len(encoded) > limit:
        raise ValueError("complete manifest exceeds the reserved package budget")
    pending = delivery/"manifest.pending.json"
    pending.write_bytes(encoded)
    if sum(path.stat().st_size for path in delivery.rglob("*") if path.is_file()) > limit:
        raise ValueError("actual delivery bytes exceed the whole-package budget")
    pending.replace(delivery/"manifest.json")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("probe", "package"))
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="plain")
    parser.add_argument("--case", choices=LOADS, default="sphere-impact")
    args = parser.parse_args()
    case = case_record(args.architecture, args.case)
    if args.action == "package":
        package(case)
        return
    if os.environ.get("GITHUB_ACTIONS") != "true" or sys.platform != "linux":
        parser.error("the monitored full-size resource probe is restricted to a Linux Actions runner")
    try:
        result = run_probe(case)
        print(json.dumps(result))
    except BaseException as error:
        engine.save(LOGS/"failure.json", {"probe_accepted": False, "completed": False,
                                        "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
