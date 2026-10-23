"""Run, review and render one bounded dynamic fibre study; no automatic publication."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_engineering_preview as engine

LOGS = ROOT / ".local/triaxial-study-ci"
DELIVERY = ROOT / ".local/triaxial-study-delivery"
SIZES = {"20x14": (.020, .014), "24x16": (.024, .016), "28x18": (.028, .018)}
READ_BYTES = 4 * 1024**2
PART_BYTES = 80 * 1024**2
TOTAL_BYTES = 192 * 1024**2
MANIFEST_RESERVE = 1024**2
REVIEW_FILES = ("tools/audit_triaxial_records.py", "tools/review_triaxial_state.py")
RECORD_INPUTS = {"plan.json", "summary.json", "status.json", "observables.csv", "diagnostics.csv"}
STATE_INPUTS = RECORD_INPUTS | {"record-audit.json", "final_state_0.npz", "particle_fields.npz",
    "trajectory_x.npy", "trajectory_damage.npy", "trajectory_failed.npy", "trajectory.npz"}
MATERIAL = {"young": 10e9, "poisson": .25, "density": 1500., "tensile_strength": 100e6,
            "fracture_energy": 100., "characteristic_length": 75e-6}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(READ_BYTES), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def selection(theta=0, law="hertz", load_transits=10., size="24x16"):
    if type(theta) is not int or theta not in range(0, 31, 5) or law not in ("hertz", "jkr"):
        raise ValueError("Select theta 0..30 in 5 degree increments and Hertz or JKR")
    if size not in SIZES:
        raise ValueError("Unknown fixed coupon size")
    load_transits = float(load_transits)
    if not math.isfinite(load_transits) or not 0 < load_transits <= 100:
        raise ValueError("load_transits must be finite and in (0,100]; 100 is a runner budget, not a material law")
    tag = str(load_transits).replace(".", "p").replace("+", "")
    output = f"outputs/triaxial-studies/{size}/transits-{tag}"
    return {"theta": theta, "law": law, "load_transits": load_transits, "size": size,
            "dimensions_m": list(SIZES[size]), "grip_length_m": .003,
            "max_nodes": 60_000_000 if size == "28x18" else 35_000_000,
            "grid_storage": "tube", "frames": 60, "output": output,
            "folder": f"{output}/formal/theta-{theta:02d}/{law}",
            "production_eligible": False, "media_policy": "Render within the same job only after full numerical review.",
            "interpretation": "One explicit dynamic numerical case. Fixed transit count changes physical loading time and strain rate with gauge length; it is not by itself a size-convergence protocol."}


def driver_arguments(case):
    return [sys.executable, "-m", "examples.triaxial_failure", "--theta", str(case["theta"]),
            "--law", case["law"], "--load-transits", str(case["load_transits"]),
            "--dimensions", *map(str, case["dimensions_m"]), "--grip-length", ".003",
            "--grid-storage", "tube", "--frames", "60", "--max-nodes", str(case["max_nodes"]),
            "--max-particles", "1000000", "--dx", "75e-6", "--particle-spacing", "45e-6",
            "--padding-cells", "6", "--packing", "1", "--diameter", "235e-6", "--pitch", ".001",
            "--crimp", "140e-6", "--clearance", "2e-6", "--young", "1e10", "--poisson", ".25",
            "--density", "1500", "--tensile-strength", "1e8", "--fracture-energy", "100",
            "--max-strain", ".02", "--coverage-skin", ".0005", "--search-window", "6",
            "--work-of-adhesion", ".1", "--output", case["output"]]


def bounded(arguments, minutes):
    return ["timeout", "--signal=INT", "--kill-after=60s", f"{minutes}m", *map(str, arguments)]


def before_deadline(arguments, deadline):
    seconds = int(deadline-time.monotonic())
    if seconds < 1:
        raise TimeoutError("The bounded rendering stage exhausted its 15 minute budget")
    return ["timeout", "--signal=INT", "--kill-after=60s", f"{seconds}s", *map(str, arguments)]


def solve_time_budget(deadline, now):
    seconds = min(265*60, int(deadline-now-35*60))
    if seconds < 1:
        raise TimeoutError("Insufficient run budget after reserving 35 minutes for review, rendering and archival")
    return seconds


def source_record():
    record = engine.shared.source_record()
    record.pop("review_source_sha256", None)
    record["runner_source_sha256"] = record["files"]["tools/run_triaxial_study.py"]
    record["review_sources"] = {name: record["files"][name] for name in REVIEW_FILES}
    record["scope"] = "Checked-out execution source; numerical acceptance is separate and production eligibility remains false."
    return record


def producer_matches(producer, source):
    files = producer.get("files", {})
    return bool(files) and all(source["files"].get(name) == value for name, value in files.items()) and (
        producer.get("fingerprint") == hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())


def validate_plan(plan, case, source):
    grid, geometry = plan["grid"], plan["geometry"]
    E, nu, rho = MATERIAL["young"], MATERIAL["poisson"], MATERIAL["density"]
    wave_speed = math.sqrt(E*(1-nu)/((1+nu)*(1-2*nu)*rho))
    dt = .15*75e-6/wave_speed
    load_time = case["load_transits"]*(case["dimensions_m"][0]-.006)/wave_speed
    steps = int(math.ceil(1.2*load_time/dt-1e-10))
    checks = {
        "full_formal_scope": plan["scope"] == "formal" and plan["probe_steps"] == 0,
        "no_envelope_claim": plan["production_eligible"] is False,
        "selection": plan["theta_deg"] == case["theta"] and plan["law"] == case["law"]
            and plan["load_transits"] == case["load_transits"],
        "geometry": geometry["dimensions_m"] == case["dimensions_m"] and geometry["grip_length_m"] == .003
            and geometry["diameter_m"] == 235e-6 and geometry["pitch_m"] == .001
            and geometry["crimp_amplitude_m"] == 140e-6 and geometry["clearance_m"] == 2e-6,
        "material": plan["material"] == MATERIAL,
        "loading_protocol": plan.get("CFL") == .15 and plan.get("hold_fraction_of_loading") == .2
            and math.isclose(plan.get("dt", -1), dt, rel_tol=1e-14)
            and math.isclose(plan["load_time_s"], load_time, rel_tol=1e-14)
            and math.isclose(plan.get("requested_duration_s", -1), 1.2*load_time, rel_tol=1e-14)
            and plan["steps"] == steps and math.isclose(plan["duration_s"], steps*dt, rel_tol=1e-14),
        "contact_parameters": plan.get("coverage_skin_m") == .0005 and plan.get("search_window") == 6
            and plan.get("requested_work_of_adhesion_J_per_m2") == .1
            and plan.get("work_of_adhesion_J_per_m2") == (.1 if case["law"] == "jkr" else 0.),
        "grid_and_sampling": grid["storage"] == "tube" and grid["dx_m"] == 75e-6
            and grid["padding_cells"] == 6 and grid["max_nodes"] == case["max_nodes"]
            and plan["particle_spacing_m"] == 45e-6 and plan["sampling_parameters"]["packing"] == 1.
            and plan["sampling_parameters"]["max_particles"] == 1_000_000,
        "scale_budget": 100_000 <= plan["particles"] <= 1_000_000 and 0 < grid["nodes"] <= case["max_nodes"],
        "unshortened_protocol": plan["stop_steps"] == plan["steps"] and plan["steps"] > 0
            and plan["saved_states"] == 61 and len(plan["frame_steps"]) == 61
            and plan["frame_steps"][-1] == plan["steps"] and plan["max_engineering_strain"] == .02,
        "producer_source": producer_matches(plan["source"], source),
    }
    if not all(checks.values()):
        raise ValueError(f"Resolved study plan mismatch: {[key for key, value in checks.items() if not value]}")
    return checks


def strict_review(report):
    checks = report.get("checks", {})
    return report.get("passed") is True and bool(checks) and all(value is True for value in checks.values())


def physical_loading(plan):
    return {"load_transits": plan["load_transits"], "load_time_s": plan["load_time_s"],
            "requested_duration_s": plan["requested_duration_s"],
            "average_engineering_strain_rate_per_s": plan["max_engineering_strain"]/plan["load_time_s"],
            "peak_prescribed_engineering_strain_rate_per_s": math.pi*plan["max_engineering_strain"]/(2*plan["load_time_s"]),
            "scope": "At fixed elastic wave speed, constant transit count makes loading time proportional to gauge length. Size comparisons must separately consider physical time and inertia; no quasi-static assumption is made."}


def inputs_match(folder, hashes):
    if not hashes:
        return False
    root = Path(folder).resolve()
    for name, expected in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or path == root or not path.is_file() or digest(path) != expected:
            return False
    return True


def acceptance(folder, case, source, planned, source_unchanged):
    folder = Path(folder)
    plan, summary, status = [read_json(folder/name) for name in ("plan.json", "summary.json", "status.json")]
    record, state = [read_json(folder/name) for name in ("record-audit.json", "state-review.json")]
    layout = read_json(folder/"layout.json")
    validate_plan(plan, case, source)
    checks = {
        "unchanged_resolved_plan": plan == planned,
        "source_before_after": source_unchanged is True,
        "complete_numerics": summary.get("completed") is True and status.get("completed") is True
            and summary.get("probe_complete") is not True and status.get("probe_complete") is not True
            and summary.get("steps") == status.get("steps") == plan["steps"]
            and math.isclose(summary.get("simulation_time", -1), plan["duration_s"], rel_tol=1e-9, abs_tol=1e-15),
        "scope_stays_dynamic": summary.get("production_eligible") is False
            and record.get("production_eligible") is False and state.get("production_eligible") is False,
        "actual_tube_layout": layout == summary.get("layout") and layout.get("storage") == "tube"
            and layout.get("dense_aabb_nodes") == plan["grid"]["nodes"]
            and 0 < layout.get("active_nodes", 0) <= layout["dense_aabb_nodes"],
        "record_review": strict_review(record),
        "record_review_source": record.get("auditor_sha256") == source["review_sources"][REVIEW_FILES[0]],
        "record_inputs": RECORD_INPUTS.issubset(record.get("input_sha256", {}))
            and inputs_match(folder, record.get("input_sha256", {})),
        "state_review": strict_review(state),
        "full_saved_history_checked": state.get("full_history_checked") is True,
        "state_review_source": state.get("review_source_sha256") == source["review_sources"][REVIEW_FILES[1]],
        "state_inputs": STATE_INPUTS.issubset(state.get("input_sha256", {}))
            and inputs_match(folder, state.get("input_sha256", {})),
        "consistent_producer_source": plan["source"] == summary.get("source") == record.get("source")
            == state.get("producer_source"),
    }
    return {"schema": "triaxial-single-case-acceptance-v1", "passed": all(checks.values()),
            "checks": checks, "numerical_completed": summary.get("completed") is True,
            "production_eligible": False, "media_created": False, "case": case,
            "media_scope": "Numerical acceptance precedes rendering; any subsequently verified media is recorded separately in media-check.json.",
            "input_sha256": {name: digest(folder/name) for name in
                ("plan.json", "summary.json", "status.json", "layout.json", "record-audit.json", "state-review.json")},
            "runner_source_sha256": source["runner_source_sha256"],
            "physical_loading": physical_loading(plan),
            "scope": "Full saved-state numerical audit of one explicit dynamic protocol; not every substep deformation history, a quasi-static strength, a converged angle envelope, or a public animation."}


def check_media(folder, source, original_inputs):
    import numpy as np
    from PIL import Image

    folder = Path(folder)
    plan, summary = read_json(folder/"plan.json"), read_json(folder/"summary.json")
    metadata, scene, video = [read_json(folder/name) for name in ("vtk_metadata.json", "render.json", "video.json")]
    vtk = read_json(folder/"vtk_check.json")
    count = plan["saved_states"]
    if not (count == 61 and metadata["frames"] == scene["frames_rendered"] == video["frames"] == count
            and metadata["particles"] == plan["particles"] == summary["particles"][0]
            and scene["renderer"] == "ParaView" and scene["field"] == "damage"
            and scene["color_range"] == [0., 1.] and scene["geometry_scale"] == 1
            and scene["axes"] is False and scene["annotations"] is False
            and scene["animation"] == "simulation.gif" and video["render_sha256"] == digest(folder/"render.json")
            and video["sha256"] == digest(folder/"simulation.mp4") and vtk.get("passed") is True
            and vtk.get("sampled_frames") == [0, 60] and vtk.get("particles_per_frame") == plan["particles"]
            and vtk.get("checker_sha256") == source["files"]["tools/check_vtk.py"]
            and vtk.get("input_sha256") == original_inputs
            and inputs_match(folder, vtk.get("export_sha256", {}))):
        raise ValueError("ParaView/media metadata does not match the complete physical result")
    with np.load(folder/"trajectory.npz", allow_pickle=False) as index:
        times = index["time"]
    pvd_times = [float(item.attrib["timestep"]) for item in ET.parse(folder/"particles.pvd").getroot().findall(".//DataSet")]
    np.testing.assert_array_equal(times, pvd_times)
    paths = [folder/"frames"/f"frame_{index:05d}.png" for index in range(count)]
    if sorted((folder/"frames").glob("frame_*.png")) != paths:
        raise ValueError("ParaView PNG sequence is incomplete or contains stale frames")
    for path in paths:
        with Image.open(path) as image:
            if image.size != (1200, 1200):
                raise ValueError("Unexpected ParaView screenshot dimensions")
            image.verify()
    animation = engine.gif_timing(folder/"simulation.gif", count)
    if not inputs_match(folder, original_inputs):
        raise ValueError("Original numerical inputs changed during rendering")
    rendering_sources = {name: source["files"][name] for name in
        ("mpm_jax/visualization.py", "mpm_jax/paraview_scene.py", "mpm_jax/export.py", "tools/check_vtk.py", "tools/encode_video.py")}
    for name, expected in rendering_sources.items():
        if digest(ROOT/name) != expected:
            raise ValueError("Renderer/checker source changed during execution")
    return {"passed": True, "production_eligible": False, "media_created": True,
        "input_sha256": original_inputs, "renderer_sources": rendering_sources,
        "output_sha256": {name: digest(folder/name) for name in
            ("render.json", "vtk_metadata.json", "particles.pvd", "vtk_check.json", "simulation.gif", "video.json", "simulation.mp4")},
        "png_sha256": {path.name: digest(path) for path in paths}, "original_png_frames": count,
        "gif": animation, "geometry_scale": 1, "axes": False, "annotations": False,
        "scope": "Actual full-history ParaView rendering with first/last VTK readback. Visual review and cross-case validation are still required before any public display."}


def render(folder, source, run_deadline):
    folder = Path(folder)
    original_inputs = dict(read_json(folder/"state-review.json")["input_sha256"])
    original_inputs.update({name: digest(folder/name) for name in ("state-review.json", "runner-acceptance.json", "numerical-source.json")})
    deadline = min(time.monotonic()+15*60, run_deadline-3*60)
    engine.command("paraview-study", before_deadline([sys.executable, "-m", "mpm_jax.visualization", str(folder),
        "--gif", "--field", "damage", "--color-range", "0", "1", "--resolution", "1200"], deadline), monitor=True)
    engine.command("vtk-check-study", before_deadline([os.environ["PARAVIEW_PYTHON"], "--force-offscreen-rendering",
        "tools/check_vtk.py", str(folder)], deadline), monitor=True)
    metadata = read_json(folder/"vtk_metadata.json")
    save(folder/"vtk_check.json", {"passed": True, "sampled_frames": [0, 60],
        "particles_per_frame": metadata["particles"], "rtol": 1e-6, "atol": 1e-7,
        "checker_sha256": source["files"]["tools/check_vtk.py"], "input_sha256": original_inputs,
        "export_sha256": {name: digest(folder/name) for name in ("vtk_metadata.json", "particles.pvd", "vtk/particles_00000.vtp", "vtk/particles_00060.vtp")},
        "scope": "Actual ParaView VTK reader checked first and last coordinates, scalars, failed mask and static integer labels against original data."})
    engine.command("video-study", before_deadline([sys.executable, "tools/encode_video.py", str(folder)], deadline), monitor=True)
    media = check_media(folder, source, original_inputs)
    if source_record() != source:
        raise ValueError("Execution source changed before media acceptance")
    save(folder/"media-check.json", media)
    return media


def run(case):
    started = time.monotonic()
    run_deadline = started+295*60
    source = source_record()
    save(LOGS/"numerical-source.json", source)
    save(LOGS/"requested-case.json", case)
    folder = ROOT/case["folder"]
    if folder.exists():
        raise ValueError("A study requires a fresh output directory; no automatic checkpoint reuse")
    after = None
    try:
        args = driver_arguments(case)
        engine.command("planning", bounded([*args, "--dry-run"], 5), monitor=True)
        planned = read_json(LOGS/"planning.log")
        validate_plan(planned, case, source)
        save(LOGS/"resolved-plan.json", planned)
        save(LOGS/"physical-loading.json", physical_loading(planned))
        solve_seconds = solve_time_budget(run_deadline, time.monotonic())
        save(LOGS/"time-budget.json", {"total_run_limit_minutes": 295, "maximum_solve_minutes": 265,
            "reserved_after_solve_minutes": 35, "actual_solve_limit_seconds": solve_seconds,
            "elapsed_before_solve_seconds": time.monotonic()-started,
            "scope": "Wall-clock admission only; no change to physical loading time, particle count or requested complete steps. Timeout remains partial evidence."})
        solve = engine.command("solve", ["timeout", "--signal=INT", "--kill-after=60s", f"{solve_seconds}s", *args], monitor=True)
        save(folder/"memory.json", {**solve["memory"], "completed": solve["completed"],
            "scope": "Solve command process tree; includes GNU timeout and the Python solver. Review and packaging are separate commands.",
            "wall_seconds": solve["wall_seconds"], "solve_walltime_limit_seconds": solve_seconds,
            "maximum_solve_walltime_minutes": 265, "total_run_walltime_limit_minutes": 295,
            "timeout_signal": "SIGINT", "timeout_kill_after_seconds": 60})
        engine.command("record-audit", bounded([sys.executable, REVIEW_FILES[0], str(folder)], 5), monitor=True, accepted_codes=(0, 1))
        engine.command("state-review", bounded([sys.executable, REVIEW_FILES[1], str(folder)], 10), monitor=True, accepted_codes=(0, 1))
        after = source_record()
        save(LOGS/"source-after.json", {"unchanged": after == source, "source": after})
        report = acceptance(folder, case, source, planned, after == source)
        save(folder/"runner-acceptance.json", report)
        save(folder/"numerical-source.json", source)
        if not report["passed"]:
            raise ValueError(f"Single-case numerical review failed: {[key for key, value in report['checks'].items() if not value]}")
        render(folder, source, run_deadline)
        after = source_record()
        save(LOGS/"source-after.json", {"unchanged": after == source, "source": after})
        save(LOGS/"run-result.json", {"completed": True, "single_case_accepted": True,
            "production_eligible": False, "media_created": True, "case": case,
            "acceptance_sha256": digest(folder/"runner-acceptance.json"), "media_check_sha256": digest(folder/"media-check.json")})
    finally:
        try:
            after = source_record()
            save(LOGS/"source-after.json", {"unchanged": after == source, "source": after})
        except Exception as error:
            save(LOGS/"source-after.json", {"unchanged": False, "verification_error": str(error)})


def valid_npz(path):
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            return bool(names) and all(name.endswith(".npy") for name in names) and archive.testzip() is None
    except (OSError, zipfile.BadZipFile, RuntimeError, EOFError, zlib.error):
        return False


def split_complete_file(source, destination, relative, *, part_bytes=PART_BYTES):
    """Stream one whole file into ordered pieces; no unlisted partial checkpoint."""
    source, destination = Path(source), Path(destination)
    if type(part_bytes) is not int or not 0 < part_bytes <= PART_BYTES:
        raise ValueError("Archive part size must be positive and at most 80 MiB")
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Archive part path must be relative and remain inside the delivery")
    before = source.stat()
    whole = hashlib.sha256()
    parts, consumed = [], 0
    with source.open("rb") as stream:
        while consumed < before.st_size:
            target = destination/f"{relative}.part-{len(parts):04d}"
            target.parent.mkdir(parents=True, exist_ok=True)
            part_hash, part_size = hashlib.sha256(), 0
            with target.open("wb") as output:
                while part_size < part_bytes:
                    block = stream.read(min(READ_BYTES, part_bytes-part_size))
                    if not block:
                        if consumed < before.st_size:
                            raise ValueError("Source ended early during bounded archival")
                        break
                    output.write(block)
                    whole.update(block)
                    part_hash.update(block)
                    part_size += len(block)
                    consumed += len(block)
            parts.append({"path": target.relative_to(destination).as_posix(), "bytes": part_size,
                          "sha256": part_hash.hexdigest()})
    after = source.stat()
    if consumed != before.st_size or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Source changed during bounded archival")
    return {"bytes": consumed, "sha256": whole.hexdigest(), "complete_file": True,
            "encoding": "Concatenate all ordered parts and verify whole-file SHA256 before opening the original NPZ.", "parts": parts}


def verify_delivery(delivery):
    delivery = Path(delivery)
    manifest = read_json(delivery/"delivery-manifest.json")
    expected = set(manifest["files"]) | {"delivery-manifest.json"}
    actual = {path.relative_to(delivery).as_posix() for path in delivery.rglob("*") if path.is_file()}
    if actual != expected:
        raise ValueError("Existing delivery contains unlisted or missing files")
    if not inputs_match(delivery, {name: record["sha256"] for name, record in manifest["files"].items()}):
        raise ValueError("Existing delivery is incomplete or changed")
    for name, record in manifest["files"].items():
        if (delivery/name).stat().st_size != record["bytes"]:
            raise ValueError("Existing delivery has a changed byte count")
    for record in manifest["state_files"].values():
        whole, size = hashlib.sha256(), 0
        if record.get("complete_file") is not True or not record.get("parts"):
            raise ValueError("State archive lacks its complete ordered part list")
        for part in record["parts"]:
            if manifest["files"].get(part["path"]) != {"bytes": part["bytes"], "sha256": part["sha256"]} or not 0 < part["bytes"] <= PART_BYTES:
                raise ValueError("State archive part does not match the delivery manifest")
            with (delivery/part["path"]).open("rb") as stream:
                for block in iter(lambda: stream.read(READ_BYTES), b""):
                    whole.update(block)
                    size += len(block)
        if size != record["bytes"] or whole.hexdigest() != record["sha256"]:
            raise ValueError("State archive whole-file checksum is invalid")
    if sum(record["bytes"] for record in manifest["files"].values()) != manifest["payload_bytes"]:
        raise ValueError("Delivery payload byte count is inconsistent")
    if sum((delivery/name).stat().st_size for name in expected) > min(TOTAL_BYTES, manifest["total_budget_bytes"]):
        raise ValueError("Existing delivery exceeds its total archive budget")
    return manifest


def package(case, *, root=ROOT, logs=LOGS, delivery=DELIVERY, total_bytes=TOTAL_BYTES, part_bytes=PART_BYTES):
    root, logs, delivery = Path(root).resolve(), Path(logs).resolve(), Path(delivery).resolve()
    if not all(path.is_relative_to(root) for path in (logs, delivery, root/case["folder"])):
        raise ValueError("Artifact paths must remain within the execution workspace")
    if not MANIFEST_RESERVE < total_bytes <= TOTAL_BYTES or not 0 < part_bytes <= PART_BYTES:
        raise ValueError("Artifact budgets exceed the 192 MiB total or 80 MiB part cap")
    if delivery.exists():
        return verify_delivery(delivery)
    delivery.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=delivery.name+"-staging-", dir=delivery.parent))
    folder = root/case["folder"]
    files, state_files, omitted = {}, {}, {}
    budget = total_bytes-MANIFEST_RESERVE
    used = 0
    candidates = [path for parent in (logs, folder) if parent.exists() for path in parent.rglob("*")
                  if path.is_file() and path.suffix in (".json", ".csv", ".log", ".pvd")]

    def copy_record(path):
        nonlocal used
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        if used+size > budget:
            omitted[relative] = {"bytes": size, "sha256": digest(path), "reason": "Metadata/log exceeds remaining bounded artifact budget."}
            return
        target = stage/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with path.open("rb") as source, target.open("wb") as output:
            shutil.copyfileobj(source, output, length=READ_BYTES)
        files[relative] = {"bytes": size, "sha256": digest(target)}
        used += size

    for path in sorted(p for p in candidates if p.suffix != ".log"):
        copy_record(path)
    numerical = read_json(folder/"runner-acceptance.json") if (folder/"runner-acceptance.json").is_file() else {}
    state_review = read_json(folder/"state-review.json") if (folder/"state-review.json").is_file() else {}
    reviewed = (strict_review(numerical) and inputs_match(folder, numerical.get("input_sha256", {}))
        and strict_review(state_review) and state_review.get("full_history_checked") is True
        and inputs_match(folder, state_review.get("input_sha256", {})))
    media = read_json(folder/"media-check.json") if (folder/"media-check.json").is_file() else {}
    media_created = (reviewed and media.get("passed") is True and media.get("media_created") is True
        and inputs_match(folder, media.get("input_sha256", {})) and inputs_match(folder, media.get("output_sha256", {})))
    priority = {"final_state_0.npz": 0, "particle_fields.npz": 1, "trajectory.npz": 2, "checkpoint.npz": 4} if reviewed else {
        "checkpoint.npz": 0, "final_state_0.npz": 1, "particle_fields.npz": 2, "trajectory.npz": 3}

    def copy_state(path):
        nonlocal used
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        if used+size > budget or not valid_npz(path):
            omitted[relative] = {"bytes": size, "sha256": digest(path), "reason":
                "The whole NPZ exceeds the remaining artifact budget; no partial file is staged." if used+size > budget
                else "NPZ is not a complete CRC-valid NumPy archive; not represented as a checkpoint."}
            return
        record = split_complete_file(path, stage, "state-parts/"+relative, part_bytes=part_bytes)
        state_files[relative] = record
        for part in record["parts"]:
            files[part["path"]] = {"bytes": part["bytes"], "sha256": part["sha256"]}
        used += record["bytes"]

    states = sorted(folder.glob("*.npz"), key=lambda p: (priority.get(p.name, 3), p.name))
    for path in states:
        if not reviewed or path.name != "checkpoint.npz":
            copy_state(path)
    selected_media = [folder/"frames"/f"frame_{index:05d}.png" for index in (0, 30, 60)] + [folder/"simulation.gif"]
    if media_created:
        for path in selected_media:
            if path.is_file():
                copy_record(path)
        video = folder/"simulation.mp4"
        if video.is_file() and video.stat().st_size <= 20*1024**2:
            copy_record(video)
        elif video.is_file():
            omitted[video.relative_to(root).as_posix()] = {"bytes": video.stat().st_size, "sha256": digest(video),
                "reason": "Optional video exceeds the 20 MiB media sub-budget; complete GIF is prioritized."}
    if reviewed:
        for path in states:
            if path.name == "checkpoint.npz":
                copy_state(path)
    for path in sorted(p for p in candidates if p.suffix == ".log"):
        copy_record(path)
    for path in sorted(folder.glob("*.npy")):
        omitted[path.relative_to(root).as_posix()] = {"bytes": path.stat().st_size, "sha256": digest(path),
            "reason": "Full mmap trajectory retained only on the runner for independent review; intentionally not packaged or represented as locally recoverable history."}
    for path in sorted(folder.glob("*.tmp")):
        omitted[path.relative_to(root).as_posix()] = {"bytes": path.stat().st_size,
            "reason": "Interrupted temporary write; never treated as a complete numerical file."}
    source = read_json(logs/"numerical-source.json") if (logs/"numerical-source.json").is_file() else {}
    manifest = {"schema": "triaxial-study-delivery-v1", "case": case, "production_eligible": False,
        "execution": {key: source[key] for key in ("repository", "commit", "tree", "run_id", "run_attempt",
            "source_kind", "fingerprint", "runner_source_sha256") if key in source},
        "media_created": media_created,
        "selected_media_fully_archived": media_created and all(path.relative_to(root).as_posix() in files for path in selected_media),
        "files": files, "state_files": state_files, "omitted": omitted,
        "payload_bytes": used, "total_budget_bytes": total_bytes, "maximum_part_bytes": part_bytes,
        "read_buffer_bytes": READ_BYTES, "checkpoint_scope": "CRC-valid archive inclusion proves file integrity only, not numerical validity or resumability. Never clear failure flags; original source, JAX version and plan checks remain mandatory.",
        "trajectory_scope": "Whole saved mmap history was available to the runner reviewer if it executed. This artifact intentionally cannot reconstruct that full history; review receipts bind its SHA256.",
        "unarchived_groups": {"vtk": "Per-frame VTP datasets are omitted; first/last readback hashes and metadata remain.",
            "frames": "Only first/middle/last PNG files are selected; the full PNG sequence remains hash listed in media-check.json if rendering passed."},
        "media_scope": "When media_created is true, the complete original animation was verified in the same job. Only first/middle/last PNGs, the GIF and optionally a small MP4 are retained within the budget. All original PNG hashes remain in media-check.json. No automatic homepage publication or converged-envelope claim is made."}
    save(stage/"delivery-manifest.json", manifest)
    manifest_bytes = (stage/"delivery-manifest.json").stat().st_size
    if used+manifest_bytes > total_bytes:
        raise ValueError("Manifest would exceed the total bounded artifact budget")
    stage.replace(delivery)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("install", "run", "package"), default="run")
    parser.add_argument("--theta", type=int, choices=range(0, 31, 5), default=0)
    parser.add_argument("--law", choices=("hertz", "jkr"), default="hertz")
    parser.add_argument("--load-transits", type=float, default=10.)
    parser.add_argument("--size", choices=tuple(SIZES), default="24x16")
    parser.add_argument("--package-only", action="store_true")
    args = parser.parse_args()
    if os.environ.get("GITHUB_ACTIONS") != "true" or platform.system() != "Linux":
        parser.error("Numerical studies require the independent Linux Actions runner")
    case = selection(args.theta, args.law, args.load_transits, args.size)
    engine.LOGS = LOGS
    LOGS.mkdir(parents=True, exist_ok=True)
    if args.action == "install":
        engine.shared.LOGS, engine.shared.command = LOGS, engine.command
        try:
            engine.shared.install()
        except BaseException as error:
            save(LOGS/"install-failure.json", {"completed": False, "error_type": type(error).__name__, "error": str(error)})
            raise
        return
    if args.package_only or args.action == "package":
        package(case)
        return
    try:
        run(case)
    except BaseException as error:
        save(LOGS/"failure.json", {"completed": False, "single_case_accepted": False,
            "production_eligible": False, "media_created": False, "case": case,
            "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
            "scope": "A timeout, guard stop or review failure is not a completed accepted protocol. Existing validly written checkpoints and partial records remain diagnostic evidence."})
        raise
    finally:
        package(case)


if __name__ == "__main__":
    main()
