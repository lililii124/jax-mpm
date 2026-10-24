"""Collect completed run summaries and diagnostics without large trajectories."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

try:
    from .public_runs import is_ready_for_publication
except ImportError:
    from public_runs import is_ready_for_publication


SOURCES = (
    "completed",
    "suite",
    "elastic_cantilever",
    "compression",
    "dense-materials",
    "material-engineering-cloud",
    "material-engineering",
    "benchmarks",
    "engineering",
    "contact-models",
    "adhesion",
    "adhesion-visible-v4-cloud",
    "adhesion-visible-v3-cloud",
    "adhesion-visible-cloud",
    "adhesion-visible",
    "woven",
    "woven-previews",
    "woven-previews-local",
    "woven-smoke",
    "production",
    "refinement",
    "visible-motion",
)
WOVEN_PREVIEW_SOURCES = {"woven-previews", "woven-previews-local", "woven-smoke"}
CLOUD_ALIASES = {"material-engineering-cloud": "material-engineering", "adhesion-visible-cloud": "adhesion-visible", "adhesion-visible-v3-cloud": "adhesion-visible", "adhesion-visible-v4-cloud": "adhesion-visible"}
ADHESION_SOURCES = ("adhesion-visible-v4-cloud", "adhesion-visible-v3-cloud", "adhesion-visible-cloud", "adhesion-visible")
ADHESION_ROLES = {"nonadhesive": "nonadhesive-pull-off", "adhesive": "adhesive-pull-off"}
COMPARISON_FILES = ("adhesion_comparison.json", "adhesion_comparison.csv", "paired-render.json")
JC_FORMAL_ID = "material-engineering--jc-necking--johnson-cook"
JC_CHAIN_FILES = ("constitutive-review.json", "jc-support-audit.json", "jc-support-history.csv", "runner-acceptance.json")
JC_ARRAY_FILES = {"trajectory.npz", "trajectory_x.npy", "trajectory_damage.npy", "trajectory_failed.npy"}


def canonical_identifier(relative):
    parts = Path(str(relative).replace("\\", "/")).parts
    if not parts or Path(*parts).is_absolute() or any(part in (".", "..") for part in parts):
        raise ValueError("Result source must be a relative output directory")
    source = "woven-preview" if parts[0] in WOVEN_PREVIEW_SOURCES else CLOUD_ALIASES.get(parts[0], parts[0])
    return "--".join((source, *parts[1:]))


def catalogue_sources(records):
    result = {}
    pair_roots = set()
    for record in records:
        identifier = record["id"]
        directory = record.get("source_directory", "outputs/" + identifier.replace("--", "/"))
        if not directory.startswith("outputs/"):
            raise ValueError(f"Result source is outside outputs: {identifier}")
        relative = directory.removeprefix("outputs/")
        if canonical_identifier(relative) != identifier or identifier in result:
            raise ValueError(f"Result source or identifier is inconsistent: {identifier}")
        result[identifier] = relative
        if identifier in {"adhesion-visible--" + name for name in ADHESION_ROLES.values()}:
            pair_roots.add(Path(relative).parent.as_posix())
    if len(pair_roots) > 1:
        raise ValueError("The adhesion pair cannot mix source directories")
    return result


def raw_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def recorded_files_match(folder, files, names):
    for name in names:
        record, path = files[name], folder / name
        if record["bytes"] != path.stat().st_size or record["sha256"] != raw_digest(path):
            return False
    return True


def complete_jc_support_chain(folder, summary, receipt, source, augmented):
    """Verify the imported runner audit; omitted arrays are not recomputed here."""
    import numpy as np

    try:
        original, audit, acceptance = [json.loads((folder / name).read_text(encoding="utf-8"))
                                       for name in (JC_CHAIN_FILES[0], JC_CHAIN_FILES[1], JC_CHAIN_FILES[3])]
        passed = lambda report: (report.get("passed") is True and isinstance(report.get("checks"), dict)
                                 and bool(report["checks"]) and all(value is True for value in report["checks"].values()))
        runner_hash = source["files"]["tools/run_engineering_preview.py"]
        reviewer_hash = source["files"]["tools/review_material_engineering.py"]
        if not (receipt.get("preset") == "jc-necking-formal" and summary["particles"] == [102320]
                and summary["steps"] == 57900 and summary["configs"][0]["n_grid"] == 100
                and math.isclose(summary["simulation_time"], 150e-6, rel_tol=1e-12, abs_tol=0)
                and passed(original) and passed(acceptance) and passed(augmented)
                and original.get("review_source_sha256", reviewer_hash) == reviewer_hash
                and augmented["review_source_sha256"] == reviewer_hash
                and audit["runner_source_sha256"] == acceptance["runner_source_sha256"] == augmented["runner_source_sha256"] == runner_hash
                and audit.get("completed") is True and audit.get("diagnostic_only") is False
                and audit.get("strict_review_passed") is True and audit["failed_checks"] == []
                and audit["grid"] == 100 and audit["particles"] == 102320
                and recorded_files_match(folder, receipt["files"], JC_CHAIN_FILES)):
            return False
        core = {"summary.json", "diagnostics.csv", "observables.csv"}
        if set(original["sources"]) != core or augmented["sources"] != original["sources"]:
            return False
        if any(augmented.get(key) != value for key, value in original.items() if key != "checks"):
            return False
        for name in core:
            if original["sources"][name] != hashlib.sha256((folder / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest():
                return False
        audit_files = core | JC_ARRAY_FILES | {"constitutive-review.json", "final_state_0.npz"}
        if set(audit["sources"]) != audit_files:
            return False
        missing = {name for name in JC_ARRAY_FILES if not (folder / name).is_file()}
        proof = receipt["source_evidence"]["jc_support_audit"]
        if not (proof.get("verified") is True and proof.get("basis") == "verified-actions-artifact"
                and proof["audit_sha256"] == raw_digest(folder / "jc-support-audit.json")
                and proof["runner_source_sha256"] == runner_hash
                and proof["array_hash_verification"] == "recorded-by-runner-not-recomputed-after-import"
                and proof["unavailable_array_hashes"] == {name: audit["sources"][name] for name in missing}):
            return False
        artifacts = [item for item in receipt["artifacts"] if item["id"] == proof["artifact_id"] and item["part"] == "main"]
        if len(artifacts) != 1 or not (artifacts[0]["zip_sha256"] == proof["artifact_zip_sha256"]
                and artifacts[0]["github_digest"] == "sha256:" + proof["artifact_zip_sha256"]):
            return False
        for name, expected in audit["sources"].items():
            if not (isinstance(expected, str) and len(expected) == 64 and all(c in "0123456789abcdef" for c in expected)):
                return False
            if name not in missing and raw_digest(folder / name) != expected:
                return False
        csv_record = audit["history_csv"]
        if csv_record["filename"] != "jc-support-history.csv" or csv_record["sha256"] != raw_digest(folder / "jc-support-history.csv"):
            return False
        with (folder / "jc-support-history.csv").open(encoding="utf-8", newline="") as stream:
            table = list(csv.DictReader(stream))
        with np.load(folder / "trajectory_index.npz", allow_pickle=False) as data:
            times = np.asarray(data["time"])
        history, cut_history = audit["saved_history"], audit["saved_positive_axial_support_cuts"]
        if not (len(table) == len(history) == len(cut_history) == csv_record["rows"] == len(times) == 61
                and times[0] == 0 and bool(np.isfinite(times).all()) and bool((np.diff(times) > 0).all())
                and math.isclose(times[-1], 150e-6, rel_tol=1e-12, abs_tol=0)):
            return False
        row_names = {"frame", "time_s", "failed_particles", "maximum_damage", "maximum_axial_gap_m", "axial_support_cuts",
                     "positive_axial_support_components", "central_active_particles", "central_geometric_rms_radius_ratio"}
        for index, (row, text, cuts) in enumerate(zip(history, table, cut_history)):
            if not (set(row) == set(text) == row_names and row["frame"] == cuts["frame"] == index
                    and row["time_s"] == cuts["time_s"] == times[index]
                    and all(type(value) in (int, float) and math.isfinite(value) and float(text[key]) == value for key, value in row.items())
                    and len(cuts["cuts"]) == row["axial_support_cuts"]
                    and row["positive_axial_support_components"] == row["axial_support_cuts"] + int(row["failed_particles"] < 102320)):
                return False
            for key in ("frame", "failed_particles", "axial_support_cuts", "positive_axial_support_components", "central_active_particles"):
                if type(row[key]) is not int or not 0 <= row[key] <= 102320:
                    return False
            for cut in cuts["cuts"]:
                left, right = cut["positive_x_node_intervals"]
                if not (len(left) == len(right) == 2 and all(type(node) is int and 0 <= node <= 100 for node in (*left, *right))
                        and left[0] <= left[1] < right[0] <= right[1]
                        and math.isfinite(cut["gap_m"]) and cut["gap_m"] > 0):
                    return False
        first_cut = next((row for row in history if row["axial_support_cuts"]), None)
        first_failure = next((row for row in history if row["failed_particles"]), None)
        first_central = next((row for row in history if row["central_active_particles"] == 0), None)
        expected_cut = None if first_cut is None else {**first_cut, "cuts": cut_history[first_cut["frame"]]["cuts"]}
        before = first_cut is not None and (first_failure is None or first_cut["frame"] < first_failure["frame"])
        before_central = first_cut is not None and (first_central is None or first_cut["frame"] < first_central["frame"])
        without_failed = any(row["axial_support_cuts"] and row["failed_particles"] == 0 for row in history)
        ordering = ("no_saved_support_disconnection" if first_cut is None else
                    "support_disconnection_observed_before_any_material_failure" if before else
                    "same_saved_interval_unresolved" if first_cut["frame"] == first_failure["frame"] else
                    "material_failure_observed_at_an_earlier_saved_state")
        if not (audit["first_axial_stencil_disconnection"] == expected_cut and audit["first_saved_material_failure"] == first_failure
                and audit["first_saved_central_full_failure"] == first_central
                and audit["support_disconnection_before_material_failure"] is before
                and audit["support_disconnection_before_central_full_failure"] is before_central
                and audit["support_disconnection_with_no_failed_material_observed"] is without_failed
                and before is False and without_failed is False
                and audit["failure_disconnection_ordering"] == acceptance["failure_disconnection_ordering"] == ordering):
            return False
        expected_checks = {"constitutive_review_passed": True, "support_audit_complete": True,
                           "no_saved_support_disconnection_before_material_failure": True}
        expected_augmented = {**original["checks"], **{"runner_" + key: value for key, value in expected_checks.items()}}
        if acceptance["checks"] != expected_checks or augmented["checks"] != expected_augmented:
            return False
        names = {"constitutive-review.json", "jc-support-audit.json", "jc-support-history.csv"}
        if acceptance["sources"] != {name: raw_digest(folder / name) for name in names}:
            return False
        if augmented["supplemental_reviews"] != {name: raw_digest(folder / name) for name in (JC_CHAIN_FILES[0], JC_CHAIN_FILES[1], JC_CHAIN_FILES[3])}:
            return False
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, OverflowError, zipfile.BadZipFile):
        return False


def complete_engineering_media(folder, summary):
    import numpy as np

    try:
        render, metadata, video, audit, vtk, status = [json.loads((folder / name).read_text(encoding="utf-8")) for name in (
            "render.json", "vtk_metadata.json", "video.json", "media-check.json", "vtk_check.json", "status.json")]
        count, particles = metadata["frames"], sum(summary["particles"])
        if not (type(count) is int and count > 1 and count == render["frames_rendered"] == video["frames"]
                and status.get("completed") is True and status["frame"] == status["frames"] == count - 1
                and render["particles"] == metadata["particles"] == particles
                and render["renderer"] == "ParaView" and render["axes"] is False
                and render["annotations"] is False and render["geometry_scale"] == 1
                and render["animation"] == "simulation.gif" and audit.get("passed") is True
                and audit["original_png_frames"] == count and vtk.get("passed") is True
                and vtk["sampled_frames"] == [0, count - 1] and vtk["particles_per_frame"] == particles):
            return False
        expected = [f"frame_{index:05d}.png" for index in range(count)]
        frames = sorted((folder / "frames").glob("frame_*.png"))
        if [path.name for path in frames] != expected or set(audit["png_sha256"]) != set(expected):
            return False
        frame_digest = hashlib.sha256()
        for path in frames:
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != audit["png_sha256"][path.name]:
                return False
            frame_digest.update(content)
        gif = audit["gif"]
        if not (0 < gif["encoded_frames"] <= count and gif["original_png_frames"] == count
                and gif["total_duration_ms"] == gif["expected_duration_ms"] == 60 * count):
            return False
        render_hash, video_hash = raw_digest(folder / "render.json"), raw_digest(folder / "simulation.mp4")
        if not (video["frames_sha256"] == frame_digest.hexdigest()
                and video["render_sha256"] == audit["render_sha256"] == render_hash
                and video["sha256"] == audit["video_sha256"] == video_hash
                and video["bytes"] == (folder / "simulation.mp4").stat().st_size):
            return False
        archive = folder / ("trajectory_index.npz" if (folder / "trajectory_index.npz").is_file() else "trajectory.npz")
        with np.load(archive, allow_pickle=False) as data:
            times = np.asarray(data["time"])
            body = np.asarray(data["body"])
        pvd = [float(item.attrib["timestep"]) for item in ET.parse(folder / "particles.pvd").getroot().findall(".//DataSet")]
        with (folder / "observables.csv").open(encoding="utf-8", newline="") as stream:
            observed_times = [float(row["time"]) for row in csv.DictReader(stream)]
        return (len(times) == len(observed_times) == count and body.shape == (particles,)
                and bool(np.isfinite(times).all()) and bool(np.array_equal(times, pvd))
                and bool(np.allclose(times, observed_times, rtol=1e-12, atol=0)))
    except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError, ET.ParseError, zipfile.BadZipFile):
        return False


def complete_engineering_import(folder, identifier, summary):
    try:
        if summary.get("completed") is not True or not complete_artifact_import(folder) or not is_ready_for_publication(folder, identifier, summary):
            return False
        receipt, source, review = [json.loads((folder / name).read_text(encoding="utf-8")) for name in (
            "import-receipt.json", "numerical-source.json", "interpretation.json")]
        checks = review.get("checks")
        if review.get("passed") is not True or not checks or not all(value is True for value in checks.values()):
            return False
        verification = receipt["source_verification"]
        for name in ("repository", "commit", "tree", "run_id", "run_attempt"):
            proof_key = "attempt" if name == "run_attempt" else name
            value = source.get(name)
            if value in (None, "") or str(receipt.get(name)) != str(value) or str(verification.get(proof_key)) != str(value):
                return False
        reviewer = "tools/review_adhesion.py" if identifier.startswith("adhesion-visible--") else "tools/review_material_engineering.py"
        expected_hash = source["files"][reviewer]
        fingerprint = hashlib.sha256(json.dumps(source["files"], sort_keys=True).encode()).hexdigest()
        if not (verification.get("source_kind") == "github-actions"
                and verification["checked_source_files"] == len(source["files"])
                and verification["fingerprint"] == source["fingerprint"] == fingerprint
                and verification["review_source_sha256"] == source["review_source_sha256"] == review.get("review_source_sha256") == expected_hash):
            return False
        if identifier == JC_FORMAL_ID and not complete_jc_support_chain(folder, summary, receipt, source, review):
            return False
        if not complete_engineering_media(folder, summary):
            return False
        vtk = json.loads((folder / "vtk_check.json").read_text(encoding="utf-8"))
        if vtk.get("checker_sha256") != source["files"].get("tools/check_vtk.py"):
            return False
        render = json.loads((folder / "render.json").read_text(encoding="utf-8"))
        count = render["frames_rendered"]
        field = "displacement" if identifier.startswith("adhesion-visible--") else "damage"
        if render["field"] != field:
            return False
        names = ("trajectory_index.npz", "summary.json", "diagnostics.csv", "observables.csv", "status.json",
                 "numerical-source.json", "interpretation.json", "final_state_0.npz", "render.json", "vtk_metadata.json",
                 "video.json", "media-check.json", "vtk_check.json", "particles.pvd", "memory.json", "scene.pvsm",
                 "final.png", "simulation.gif", "simulation.mp4")
        if identifier == "material-engineering--concrete-bending-symmetric--hjc":
            names += ("runtime-symmetry.json",)
        return recorded_files_match(folder, receipt["files"], (*names, *(f"frames/frame_{index:05d}.png" for index in range(count))))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def adhesion_comparison_ready(folder, *, cloud=False):
    try:
        report = json.loads((folder / "adhesion_comparison.json").read_text(encoding="utf-8"))
        checks = report.get("checks")
        if not (report.get("passed") is True and checks and all(value is True for value in checks.values())
                and checks.get("lift_hold_has_visible_deformation_contrast") is True and set(report["sources"]) == set(ADHESION_ROLES)):
            return False
        identity = None
        renders, metadata = [], []
        for role, name in ADHESION_ROLES.items():
            current = folder / name
            hashes = report["sources"][role]["files"]
            expected = {"summary.json", "diagnostics.csv", "observables.csv", "interpretation.json", "status.json"}
            if set(hashes) != expected:
                return False
            for filename in expected:
                if hashes[filename] != hashlib.sha256((current / filename).read_bytes().replace(b"\r\n", b"\n")).hexdigest():
                    return False
            if cloud:
                receipt = json.loads((current / "import-receipt.json").read_text(encoding="utf-8"))
                current_identity = tuple(str(receipt[key]) for key in ("repository", "commit", "tree", "run_id", "run_attempt"))
                if identity is not None and identity != current_identity:
                    return False
                identity = current_identity
                if set(receipt["comparison_files"]) != set(COMPARISON_FILES) or not recorded_files_match(folder, receipt["comparison_files"], COMPARISON_FILES):
                    return False
                renders.append(json.loads((current / "render.json").read_text(encoding="utf-8")))
                metadata.append(json.loads((current / "vtk_metadata.json").read_text(encoding="utf-8")))
        table = report["csv"]
        if table["filename"] != "adhesion_comparison.csv":
            return False
        content = (folder / table["filename"]).read_text(encoding="utf-8")
        rows = list(csv.DictReader(io.StringIO(content)))
        if not (hashlib.sha256(content.encode()).hexdigest() == table["lf_sha256"]
                and len(rows) == table["rows"] == len(report["sampled_time_s"]) and rows
                and all(math.isfinite(float(value)) for row in rows for value in row.values())
                and [float(row["time_s"]) for row in rows] == report["sampled_time_s"]):
            return False
        if cloud:
            pair = json.loads((folder / "paired-render.json").read_text(encoding="utf-8"))
            for key in ("camera_position", "camera_focal_point", "camera_parallel_scale", "color_range"):
                if renders[0][key] != renders[1][key]:
                    return False
            expected_range = [0., max(item["field_ranges"]["displacement"][1] for item in metadata)]
            if pair["color_range"] != expected_range or renders[0]["color_range"] != expected_range:
                return False
            if pair.get("renderer_argument") == "--focus-bounds" and any(
                render.get("camera_bounds") != pair["scene_bounds"] or render.get("render_overrides", {}).get("focus_bounds") != pair["scene_bounds"] for render in renders
            ):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def selected_adhesion_source(root):
    for source in ADHESION_SOURCES:
        if source not in SOURCES:
            continue
        family = root / "outputs" / source
        ready = True
        for name in ADHESION_ROLES.values():
            folder = family / name
            try:
                summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                ready = False
                break
            identifier = "adhesion-visible--" + name
            if source in CLOUD_ALIASES:
                ready = complete_engineering_import(folder, identifier, summary)
            else:
                ready = ((folder / "trajectory.npz").is_file() or complete_artifact_import(folder)) and is_ready_for_publication(folder, identifier, summary)
            if not ready:
                break
        if ready and adhesion_comparison_ready(family, cloud=source in CLOUD_ALIASES):
            return source
    return None


def complete_artifact_import(folder):
    """A small trajectory index must belong to a fully verified Actions import."""
    try:
        receipt = json.loads((folder / "import-receipt.json").read_text(encoding="utf-8"))
        source = json.loads((folder / "numerical-source.json").read_text(encoding="utf-8"))
        if not (receipt.get("verified") is True and source.get("source_kind") == "github-actions"
                and receipt.get("commit") == source.get("commit")
                and str(receipt.get("run_id")) == str(source.get("run_id"))
                and str(receipt.get("run_attempt")) == str(source.get("run_attempt"))):
            return False
        names = ("trajectory_index.npz", "summary.json", "diagnostics.csv", "observables.csv", "numerical-source.json")
        for name in names:
            if receipt["files"][name]["sha256"] != hashlib.sha256((folder / name).read_bytes()).hexdigest():
                return False
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False
    return True


def complete_preview_media(folder):
    try:
        render, metadata, video = [json.loads((folder / name).read_text(encoding="utf-8"))
                                   for name in ("render.json", "vtk_metadata.json", "video.json")]
        count = metadata["frames"]
        if not (count > 1 and count == render["frames_rendered"] == video["frames"]
                and render["particles"] == metadata["particles"]
                and render["renderer"] == "ParaView" and render["field"] == "family_id"
                and render["axes"] is False and render["annotations"] is False
                and render["geometry_scale"] == 1):
            return False
        frames = sorted((folder / "frames").glob("frame_*.png"))
        if [path.name for path in frames] != [f"frame_{i:05d}.png" for i in range(count)]:
            return False
        digest = hashlib.sha256()
        for path in frames:
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024**2), b""):
                    digest.update(block)
        return (video["frames_sha256"] == digest.hexdigest()
                and video["render_sha256"] == hashlib.sha256((folder / "render.json").read_bytes()).hexdigest()
                and video["bytes"] == (folder / "simulation.mp4").stat().st_size
                and video["sha256"] == hashlib.sha256((folder / "simulation.mp4").read_bytes()).hexdigest())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def verify_comparison_sources(root, destination, indexed, identifier, relative, hashes):
    folder = root / "outputs" / relative
    filenames = {
        "summary.json", "observables.csv", "diagnostics.csv",
        "interpretation.json", "status.json",
    }
    if identifier not in indexed or not isinstance(hashes, dict) or set(hashes) != filenames:
        raise ValueError(f"Comparison references an uncollected run or source: {identifier}")
    for filename in filenames:
        content = (folder / filename).read_text(encoding="utf-8").encode("utf-8")
        digest = hashlib.sha256(content).hexdigest()
        if hashes[filename] != digest:
            raise ValueError(f"Comparison source changed: {identifier}/{filename}")
        if filename != "status.json":
            collected = destination / identifier / filename
            if (
                indexed[identifier]["sha256"].get(filename) != digest
                or hashlib.sha256(collected.read_bytes()).hexdigest() != digest
            ):
                raise ValueError(f"Collected comparison source changed: {identifier}/{filename}")
    return folder.relative_to(root).as_posix()


def collect_wave_comparison(root, destination, records):
    source = root / "outputs/visible-motion/wave_comparison.json"
    if not source.exists():
        return
    report = json.loads(source.read_text(encoding="utf-8"))
    checks = report.get("checks")
    if not (
        report.get("passed") is True
        and isinstance(checks, dict)
        and checks
        and all(value is True for value in checks.values())
        and report.get("comparison_family") == "finite-transverse-shear-wave-v1"
    ):
        raise ValueError("The transverse-wave comparison has not passed its review")
    expected = {"wave-grid32", "wave-grid48", "wave-grid64", "wave-half-dt"}
    runs = report.get("runs")
    if not isinstance(runs, dict) or set(runs) != expected:
        raise ValueError("The transverse-wave comparison must contain all four runs")
    indexed = {record["id"]: record for record in records}
    for name, run in runs.items():
        identifier = f"visible-motion--{name}--neo-hookean"
        run["source_directory"] = verify_comparison_sources(
            root, destination, indexed, identifier,
            f"visible-motion/{name}/neo-hookean", run.get("lf_text_sha256"),
        )
    write_if_changed(
        destination / "visible-wave-comparison.json",
        json.dumps(report, indent=2, allow_nan=False) + "\n",
    )


def collect_adhesion_comparison(root, destination, records, family="adhesion"):
    indexed = {record["id"]: record for record in records}
    source_family = family
    if family == "adhesion-visible":
        pair_ids = [family + "--" + name for name in ADHESION_ROLES.values()]
        if not any(identifier in indexed for identifier in pair_ids):
            return
        if not all(identifier in indexed for identifier in pair_ids):
            raise ValueError("The lift-hold pair must be collected together")
        paths = catalogue_sources(records)
        parents = {Path(paths[identifier]).parent.as_posix() for identifier in pair_ids}
        if len(parents) != 1:
            raise ValueError("The lift-hold pair cannot mix source roots")
        source_family = parents.pop()
    source = root / "outputs" / source_family / "adhesion_comparison.json"
    if not source.exists():
        if family == "adhesion-visible" and any(record["id"].startswith(family + "--") for record in records):
            raise ValueError("The lift-hold pair requires its completed comparison review")
        return
    report = json.loads(source.read_text(encoding="utf-8"))
    checks = report.get("checks")
    if not (
        report.get("passed") is True
        and isinstance(checks, dict)
        and checks
        and all(value is True for value in checks.values())
    ):
        raise ValueError("The adhesion comparison has not passed its review")
    if family == "adhesion-visible" and checks.get("lift_hold_has_visible_deformation_contrast") is not True:
        raise ValueError("The lift-hold deformation comparison has not passed")
    sources = report.get("sources")
    roles = {"nonadhesive": "nonadhesive-pull-off", "adhesive": "adhesive-pull-off"}
    if not isinstance(sources, dict) or set(sources) != set(roles):
        raise ValueError("The adhesion comparison must contain the matched pair")
    for role, name in roles.items():
        run = sources[role]
        run["source_directory"] = verify_comparison_sources(
            root, destination, indexed, f"{family}--{name}", f"{source_family}/{name}", run.get("files"),
        )
    table = report.get("csv")
    if not isinstance(table, dict) or table.get("filename") != "adhesion_comparison.csv":
        raise ValueError("The adhesion comparison has no matching CSV record")
    content = (source.parent / table["filename"]).read_text(encoding="utf-8")
    if hashlib.sha256(content.encode("utf-8")).hexdigest() != table.get("lf_sha256"):
        raise ValueError("The adhesion comparison CSV hash has changed")
    rows = list(csv.DictReader(io.StringIO(content)))
    if len(rows) != table.get("rows") or len(rows) != len(report.get("sampled_time_s", [])):
        raise ValueError("The adhesion comparison CSV has a different sample count")
    if not rows or not all(math.isfinite(float(value)) for row in rows for value in row.values()):
        raise ValueError("The adhesion comparison CSV contains missing or nonfinite values")
    if [float(row["time_s"]) for row in rows] != report["sampled_time_s"]:
        raise ValueError("The adhesion comparison CSV has a different time schedule")
    public_name = family + "-comparison"
    table["filename"] = public_name + ".csv"
    report_text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    write_if_changed(destination / (public_name + ".csv"), content)
    write_if_changed(destination / (public_name + ".json"), report_text)


def write_if_changed(path, content):
    data = content.encode("utf-8") if isinstance(content, str) else content
    if not path.exists() or path.read_bytes() != data:
        path.write_bytes(data)


def collect(root, destination):
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    completed = 0
    selected = set()
    adhesion_source = selected_adhesion_source(root)
    for source in SOURCES:
        if source in ADHESION_SOURCES and source != adhesion_source:
            continue
        for path in sorted((root / "outputs" / source).rglob("summary.json")):
            folder = path.parent
            if source in CLOUD_ALIASES and not (folder / "import-receipt.json").is_file():
                continue
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                if source in CLOUD_ALIASES:
                    continue
                raise
            identifier = canonical_identifier(folder.relative_to(root / "outputs"))
            has_trajectory = ((folder / "trajectory.npz").is_file() or complete_artifact_import(folder)
                              or (source == "completed" and is_ready_for_publication(folder, identifier, summary)))
            if source in CLOUD_ALIASES and not complete_engineering_import(folder, identifier, summary):
                continue
            if not has_trajectory or summary.get("completed") is False:
                continue
            completed += 1
            if source in WOVEN_PREVIEW_SOURCES:
                if not complete_preview_media(folder):
                    continue
            if identifier in selected:
                continue
            if not is_ready_for_publication(folder, identifier, summary):
                continue
            selected.add(identifier)
            target = destination / identifier
            target.mkdir(exist_ok=True)
            hashes = {}
            names = (
                "summary.json",
                "diagnostics.csv",
                "observables.csv",
                "render.json",
                "vtk_metadata.json",
                "video.json",
                "jkr_law_cycle.csv",
                "jkr_law_parameters.json",
                "law_comparison.csv",
                "memory.json",
                "vtk_check.json",
                "interpretation.json",
                "numerical-source.json",
                "media-check.json",
                "runtime-symmetry.json",
                "preview-review.json",
                "preview-eligibility.json",
                "publication-review.json",
                "full-contact-review.json", "vtk-full-mesh-audit.json", "vtk-child-receipt.json",
                "read-only-receipt.json", "admission.json", "media-audit.json",
                "status.json",
                "physical-review.json",
                "material-review.json",
                "contact-controller-binding.json",
                "prefix-review.json",
                "provenance.json",
                "original-physical-review.json",
                "signature-reconciliation.json",
                *JC_CHAIN_FILES,
            )
            for name in names:
                original = folder / name
                if original.exists():
                    content = original.read_bytes() if name in JC_CHAIN_FILES else original.read_text(encoding="utf-8").encode("utf-8")
                    if name == "vtk_check.json":
                        verification = json.loads(content)
                        if "command" in verification:
                            verification["command"] = [
                                "pvpython",
                                "tools/check_vtk.py",
                                folder.relative_to(root).as_posix(),
                            ]
                        content = (json.dumps(verification, indent=2) + "\n").encode("utf-8")
                    write_if_changed(target / name, content)
                    hashes[name] = hashlib.sha256(content).hexdigest()
            if source in CLOUD_ALIASES and identifier == JC_FORMAL_ID:
                receipt = json.loads((folder / "import-receipt.json").read_text(encoding="utf-8"))
                proof = receipt["source_evidence"]["jc_support_audit"]
                artifact = next(item for item in receipt["artifacts"] if item["id"] == proof["artifact_id"])
                evidence = {key: receipt[key] for key in ("verified", "repository", "commit", "tree", "run_id", "run_attempt", "preset")}
                evidence.update(source_evidence=receipt["source_evidence"],
                    source_verification=receipt["source_verification"],
                    import_receipt_sha256=raw_digest(folder / "import-receipt.json"),
                    artifact={key: artifact[key] for key in ("id", "name", "part", "github_digest", "zip_sha256")},
                    scope="The imported artifact and its recorded runner audit were verified. Missing trajectory arrays were not independently reconstructed or recomputed during import.")
                content = (json.dumps(evidence, indent=2, allow_nan=False) + "\n").encode()
                write_if_changed(target / "source-evidence.json", content)
                hashes["source-evidence.json"] = hashlib.sha256(content).hexdigest()
            render = (
                json.loads((folder / "render.json").read_text())
                if (folder / "render.json").exists()
                else {}
            )
            records.append(
                {
                    "id": identifier,
                    "case": summary["case"],
                    "source_directory": folder.relative_to(root).as_posix(),
                    "particles": sum(summary["particles"]),
                    "steps": summary["steps"],
                    "simulation_time": summary["simulation_time"],
                    "min_J": (json.loads((folder / "publication-review.json").read_text())["catalogue_metrics"]["min_J"]
                              if source == "completed" else summary["min_J"]),
                    "max_damage": (json.loads((folder / "publication-review.json").read_text())["catalogue_metrics"]["max_damage"]
                                   if source == "completed" else summary["max_damage"]),
                    "solver_seconds": summary["solver_seconds"],
                    "renderer": render.get("renderer"),
                    "sha256": hashes,
                }
            )
            if source in WOVEN_PREVIEW_SOURCES:
                records[-1]["publication_scope"] = "coarse-preview"
    if "visible-motion" in SOURCES:
        collect_wave_comparison(root, destination, records)
    if "adhesion" in SOURCES:
        collect_adhesion_comparison(root, destination, records)
    if any(source in SOURCES for source in ADHESION_SOURCES):
        collect_adhesion_comparison(root, destination, records, "adhesion-visible")
    for source, name in (
        ("contact-models", "contact-review.json"),
        ("benchmarks", "dynamics-review.json"),
    ):
        review = root / "outputs" / source / name
        if review.exists():
            write_if_changed(destination / name, review.read_text(encoding="utf-8"))
    write_catalogue(records, destination)
    print(f"Collected {len(records)} public runs from {completed} completed local runs in {destination}")


def write_catalogue(records, destination):
    write_if_changed(destination / "index.json", json.dumps(records, indent=2) + "\n")
    lines = [
        "# Published runs",
        "",
        "Each row links to a selected completed run's configuration and numerical results. Particle trajectories",
        "and ParaView datasets remain in the local output directory. Solver time excludes",
        "compilation, checkpoint I/O and rendering; these CPU runs shared the host with other jobs.",
        "",
        "Public-example adaptations are described in [EXAMPLES.md](EXAMPLES.md). Finite states",
        "and a positive determinant establish numerical admissibility, not experimental validation.",
        "",
        "| Run | Particles | Steps | Physical time (s) | Minimum J | Maximum damage | Rendering |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in records:
        label = item["id"] + (" · coarse preview" if item.get("publication_scope") == "coarse-preview" else "")
        lines.append(
            f"| [{label}](results/{item['id']}/summary.json) | {item['particles']:,} | "
            f"{item['steps']:,} | {item['simulation_time']:.6g} | {item['min_J']:.6g} | "
            f"{item['max_damage']:.6g} | {item['renderer'] or 'Pending'} |"
        )
    lines += [
        "",
        "The catalogue contains completed runs only. An active run's `status.json` and checkpoints",
        "are progress records and are not included here. [VALIDATION.md](../VALIDATION.md)",
        "describes the separate constitutive and convergence checks.",
        "",
    ]
    if any(item.get("publication_scope") == "coarse-preview" for item in records):
        lines += [
            "Rows marked **coarse preview** retain their unsuccessful production-resolution review.",
            "A separate eligibility record permits visualization with explicit resolution and tracking",
            "limitations; it does not convert that review into a production pass.",
            "See [WOVEN_PREVIEWS.md](WOVEN_PREVIEWS.md).",
            "",
        ]
    if (destination / "visible-wave-comparison.json").exists():
        lines += [
            "The [transverse shear-wave comparison](results/visible-wave-comparison.json)",
            "compares the new finite-motion runs with one another. The original axial-wave",
            "analytical checks remain separate in [VALIDATION.md](../VALIDATION.md).",
            "",
        ]
    if (destination / "adhesion-comparison.json").exists():
        lines += [
            "The [matched adhesion comparison](results/adhesion-comparison.json) and",
            "[response table](results/adhesion-comparison.csv) use the same sphere path",
            "and dynamic body with zero and positive work of adhesion.",
            "",
        ]
    if (destination / "adhesion-visible-comparison.json").exists():
        lines += [
            "The [lift-hold adhesion comparison](results/adhesion-visible-comparison.json)",
            "checks the force-free baseline against a surface still pulled upward by adhesion",
            "at matching times. Its [response table](results/adhesion-visible-comparison.csv)",
            "retains the measured deformation difference.",
            "",
        ]
    write_if_changed(destination.parent / "RESULTS.md", "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("docs/results"))
    args = parser.parse_args()
    collect(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
