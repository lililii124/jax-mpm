"""Collect completed run summaries and diagnostics without large trajectories."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

from public_runs import is_ready_for_publication


SOURCES = (
    "suite",
    "elastic_cantilever",
    "compression",
    "dense-materials",
    "material-engineering",
    "benchmarks",
    "engineering",
    "contact-models",
    "adhesion",
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
    except (OSError, ValueError, KeyError, TypeError):
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
    source = root / "outputs" / family / "adhesion_comparison.json"
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
    indexed = {record["id"]: record for record in records}
    for role, name in roles.items():
        run = sources[role]
        run["source_directory"] = verify_comparison_sources(
            root, destination, indexed, f"{family}--{name}", f"{family}/{name}", run.get("files"),
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
    for source in SOURCES:
        for path in sorted((root / "outputs" / source).rglob("summary.json")):
            folder = path.parent
            summary = json.loads(path.read_text(encoding="utf-8"))
            has_trajectory = (folder / "trajectory.npz").is_file() or complete_artifact_import(folder)
            if not has_trajectory or summary.get("completed") is False:
                continue
            completed += 1
            identifier = "--".join(folder.relative_to(root / "outputs").parts)
            if source in WOVEN_PREVIEW_SOURCES:
                identifier = "woven-preview--" + identifier.removeprefix(source + "--")
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
                "preview-review.json",
                "preview-eligibility.json",
            )
            for name in names:
                original = folder / name
                if original.exists():
                    content = original.read_text(encoding="utf-8").encode("utf-8")
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
                    "min_J": summary["min_J"],
                    "max_damage": summary["max_damage"],
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
    if "adhesion-visible" in SOURCES:
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
