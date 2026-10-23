"""Collect completed run summaries and diagnostics without large trajectories."""

import argparse
import hashlib
import json
from pathlib import Path

from public_runs import is_public_run


SOURCES = (
    "suite",
    "elastic_cantilever",
    "compression",
    "dense-materials",
    "material-engineering",
    "benchmarks",
    "engineering",
    "contact-models",
    "production",
    "refinement",
)


def write_if_changed(path, content):
    data = content.encode("utf-8") if isinstance(content, str) else content
    if not path.exists() or path.read_bytes() != data:
        path.write_bytes(data)


def collect(root, destination):
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    completed = 0
    for source in SOURCES:
        for path in sorted((root / "outputs" / source).rglob("summary.json")):
            folder = path.parent
            summary = json.loads(path.read_text(encoding="utf-8"))
            if not (folder / "trajectory.npz").is_file() or summary.get("completed") is False:
                continue
            completed += 1
            identifier = "--".join(folder.relative_to(root / "outputs").parts)
            if not is_public_run(identifier):
                continue
            target = destination / identifier
            target.mkdir(exist_ok=True)
            hashes = {}
            for name in (
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
            ):
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
        lines.append(
            f"| [{item['id']}](results/{item['id']}/summary.json) | {item['particles']:,} | "
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
    write_if_changed(destination.parent / "RESULTS.md", "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("docs/results"))
    args = parser.parse_args()
    collect(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
