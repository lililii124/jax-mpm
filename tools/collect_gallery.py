"""Copy completed ParaView images into the documentation gallery."""

import json
import csv
import hashlib
from pathlib import Path
import shutil

from public_runs import HOME_REPLACEMENTS, homepage_records, is_public_run, is_ready_for_publication


ROOT = Path(__file__).resolve().parents[1]


def copy_if_changed(source, target):
    if target.exists():
        source_stat, target_stat = source.stat(), target.stat()
        if (source_stat.st_size, source_stat.st_mtime_ns) == (
            target_stat.st_size, target_stat.st_mtime_ns
        ):
            return
    shutil.copy2(source, target)


def compression_source(model):
    dense = f"dense-materials/compression/{model}"
    folder = ROOT / "outputs" / dense
    if all((folder / name).is_file() for name in ("summary.json", "render.json", "final.png")):
        render = json.loads((folder / "render.json").read_text(encoding="utf-8"))
        if render.get("renderer") == "ParaView" and render.get("style_revision", 0) >= 3:
            return dense
    return f"compression/compression/{model}"


GROUPS = {
    "Material engineering": [
        ("jc-tension", "material-engineering/jc-tension/johnson-cook"),
        ("concrete-bending", "material-engineering/concrete-bending/hjc"),
        ("ceramic-indentation", "material-engineering/ceramic-indentation/jh2"),
    ],
    "Adhesion comparison": [],
    "Woven composites": [],
    "Engineering structures and tanks": [
        ("perforated-panel", "engineering/perforated-panel/neo-hookean"),
        ("cellular-cushion", "engineering/cellular-cushion/neo-hookean"),
        ("baffled-tank", "engineering/baffled-tank/neo-hookean"),
        ("unbaffled-tank", "engineering/unbaffled-tank/neo-hookean"),
    ],
    "Dynamics and contact": [
        ("dugdale-array", "adhesion/dugdale-array"),
        ("sphere-contact", "contact-models/sphere-contact"),
        ("jkr-footprint", "contact-models/jkr-footprint"),
        ("wave", "suite/wave/neo-hookean"),
        ("cantilever", "suite/cantilever/neo-hookean"),
        ("torsion", "suite/torsion/neo-hookean"),
        ("oblique", "suite/oblique/neo-hookean"),
        ("sliding", "suite/sliding/neo-hookean"),
        ("rebound", "suite/rebound/neo-hookean"),
        ("rubber-ball", "benchmarks/rubber-ball/neo-hookean"),
        ("ball-beam", "benchmarks/ball-beam/neo-hookean"),
        ("large-bending", "benchmarks/large-bending/neo-hookean"),
        ("large-twist", "benchmarks/large-twist/neo-hookean"),
    ],
    "Fluids and fluid–solid interaction": [
        ("dam-break", "suite/dam-break/neo-hookean"),
        ("sloshing", "suite/sloshing/neo-hookean"),
        ("fsi", "suite/fsi/neo-hookean"),
        ("elastic-gate", "benchmarks/elastic-gate/neo-hookean"),
        ("floating-block", "benchmarks/floating-block/neo-hookean"),
        ("water-entry", "production/water-entry/neo-hookean"),
    ],
    "Finite deformation and prescribed loading": [
        ("visible-cantilever", "visible-motion/cantilever/neo-hookean"),
        ("visible-shear", "visible-motion/shear/neo-hookean"),
        ("visible-torsion", "visible-motion/torsion/neo-hookean"),
        ("visible-surface-load", "visible-motion/surface-load/neo-hookean"),
    ],
    "Transverse shear-wave comparisons": [
        ("visible-wave-grid32", "visible-motion/wave-grid32/neo-hookean"),
        ("visible-wave-grid48", "visible-motion/wave-grid48/neo-hookean"),
        ("visible-wave-grid64", "visible-motion/wave-grid64/neo-hookean"),
        ("visible-wave-half-dt", "visible-motion/wave-half-dt/neo-hookean"),
    ],
    "Constitutive examples": [
        ("orthotropic", "elastic_cantilever/cantilever/orthotropic"),
        ("transverse-isotropic", "elastic_cantilever/cantilever/transverse-isotropic"),
        ("johnson-cook", compression_source("johnson-cook")),
        ("jh2", compression_source("jh2")),
        ("hjc", compression_source("hjc")),
        ("rht", compression_source("rht")),
    ],
}
ANIMATIONS = {
    "dugdale-array",
    "jc-tension",
    "jc-necking",
    "concrete-bending",
    "concrete-bending-large",
    "ceramic-indentation",
    "ceramic-indentation-wide",
    "sphere-contact",
    "jkr-footprint",
    "perforated-panel",
    "cellular-cushion",
    "baffled-tank",
    "unbaffled-tank",
    "oblique",
    "fsi",
    "elastic-gate",
    "floating-block",
    "water-entry",
    "ball-beam",
    "rubber-ball",
    "large-bending",
    "large-twist",
    "dam-break",
    "johnson-cook",
    "jh2",
    "hjc",
    "rht",
    "visible-cantilever",
    "visible-shear",
    "visible-torsion",
    "visible-surface-load",
    "visible-wave-grid32",
    "visible-wave-grid48",
    "visible-wave-grid64",
    "visible-wave-half-dt",
}


def gallery_groups(catalogue):
    groups = {heading: list(entries) for heading, entries in GROUPS.items()}
    present = {record["id"] for record in catalogue}
    for heading, entries in groups.items():
        for index, (name, relative) in enumerate(entries):
            replacement = HOME_REPLACEMENTS.get(relative.replace("/", "--"))
            if replacement in present:
                parts = replacement.split("--")
                if parts[0] == "material-engineering":
                    entries[index] = (parts[1], "/".join(parts))
    listed = {relative for entries in groups.values() for _, relative in entries}
    for record in catalogue:
        parts = record["id"].split("--")
        if parts[0] == "woven":
            relative = "/".join(parts)
            if relative not in listed:
                groups["Woven composites"].append(("-".join(parts), relative))
                listed.add(relative)
            continue
        if parts[0] not in ("adhesion", "adhesion-visible"):
            continue
        relative = "/".join(parts)
        if relative not in listed:
            heading = "Adhesion comparison" if parts[0] == "adhesion-visible" else "Dynamics and contact"
            groups[heading].append((parts[0] + "-" + "-".join(parts[1:]), relative))
            listed.add(relative)
    groups["Adhesion comparison"].sort(key=lambda entry: ("nonadhesive" not in entry[0], entry[0]))
    return groups


def preview(folder, name, render):
    """Select the largest response when a completed animation is available."""
    if folder.parent.name == "adhesion-visible":
        comparison = json.loads((folder.parent / "adhesion_comparison.json").read_text(encoding="utf-8"))
        intervals = comparison["deformation_contrast"]["qualifying_saved_intervals"]
        if comparison.get("passed") is not True or not intervals:
            raise ValueError("A contact-detail preview requires a passed deformation comparison")
        interval = max(intervals, key=lambda item: item["saved_span_s"])
        middle = 0.5 * (interval["first_saved_time_s"] + interval["last_saved_time_s"])
        times = comparison["sampled_time_s"]
        frame = min(range(len(times)), key=lambda i: abs(times[i] - middle))
        return folder / "frames" / f"frame_{frame:05d}.png", frame, float(times[frame])
    if name == "water-entry" and render.get("frames_rendered", 0) > 1:
        frame = round((render["frames_rendered"] - 1) * 2 / 3)
        image = folder / "frames" / f"frame_{frame:05d}.png"
        observable = folder / "observables.csv"
        if image.exists() and observable.exists():
            with observable.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            if frame < len(rows):
                return image, frame, float(rows[frame]["time"])
    columns = {
        "dugdale-array": "surface_displacement_y_min_m",
        "perforated-panel": "maximum_displacement",
        "cellular-cushion": "cushion_upper_displacement_y",
        "baffled-tank": "maximum_baffle_displacement",
        "sphere-contact": "nominal_indentation_m",
        "jkr-footprint": "prescribed_contact_radius_m",
        "large-bending": "tracked_displacement_y",
        "large-twist": "tip_rotation_x_rad",
        "ceramic-indentation": "nominal_indentation_m",
        "ceramic-indentation-wide": "nominal_indentation_m",
        "concrete-bending-large": "midspan_bottom_deflection_m",
    }
    if name.startswith("visible-"):
        columns[name] = (
            "tip_rotation_x_rad" if name == "visible-torsion" else "maximum_displacement_m"
        )
    if name.startswith("woven-"):
        columns[name] = "maximum_displacement_m"
    if name.startswith("adhesion-") or folder.parent.name == "adhesion":
        columns[name] = "surface_displacement_y_min_m"
    observable = folder / "observables.csv"
    if name in columns and observable.exists() and render.get("frames_rendered", 0) > 1:
        with observable.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if rows and columns[name] in rows[0]:
            frame = max(range(len(rows)), key=lambda i: abs(float(rows[i][columns[name]])))
            image = folder / "frames" / f"frame_{frame:05d}.png"
            if image.exists():
                return image, frame, float(rows[frame]["time"])
    return folder / "final.png", None, None


def main():
    target = ROOT / "docs/images"
    target.mkdir(exist_ok=True)
    catalogue = json.loads((ROOT / "docs/results/index.json").read_text(encoding="utf-8"))
    selected_ids = {record["id"] for record in homepage_records(ROOT, catalogue)}
    lines = [
        "# Gallery",
        "",
        "Selected ParaView frames and videos from completed runs. Every material point is rendered.",
        "Body colours distinguish contacting bodies; damage colours use a fixed range",
        "over time. [Run records](RESULTS.md) contain",
        "the parameters and numerical results.",
        "",
    ]
    manifest = []
    for heading, entries in gallery_groups(catalogue).items():
        cells = []
        for name, relative in entries:
            if not is_public_run(relative) or relative.replace("/", "--") not in selected_ids:
                continue
            folder = ROOT / "outputs" / relative
            if not all((folder / f).exists() for f in ("summary.json", "render.json", "final.png")):
                continue
            render = json.loads((folder / "render.json").read_text())
            if render.get("renderer") != "ParaView" or render.get("style_revision", 0) < 3:
                continue
            summary = json.loads((folder / "summary.json").read_text())
            if not is_ready_for_publication(folder, relative, summary):
                continue
            selected, frame, time = preview(folder, name, render)
            copy_if_changed(selected, target / f"{name}.png")
            animation = (
                (name in ANIMATIONS or relative.startswith(("adhesion/", "adhesion-visible/", "woven/")))
                and (folder / "video.json").exists()
                and (folder / "simulation.mp4").exists()
            )
            if animation:
                video = json.loads((folder / "video.json").read_text(encoding="utf-8"))
                animation = (
                    video.get("render_sha256")
                    == hashlib.sha256((folder / "render.json").read_bytes()).hexdigest()
                    and video.get("bytes") == (folder / "simulation.mp4").stat().st_size
                    and video.get("sha256")
                    == hashlib.sha256((folder / "simulation.mp4").read_bytes()).hexdigest()
                )
            if animation:
                copy_if_changed(folder / "simulation.mp4", target / f"{name}.mp4")
            count = sum(summary["particles"])
            identifier = "--".join(Path(relative).parts)
            video_link = f" · [Video](images/{name}.mp4)" if animation else ""
            cells.append(
                f"![{name}](images/{name}.png)<br>**{name}** · {count:,} particles<br>"
                f"[Results](results/{identifier}/summary.json){video_link}"
            )
            manifest.append(
                {
                    "name": name,
                    "source": relative,
                    "particles": count,
                    "field": render["field"],
                    "renderer": render["renderer"],
                    "version": render["version"],
                    "animation": animation,
                    "preview_frame": frame,
                    "preview_time": time,
                    "auxiliary_geometry": render.get("auxiliary_geometry", []),
                }
            )
        if cells:
            columns = 3 if heading == "Woven composites" else 2
            lines += [f"## {heading}", "", "|" + " |" * columns, "|" + "---|" * columns]
            for i in range(0, len(cells), columns):
                pair = cells[i : i + columns]
                lines.append("| " + " | ".join(pair + [""] * (columns - len(pair))) + " |")
            lines.append("")
    (ROOT / "docs/GALLERY.md").write_text("\n".join(lines), encoding="utf-8")
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Collected {len(manifest)} ParaView scenes")


if __name__ == "__main__":
    main()
