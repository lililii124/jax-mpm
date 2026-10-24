"""Copy completed ParaView images into the documentation gallery."""

import re
import json
import csv
import hashlib
from pathlib import Path
import shutil

try:
    from .public_runs import HIDDEN_HOME_IDS, HOME_REPLACEMENTS, homepage_records, is_public_run, is_ready_for_publication
    from .collect_results import ADHESION_SOURCES, canonical_identifier, catalogue_sources
except ImportError:
    from public_runs import HIDDEN_HOME_IDS, HOME_REPLACEMENTS, homepage_records, is_public_run, is_ready_for_publication
    from collect_results import ADHESION_SOURCES, canonical_identifier, catalogue_sources


try:
    from .public_themes import THEMES, theme, order, metal_stage_cell, ceramic_section_cell
except ImportError:
    from public_themes import THEMES, theme, order, metal_stage_cell, ceramic_section_cell

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
    "Woven composites · coarse previews": [],
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
    "concrete-bending-symmetric",
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
    sources = catalogue_sources(catalogue)
    groups = {heading: list(entries) for heading, entries in GROUPS.items()}
    present = {record["id"] for record in catalogue}
    for heading, entries in groups.items():
        resolved = []
        for name, relative in entries:
            identifier = relative.replace("/", "--")
            visited = {identifier}
            if identifier.startswith("material-engineering--"):
                while (replacement := HOME_REPLACEMENTS.get(identifier)) in present:
                    if replacement in visited:
                        raise ValueError(f"Cyclic gallery replacement: {replacement}")
                    visited.add(replacement)
                    identifier = replacement
                    parts = identifier.split("--")
                    name, relative = parts[1], "/".join(parts)
            if identifier not in HIDDEN_HOME_IDS:
                resolved.append((name, relative))
        groups[heading] = resolved
    listed = {relative for entries in groups.values() for _, relative in entries}
    for record in catalogue:
        parts = record["id"].split("--")
        if parts[0] == "completed":
            relative = sources[record["id"]]
            if relative not in listed:
                groups["Dynamics and contact"].append((parts[1], relative))
                listed.add(relative)
            continue
        if parts[0] in ("woven", "woven-preview"):
            relative = record.get("source_directory", "outputs/" + "/".join(parts)).removeprefix("outputs/")
            if relative not in listed:
                heading = "Woven composites · coarse previews" if parts[0] == "woven-preview" else "Woven composites"
                groups[heading].append(("-".join(parts), relative))
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
    for heading, entries in groups.items():
        groups[heading] = [(name, sources.get(canonical_identifier(relative), relative)) for name, relative in entries]
    symmetry = 'material-engineering--concrete-bending-symmetric--hjc'
    if symmetry in sources and not any('concrete-bending-symmetric' in name for entries in groups.values() for name, _ in entries):
        groups['Material engineering'].append(('concrete-bending-symmetric', sources[symmetry]))
    themed = {heading: [] for heading in THEMES}
    for entries in groups.values():
        for name, relative in entries:
            themed[theme(name + ' ' + relative)].append((name, relative))
    for entries in themed.values():
        entries.sort(key=lambda item: order(item[0]))
    return themed


def preview(folder, name, render):
    """Select the largest response when a completed animation is available."""
    if folder.parent.name in ADHESION_SOURCES:
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
        "concrete-bending-symmetric": "midspan_bottom_deflection_m",
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
    source_ids = {
        record.get("source_directory", "outputs/" + record["id"].replace("--", "/")).removeprefix("outputs/"): record["id"]
        for record in catalogue
    }
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
            identifier = source_ids.get(relative, relative.replace("/", "--"))
            if not is_public_run(identifier) or identifier not in selected_ids:
                continue
            folder = ROOT / "outputs" / relative
            if not all((folder / f).exists() for f in ("summary.json", "render.json", "final.png")):
                continue
            render = json.loads((folder / "render.json").read_text())
            if render.get("renderer") != "ParaView" or render.get("style_revision", 0) < 3:
                continue
            summary = json.loads((folder / "summary.json").read_text())
            if not is_ready_for_publication(folder, identifier, summary):
                continue
            selected, frame, time = preview(folder, name, render)
            copy_if_changed(selected, target / f"{name}.png")
            animation = (
                (name in ANIMATIONS or identifier.startswith(("adhesion--", "adhesion-visible--", "woven--", "woven-preview--", "completed--")))
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
            video_link = f" · [Video](images/{name}.mp4)" if animation else ""
            if identifier.startswith("completed--") and (ROOT / f"docs/images/home/{identifier}.gif").is_file():
                video_link += f" · [GIF](images/home/{identifier}.gif)"
            label = name + (" · Coarse preview" if identifier.startswith("woven-preview--") else "")
            cells.append(
                f'<img src="images/{name}.png" alt="{name}" width="100%"><br>**{label}** · {count:,} particles<br>'
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
        if heading == THEMES[0] and (ROOT / "docs/images/stages/metal-impact-f16.gif").is_file():
            cells.append(metal_stage_cell(""))
        if heading == THEMES[1] and (ROOT / "docs/images/views/ceramic-cross-section.gif").is_file():
            cells.append(ceramic_section_cell(""))
        if cells:
            columns = 3
            lines += [f"## {heading}", "", '<table width="100%">']
            for i in range(0, len(cells), columns):
                pair = cells[i : i + columns]
                lines.append('<tr>')
                for cell in pair + [''] * (columns - len(pair)):
                    cell = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', cell)
                    cell = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', cell)
                    lines.append(f'<td width="{100//columns}%" align="center">{cell}</td>')
                lines.append('</tr>')
            lines += ['</table>', '']
    (ROOT / "docs/GALLERY.md").write_text("\n".join(lines), encoding="utf-8")
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Collected {len(manifest)} ParaView scenes")


if __name__ == "__main__":
    main()
