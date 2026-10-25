"""Build the README animation catalogue from completed run records."""

import re
import argparse
import hashlib
import json
from pathlib import Path

try:
    from .public_runs import homepage_records, is_public_run, is_ready_for_publication
    from .collect_results import CLOUD_ALIASES, catalogue_sources
except ImportError:
    from public_runs import homepage_records, is_public_run, is_ready_for_publication
    from collect_results import CLOUD_ALIASES, catalogue_sources


ROOT = Path(__file__).resolve().parents[1]
BEGIN = "<!-- BEGIN EXAMPLE GIFS -->"
END = "<!-- END EXAMPLE GIFS -->"
GROUPS = (
    "Material engineering",
    "Structures and tanks",
    "Dynamics and contact",
    "Fluids and fluid–solid interaction",
    "Elasticity and prescribed loading",
    "Constitutive compression",
    "Wave refinement",
    "Shear-wave comparisons",
    "Adhesion comparison",
    "Woven composites",
    "Woven composites · coarse previews",
)
WOVEN_ARCHITECTURES = {"plain": "0°/90° plain weave", "triaxial": "0°/±60° weave", "twill-2x2": "2×2 twill"}
WOVEN_LOADS = {
    "warp-tension": "Warp tension", "weft-tension": "Transverse in-plane tension",
    "in-plane-shear": "In-plane shear", "transverse-compression": "Through-thickness compression",
    "transverse-bending": "Out-of-plane bending",
    "hemispherical-compression": "Hemispherical compression",
}
TITLES = {
    "jc-tension": "JC round-bar tension",
    "jc-necking": "JC steel necking",
    "concrete-bending": "HJC concrete bending",
    "concrete-bending-large": "HJC large-displacement bending",
    "concrete-bending-symmetric": "HJC bending with symmetry constraints",
    "ceramic-indentation": "JH-2 ceramic indentation",
    "ceramic-indentation-wide": "JH-2 plate indentation",
    "cellular-cushion": "Cellular cushioning",
    "perforated-panel": "Perforated panel",
    "baffled-tank": "Flexible baffles",
    "unbaffled-tank": "Unbaffled tank",
    "large-bending": "Large bending",
    "large-twist": "Large twist",
    "head-on": "Head-on contact",
    "oblique": "Oblique contact",
    "sliding": "Frictional sliding",
    "rebound": "Rebound",
    "rubber-ball": "Rubber-ball drop",
    "ball-beam": "Ball–beam contact",
    "sphere-contact": "Spherical indentation",
    "jkr-footprint": "JKR pressure footprint",
    "dugdale-array": "Dugdale adhesive array",
    "nonadhesive-pull-off": "Nonadhesive pull-off",
    "adhesive-pull-off": "Adhesive pull-off",
    "lateral-release": "Lateral release",
    "three-sphere-array": "Three-sphere array",
    "dam-break": "Dam break",
    "sloshing": "Sloshing",
    "fsi": "Flow past an elastic obstacle",
    "elastic-gate": "Elastic gate",
    "floating-block": "Floating block",
    "water-entry": "Water entry",
    "cantilever": "Cantilever",
    "shear": "Simple shear",
    "torsion": "Torsional release",
    "surface-load": "Pressure-loaded panel",
    "compression": "Compression",
    "wave": "Longitudinal wave",
}
MATERIALS = {
    "linear": "Linear elastic",
    "neo-hookean": "Neo-Hookean",
    "blatz-ko": "Blatz–Ko",
    "orthotropic": "Orthotropic",
    "transverse-isotropic": "Transverse isotropic",
    "bilinear": "Bilinear J2",
    "isotropic-plastic": "Isotropic J2",
    "johnson-cook": "Johnson–Cook",
    "steinberg-guinan": "Steinberg–Guinan",
    "thermal-plastic": "Thermal plasticity",
    "hjc": "HJC",
    "jh2": "JH-2",
    "rht": "RHT",
}


def legacy_describe(identifier):
    completed = {
        "fibres--single-fibre-tension": (GROUPS[2], "Single fibre tension"),
        "fibres--dry-three-strand-forming": (GROUPS[2], "Dry three-strand forming"),
        "fibres--three-strand-forming": (GROUPS[2], "Three-strand forming"),
        "fibres--three-strand-tension": (GROUPS[2], "Three-strand tension"),
        "completed--jc-necking-refined": (GROUPS[0], "JC round-bar necking"),
        "completed--radial-sinusoidal-pull-off": (GROUPS[8], "Radial sinusoidal pull-off"),
        "completed--woven-plate-impact": (GROUPS[2], "Clamped woven plate impact"),
        "completed--crossed-fibre-hertz": (GROUPS[2], "Crossed fibre Hertz contact"),
        "completed--crossed-fibre-jkr": (GROUPS[2], "Crossed fibre JKR contact"),
        "completed--crossed-fibre-md": (GROUPS[2], "Crossed fibre MD contact"),
        "completed--crossed-fibre-self-consistent": (GROUPS[2], "Crossed fibre SC local BEM contact"),
        "completed--phase-field-at2-tension": (GROUPS[0], "AT2 phase-field crack"),
        "completed--ceramic-impact-debris": (GROUPS[0], "JH2 ceramic impact and debris"),
    }
    if identifier in completed:
        return completed[identifier]
    parts = identifier.split("--")
    source = parts[0]
    if source in ("woven", "woven-preview"):
        title = f"{WOVEN_ARCHITECTURES[parts[1]]} · {WOVEN_LOADS[parts[2]]}"
        return (GROUPS[10], title + " · Coarse preview") if source == "woven-preview" else (GROUPS[9], title)
    scene = parts[-1] if source in ("contact-models", "adhesion", "adhesion-visible") else parts[-2]
    if source == "adhesion-visible":
        return GROUPS[8], TITLES[scene]
    if source == "adhesion":
        return GROUPS[2], TITLES.get(scene, scene.replace("-", " ").capitalize())
    if source == "visible-motion":
        if scene.startswith("wave-"):
            variant = {
                "wave-grid32": "coarse grid",
                "wave-grid48": "reference grid",
                "wave-grid64": "fine grid",
                "wave-half-dt": "half time step",
            }.get(scene, scene.removeprefix("wave-"))
            return GROUPS[7], f"Transverse shear wave · {variant}"
        title = {
            "cantilever": "Cyclic bending",
            "shear": "Large simple shear",
            "torsion": "Cyclic torsion",
            "surface-load": "Pressure-loaded panel",
        }[scene]
        return GROUPS[4], title
    title = TITLES[scene]
    if source == "material-engineering":
        return GROUPS[0], title
    if source == "engineering" or scene in ("large-bending", "large-twist"):
        return GROUPS[1], title
    if source in ("contact-models", "adhesion") or scene in (
        "head-on", "oblique", "sliding", "rebound", "rubber-ball", "ball-beam"
    ):
        return GROUPS[2], title
    if scene in ("dam-break", "sloshing", "fsi", "elastic-gate", "floating-block", "water-entry"):
        if scene == "water-entry":
            title += " · refined" if source == "production" else " · reference"
        return GROUPS[3], title
    if source in ("compression", "dense-materials"):
        title = "Compression · " + MATERIALS[parts[-1]]
        if source == "dense-materials":
            title += " · refined"
        return GROUPS[5], title
    if scene == "wave":
        variant = {
            "grid32": "coarse grid", "grid64": "fine grid", "half_dt": "half time step"
        }.get(parts[1], "reference grid")
        return GROUPS[6], f"{title} · {variant}"
    if scene == "cantilever":
        title += " · " + MATERIALS[parts[-1]]
    elif scene == "compression":
        title = "Elastic compression"
    return GROUPS[4], title


try:
    from .public_themes import THEMES, HIDDEN_DISPLAY_IDS, theme, order
except ImportError:
    from public_themes import THEMES, HIDDEN_DISPLAY_IDS, theme, order

def describe(identifier):
    _, title = legacy_describe(identifier)
    return theme(identifier), title


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / "docs/images/home/manifest.json"
    payload = json.loads((args.audit or manifest_path).read_text(encoding="utf-8"))
    entries = payload["runs"] if isinstance(payload, dict) else payload
    index_path = root / "docs/results/index.json"
    catalogue = json.loads(index_path.read_text(encoding="utf-8"))
    sources = catalogue_sources(catalogue)
    if any(not is_public_run(record["id"]) for record in catalogue + entries):
        raise ValueError("GIF catalogue contains a run excluded from public documentation")
    if isinstance(payload, dict):
        if not payload.get("completed"):
            raise ValueError("GIF audit is incomplete")
        if payload["index_sha256"] != hashlib.sha256(index_path.read_bytes()).hexdigest():
            raise ValueError("Run catalogue changed after the GIF audit")
    expected = {record["id"] for record in homepage_records(root, catalogue)}
    found = [record["id"] for record in entries]
    if len(found) != len(set(found)) or set(found) != expected:
        raise ValueError(f"GIF catalogue mismatch: missing={expected - set(found)}, extra={set(found) - expected}")
    public = []
    groups = {name: [] for name in THEMES}
    paths = set()
    home = (root / "docs/images/home").resolve()
    for entry in entries:
        relative = Path(entry["gif_path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or path.parent != home or path.suffix != ".gif":
            raise ValueError(f"GIF must be in docs/images/home: {relative}")
        if path in paths:
            raise ValueError(f"Duplicate GIF path: {relative}")
        paths.add(path)
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"GIF size changed: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["gif_sha256"]:
            raise ValueError(f"GIF hash changed: {path}")
        group, title = describe(entry["id"])
        result = f"docs/results/{entry['id']}/summary.json"
        summary = json.loads((root / result).read_text(encoding="utf-8"))
        if not is_ready_for_publication((root / result).parent, entry["id"], summary):
            raise ValueError(f"Run has no matching passed review: {entry['id']}")
        render_path = root / f"docs/results/{entry['id']}/render.json"
        render_hash = hashlib.sha256(render_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        expected_render_hash = entry.get("render_lf_sha256", entry["render_sha256"])
        if render_hash != expected_render_hash:
            raise ValueError(f"Rendering record changed: {render_path}")
        if entry.get("display_view_review"):
            view_path = (root / entry["display_view_review"]).resolve()
            if not view_path.is_relative_to((root / "docs/views").resolve()):
                raise ValueError("Display view review must be archived under docs/views")
            view = json.loads(view_path.read_text(encoding="utf-8"))
            display = view["media"][entry["id"]]
            if (view.get("passed") is not True or view.get("visual_review_passed") is not True
                    or display["gif_sha256"] != entry["gif_sha256"]
                    or display["source_frames"] != entry["source_frames"]):
                raise ValueError("Display view differs from its independent media admission")
        if entry["id"] not in HIDDEN_DISPLAY_IDS:
            groups[group].append((title, f'<img src="{entry["gif_path"]}" alt="{title}" width="100%"><br>[{title}]({result})'))
        record = {
            key: entry[key] for key in (
                "id", "gif_path", "gif_sha256", "bytes",
                "dimensions", "source_frames", "gif_frames",
            )
        }
        record["render_sha256"] = render_hash
        record["source_render_sha256"] = entry.get("source_render_sha256", entry["render_sha256"])
        if entry.get("display_view_review"):
            record["display_view_review"] = entry["display_view_review"]
        relative_source = sources[entry["id"]]
        if relative_source.split("/")[0] in CLOUD_ALIASES:
            directory = "outputs/" + relative_source
            if entry.get("source_directory", directory) != directory:
                raise ValueError(f"GIF source differs from the selected cloud result: {entry['id']}")
            record["source_directory"] = directory
        public.append(record)
    lines = [BEGIN, "", "[Gallery](docs/GALLERY.md) · [Examples](docs/EXAMPLES.md) · [Verification](VALIDATION.md)", ""]
    for heading in THEMES:
        items = groups[heading]
        if not items:
            continue
        columns = 3
        lines += [f"### {heading}", "", '<table width="100%">']
        ordered = sorted(items, key=lambda item: order(item[0]))
        cells = [cell for _, cell in ordered]
        rows = [cells[start:start + columns] for start in range(0, len(cells), columns)]
        if heading == "Fibres" and ordered and ordered[0][0] == "Single fibre tension":
            rows = [cells[:1]] + [cells[start:start + columns] for start in range(1, len(cells), columns)]
        for row in rows:
            lines.append('<tr>')
            for cell in row + [''] * (columns - len(row)):
                cell = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', cell)
                lines.append(f'<td width="{100//columns}%" align="center">{cell}</td>')
            lines.append('</tr>')
        lines += ['</table>', '']
    lines += [END, ""]
    block = "\n".join(lines)
    readme_path = root / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    if BEGIN in readme and END in readme:
        start, stop = readme.index(BEGIN), readme.index(END) + len(END)
    else:
        start, stop = readme.index("| Cellular cushioning |"), readme.index("## Installation")
    readme_path.write_text(readme[:start] + block + "\n" + readme[stop:].lstrip("\n"), encoding="utf-8")
    manifest_path.write_text(json.dumps(public, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"README: {sum(len(items) for items in groups.values())} displayed GIFs")


if __name__ == "__main__":
    main()
