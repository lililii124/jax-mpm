"""Build the README animation catalogue from completed run records."""

import argparse
import hashlib
import json
from pathlib import Path

from public_runs import homepage_records, is_public_run, is_ready_for_publication


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
)
WOVEN_ARCHITECTURES = {"plain": "0°/90° plain weave", "triaxial": "0°/±60° weave", "twill-2x2": "2×2 twill"}
WOVEN_LOADS = {
    "warp-tension": "Warp tension", "weft-tension": "Transverse in-plane tension",
    "in-plane-shear": "In-plane shear", "transverse-compression": "Through-thickness compression",
    "transverse-bending": "Out-of-plane bending",
}
TITLES = {
    "jc-tension": "JC round-bar tension",
    "jc-necking": "JC steel necking",
    "concrete-bending": "HJC concrete bending",
    "concrete-bending-large": "HJC large-displacement bending",
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


def describe(identifier):
    parts = identifier.split("--")
    source = parts[0]
    if source == "woven":
        return GROUPS[9], f"{WOVEN_ARCHITECTURES[parts[1]]} · {WOVEN_LOADS[parts[2]]}"
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
    groups = {name: [] for name in GROUPS}
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
        groups[group].append((title, f"![{title}]({entry['gif_path']})<br>[{title}]({result})"))
        record = {
            key: entry[key] for key in (
                "id", "gif_path", "gif_sha256", "bytes",
                "dimensions", "source_frames", "gif_frames",
            )
        }
        record["render_sha256"] = render_hash
        record["source_render_sha256"] = entry.get("source_render_sha256", entry["render_sha256"])
        public.append(record)
    lines = [BEGIN, "", "[Gallery](docs/GALLERY.md) · [Examples](docs/EXAMPLES.md) · [Verification](VALIDATION.md)", ""]
    for heading in (GROUPS[0], GROUPS[8], GROUPS[9], *GROUPS[1:8]):
        items = groups[heading]
        if not items:
            continue
        columns = 2 if heading == GROUPS[8] else 3
        lines += [f"### {heading}", "", "|" + " |" * columns, "|" + "---|" * columns]
        ordered = sorted(items, key=lambda item: (not item[0].startswith("Nonadhesive"), item[0])) if heading == GROUPS[8] else sorted(items)
        if heading == GROUPS[9]:
            ordered = sorted(items, key=lambda item: (
                tuple(WOVEN_LOADS.values()).index(item[0].split(" · ")[1]),
                tuple(WOVEN_ARCHITECTURES.values()).index(item[0].split(" · ")[0]),
            ))
        cells = [cell for _, cell in ordered]
        for start in range(0, len(cells), columns):
            row = cells[start:start + columns]
            lines.append("| " + " | ".join(row + [""] * (columns - len(row))) + " |")
        lines.append("")
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
    print(f"README: {len(entries)} completed runs with GIFs")


if __name__ == "__main__":
    main()
