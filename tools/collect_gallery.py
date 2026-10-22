"""Copy completed ParaView images into the documentation gallery."""

import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "Dynamics and contact": [
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
    "Constitutive examples": [
        ("orthotropic", "elastic_cantilever/cantilever/orthotropic"),
        ("transverse-isotropic", "elastic_cantilever/cantilever/transverse-isotropic"),
        ("johnson-cook", "compression/compression/johnson-cook"),
        ("jh2", "compression/compression/jh2"),
        ("hjc", "compression/compression/hjc"),
        ("rht", "compression/compression/rht"),
    ],
}
ANIMATIONS = {
    "oblique",
    "fsi",
    "elastic-gate",
    "floating-block",
    "water-entry",
    "ball-beam",
    "large-twist",
}


def main():
    target = ROOT / "docs/images"
    target.mkdir(exist_ok=True)
    lines = [
        "# Gallery",
        "",
        "ParaView screenshots of completed runs. Every material point is rendered.",
        "Body colours distinguish contacting bodies; the brittle-material compression",
        "images show damage on a fixed range over time. [Run records](RESULTS.md) contain",
        "the parameters and numerical results.",
        "",
    ]
    manifest = []
    for heading, entries in GROUPS.items():
        cells = []
        for name, relative in entries:
            folder = ROOT / "outputs" / relative
            if not all((folder / f).exists() for f in ("summary.json", "render.json", "final.png")):
                continue
            render = json.loads((folder / "render.json").read_text())
            if render.get("renderer") != "ParaView" or render.get("style_revision", 0) < 2:
                continue
            summary = json.loads((folder / "summary.json").read_text())
            shutil.copy2(folder / "final.png", target / f"{name}.png")
            animation = name in ANIMATIONS and (folder / "simulation.gif").exists()
            if animation:
                shutil.copy2(folder / "simulation.gif", target / f"{name}.gif")
            count = sum(summary["particles"])
            identifier = "--".join(Path(relative).parts)
            suffix = "gif" if animation else "png"
            cells.append(
                f"![{name}](images/{name}.{suffix})<br>**{name}** · {count:,} particles<br>"
                f"[Results](results/{identifier}/summary.json)"
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
                }
            )
        if cells:
            lines += [f"## {heading}", "", "| | |", "|---|---|"]
            for i in range(0, len(cells), 2):
                pair = cells[i : i + 2]
                lines.append("| " + " | ".join(pair + [""] * (2 - len(pair))) + " |")
            lines.append("")
    (ROOT / "docs/GALLERY.md").write_text("\n".join(lines), encoding="utf-8")
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Collected {len(manifest)} ParaView scenes")


if __name__ == "__main__":
    main()
