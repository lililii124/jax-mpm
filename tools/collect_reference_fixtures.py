"""Package previously generated reference traces without shipping reference binaries."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("tests/data/materials"))
    parser.add_argument("--jh2-source", type=Path, required=True)
    parser.add_argument("--rht-source", type=Path, required=True)
    parser.add_argument("--hjc-source", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "description": "Material-point traces from the supplied reference implementations; no solver-wide equivalence claim.",
        "sources": {},
        "fixtures": [],
    }
    for model, source in (
        ("jh2", args.jh2_source),
        ("rht", args.rht_source),
        ("hjc", args.hjc_source),
    ):
        if source is None:
            continue
        manifest["sources"][model] = {
            "file": source.name,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    for path in sorted(args.input.glob("*.npz")):
        model = path.stem.split("_")[0]
        if model not in manifest["sources"]:
            continue
        data = np.load(path)
        reference = data["reference"]
        if model == "jh2":
            s = reference[:, :6]
            stress = np.zeros((len(s), 3, 3))
            stress[:, 0, 0], stress[:, 1, 1], stress[:, 2, 2] = s[:, 0], s[:, 1], s[:, 2]
            stress[:, 0, 1] = stress[:, 1, 0] = s[:, 3]
            stress[:, 1, 2] = stress[:, 2, 1] = s[:, 4]
            stress[:, 0, 2] = stress[:, 2, 0] = s[:, 5]
            fields = ("plastic_strain", "damage", "bulking", "pressure")
            values = reference[:, [12, 13, 14, 23]]
            dt = 2e-8
        elif model == "rht":
            stress = data["stress"]
            fields = ("alpha", "plastic_strain", "damage", "hardening", "pressure", "energy")
            values = reference[:, [0, 1, 2, 4, 7, 14]]
            dt = 1e-6
        else:
            stress = data["stress"]
            fields = (
                "plastic_strain",
                "damage",
                "plastic_volume",
                "max_compression",
                "compaction",
                "pressure",
            )
            values = reference[:, [18, 10, 6, 15, 16, 12]]
            dt = 1e-6
        target = args.output / path.name
        np.savez_compressed(
            target, increments=data["increments"], stress=stress, values=values, dt=dt
        )
        manifest["fixtures"].append(
            {
                "file": path.name,
                "model": model,
                "fields": fields,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
