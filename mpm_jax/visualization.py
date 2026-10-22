"""ParaView batch rendering and animation assembly."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from .export import export_series


def find_pvpython():
    configured = os.environ.get("PARAVIEW_PYTHON")
    if configured and Path(configured).is_file():
        return configured
    installed = shutil.which("pvpython")
    if installed:
        return installed
    root = Path(__file__).resolve().parents[1]
    candidates = list((root / ".local/ParaView").glob("**/pvpython.exe"))
    if candidates:
        return str(candidates[0])
    raise FileNotFoundError(
        "ParaView pvpython was not found. Set PARAVIEW_PYTHON or use --no-render."
    )


def render_directory(output, *, gif=False, field="body", resolution=1200, final_only=False):
    output = Path(output).resolve()
    pvpython = find_pvpython()
    command = [
        pvpython,
        "--force-offscreen-rendering",
        str(Path(__file__).with_name("paraview_scene.py")),
        str(output),
        "--field",
        field,
        "--resolution",
        str(resolution),
    ]
    if final_only or not gif:
        command.append("--final-only")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    ) as process:
        with (output / "render.log").open("w", encoding="utf-8") as log:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
        if process.wait():
            raise RuntimeError(f"ParaView rendering failed; see {output / 'render.log'}")
    if gif:
        from PIL import Image

        paths = sorted((output / "frames").glob("frame_*.png"))
        frames = [Image.open(path).convert("RGB") for path in paths]
        frames[0].save(
            output / "simulation.gif",
            save_all=True,
            append_images=frames[1:],
            duration=60,
            loop=0,
            optimize=False,
            disposal=2,
        )
        for frame in frames:
            frame.close()


def render_trajectory(output, *, gif=False, field="body", resolution=1200, final_only=False):
    output = Path(output)
    with np.load(output / "trajectory.npz") as data:
        positions = data["x"]
        times = data["time"]
        fields = {
            key: data[key] for key in ("speed", "damage", "temperature", "body") if key in data
        }
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    config = summary.get("configs", [summary.get("config", {})])[0]
    dim = config.get("dim", positions.shape[-1])
    spacing = config.get("domain_size", 1.0) / config.get("n_grid", 48)
    spacing /= summary.get("definition", {}).get(
        "particles_per_cell_per_axis", summary.get("particles_per_cell_per_axis", 2)
    )
    if "body" not in fields:
        fields["body"] = np.zeros(positions.shape[1], dtype=np.int32)
    export_series(positions, fields, times, output, spacing=spacing, dim=dim)
    del positions, fields
    render_directory(output, gif=gif, field=field, resolution=resolution, final_only=final_only)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gif", action="store_true")
    parser.add_argument(
        "--field", choices=("body", "speed", "damage", "temperature"), default="body"
    )
    parser.add_argument("--resolution", type=int, default=1200)
    parser.add_argument("--final-only", action="store_true")
    args = parser.parse_args()
    render_trajectory(
        args.output,
        gif=args.gif,
        field=args.field,
        resolution=args.resolution,
        final_only=args.final_only,
    )


if __name__ == "__main__":
    main()
