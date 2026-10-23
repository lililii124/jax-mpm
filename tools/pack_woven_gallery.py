"""Verify five completed coarse artifacts and relay their original frames as small GIFs."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.public_runs import is_woven_preview_ready

REPOSITORY = "lililii124/jax-mpm"
SOURCE_RUN = 39952807550
SOURCE_COMMIT = "4c71466cf16445aa36398bf7f7a64a8c78d138c0"
CASES = (("plain", "warp-tension"), ("plain", "weft-tension"),
         ("plain", "transverse-bending"), ("triaxial", "in-plane-shear"),
         ("twill-2x2", "transverse-compression"))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def verify_manifest(folder):
    manifest = read(folder / "delivery-manifest.json")
    for key, value in (("repository", REPOSITORY), ("commit", SOURCE_COMMIT), ("run_id", str(SOURCE_RUN)),
                       ("run_attempt", "1"), ("preset", "coarse"), ("strict_review_passed", False), ("preview_eligible", True)):
        if manifest.get(key) != value:
            raise ValueError(f"Unexpected original manifest {key}")
    listed = set(manifest["files"])
    actual = {path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()}
    if actual != listed | {"delivery-manifest.json"}:
        raise ValueError("Original artifact file list differs from its manifest")
    for name, entry in manifest["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid artifact path")
        path = folder / name
        if path.is_symlink() or path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError(f"Original artifact hash mismatch: {name}")
    return manifest


def encode(paths, original_gif, destination):
    with Image.open(original_gif) as original:
        if original.n_frames != 61:
            raise ValueError("The producer's GIF did not retain its 61 original frames")
        durations = []
        for index in range(61):
            original.seek(index)
            durations.append(original.info["duration"])
    images = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((400, 400), Image.Resampling.LANCZOS)
            images.append(image)
    montage = Image.new("RGB", (8 * 80, 8 * 80), "white")
    for index, image in enumerate(images):
        sample = image.resize((80, 80), Image.Resampling.LANCZOS)
        montage.paste(sample, ((index % 8) * 80, (index // 8) * 80))
        sample.close()
    palette = montage.quantize(colors=96, method=Image.Quantize.MEDIANCUT)
    frames = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
    frames[0].save(destination, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=False, disposal=2)
    decoded_durations, decoded_hashes = [], []
    with Image.open(destination) as decoded:
        dimensions, count = list(decoded.size), decoded.n_frames
        if decoded.info.get("loop") != 0:
            raise ValueError("Compact GIF does not loop")
        for index in range(count):
            decoded.seek(index)
            decoded_durations.append(decoded.info["duration"])
            decoded_hashes.append(hashlib.sha256(decoded.convert("RGB").tobytes()).hexdigest())
    if sum(decoded_durations) != sum(durations) or len(set(decoded_hashes)) < 2:
        raise ValueError("Compact GIF changed total playback duration or became static")
    elapsed, decoded_index, decoded_end = 0, 0, decoded_durations[0]
    for frame, duration in zip(frames, durations):
        while elapsed >= decoded_end:
            decoded_index += 1
            decoded_end += decoded_durations[decoded_index]
        if hashlib.sha256(frame.convert("RGB").tobytes()).hexdigest() != decoded_hashes[decoded_index]:
            raise ValueError("Compact GIF changed the ordered quantized frame sequence")
        elapsed += duration
    for image in (*images, *frames, montage, palette):
        image.close()
    return {"source_frames": 61, "decoded_gif_frames": count, "dimensions": dimensions,
            "source_frame_durations_ms": durations, "decoded_frame_durations_ms": decoded_durations,
            "total_duration_ms": sum(durations), "sha256": digest(destination), "bytes": destination.stat().st_size,
            "encoding": "Pillow LANCZOS spatial downsampling, one shared 96-color palette, no dithering; original order and playback duration, no generated time frames",
            "repeated_frames": "Identical quantized frames may merge with their durations added"}


def build(source, output, relay):
    run = read(source / "source-run.json")
    if not (run["id"] == SOURCE_RUN and run["run_attempt"] == 1 and run["head_sha"] == SOURCE_COMMIT
            and run["conclusion"] == "success" and run["status"] == "completed"
            and run["repository"]["full_name"] == REPOSITORY
            and run["path"] == ".github/workflows/woven-previews.yml"):
        raise ValueError("The selected producer is not the fixed successful coarse workflow")
    artifacts = {item["name"]: item for item in read(source / "source-artifacts.json")["artifacts"]}
    records = []
    for architecture, loading in CASES:
        name = f"woven-{architecture}-{loading}-{SOURCE_RUN}-1"
        artifact = artifacts[name]
        if artifact["expired"] or artifact["workflow_run"]["id"] != SOURCE_RUN:
            raise ValueError("Producer artifact is expired or belongs to another run")
        root = source / "artifacts" / name
        verify_manifest(root)
        folder = root / "outputs/woven-smoke" / architecture / loading
        summary, render, vtk = (read(folder / filename) for filename in ("summary.json", "render.json", "vtk_metadata.json"))
        if not is_woven_preview_ready(folder, f"woven-preview--{architecture}--{loading}", summary):
            raise ValueError("Original preview eligibility check failed")
        if not (render["renderer"] == "ParaView" and render["field"] == "family_id"
                and render["axes"] is False and render["annotations"] is False and render["geometry_scale"] == 1
                and render["frames_rendered"] == vtk["frames"] == 61 and render["woven_display"]["all_particles_retained"] is True
                and read(folder / "vtk_check.json")["passed"] is True):
            raise ValueError("Original ParaView evidence is incomplete")
        paths = [folder / "frames" / f"frame_{index:05d}.png" for index in range(61)]
        if set(paths) != set((folder / "frames").glob("*.png")):
            raise ValueError("Expected precisely 61 original PNG frames")
        png_stream = hashlib.sha256()
        for path in paths:
            with Image.open(path) as image:
                if list(image.size) != render["resolution"]:
                    raise ValueError("Original frame dimensions differ from render.json")
                image.verify()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024**2), b""):
                    png_stream.update(block)
        if png_stream.hexdigest() != read(folder / "video.json")["frames_sha256"]:
            raise ValueError("Original PNG sequence differs from its video provenance")
        with (folder / "observables.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        metric = "center_mean_displacement_y_m" if loading == "transverse-bending" else "maximum_displacement_m"
        values = [abs(float(row[metric])) for row in rows]
        times = [float(row["time"]) for row in rows]
        if len(rows) != 61 or not all(math.isfinite(value) for value in (*values, *times)) or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("Saved observations do not match 61 ordered output times")
        peak = max(range(61), key=values.__getitem__)
        destination = output / "outputs/woven-previews" / architecture / loading
        destination.mkdir(parents=True, exist_ok=True)
        animation = encode(paths, folder / "simulation.gif", destination / "preview.gif")
        selected = {}
        for label, index in (("initial", 0), ("peak", peak), ("final", 60)):
            shutil.copy2(paths[index], destination / f"{label}.png")
            selected[label] = {"file": f"{label}.png", "source_frame": index, "physical_time_s": times[index],
                               "source_png_sha256": digest(paths[index])}
        for filename in ("summary.json", "diagnostics.csv", "observables.csv", "interpretation.json", "preview-review.json",
                         "preview-eligibility.json", "coarse-preview.json", "render.json", "vtk_metadata.json", "vtk_check.json", "video.json"):
            shutil.copy2(folder / filename, destination / filename)
        for original, copied in (("delivery-manifest.json", "producer-delivery-manifest.json"),
                                 ("provenance.json", "producer-provenance.json"), ("numerical-source.json", "numerical-source.json")):
            shutil.copy2(root / original, destination / copied)
        record = {"id": f"woven-preview--{architecture}--{loading}", "producer_repository": REPOSITORY,
                  "producer_commit": SOURCE_COMMIT, "producer_run_id": SOURCE_RUN, "producer_run_attempt": 1,
                  "producer_artifact": {key: artifact[key] for key in ("id", "name", "size_in_bytes", "digest") if key in artifact},
                  "producer_manifest_sha256": digest(root / "delivery-manifest.json"), "relay": relay,
                  "gif_path": (destination / "preview.gif").relative_to(output).as_posix(), "animation": animation,
                  "source_png_sequence_sha256": png_stream.hexdigest(), "source_png_sequence_hash_definition": "SHA-256 of all 61 PNG byte streams concatenated in frame-number order; matches original video.json frames_sha256",
                  "physical_output_times_s": times, "selected_pngs": selected,
                  "peak_metric": metric, "peak_selection": "maximum absolute saved observable",
                  "strict_review_passed": False, "preview_eligible": True,
                  "scope": "Transfer and spatial image encoding only. No solver or renderer rerun; preview.gif is a new derivative, not the original simulation.gif bytes. Full numerical artifacts remain separate."}
        save(destination / "source.json", record)
        records.append(record)
        print(f"{architecture}/{loading}: {animation['bytes']} bytes, {animation['decoded_gif_frames']} GIF frames", flush=True)
    for filename in ("source-run.json", "source-artifacts.json"):
        shutil.copy2(source / filename, output / filename)
    files = {path.relative_to(output).as_posix(): {"bytes": path.stat().st_size, "sha256": digest(path)}
             for path in sorted(output.rglob("*")) if path.is_file()}
    total = sum(item["bytes"] for item in files.values())
    if total >= 30 * 1024**2:
        raise ValueError(f"Light gallery exceeds 30 MiB: {total}")
    save(output / "relay-manifest.json", {"relay": relay, "producer_run_id": SOURCE_RUN, "producer_commit": SOURCE_COMMIT,
                                         "runs": records, "files": files, "total_file_bytes": total})
    if sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) >= 30 * 1024**2:
        raise ValueError("Complete light gallery including its manifest exceeds 30 MiB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / ".local/woven-gallery-input")
    parser.add_argument("--output", type=Path, default=ROOT / ".local/woven-gallery")
    args = parser.parse_args()
    if os.environ.get("GITHUB_REPOSITORY") != REPOSITORY:
        parser.error("This relay only accepts the fixed producer from this repository")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if commit != os.environ["GITHUB_SHA"]:
        raise ValueError("Relay checkout differs from its run commit")
    relay = {"repository": REPOSITORY, "commit": commit, "run_id": os.environ["GITHUB_RUN_ID"],
             "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"], "tool_sha256": digest(Path(__file__)),
             "pillow_version": Image.__version__}
    build(args.source, args.output, relay)


if __name__ == "__main__":
    main()
