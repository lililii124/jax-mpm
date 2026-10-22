"""Encode a complete ParaView screenshot sequence as a compact H.264 video."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def encode(folder, fps=20, crf=19):
    folder = Path(folder).resolve()
    metadata = json.loads((folder / "render.json").read_text(encoding="utf-8"))
    render_hash = hashlib.sha256((folder / "render.json").read_bytes()).hexdigest()
    vtk = json.loads((folder / "vtk_metadata.json").read_text(encoding="utf-8"))
    count = vtk["frames"]
    if (
        metadata.get("renderer") != "ParaView"
        or metadata.get("style_revision", 0) < 3
        or metadata["frames_rendered"] != count
    ):
        raise ValueError("a complete ParaView screenshot sequence is required")
    digest = hashlib.sha256()
    for frame in range(count):
        path = folder / "frames" / f"frame_{frame:05d}.png"
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    signature = digest.hexdigest()
    existing = folder / "video.json"
    video = folder / "simulation.mp4"
    if existing.exists() and video.exists():
        old = json.loads(existing.read_text(encoding="utf-8"))
        if (
            old.get("frames_sha256") == signature
            and old.get("fps") == fps
            and old.get("crf") == crf
            and old.get("render_sha256") == render_hash
        ):
            return video
    executable = shutil.which("ffmpeg")
    if not executable:
        raise FileNotFoundError("ffmpeg is required to encode MP4 files")
    temporary = folder / "simulation.tmp.mp4"
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-framerate",
        str(fps),
        "-start_number",
        "0",
        "-i",
        str(folder / "frames" / "frame_%05d.png"),
        "-frames:v",
        str(count),
        "-c:v",
        "libx264",
        "-threads",
        "2",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    subprocess.run(command, check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    temporary.replace(video)
    result = {
        "file": video.name,
        "renderer": "ParaView",
        "encoder": "ffmpeg/libx264",
        "frames": count,
        "fps": fps,
        "crf": crf,
        "frames_sha256": signature,
        "render_sha256": render_hash,
        "bytes": video.stat().st_size,
        "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
    }
    existing.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{video}: {video.stat().st_size / 1024**2:.2f} MiB, {count} frames")
    return video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--crf", type=int, default=19)
    args = parser.parse_args()
    if args.fps < 1 or not 0 <= args.crf <= 51:
        parser.error("fps >= 1 and 0 <= crf <= 51 are required")
    encode(args.output, args.fps, args.crf)


if __name__ == "__main__":
    main()
