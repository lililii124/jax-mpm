"""Run the fixed woven preview pair on GitHub's CPU runners."""

import argparse
from collections import deque
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / ".local/woven-ci"
DELIVERY = ROOT / ".local/woven-delivery"
LIMIT = 100 * 1024**2
PAIRS = {("plain", "transverse-bending"), ("triaxial", "in-plane-shear")}
PV_VERSION = "6.2.0"
PV_NAME = f"ParaView-{PV_VERSION}-MPI-Linux-Python3.12-x86_64"
PV_URL = f"https://www.paraview.org/files/v6.2/{PV_NAME}.tar.gz"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def identity():
    return {
        "source_kind": "github-actions", "repository": os.environ["GITHUB_REPOSITORY"],
        "commit": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}"),
        "run_id": os.environ["GITHUB_RUN_ID"], "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
    }


def command(name, arguments, *, monitor=False):
    import psutil

    LOGS.mkdir(parents=True, exist_ok=True)
    tail = deque(maxlen=3000)
    started = time.monotonic()
    record = {"command": [str(arg) for arg in arguments], "completed": False}
    memory = {"scope": "this command process tree; sampled once per second; summed RSS may count shared pages more than once", "peak_tree_rss_bytes": 0, "peak_tree_uss_bytes": 0, "minimum_system_available_bytes": None, "samples": 0, "stopped_for_memory": False, "stop_system_available_bytes": 2 * 1024**3}
    environment = os.environ.copy()
    if name.startswith("paraview") or name == "vtk-check":
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
    with subprocess.Popen(arguments, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, errors="replace") as process:
        finished = threading.Event()

        def sample():
            while not finished.wait(1):
                try:
                    parent = psutil.Process(process.pid)
                    tree = [parent, *parent.children(recursive=True)]
                    infos = []
                    for child in tree:
                        try:
                            infos.append(child.memory_full_info())
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    available = psutil.virtual_memory().available
                    memory["samples"] += 1
                    memory["peak_tree_rss_bytes"] = max(memory["peak_tree_rss_bytes"], sum(info.rss for info in infos))
                    memory["peak_tree_uss_bytes"] = max(memory["peak_tree_uss_bytes"], sum(getattr(info, "uss", 0) for info in infos))
                    memory["minimum_system_available_bytes"] = min(memory["minimum_system_available_bytes"] or available, available)
                    if available < memory["stop_system_available_bytes"]:
                        memory["stopped_for_memory"] = True
                        for child in reversed(tree):
                            try:
                                child.kill()
                            except psutil.NoSuchProcess:
                                pass
                        break
                except psutil.NoSuchProcess:
                    break

        watcher = threading.Thread(target=sample, daemon=True) if monitor else None
        if watcher:
            watcher.start()
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                tail.append(line)
            code = process.wait()
        finally:
            finished.set()
            if watcher:
                watcher.join(timeout=3)
            (LOGS / f"{name}.log").write_text("".join(tail)[-200_000:], encoding="utf-8")
            record.update(returncode=process.poll(), wall_seconds=time.monotonic() - started)
            record["completed"] = record["returncode"] == 0
            if monitor:
                record["memory"] = memory
            save(LOGS / f"{name}.json", record)
    if code:
        raise subprocess.CalledProcessError(code, arguments)
    return record


def install():
    target = Path(os.environ["RUNNER_TEMP"]) / "woven-paraview"
    target.mkdir(exist_ok=True)
    archive = target / f"{PV_NAME}.tar.gz"
    command("download-paraview", ["curl", "--fail", "--location", "--retry", "3", "--connect-timeout", "30", "--max-time", "900", PV_URL, "--output", str(archive)])
    record = {"version": PV_VERSION, "url": PV_URL, "bytes": archive.stat().st_size,
              "sha256": digest(archive), "independent_checksum_verified": False,
              "checksum_scope": "SHA-256 of the official HTTPS download; no separate publisher checksum asserted",
              "headless_reference": "https://www.paraview.org/paraview-docs/latest/cxx/Offscreen.html"}
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(target, filter="data")
    archive.unlink()
    executable = target / PV_NAME / "bin/pvpython"
    if not executable.is_file():
        raise FileNotFoundError(executable)
    record["pvpython"] = str(executable)
    save(LOGS / "paraview-install.json", record)
    with Path(os.environ["GITHUB_ENV"]).open("a", encoding="utf-8") as stream:
        stream.write(f"PARAVIEW_PYTHON={executable}\n")
    command("paraview-version", [str(executable), "--version"])
    command("paraview-information", [str(executable), "--force-offscreen-rendering", "-m", "paraview.info"])


def source_record():
    paths = ("mpm_jax", "examples", "tools", ".github/workflows", "pyproject.toml")
    if git("diff", "HEAD", "--", *paths):
        raise ValueError("Runtime source differs from the checked-out commit")
    files, blobs, normalized = {}, {}, {}
    for entry in git("ls-tree", "-r", "HEAD", "--", *paths).splitlines():
        header, name = entry.split("\t", 1)
        if Path(name).suffix not in (".py", ".toml", ".yml", ".yaml"):
            continue
        content = (ROOT / name).read_bytes()
        files[name] = hashlib.sha256(content).hexdigest()
        normalized[name] = hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()
        blobs[name] = header.split()[2]
    record = {**identity(), "files": files, "git_blob_sha": blobs, "lf_sha256": normalized,
              "fingerprint": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
              "review_source_sha256": files["tools/review_woven.py"]}
    if record["commit"] != os.environ["GITHUB_SHA"]:
        raise ValueError("Checked-out commit differs from the dispatched commit")
    return record


def manifest(folder):
    files = {path.relative_to(folder).as_posix(): {"bytes": path.stat().st_size, "sha256": digest(path)}
             for path in sorted(folder.rglob("*")) if path.is_file() and path.name != "delivery-manifest.json"}
    record = {**identity(), "files": files, "total_bytes": sum(item["bytes"] for item in files.values())}
    if record["total_bytes"] > LIMIT:
        raise ValueError(f"Artifact exceeds 100 MiB: {folder}")
    save(folder / "delivery-manifest.json", record)


def package(folder, provenance):
    import numpy as np

    main = DELIVERY / "main"
    relative = folder.relative_to(ROOT)
    result = main / relative
    result.mkdir(parents=True, exist_ok=True)
    names = ("summary.json", "diagnostics.csv", "observables.csv", "interpretation.json", "status.json",
             "particle_fields.npz", "final_state_0.npz", "particles.pvd", "vtk_metadata.json", "render.json",
             "video.json", "vtk_check.json", "memory.json", "scene.pvsm", "final.png", "simulation.gif", "simulation.mp4")
    for name in names:
        shutil.copy2(folder / name, result / name)
    shutil.copytree(folder / "frames", result / "frames")
    with np.load(folder / "trajectory.npz") as archive:
        np.savez_compressed(result / "trajectory_index.npz", time=archive["time"], body=archive["body"])
    provenance["trajectory_index"] = {"file": "trajectory_index.npz", "fields": ["time", "body"], "source": "trajectory.npz", "source_sha256": digest(folder / "trajectory.npz"), "source_bytes": (folder / "trajectory.npz").stat().st_size}
    provenance["omitted_large_data"] = "Full trajectory archives, memory-mapped trajectories and VTK frames remain on the runner; strict numerical review and actual VTK first/last readback ran before packaging. The PVD and scene retain their original references, so they require the omitted VTK series to reopen."
    shutil.copy2(LOGS / "numerical-source.json", main / "numerical-source.json")
    shutil.copy2(LOGS / "paraview-install.json", main / "paraview-install.json")
    save(main / "provenance.json", provenance)
    if sum(path.stat().st_size for path in main.rglob("*") if path.is_file()) > LIMIT:
        separate = DELIVERY / "final-state"
        destination = separate / relative
        destination.mkdir(parents=True, exist_ok=True)
        shutil.move(str(result / "final_state_0.npz"), destination / "final_state_0.npz")
        shutil.copy2(main / "numerical-source.json", separate / "numerical-source.json")
        save(separate / "provenance.json", {**provenance, "scope": "final state accompanying the media artifact"})
        provenance["separate_final_state_artifact"] = True
        save(main / "provenance.json", provenance)
        manifest(separate)
    manifest(main)


def run(architecture, loading):
    from PIL import Image

    if (architecture, loading) not in PAIRS:
        raise ValueError("Only the reviewed first two preview cases are enabled")
    source = source_record()
    save(LOGS / "numerical-source.json", source)
    folder = ROOT / "outputs/woven" / architecture / loading
    provenance = {**identity(), "parameters": {"architecture": architecture, "loading": loading, "grid": 96, "ppc": 2, "frames": 60},
                  "runtime": {"python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(),
                              "packages": {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "numpy", "pillow", "psutil")}},
                  "disk_free_bytes_before_solve": shutil.disk_usage(ROOT).free, "completed": False}
    save(LOGS / "provenance.json", provenance)
    if provenance["disk_free_bytes_before_solve"] < 8 * 1024**3:
        raise RuntimeError("Less than 8 GiB free before solving")
    solve = command("solve", [sys.executable, "-m", "examples.woven_composite", "--architecture", architecture,
                              "--loading", loading, "--grid", "96", "--ppc", "2", "--frames", "60", "--output", "outputs/woven", "--no-render"], monitor=True)
    save(folder / "memory.json", solve["memory"])
    command("review", [sys.executable, "tools/review_woven.py", str(folder)], monitor=True)
    review = json.loads((folder / "interpretation.json").read_text())
    if review.get("passed") is not True or not review.get("checks") or not all(value is True for value in review["checks"].values()):
        raise ValueError("Strict numerical review did not pass")
    review["review_source_sha256"] = source["review_source_sha256"]
    save(folder / "interpretation.json", review)
    command("render", [sys.executable, "-m", "mpm_jax.visualization", str(folder), "--gif", "--field", "family_id", "--resolution", "1200", "--matrix-opacity", "0.008"], monitor=True)
    command("video", [sys.executable, "tools/encode_video.py", str(folder)])
    command("vtk-check", [os.environ["PARAVIEW_PYTHON"], "--force-offscreen-rendering", "tools/check_vtk.py", str(folder)])
    summary = json.loads((folder / "summary.json").read_text())
    render = json.loads((folder / "render.json").read_text())
    vtk = json.loads((folder / "vtk_metadata.json").read_text())
    video = json.loads((folder / "video.json").read_text())
    if not (summary["completed"] and vtk["frames"] == render["frames_rendered"] == video["frames"] == 61
            and render["renderer"] == "ParaView" and render["field"] == "family_id"
            and render["geometry_scale"] == 1 and render["axes"] is False and render["annotations"] is False
            and render["woven_display"]["all_particles_retained"] is True
            and render["animation"] == "simulation.gif"
            and video["render_sha256"] == digest(folder / "render.json")
            and video["sha256"] == digest(folder / "simulation.mp4")):
        raise ValueError("Media provenance check failed")
    frames = sorted((folder / "frames").glob("frame_*.png"))
    if len(frames) != 61:
        raise ValueError("Expected 61 original ParaView screenshots")
    for i, path in enumerate(frames):
        if path.name != f"frame_{i:05d}.png":
            raise ValueError("Non-contiguous screenshot sequence")
        with Image.open(path) as image:
            if image.size != (1200, 1200):
                raise ValueError("Unexpected screenshot dimensions")
            image.verify()
    with Image.open(folder / "simulation.gif") as image:
        if image.n_frames != 61 or image.info.get("loop") != 0:
            raise ValueError("GIF omitted original frames or does not loop")
    save(folder / "vtk_check.json", {"passed": True, "sampled_frames": [0, 60], "particles_per_frame": vtk["particles"],
                                    "rtol": 1e-6, "atol": 1e-7, "checker_sha256": source["files"]["tools/check_vtk.py"],
                                    "scope": "actual VTK reader comparisons with original coordinates/scalars and exact static integer fields"})
    if source_record() != source:
        raise ValueError("Source changed during the run")
    provenance.update(completed=True, strict_review_passed=True, vtk_readback_passed=True,
                      review_sha256=digest(folder / "interpretation.json"),
                      solver_seconds=summary["solver_seconds"], render_version=render["version"],
                      disk_free_bytes_after_render=shutil.disk_usage(ROOT).free)
    save(LOGS / "provenance.json", provenance)
    package(folder, provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "run"))
    parser.add_argument("--architecture", choices=("plain", "triaxial"))
    parser.add_argument("--loading", choices=("transverse-bending", "in-plane-shear"))
    args = parser.parse_args()
    if os.environ.get("GITHUB_ACTIONS") != "true" or platform.system() != "Linux":
        parser.error("This helper only runs on the Linux GitHub Actions runner")
    LOGS.mkdir(parents=True, exist_ok=True)
    try:
        install() if args.action == "install" else run(args.architecture, args.loading)
    except Exception as error:
        save(LOGS / "failure.json", {"error": repr(error), "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        for path in (ROOT / "outputs/woven").glob("*/*/*"):
            if path.is_file() and path.name in ("summary.json", "status.json", "interpretation.json", "diagnostics.csv", "observables.csv") and path.stat().st_size < 500_000:
                shutil.copy2(path, LOGS / path.name)
        raise


if __name__ == "__main__":
    main()
