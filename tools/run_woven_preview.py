"""Run reviewed woven load cases on GitHub's CPU runners."""

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
sys.path.insert(0, str(ROOT))
LOGS = ROOT / ".local/woven-ci"
DELIVERY = ROOT / ".local/woven-delivery"
LIMIT = 100 * 1024**2
ARCHITECTURES = ("plain", "triaxial", "twill-2x2")
LOADINGS = ("warp-tension", "weft-tension", "in-plane-shear", "transverse-compression", "transverse-bending", "hemispherical-compression")
INITIAL_PAIRS = {("plain", "transverse-bending"), ("triaxial", "in-plane-shear")}
PAIRS = {(architecture, loading) for architecture in ARCHITECTURES for loading in LOADINGS}
COARSE_PAIRS = INITIAL_PAIRS | {("plain", "warp-tension"), ("plain", "weft-tension"), ("twill-2x2", "transverse-compression")}
BATCHES = ("initial", "shear", "remaining", "hemisphere")
PV_VERSION = "6.2.0"
PV_NAME = f"ParaView-{PV_VERSION}-MPI-Linux-Python3.12-x86_64"
PV_URL = f"https://www.paraview.org/files/v6.2/{PV_NAME}.tar.gz"


def case_matrix(preset, batch):
    if preset not in ("formal", "coarse") or batch not in BATCHES:
        raise ValueError("Unknown woven run preset or batch")
    if preset == "coarse":
        if batch != "initial":
            raise ValueError("Only the original five-case coarse preview batch is enabled")
        pairs = COARSE_PAIRS
    elif batch == "initial":
        pairs = INITIAL_PAIRS
    elif batch == "shear":
        pairs = {("triaxial", "in-plane-shear")}
    elif batch == "remaining":
        pairs = {pair for pair in PAIRS - INITIAL_PAIRS if pair[1] != "hemispherical-compression"}
    else:
        pairs = {pair for pair in PAIRS if pair[1] == "hemispherical-compression"}
    return [{"architecture": architecture, "loading": loading} for architecture, loading in sorted(pairs)]


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


def command(name, arguments, *, monitor=False, accepted_codes=(0,)):
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
    if code not in accepted_codes:
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
    provenance = folder / "provenance.json"
    if provenance.is_file():
        metadata = json.loads(provenance.read_text())
        record.update({key: metadata[key] for key in ("preset", "strict_review_passed", "preview_eligible") if key in metadata})
    if record["total_bytes"] > LIMIT:
        raise ValueError(f"Artifact exceeds 100 MiB: {folder}")
    save(folder / "delivery-manifest.json", record)


def prepare_coarse_preview(folder, architecture, loading):
    from tools.public_runs import is_woven_preview_ready

    original = (folder / "interpretation.json").read_bytes()
    review = json.loads(original)
    summary = json.loads((folder / "summary.json").read_text())
    allowed = {"at_least_one_hundred_thousand_particles", "grid_resolves_yarn_height"}
    if loading == "transverse-compression":
        allowed.add("required_moving_samples_exist")
    checks = review.get("checks", {})
    failed = {name for name, passed in checks.items() if passed is False}
    required = ("positive_final_stored_elastic_energy", "final_piola_section_forces_match", "final_reference_section_moment_matches")
    if (review.get("passed") is not False or not failed or not failed <= allowed
            or any(type(value) is not bool for value in checks.values())
            or not all(checks.get(name) is True for name in required)):
        raise ValueError("Coarse preview has an unapproved physical or integrity failure")
    limitations = ["Coarse grid and particle sampling do not satisfy the production particle-count and yarn-height resolution requirements; no convergence claim."]
    if "required_moving_samples_exist" in failed:
        limitations.append("No fully supported interior moving-platen samples exist on this coarse grid; exact particle boundary tracking is not established.")
    (folder / "preview-review.json").write_bytes(original)
    names = ("summary.json", "diagnostics.csv", "observables.csv", "interpretation.json", "preview-review.json")
    eligibility = {"kind": "coarse-woven-preview-v1", "eligible": True,
                   "review_file": "preview-review.json", "source_review_passed": False,
                   "allowed_failed_checks": sorted(allowed), "observed_failed_checks": sorted(failed),
                   "limitations": limitations,
                   "sources": {name: hashlib.sha256((folder / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for name in names}}
    save(folder / "preview-eligibility.json", eligibility)
    if not is_woven_preview_ready(folder, f"woven-preview--{architecture}--{loading}", summary):
        raise ValueError("The public coarse-preview eligibility gate rejected this run")
    if (folder / "interpretation.json").read_bytes() != original:
        raise ValueError("Original numerical review was modified")
    return eligibility


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
    if provenance.get("preset") == "coarse":
        for name in ("preview-review.json", "preview-eligibility.json", "coarse-preview.json"):
            shutil.copy2(folder / name, result / name)
    shutil.copytree(folder / "frames", result / "frames")
    with np.load(folder / "trajectory.npz") as archive:
        np.savez_compressed(result / "trajectory_index.npz", time=archive["time"], body=archive["body"])
    provenance["trajectory_index"] = {"file": "trajectory_index.npz", "fields": ["time", "body"], "source": "trajectory.npz", "source_sha256": digest(folder / "trajectory.npz"), "source_bytes": (folder / "trajectory.npz").stat().st_size}
    provenance["omitted_large_data"] = "Unpackaged trajectory archives, memory-mapped trajectories and VTK frames remain on the runner; numerical review and actual VTK first/last readback ran before packaging. The PVD and scene retain their original references, so they require the omitted VTK series to reopen."
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
    if provenance.get("preset") == "coarse":
        extra = sorted(folder.glob("trajectory*.np*"))
        remaining = LIMIT - sum(path.stat().st_size for path in main.rglob("*") if path.is_file()) - 32768
        included = sum(path.stat().st_size for path in extra) <= remaining
        if included:
            for path in extra:
                shutil.copy2(path, result / path.name)
        provenance["coarse_trajectory_bundle"] = {
            "included": included,
            "files": {path.name: {"bytes": path.stat().st_size, "sha256": digest(path)} for path in extra},
            "scope": "Original complete archive and saved arrays; included together when the media artifact remains below 100 MiB",
        }
        save(main / "provenance.json", provenance)
    manifest(main)


def run(architecture, loading, preset="formal"):
    from PIL import Image

    if (architecture, loading) not in (COARSE_PAIRS if preset == "coarse" else PAIRS):
        raise ValueError("This architecture/loading pair is not enabled for the selected preset")
    grid, ppc = (48, 1) if preset == "coarse" else ((192, 1) if loading == "hemispherical-compression" else (96, 2))
    output_root = "outputs/woven-smoke" if preset == "coarse" else "outputs/woven"
    source = source_record()
    save(LOGS / "numerical-source.json", source)
    folder = ROOT / output_root / architecture / loading
    provenance = {**identity(), "preset": preset, "parameters": {"architecture": architecture, "loading": loading, "grid": grid, "ppc": ppc, "frames": 60},
                  "runtime": {"python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(),
                              "packages": {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "numpy", "pillow", "psutil")}},
                  "disk_free_bytes_before_solve": shutil.disk_usage(ROOT).free, "completed": False}
    save(LOGS / "provenance.json", provenance)
    if provenance["disk_free_bytes_before_solve"] < 8 * 1024**3:
        raise RuntimeError("Less than 8 GiB free before solving")
    solve = command("solve", [sys.executable, "-m", "examples.woven_composite", "--architecture", architecture,
                              "--loading", loading, "--grid", str(grid), "--ppc", str(ppc), "--frames", "60", "--output", output_root, "--no-render"], monitor=True)
    save(folder / "memory.json", solve["memory"])
    command("review", [sys.executable, "tools/review_woven.py", str(folder)], monitor=True,
            accepted_codes=(0, 1) if preset == "coarse" else (0,))
    review = json.loads((folder / "interpretation.json").read_text())
    if preset == "coarse":
        eligibility = prepare_coarse_preview(folder, architecture, loading)
    else:
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
    provenance.update(completed=True, strict_review_passed=review["passed"],
                      preview_eligible=preset == "coarse", vtk_readback_passed=True,
                      review_sha256=digest(folder / "interpretation.json"),
                      solver_seconds=summary["solver_seconds"], render_version=render["version"],
                      disk_free_bytes_after_render=shutil.disk_usage(ROOT).free)
    if preset == "coarse":
        save(folder / "coarse-preview.json", {
            "scope": "Full-duration coarse preview; not production-resolution validation",
            "production_eligible": False, "original_review_passed": False, "current_review_passed": False,
            "coarse_preview_authorized": True, "preview_eligible": True,
            "grid": grid, "particles_per_cell_per_axis": ppc, "frames": 61,
            "particles_per_frame": vtk["particles"],
            "failed_production_checks_preserved": eligibility["observed_failed_checks"],
            "limitations": eligibility["limitations"], "review_tool_sha256": source["review_source_sha256"],
            "review_sha256": digest(folder / "preview-review.json"),
            "render_sha256": digest(folder / "render.json"), "gif_sha256": digest(folder / "simulation.gif"),
            "video_sha256": digest(folder / "simulation.mp4"),
        })
    save(LOGS / "provenance.json", provenance)
    package(folder, provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "run"))
    parser.add_argument("--preset", choices=("formal", "coarse"), default="formal")
    parser.add_argument("--architecture", choices=ARCHITECTURES)
    parser.add_argument("--loading", choices=LOADINGS)
    args = parser.parse_args()
    if os.environ.get("GITHUB_ACTIONS") != "true" or platform.system() != "Linux":
        parser.error("This helper only runs on the Linux GitHub Actions runner")
    LOGS.mkdir(parents=True, exist_ok=True)
    try:
        install() if args.action == "install" else run(args.architecture, args.loading, args.preset)
    except Exception as error:
        save(LOGS / "failure.json", {"error": repr(error), "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        output_root = "outputs/woven-smoke" if args.preset == "coarse" else "outputs/woven"
        for path in (ROOT / output_root).glob("*/*/*"):
            if path.is_file() and path.name in ("summary.json", "status.json", "interpretation.json", "diagnostics.csv", "observables.csv") and path.stat().st_size < 500_000:
                shutil.copy2(path, LOGS / path.name)
        raise


if __name__ == "__main__":
    main()
