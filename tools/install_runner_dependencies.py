"""Install CPU and rendering dependencies from explicit Ubuntu repositories."""

import argparse
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time


PACKAGES = (
    "ffmpeg", "libosmesa6", "libosmesa6-dev", "libgl1", "libglu1-mesa",
    "libx11-6", "libx11-xcb1", "libxcb1", "libxcb-cursor0", "libxrender1",
    "libxi6", "libxkbcommon0", "libxkbcommon-x11-0", "libsm6", "libice6", "libxext6",
)
REPOSITORIES = (
    "deb [signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] https://archive.ubuntu.com/ubuntu/ noble main restricted universe multiverse",
    "deb [signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] https://archive.ubuntu.com/ubuntu/ noble-updates main restricted universe multiverse",
    "deb [signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] https://security.ubuntu.com/ubuntu/ noble-security main restricted universe multiverse",
)


def installation_plan(source, interpreter):
    options = ["-o", f"Dir::Etc::sourcelist={source}", "-o", "Dir::Etc::sourceparts=-",
               "-o", "Acquire::Retries=3", "-o", "Acquire::https::Timeout=45",
               "-o", "Acquire::http::Timeout=45", "-o", "Acquire::ForceIPv4=true"]
    return (
        ("apt-update", 240, ["sudo", "apt-get", *options, "update"]),
        ("apt-install", 900, ["sudo", "apt-get", *options, "install", "-y", "--no-install-recommends", *PACKAGES]),
        ("pip-install", 600, [interpreter, "-m", "pip", "--disable-pip-version-check", "--timeout", "60", "--retries", "3", "install", "-e", ".", "psutil"]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or os.environ.get("GITHUB_ACTIONS") != "true":
        parser.error("This installer is restricted to the Linux Actions runner")
    release = {}
    for line in Path("/etc/os-release").read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            release[key] = value.strip('"')
    if release.get("ID") != "ubuntu" or release.get("VERSION_ID") != "24.04":
        parser.error("This dependency plan requires Ubuntu 24.04")
    source = Path(os.environ["RUNNER_TEMP"]) / "jax-mpm-ubuntu.list"
    source.write_text("\n".join(REPOSITORIES) + "\n", encoding="utf-8")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    record = {"repositories": list(REPOSITORIES), "source_file": str(source),
              "scope": "Explicit sources for these apt commands; host source configuration is unchanged",
              "completed": False, "stages": []}
    destination = args.log_dir / "dependency-install.json"

    def save():
        destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    environment = {**os.environ, "DEBIAN_FRONTEND": "noninteractive"}
    save()
    for name, limit, command in installation_plan(source, sys.executable):
        stage = {"name": name, "command": command, "timeout_seconds": limit, "completed": False}
        record["stages"].append(stage)
        save()
        print(f"Starting {name}, limit {limit}s", flush=True)
        started = time.monotonic()
        process = subprocess.Popen(command, env=environment, start_new_session=True)
        try:
            code = process.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            stage["timed_out"] = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            stage.update(returncode=process.returncode, wall_seconds=time.monotonic() - started)
            save()
            raise RuntimeError(f"{name} exceeded its {limit}s limit") from None
        stage.update(returncode=code, wall_seconds=time.monotonic() - started, completed=code == 0)
        save()
        if code:
            raise subprocess.CalledProcessError(code, command)
    record["completed"] = True
    save()


if __name__ == "__main__":
    main()
