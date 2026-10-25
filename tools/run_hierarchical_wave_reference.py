"""Bound one fine-grid wave reference on a Linux CPU runner."""

import ctypes
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_phase_field_visible_crack import owned_session_uss, terminate_owned_group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", choices=("fine", "coarse", "hierarchical"), default="fine")
    parser.add_argument("--coarse-cells", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--transverse-padding-cells", type=int)
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--seconds-limit", type=int, default=2400)
    parser.add_argument("--output-name", default="hierarchical-wave-reference")
    args = parser.parse_args()
    if not 1 <= args.seconds_limit <= 7200 or not args.output_name.replace('-', '').isalnum():
        parser.error("Invalid time limit or output directory name")
    if sys.platform != "linux" or ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise RuntimeError("Linux owned-session guard required")
    output = ROOT / "outputs" / args.output_name
    output.mkdir(parents=True, exist_ok=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != os.environ.get("GITHUB_SHA", head):
        raise RuntimeError("Checkout differs from dispatched source")
    command = [sys.executable, "examples/hierarchical_wave.py", "--resolution", args.resolution,
               "--coarse-cells", str(args.coarse_cells), "--transverse-ppc", "1", "--output", str(output),
               "--batch-size", str(args.batch_size)]
    if args.transverse_padding_cells is not None:
        command += ["--transverse-padding-cells", str(args.transverse_padding_cells)]
    if args.stop_after_steps is not None:
        command += ["--stop-after-steps", str(args.stop_after_steps)]
    record = {"source": head, "command": command, "seconds_limit": args.seconds_limit,
              "uss_limit_bytes": int(5.5 * 2**30), "minimum_available_bytes": int(1.5 * 2**30),
              "peak_uss_bytes": 0, "guard_reason": None}
    def save():
        path = output / "resource-guard.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    started = time.monotonic()
    process = None
    code = None
    with (output / "solver.log").open("wb") as stream:
        try:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            while process.poll() is None:
                uss = owned_session_uss(process.pid, psutil)
                record["peak_uss_bytes"] = max(record["peak_uss_bytes"], uss)
                reason = ("wall time" if time.monotonic() - started > record["seconds_limit"] else
                          "USS" if uss > record["uss_limit_bytes"] else
                          "available memory" if psutil.virtual_memory().available < record["minimum_available_bytes"] else None)
                if reason:
                    record["guard_reason"] = reason
                    break
                save()
                time.sleep(.5)
            code = process.poll()
        except BaseException as error:
            record["guard_reason"] = repr(error)
            raise
        finally:
            try:
                terminate_owned_group(process)
            finally:
                record.update(returncode=process.returncode if process else None,
                              wall_seconds=time.monotonic() - started)
                save()
    if code != 0 or record["guard_reason"]:
        raise RuntimeError("Reference interrupted; completed chunks and diagnostics retained")
    review = json.loads((output / "review.json").read_text())
    expected_steps = args.stop_after_steps if args.stop_after_steps is not None else review["planned_steps"]
    assert review["steps"] == expected_steps and review["commit"] == head
    print(json.dumps({key: review[key] for key in ("steps", "duration_s", "min_J", "max_cfl_bound",
          "linear_reference_velocity_relative_L2", "relative_energy_change")}), flush=True)


if __name__ == "__main__":
    main()
