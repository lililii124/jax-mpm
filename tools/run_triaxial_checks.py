"""Run a large fibre memory probe or a small coupled dynamics check on Actions."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_engineering_preview as engine

LOGS = ROOT / ".local/triaxial-ci"
DELIVERY = ROOT / ".local/triaxial-delivery"


def source_record():
    result = engine.shared.source_record()
    result.pop("review_source_sha256", None)
    result["runner_source_sha256"] = result["files"]["tools/run_triaxial_checks.py"]
    result["scope"] = "Execution-source identity; this is not an independent failure-envelope review."
    return result


def run(preset, theta, law):
    source = source_record()
    engine.save(LOGS / "numerical-source.json", source)
    output = "outputs/triaxial-checks"
    arguments = [sys.executable, "-m", "examples.triaxial_failure", "--theta", str(theta),
                 "--law", law, "--output", output]
    arguments += ["--probe-steps", "8"] if preset == "probe" else ["--smoke", "--frames", "12"]
    result = engine.command("solve", arguments, monitor=True)
    summaries = list((ROOT / output).rglob("summary.json"))
    if len(summaries) != 1:
        raise ValueError("Expected one independently completed check")
    summary = json.loads(summaries[0].read_text())
    status = json.loads((summaries[0].parent / "status.json").read_text())
    if summary.get("production_eligible") is not False:
        raise ValueError("A probe or smoke test must not be labelled a production envelope")
    if preset == "probe":
        if not (summary.get("probe_complete") is True and summary.get("completed") is False
                and status.get("probe_complete") is True):
            raise ValueError("The memory probe did not complete its requested steps")
    elif not (summary.get("completed") is True and status.get("completed") is True):
        raise ValueError("The coupled dynamics smoke check did not complete")
    if source_record() != source:
        raise ValueError("Execution source changed during the check")
    engine.save(LOGS / "check-result.json", {**engine.identity(), "preset": preset, "theta": theta,
        "law": law, "completed": True, "production_eligible": False, "memory": result["memory"],
        "summary_sha256": engine.digest(summaries[0]), "runner_source_sha256": source["runner_source_sha256"]})


def package():
    DELIVERY.mkdir(parents=True, exist_ok=True)
    limit, included, omitted = 100 * 1024**2, {}, {}
    candidates = [path for parent in (LOGS, ROOT / "outputs/triaxial-checks") for path in parent.rglob("*")
                  if path.is_file() and path.suffix in (".json", ".csv")]
    for path in sorted(candidates):
        relative = path.relative_to(ROOT)
        target = DELIVERY / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        included[relative.as_posix()] = {"sha256": engine.digest(path), "bytes": path.stat().st_size}
    used = sum(item["bytes"] for item in included.values())
    for path in sorted((ROOT / "outputs/triaxial-checks").rglob("*.npz")):
        relative = path.relative_to(ROOT)
        record = {"sha256": engine.digest(path), "bytes": path.stat().st_size}
        if used + record["bytes"] + 128 * 1024 > limit:
            omitted[relative.as_posix()] = {**record, "reason": "The complete file exceeds the remaining bounded check artifact budget; not represented as a restored checkpoint."}
            continue
        target = DELIVERY / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        included[relative.as_posix()] = record
        used += record["bytes"]
    engine.save(DELIVERY / "delivery-manifest.json", {**engine.identity(), "production_eligible": False,
        "scope": "Component integration and memory checks only; no accepted angular failure envelope or public animation.",
        "files": included, "omitted_npz": omitted,
        "trajectory_scope": "Memory-mapped trajectory duplicates remain on the runner; compact check records and any fully included NPZ files are listed individually."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("probe", "smoke"), required=True)
    parser.add_argument("--theta", type=int, choices=range(0, 31, 5), default=0)
    parser.add_argument("--law", choices=("hertz", "jkr"), default="hertz")
    args = parser.parse_args()
    if os.environ.get("GITHUB_ACTIONS") != "true" or platform.system() != "Linux":
        parser.error("This check requires the independent Linux Actions runner")
    engine.LOGS = LOGS
    LOGS.mkdir(parents=True, exist_ok=True)
    try:
        run(args.preset, args.theta, args.law)
    except BaseException as error:
        engine.save(LOGS / "failure.json", {"completed": False, "error_type": type(error).__name__, "error": str(error)})
        raise
    finally:
        package()


if __name__ == "__main__":
    main()
