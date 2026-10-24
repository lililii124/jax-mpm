"""Reconcile the ceramic plan signature after finite JSON serialization."""

import argparse
import copy
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def reconcile(folder, original_review):
    folder = Path(folder)
    summary_path = folder / "summary.json"
    checkpoint_path = folder / "checkpoint.json"
    before = {name: digest(path) for name, path in (
        ("summary.json", summary_path), ("checkpoint.json", checkpoint_path))}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    review = json.loads(Path(original_review).read_text(encoding="utf-8"))
    if (checkpoint.get("frame") != 60 or review.get("accepted_saved_frame") != 60
            or review.get("passed") is not False):
        raise ValueError("Expected the original rejected full ceramic review")
    plan = {
        "configs": copy.deepcopy(summary["configs"]),
        "materials": summary["materials"],
        "frames": 60,
        "substeps": summary["steps"] // 60,
        "label": summary["case"],
        "definition": summary["definition"],
    }
    if summary["steps"] != 6120 or len(plan["configs"]) != 1:
        raise ValueError("Unexpected full ceramic protocol")
    reported = signature(plan)
    boundaries = plan["configs"][0]["boundaries"]
    if len(boundaries) != 5 or any(boundary.get("end", "missing") is not None
                                   for boundary in boundaries[:4]):
        raise ValueError("Expected exactly four finite-JSON fixed boundary ends")
    if "end" in boundaries[4]:
        raise ValueError("Unexpected end value on the rigid sphere")
    for boundary in boundaries[:4]:
        boundary["end"] = float("inf")
    corrected = signature(plan)
    checks = dict(review["checks"])
    if checks.get("full_plan_signature_matches_checkpoint") is not False:
        raise ValueError("Original reviewer did not reject the signature")
    if corrected != checkpoint["signature"] or reported == corrected:
        raise ValueError("The four serialized infinities do not explain the signature")
    checks["full_plan_signature_matches_checkpoint"] = True
    after = {name: digest(path) for name, path in (
        ("summary.json", summary_path), ("checkpoint.json", checkpoint_path))}
    if before != after:
        raise ValueError("Original evidence changed during signature reconciliation")
    return {
        "schema": "ceramic-signature-reconciliation-v1",
        "raw_files_sha256": before,
        "original_review_sha256": digest(original_review),
        "original_summary_signature": reported,
        "checkpoint_signature": checkpoint["signature"],
        "corrected_summary_signature": corrected,
        "serialized_boundary_end_count": 4,
        "remaining_failed_checks": [key for key, value in checks.items() if value is not True],
        "all_original_physical_checks_after_correction": all(value is True for value in checks.values()),
        "scope": "Read-only reviewer serialization correction; original result and rejected receipt remain unchanged.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("original_review", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = reconcile(args.folder, args.original_review)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"remaining_failed_checks": result["remaining_failed_checks"],
                      "corrected": result["all_original_physical_checks_after_correction"]}))


if __name__ == "__main__":
    main()
