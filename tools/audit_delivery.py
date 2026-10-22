"""Audit current-source Final Check records without running a legacy solver."""

import argparse
import hashlib
import json
from pathlib import Path


def audit(root):
    records = []
    for folder in sorted(root.glob("MAT[0-9][0-9][0-9]")):
        manifest = folder / "manifest.json"
        if not manifest.exists():
            continue
        data = json.loads(manifest.read_text(encoding="utf-8-sig"))
        status = data.get("complex_final_check_status", "")
        if "PASS" not in status or any(
            s in status for s in ("PENDING", "NOT_VALIDATED", "HISTORICAL")
        ):
            continue
        source = folder / data.get("source_file", "")
        if not source.is_file():
            continue
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        evidence = data.get("strict_check_evidence") or {}
        acceptance_sha = evidence.get("source_sha256")
        acceptance_file_valid = True
        if "path" in evidence:
            record = folder / evidence["path"]
            if record.is_file():
                acceptance_file_valid = hashlib.sha256(
                    record.read_bytes()
                ).hexdigest() == evidence.get("sha256")
                accepted = json.loads(record.read_text(encoding="utf-8-sig"))
                acceptance_sha = accepted.get("source", {}).get("sha256")
            else:
                acceptance_file_valid = False
        records.append(
            {
                "material": folder.name,
                "name": data.get("catalog_name"),
                "status": status,
                "source_sha256": actual,
                "manifest_source_matches": actual == data.get("source_sha256"),
                "acceptance_source_matches": actual == acceptance_sha,
                "acceptance_file_matches": acceptance_file_valid,
            }
        )
    return {
        "delivery": root.name,
        "materials": records,
        "eligible": [
            r["material"]
            for r in records
            if r["manifest_source_matches"]
            and r["acceptance_source_matches"]
            and r["acceptance_file_matches"]
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("delivery", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/source_audit.json"))
    args = parser.parse_args()
    result = audit(args.delivery)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Current-source Final Check records:", ", ".join(result["eligible"]))
