"""Select completed runs for the public documentation."""

from fnmatch import fnmatchcase
import hashlib
import json
from pathlib import Path


HIDDEN_HOME_IDS = {
    "refinement--grid32--wave--neo-hookean",
    "refinement--grid64--wave--linear",
    "refinement--half_dt--wave--neo-hookean",
    "suite--wave--neo-hookean",
    "visible-motion--wave-grid32--neo-hookean",
    "visible-motion--wave-grid48--neo-hookean",
    "visible-motion--wave-grid64--neo-hookean",
    "visible-motion--wave-half-dt--neo-hookean",
}


HOME_REPLACEMENTS = {
    "suite--wave--neo-hookean": "visible-motion--wave-grid48--neo-hookean",
    "refinement--grid32--wave--neo-hookean": "visible-motion--wave-grid32--neo-hookean",
    "refinement--grid64--wave--linear": "visible-motion--wave-grid64--neo-hookean",
    "refinement--half_dt--wave--neo-hookean": "visible-motion--wave-half-dt--neo-hookean",
    "suite--cantilever--neo-hookean": "visible-motion--cantilever--neo-hookean",
    "suite--shear--neo-hookean": "visible-motion--shear--neo-hookean",
    "suite--torsion--neo-hookean": "visible-motion--torsion--neo-hookean",
    "suite--surface-load--neo-hookean": "visible-motion--surface-load--neo-hookean",
}


def is_public_run(identifier):
    parts = str(identifier).replace("\\", "/").replace("/", "--").lower().split("--")
    excluded_sources = ("compression", "compression_*", "dense*", "elastic_*", "elatic_*")
    return not (
        any(fnmatchcase(parts[0], pattern) for pattern in excluded_sources)
        or parts[:2] == ["suite", "compression"]
        or parts == ["adhesion", "dugdale-array"]
    )


def is_ready_for_publication(folder, identifier, summary):
    if not is_public_run(identifier) or summary.get("completed") is False:
        return False
    source = str(identifier).replace("\\", "/").replace("/", "--").lower().split("--")[0]
    if source not in ("adhesion", "visible-motion"):
        return True
    if summary.get("completed") is not True:
        return False
    folder = Path(folder)
    try:
        review = json.loads((folder / "interpretation.json").read_text(encoding="utf-8"))
        if not isinstance(review, dict) or review.get("passed") is not True:
            return False
        checks = review.get("checks")
        sources = review.get("sources")
        if not isinstance(checks, dict) or not checks or not isinstance(sources, dict):
            return False
        if not all(value is True for value in checks.values()):
            return False
        for name in ("summary.json", "diagnostics.csv", "observables.csv"):
            content = (folder / name).read_bytes().replace(b"\r\n", b"\n")
            if sources.get(name) != hashlib.sha256(content).hexdigest():
                return False
    except (OSError, ValueError, TypeError):
        return False
    return True


def homepage_records(root, catalogue):
    """Select reviewed replacements while retaining their numerical references."""
    identifiers = [record["id"] for record in catalogue]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Run catalogue contains duplicate identifiers")
    for identifier in identifiers:
        folder = Path(root) / "docs/results" / identifier
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        if not is_ready_for_publication(folder, identifier, summary):
            raise ValueError(f"Run has no matching passed review: {identifier}")
    present = set(identifiers)
    replaced = {old for old, new in HOME_REPLACEMENTS.items() if new in present}
    return [
        record for record in catalogue
        if record["id"] not in replaced and record["id"] not in HIDDEN_HOME_IDS
    ]
