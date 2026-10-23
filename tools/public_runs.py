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
    "material-engineering--jc-tension--johnson-cook": "material-engineering--jc-necking--johnson-cook",
    "material-engineering--concrete-bending--hjc": "material-engineering--concrete-bending-large--hjc",
    "material-engineering--ceramic-indentation--jh2": "material-engineering--ceramic-indentation-wide--jh2",
    "adhesion--nonadhesive-pull-off": "adhesion-visible--nonadhesive-pull-off",
    "adhesion--adhesive-pull-off": "adhesion-visible--adhesive-pull-off",
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
    canonical = str(identifier).replace("\\", "/").replace("/", "--").lower()
    source = canonical.split("--")[0]
    revised_material = canonical in (
        "material-engineering--jc-necking--johnson-cook",
        "material-engineering--concrete-bending-large--hjc",
        "material-engineering--ceramic-indentation-wide--jh2",
    )
    if source not in ("adhesion", "adhesion-visible", "visible-motion", "woven") and not revised_material:
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


def check_home_adhesion_pair(root, catalogue):
    indexed = {record["id"]: record for record in catalogue}
    roles = {"nonadhesive": "nonadhesive-pull-off", "adhesive": "adhesive-pull-off"}
    identifiers = {role: "adhesion-visible--" + name for role, name in roles.items()}
    if not any(identifier in indexed for identifier in identifiers.values()):
        return
    if not all(identifier in indexed for identifier in identifiers.values()):
        raise ValueError("Both members of the lift-hold pair must be published together")
    results = Path(root) / "docs/results"
    try:
        report = json.loads((results / "adhesion-visible-comparison.json").read_text(encoding="utf-8"))
        checks = report["checks"]
        if report.get("passed") is not True or not checks or not all(value is True for value in checks.values()) or checks.get("lift_hold_has_visible_deformation_contrast") is not True:
            raise ValueError("The lift-hold pair has no passed deformation comparison")
        if set(report["sources"]) != set(roles):
            raise ValueError("The lift-hold pair has different comparison roles")
        for role, identifier in identifiers.items():
            hashes = report["sources"][role]["files"]
            for name in ("summary.json", "observables.csv", "diagnostics.csv", "interpretation.json"):
                digest = hashlib.sha256((results / identifier / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                if hashes.get(name) != digest or indexed[identifier]["sha256"].get(name) != digest:
                    raise ValueError(f"Lift-hold comparison source changed: {identifier}/{name}")
        table = report["csv"]
        if table["filename"] != "adhesion-visible-comparison.csv":
            raise ValueError("The lift-hold response table has a different name")
        digest = hashlib.sha256((results / table["filename"]).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if table.get("lf_sha256") != digest:
            raise ValueError("The lift-hold response table has changed")
    except (OSError, KeyError, TypeError) as error:
        raise ValueError("The lift-hold pair requires its complete archived comparison") from error


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
    check_home_adhesion_pair(root, catalogue)
    replaced = {old for old, new in HOME_REPLACEMENTS.items() if new in present}
    return [
        record for record in catalogue
        if record["id"] not in replaced and record["id"] not in HIDDEN_HOME_IDS
    ]
