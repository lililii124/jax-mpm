"""Select completed runs for the public documentation."""

from fnmatch import fnmatchcase


def is_public_run(identifier):
    parts = str(identifier).replace("\\", "/").replace("/", "--").lower().split("--")
    excluded_sources = ("compression", "compression_*", "dense*", "elastic_*", "elatic_*")
    return not (
        any(fnmatchcase(parts[0], pattern) for pattern in excluded_sources)
        or parts[:2] == ["suite", "compression"]
    )
