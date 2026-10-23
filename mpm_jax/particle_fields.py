"""Compact metadata and static particle labels."""

import hashlib
import re

import numpy as np


def array_descriptor(value):
    array = np.ascontiguousarray(value)
    return {
        "shape": list(np.shape(value)),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest(),
    }


def material_metadata(material):
    """Keep small parameters inline and fingerprint particle-sized arrays."""
    result = {}
    for name, value in material._asdict().items():
        array = np.asarray(value)
        result[name] = array.tolist() if array.size <= 256 else array_descriptor(array)
    return result


def checked_particle_fields(fields, count):
    reserved = {
        "x", "speed", "damage", "temperature", "failed", "time", "body",
        "contact_impulse", "displacement", "Position",
    }
    result = {}
    for name, value in sorted((fields or {}).items()):
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError("particle field names must be identifiers starting with a letter")
        if name in reserved:
            raise ValueError(f"particle field {name} is reserved for trajectory data")
        array = np.asarray(value)
        if array.shape != (count,) or array.dtype.kind not in "biuf":
            raise ValueError(f"particle field {name} must be a numeric scalar for each particle")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"particle field {name} contains nonfinite values")
        result[name] = np.ascontiguousarray(array)
    return result
