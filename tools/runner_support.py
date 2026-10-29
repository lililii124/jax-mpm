"""Shared file, checkpoint and environment helpers for case runners."""


import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import zipfile
import zlib


READ_BYTES = 4 * 1024**2

PART_BYTES = 80 * 1024**2

TOTAL_BYTES = 192 * 1024**2

def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(READ_BYTES), b""):
            result.update(block)
    return result.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)

def strict_review(report):
    checks = report.get("checks", {})
    return report.get("passed") is True and bool(checks) and all(value is True for value in checks.values())

def inputs_match(folder, hashes):
    if not hashes:
        return False
    root = Path(folder).resolve()
    for name, expected in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or path == root or not path.is_file() or digest(path) != expected:
            return False
    return True

def valid_npz(path):
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            return bool(names) and all(name.endswith(".npy") for name in names) and archive.testzip() is None
    except (OSError, zipfile.BadZipFile, RuntimeError, EOFError, zlib.error):
        return False

def split_complete_file(source, destination, relative, *, part_bytes=PART_BYTES):
    """Stream one whole file into ordered pieces; no unlisted partial checkpoint."""
    source, destination = Path(source), Path(destination)
    if type(part_bytes) is not int or not 0 < part_bytes <= PART_BYTES:
        raise ValueError("Archive part size must be positive and at most 80 MiB")
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Archive part path must be relative and remain inside the delivery")
    before = source.stat()
    whole = hashlib.sha256()
    parts, consumed = [], 0
    with source.open("rb") as stream:
        while consumed < before.st_size:
            target = destination/f"{relative}.part-{len(parts):04d}"
            target.parent.mkdir(parents=True, exist_ok=True)
            part_hash, part_size = hashlib.sha256(), 0
            with target.open("wb") as output:
                while part_size < part_bytes:
                    block = stream.read(min(READ_BYTES, part_bytes-part_size))
                    if not block:
                        if consumed < before.st_size:
                            raise ValueError("Source ended early during bounded archival")
                        break
                    output.write(block)
                    whole.update(block)
                    part_hash.update(block)
                    part_size += len(block)
                    consumed += len(block)
            parts.append({"path": target.relative_to(destination).as_posix(), "bytes": part_size,
                          "sha256": part_hash.hexdigest()})
    after = source.stat()
    if consumed != before.st_size or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Source changed during bounded archival")
    return {"bytes": consumed, "sha256": whole.hexdigest(), "complete_file": True,
            "encoding": "Concatenate all ordered parts and verify whole-file SHA256 before opening the original NPZ.", "parts": parts}

def file_record(path):
    before = Path(path).stat()
    value = {'bytes': before.st_size, 'sha256': digest(path)}
    after = Path(path).stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('File changed while hashing')
    return value

def cpu_record():
    path = Path('/proc/cpuinfo')
    fields = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            key, sep, value = line.partition(':')
            if sep and key.strip() in ('vendor_id', 'model name', 'cpu family', 'model', 'stepping', 'flags'):
                fields.setdefault(key.strip(), value.strip())
    return {'machine': platform.machine(), 'logical_cpus': os.cpu_count(), 'fields': fields,
            'kernel': platform.release(), 'scope': 'Recorded per job; CPU identity is not assumed equal across hosted jobs.'}

def environment():
    versions = {}
    for distribution in importlib.metadata.distributions():
        name = re.sub(r'[-_.]+', '-', distribution.metadata['Name']).lower()
        if name in ('pip', 'setuptools', 'wheel', 'jax-mpm'):
            continue
        version = distribution.version
        if not re.fullmatch(r'[a-z0-9-]+', name) or not re.fullmatch(r'[A-Za-z0-9.!+_-]+', version):
            raise ValueError('Unsafe environment pin')
        if name in versions and versions[name] != version:
            raise ValueError('Ambiguous installed distribution')
        versions[name] = version
    if not {'jax', 'jaxlib', 'numpy', 'scipy', 'psutil', 'pillow'}.issubset(versions):
        raise ValueError('Incomplete scientific environment')
    os_release = platform.freedesktop_os_release() if hasattr(platform, 'freedesktop_os_release') and platform.system() == 'Linux' else {}
    return {'python': platform.python_version(), 'implementation': platform.python_implementation(),
            'system': platform.system(), 'machine': platform.machine(),
            'os_id': os_release.get('ID'), 'os_version_id': os_release.get('VERSION_ID'),
            'packages': dict(sorted(versions.items()))}
