#!/usr/bin/env python3
"""Build a hash-bound continuation capsule from one accepted C2 run window.

This is packaging and admission validation only. It does not alter a checkpoint,
run solver code, or make a support split; it carries forward the exact accepted
state and source-generated receipts that the runner validates on resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np


REQUIRED_CHECKPOINT_FIELDS = {
    'displacement', 'velocity', 'F', 'tau', 'ep', 'cold', 'fracture',
    'damage', 'internal', 'time_s', 'step', 'initial_P', 'initial_L',
    'initial_K', 'viscous_loss', 'contact_work',
}
MATERIAL_FIELDS = ('F', 'tau', 'ep', 'cold', 'fracture', 'damage', 'internal')
LEDGER_FIELDS = ('time_s', 'step', 'initial_P', 'initial_L', 'initial_K',
                 'viscous_loss', 'contact_work')
RECEIPTS = (
    ('prior_contact_history_sha256', 'contact-history.jsonl', True),
    ('prior_first_event_sha256', 'first-domain-intersection.json', True),
    ('prior_e0_activation_sha256', 'e0-activation.json', True),
    ('prior_damage_history_sha256', 'damage-history.jsonl', False),
    ('prior_first_failed_sha256', 'first-fully-failed-point.json', False),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_file(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def source_hashes(source_root: Path, expected: dict) -> None:
    if not isinstance(expected, dict) or not expected:
        raise ValueError('run.json has no nonempty source_sha256 map')
    actual = {}
    root = source_root.resolve(strict=True)
    for name in sorted(expected):
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError(f'Unsafe source path in run.json: {name!r}')
        source = (root / rel).resolve(strict=True)
        if root not in source.parents:
            raise ValueError(f'Source path escapes --source-root: {name!r}')
        actual[name] = sha256(source)
    if actual != expected:
        mismatch = [name for name in expected if actual.get(name) != expected[name]]
        raise ValueError('Source tree differs from the accepted run: ' + ', '.join(mismatch[:8]))


def read_checkpoint(path: Path):
    archive = np.load(path, allow_pickle=False)
    fields = set(archive.files)
    if not REQUIRED_CHECKPOINT_FIELDS.issubset(fields):
        archive.close()
        missing = sorted(REQUIRED_CHECKPOINT_FIELDS - fields)
        raise ValueError(f'Accepted checkpoint is incomplete: {missing}')
    return archive


def scalar(archive, name):
    value = np.asarray(archive[name])
    if value.size != 1:
        raise ValueError(f'Checkpoint {name} is not scalar')
    return value.item()


def truncate_jsonl(source: Path, target: Path, step: int) -> tuple[int, int]:
    kept, dropped = [], 0
    for line_number, line in enumerate(source.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'Malformed {source.name} line {line_number}') from exc
        row_step = row.get('step', 0)
        if not isinstance(row_step, int):
            raise ValueError(f'{source.name} line {line_number} has noninteger step')
        if row_step <= step:
            kept.append(json.dumps(row, allow_nan=False, separators=(',', ':')))
        else:
            dropped += 1
    target.write_text(''.join(row + '\n' for row in kept), encoding='utf-8', newline='\n')
    return len(kept), dropped


def verify_support_split(result: Path, accepted: Path, archive, step: int):
    if 'metadata_json' not in archive.files:
        return None
    metadata = json.loads(str(scalar(archive, 'metadata_json')))
    if metadata.get('schema') != 'tl-fracture-support-checkpoint-v1':
        raise ValueError('Accepted checkpoint has an unrecognized support adapter schema')
    parent = result / f'pre-split-step{step}.npz'
    audit = result / f'support-split-step{step}.json'
    if not parent.is_file() or not audit.is_file():
        raise ValueError('Support-split checkpoint lacks its exact parent or split audit')
    parent_hash = sha256(parent)
    child_hash = sha256(accepted)
    receipt = json_file(audit)
    if receipt.get('schema') != 'c2-tl-component-support-event-v1':
        raise ValueError('Support-split audit schema is not recognized')
    manifest_hash = metadata.get('source_manifest_sha256')
    if (receipt.get('status') != 'accepted-zero-time-conservative-support-split'
            or receipt.get('step') != step
            or receipt.get('time_s') != float(scalar(archive, 'time_s'))
            or receipt.get('parent_checkpoint') != parent.name
            or receipt.get('parent_checkpoint_sha256') != parent_hash
            or receipt.get('support_checkpoint') != 'accepted.npz'
            or receipt.get('support_checkpoint_sha256') != child_hash
            or receipt.get('source_manifest_sha256') != manifest_hash
            or receipt.get('parent_source_manifest_sha256') != manifest_hash
            or receipt.get('support_metadata') != metadata.get('support_hash')
            or receipt.get('topology_hash') != metadata.get('topology_hash')
            or metadata.get('parent_checkpoint_sha256') != parent_hash):
        raise ValueError('Support-split audit, parent, and accepted state are not mutually bound')
    with read_checkpoint(parent) as raw:
        for name in MATERIAL_FIELDS + LEDGER_FIELDS:
            if not np.array_equal(archive[name], raw[name]):
                raise ValueError(f'Support adapter changed accepted particle/ledger field {name}')
    return parent, audit, parent_hash, receipt


def prepare(args) -> dict:
    run_dir = args.run_dir.resolve(strict=True)
    result = run_dir if (run_dir / 'accepted.npz').is_file() else run_dir / 'result'
    accepted = result / 'accepted.npz'
    run_path = result / 'run.json'
    if not accepted.is_file() or not run_path.is_file():
        raise FileNotFoundError('--run-dir must name a run output or its result directory')
    run = json_file(run_path)
    hashes = run.get('source_sha256')
    source_hashes(args.source_root, hashes)
    accepted_hash = sha256(accepted)
    with read_checkpoint(accepted) as archive:
        step = int(scalar(archive, 'step'))
        time_s = float(scalar(archive, 'time_s'))
        if step < 1 or not math.isfinite(time_s) or time_s <= 0:
            raise ValueError('Accepted checkpoint has an invalid endpoint')
        if int(run.get('steps', -1)) != step or not math.isclose(
                float(run.get('time_s', math.nan)), time_s, rel_tol=0., abs_tol=1e-18):
            raise ValueError('run.json endpoint does not match accepted.npz')
        if run.get('status') not in ('timeout-preserved', 'completed-short-window',
                                      'stopped-after-accepted-checkpoint',
                                      'read-only-checkpoint-audit-complete'):
            raise ValueError(f'Run is not at an admitted bounded checkpoint: {run.get("status")!r}')
        requested_end = run.get('requested_end_time_s')
        if requested_end is not None and time_s >= float(requested_end):
            raise ValueError('Accepted run already reached its requested endpoint; no continuation capsule is needed')
        support = verify_support_split(result, accepted, archive, step)
        support_map_hash = (str(scalar(archive, 'support_map_hash'))
                            if 'support_map_hash' in archive.files else None)
        particles = int(len(archive['damage']))

    previous = json_file(args.previous_bridge) if args.previous_bridge else {}
    if previous and previous.get('continuation_source_sha256') != hashes:
        raise ValueError('Previous bridge is bound to a different executing source tree')
    bridge = dict(previous)
    bridge.update(schema='metal-tl-source-bridge-v1', case_geometry_particles=particles,
                  continuation_source_sha256=hashes, checkpoint_sha256=accepted_hash,
                  support_checkpoint_sha256=accepted_hash, support_map_hash=support_map_hash,
                  support_parent_checkpoint_sha256=None, support_audit_sha256=None)
    trim = {}

    output = args.output.resolve()
    if output == result or result in output.parents:
        raise ValueError('Keep the continuation capsule outside the accepted run directory')
    if output.exists():
        raise FileExistsError(f'Refusing to replace existing capsule directory: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + '.tmp-', dir=output.parent))
    try:
        shutil.copyfile(accepted, staging / 'resume-checkpoint.npz')
        if support:
            parent, audit, parent_hash, _ = support
            shutil.copyfile(parent, staging / 'resume-parent-checkpoint.npz')
            shutil.copyfile(audit, staging / 'resume-support-audit.json')
            bridge['support_parent_checkpoint_sha256'] = parent_hash
            bridge['support_audit_sha256'] = sha256(staging / 'resume-support-audit.json')

        for key, filename, required in RECEIPTS:
            src = result / filename
            dst = staging / ('resume-' + filename)
            if not src.is_file():
                if required:
                    raise FileNotFoundError(f'Required source receipt is missing: {src}')
                bridge[key] = None
                continue
            if filename.endswith('.jsonl'):
                kept, dropped = truncate_jsonl(src, dst, step)
                trim[filename] = dropped
            else:
                shutil.copyfile(src, dst)
                json_file(dst)  # Fail closed on malformed JSON receipts.
            bridge[key] = sha256(dst)

        bridge['newly_active_e0_checked'] = bool(run.get('newly_active_e0_checked', False))
        bridge['prior_source_sha256'] = hashes
        bridge['prior_run_json_sha256'] = sha256(run_path)
        bridge['prior_metrics'] = {key: run.get(key, 0.) for key in
                                   ('max_linear_relative', 'max_angular_relative',
                                    'max_energy_error_J')}
        controller_path = result / 'numerical-controller.json'
        controller = json_file(controller_path) if controller_path.is_file() else {}
        bridge['numerical_dt_cap_s'] = controller.get('dt_cap_s') or run.get('dt_last_s')
        if not bridge['numerical_dt_cap_s'] or not math.isfinite(float(bridge['numerical_dt_cap_s'])):
            raise ValueError('No finite accepted numerical timestep ceiling in run/controller receipts')
        bridge.update(prior_step=step, prior_time_s=time_s,
                      bridge_reason=('Exact accepted checkpoint continuation after bounded wall window. '
                          'Same geometry, physical law and admission gates. This corrected geometry history '
                          'starts at virgin; same qualified quarter CFL and unchanged sources throughout. '
                          'Uncommitted diagnostic rows beyond checkpoint omitted.'),
                      uncommitted_history_rows_omitted=trim)
        (staging / 'resume-bridge.json').write_text(
            json.dumps(bridge, indent=2, allow_nan=False),
            encoding='utf-8', newline='\n')
        manifest = dict(schema='c2-domain-impact-resume-capsule-v1',
            accepted_run_status=run['status'], accepted_step=step, accepted_time_s=time_s,
            accepted_checkpoint_sha256=accepted_hash, continuation_source_sha256=hashes,
            support_split=bool(support), receipts=trim,
            files={p.name: sha256(p) for p in sorted(staging.iterdir()) if p.is_file()},
            contract='The public runner revalidates every bridge hash and, for support splits, the parent/audit/state duplication.')
        (staging / 'capsule-manifest.json').write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n',
            encoding='utf-8', newline='\n')
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main():
    script_path = Path(__file__).resolve()
    default_source = (script_path.parents[2] / 'examples' / 'domain_impact' / 'vendor'
                     if len(script_path.parents) > 2 else None)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True,
                        help='Run output directory or its result/ directory')
    parser.add_argument('--source-root', type=Path, default=default_source,
                        help='Exact executing source tree; must match run.json source_sha256')
    parser.add_argument('--previous-bridge', type=Path,
                        help='Prior bridge from this same source chain; preserves ancestry fields')
    parser.add_argument('--output', type=Path, required=True,
                        help='New, nonexistent directory for the complete resume capsule')
    args = parser.parse_args()
    if args.source_root is None:
        parser.error('--source-root is required when the tool is run outside the repository tree')
    manifest = prepare(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
