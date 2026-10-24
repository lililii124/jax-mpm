"""Secondary read-only transparent impact overview; exact original JH2 f60 raw and v2 proof."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

import finalize_ceramic_media as base

c = base.c
base.OUT = c.ROOT / '.local/ceramic-detail-media'
base.TEMPLATE = '.github/workflows/ceramic-detail-media.yml'
base.TEMPLATE_NAME = 'feature ceramic read-only detail media'
base.FEATURE_REF = 'refs/heads/work/ceramic-fragmentation-prefix'
base.EXTRA_SOURCES = ('tools/render_ceramic_detail.py',)
base.EXPECTED_SCOPE = 'complete scene'
DELIVERY = c.ROOT / '.local/ceramic-detail-delivery'
SCENE_BOUNDS = [.030, .220, .060, .180, .030, .220]
OPACITY = .08
LIMIT = 100_000_000


def render():
    base.controller()
    base.verify_raw()
    subprocess.run([sys.executable, '-m', 'mpm_jax.visualization', str(base.OUT/'result'),
        '--gif', '--field', 'damage', '--color-range', '0', '1', '--resolution', '1200',
        '--debris-style', 'highlight', '--sphere-opacity', str(OPACITY),
        '--scene-bounds', *map(str, SCENE_BOUNDS)], cwd=c.ROOT, check=True, timeout=4200)
    install = c.read(base.OUT/'paraview-install.json')
    if c.digest(install['pvpython']) != install['executable_sha256']:
        raise ValueError('ParaView executable changed')
    subprocess.run([install['pvpython'], '--force-offscreen-rendering', str(Path(__file__).resolve()),
                    'vtk'], cwd=c.ROOT, check=True, timeout=600)
    audit()


def audit():
    base.audit()
    record = c.read(base.OUT/'result/render.json')
    if (record['camera_bounds'] != SCENE_BOUNDS
            or record['render_overrides']['sphere_opacity'] != OPACITY
            or record['render_overrides']['scene_bounds'] != SCENE_BOUNDS
            or record['camera_bounds_source'] != '--scene-bounds'):
        raise ValueError('Overview camera or transparency differs')
    actual = record['actual_scene_bounds']
    if any(SCENE_BOUNDS[2*i] > actual[2*i] or SCENE_BOUNDS[2*i+1] < actual[2*i+1] for i in range(3)):
        raise ValueError('Overview bounds must contain the entire original scene')
    spheres = [g for g in record['auxiliary_geometry'] if 'sphere' in g['type']]
    if not spheres or any(g['display_opacity'] != OPACITY for g in spheres):
        raise ValueError('Actual rigid sphere display opacity differs')
    c.save(base.OUT/'detail-audit.json', dict(passed=True, secondary_detail_view=True,
        original_primary_media_run='40076682957', primary_media_untouched=True,
        full_saved_history_frames=61, scene_bounds=SCENE_BOUNDS, sphere_opacity=OPACITY,
        complete_plate_and_sphere_in_frame=True, geometry_outside_camera_may_be_cropped=False,
        sphere_physical_radius_unchanged=True, numerical_solver_rerun=False,
        note='JH2 failed-particle debris; no cohesive fragment surfaces; visual acceptance pending.'))


def package():
    audit()
    if DELIVERY.exists():
        raise FileExistsError('Detail delivery requires fresh output')
    DELIVERY.mkdir(parents=True)
    names = ['admission.json', 'independent-physical-review-v2.json', 'original-full-physical-review.json',
             'signature-reconciliation.json', 'paraview-install.json', 'renderer-requirements.txt',
             'media-audit.json', 'detail-audit.json',
             'result/render.json', 'result/vtk-full-audit.json', 'result/simulation.gif']
    metadata = [p for p in sorted((base.OUT/'sealed-cache-metadata').rglob('*')) if p.is_file()]
    names += [p.relative_to(base.OUT).as_posix() for p in metadata if p.suffix in ('.json', '.txt')]
    omitted_metadata = {p.relative_to(base.OUT).as_posix():c.file_record(p)
                        for p in metadata if p.suffix not in ('.json', '.txt')}
    names += [f'result/frames/frame_{i:05d}.png' for i in (0, 10, 20, 30, 40, 50, 60)]
    for name in names:
        target = DELIVERY/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(c.local_file(base.OUT, name), target)
    files = {n:c.file_record(DELIVERY/n) for n in names}
    total = sum(v['bytes'] for v in files.values())
    if total > LIMIT:
        raise ValueError('Detail delivery exceeds 100 MB bound')
    receipt = dict(schema='ceramic-detail-delivery-v1', files=files, bytes=total,
                   bound_bytes=LIMIT, raw_retained_in_exact_sealed_cache=True,
                   permanent_raw_backup_verified=False, solver_executed=False,
                   omitted_binary_raw_metadata=omitted_metadata,
                   complete_raw_artifact_id='12553597633')
    c.save(DELIVERY/'delivery-receipt.json', receipt)
    c.save(base.OUT/'delivery-receipt.json', receipt)
    base.verify_raw()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','install','render','vtk','package','retrieved'))
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--proof', type=Path)
    parser.add_argument('--original-review', type=Path)
    parser.add_argument('--review-sha')
    parser.add_argument('--provenance-sha')
    parser.add_argument('--retrieved', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare': base.prepare(args)
    elif args.action == 'install': base.install()
    elif args.action == 'vtk':
        from check_jc_final_media import vtk_audit
        report = vtk_audit(base.OUT/'result')
        report['schema'] = 'ceramic-detail-vtk-audit-v1'
        c.save(base.OUT/'result/vtk-full-audit.json', report)
    elif args.action == 'retrieved': base.retrieved(args.retrieved)
    else: globals()[args.action]()
