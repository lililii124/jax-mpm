"""Read-only 61-state cross section of the reconciled original JH2 run."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import finalize_ceramic_media as base

c = base.c
base.OUT = c.ROOT/'.local/ceramic-section-media'
base.TEMPLATE = '.github/workflows/ceramic-section-media.yml'
base.TEMPLATE_NAME = 'feature ceramic read-only section media'
base.FEATURE_REF = 'refs/heads/work/ceramic-fragmentation-prefix'
base.EXTRA_SOURCES = ('tools/render_ceramic_section.py', 'tools/paraview_ceramic_section.py')
DELIVERY = c.ROOT/'.local/ceramic-section-delivery'
LIMIT = 100_000_000


def render():
    base.controller()
    base.verify_raw()
    install = c.read(base.OUT/'paraview-install.json')
    if c.digest(install['pvpython']) != install['executable_sha256']:
        raise ValueError('ParaView executable changed')
    subprocess.run([install['pvpython'], '--force-offscreen-rendering',
        str(c.ROOT/'tools/paraview_ceramic_section.py'), str(base.OUT/'result')],
        cwd=c.ROOT, check=True, timeout=4200)
    from PIL import Image
    frames = []
    for i in range(61):
        with Image.open(base.OUT/'result/frames'/f'frame_{i:05d}.png') as image:
            frames.append(image.convert('RGB'))
    frames[0].save(base.OUT/'result/simulation.gif', save_all=True,
                   append_images=frames[1:], duration=60, loop=0, optimize=False, disposal=2)
    audit()


def audit():
    import numpy as np
    from PIL import Image
    base.controller()
    folder = base.OUT/'result'
    r = c.read(folder/'section-render.json')
    with np.load(folder/'frames_data/frame_00000.npz', allow_pickle=False) as f:
        ids = np.flatnonzero(np.abs(f['x'][:,0]-.125) <= .00125+1e-12)
    if not (r['renderer']=='ParaView' and r['frames_rendered']==61 and
        r['axes'] is False and r['annotations'] is False and r['geometry_scale']==1
        and r['resolution']==[1200,1200] and r['field']=='damage' and r['color_range']==[0,1]
        and r['selected_particle_ids']==ids.tolist() and len(ids)>0
        and r['reference_plane_x_m']==.125 and r['reference_half_thickness_m']==.00125
        and r['view_scope']=='central material cross section' and len(r['displayed_counts'])==61):
        raise ValueError('Cross section provenance differs')
    expected = [f'frame_{i:05d}.png' for i in range(61)]
    if sorted(p.name for p in (folder/'frames').glob('*.png')) != expected:
        raise ValueError('Exactly 61 PNG frames required')
    for i in range(61):
        with np.load(folder/'frames_data'/f'frame_{i:05d}.npz', allow_pickle=False) as f:
            failed = int(np.count_nonzero(f['failed'][ids]))
            if r['displayed_counts'][i] != dict(active_particles=len(ids)-failed, debris_particles=failed):
                raise ValueError('Cross section lost selected active or failed particles')
        with Image.open(folder/'frames'/expected[i]) as png:
            png.load()
            if png.size!=(1200,1200) or png.convert('RGB').getextrema()==((255,255),)*3:
                raise ValueError('Invalid section PNG')
    with Image.open(folder/'simulation.gif') as gif:
        if not 2<=gif.n_frames<=61 or gif.size!=(1200,1200):
            raise ValueError('GIF must show changing saved states')
        duration_ms = 0
        for i in range(61):
            if i>=gif.n_frames:
                break
            gif.seek(i)
            gif.load()
            duration = gif.info.get('duration')
            if not isinstance(duration,int) or duration<60 or duration%60:
                raise ValueError('Unexpected GIF timing')
            duration_ms += duration
        if duration_ms!=61*60:
            raise ValueError('GIF duration differs from all 61 saved states')
    for name, record in c.read(base.OUT/'admission.json')['raw_files'].items():
        if name.startswith('result/') and c.file_record(base.OUT/name)!=record:
            raise ValueError('Original numerical bytes changed: '+name)
    base.verify_raw()
    c.save(base.OUT/'section-audit.json', dict(passed=True, saved_states=61,
        gif_encoded_frames=gif.n_frames, gif_duration_ms=duration_ms,
        material_selection_from_reference_x=True, section_plane_x_m=.125,
        slab_total_thickness_m=.0025, no_axes_or_text=True, raw_unchanged=True,
        solver_executed=False, production_eligible=False, public_visual_review_passed=False))


def package():
    audit()
    if DELIVERY.exists():
        raise FileExistsError('Section delivery requires fresh directory')
    names = ['admission.json', 'independent-physical-review-v2.json',
        'original-full-physical-review.json', 'signature-reconciliation.json',
        'paraview-install.json', 'renderer-requirements.txt', 'section-audit.json',
        'result/section-render.json', 'result/section-scene.pvsm', 'result/simulation.gif']
    names += [f'result/frames/frame_{i:05d}.png' for i in (0, 10, 20, 30, 40, 50, 60)]
    for name in names:
        target = DELIVERY/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(c.local_file(base.OUT,name),target)
    files = {n:c.file_record(DELIVERY/n) for n in names}
    total = sum(v['bytes'] for v in files.values())
    if total>LIMIT:
        raise ValueError('Section delivery exceeds 100 MB')
    receipt = dict(schema='ceramic-section-delivery-v1',files=files,bytes=total,
        bound_bytes=LIMIT, raw_retained_in_exact_sealed_cache=True,
        permanent_raw_backup_verified=False, solver_executed=False)
    c.save(DELIVERY/'delivery-receipt.json',receipt)
    c.save(base.OUT/'delivery-receipt.json',receipt)
    base.verify_raw()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','install','render','audit','package','retrieved'))
    for name in ('evidence','proof','original-review','retrieved'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--review-sha')
    parser.add_argument('--provenance-sha')
    args = parser.parse_args()
    if args.action=='prepare': base.prepare(args)
    elif args.action=='install': base.install()
    elif args.action=='retrieved': base.retrieved(args.retrieved)
    else: globals()[args.action]()
