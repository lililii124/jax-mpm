"""Render only completely audited actual HJC61 arrays with the old camera."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def render(audit, output):
    report = json.loads((audit/'independent-hjc60-review.json').read_text())
    assert report['passed'] and report['frames'] == 61 and report['steps'] == 10500
    assert report['child_archive']['official_digest_matched'] and report['child_archive']['all_member_crc_checked']
    source = audit/'child/restored'
    reference_path = ROOT/'docs/results/material-engineering--concrete-bending-symmetric--hjc/render.json'
    reference = json.loads(reference_path.read_text())
    output.mkdir(parents=True,exist_ok=False)
    summary = (source/'raw/summary.json').read_bytes()
    (output/'summary.json').write_bytes(summary)
    initial_hash = hashlib.sha256((source/'raw/checkpoint.json').read_bytes()).hexdigest()
    # Load the pure NumPy exporter directly, avoiding package __init__/JAX.
    spec = importlib.util.spec_from_file_location('pure_vtk_export',ROOT/'mpm_jax/export.py')
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    positions = np.load(source/'raw/trajectory_x.npy',mmap_mode='r',allow_pickle=False)
    fields = {name:np.load(source/f'raw/trajectory_{name}.npy',mmap_mode='r',allow_pickle=False)
              for name in ('speed','damage','temperature')}
    fields['body'] = np.zeros(121104,dtype=np.int32)
    plan = json.loads((source/'plan.json').read_text())
    spacing = plan['definition']['grid_spacing_m']/plan['definition']['particles_per_cell_per_axis']
    times = np.arange(61)*25e-6
    exporter.export_series(positions,fields,times,output,spacing=spacing,dim=3)
    command = [os.environ['PARAVIEW_PYTHON'],'--force-offscreen-rendering',str(ROOT/'mpm_jax/paraview_scene.py'),
        str(output),'--field','damage','--resolution','400','--color-range','0','1',
        '--focus-bounds',*map(str,reference['camera_bounds'])]
    env = dict(os.environ)
    env.pop('PYTHONPATH',None)
    env.pop('PYTHONHOME',None)
    with (output/'render.log').open('w') as log:
        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    record = json.loads((output/'render.json').read_text())
    assert record['frames_rendered'] == 61 and record['particles'] == 121104
    assert record['resolution'] == [400,400] and record['geometry_scale'] == 1
    assert not record['axes'] and not record['annotations'] and record['color_range'] == [0,1]
    for name in ('camera_position','camera_focal_point','camera_parallel_scale'):
        np.testing.assert_allclose(record[name],reference[name],rtol=0,atol=1e-12)
    from PIL import Image
    images = []
    for index in range(61):
        with Image.open(output/'frames'/f'frame_{index:05d}.png') as picture:
            assert picture.size == (400,400)
            images.append(picture.convert('RGB'))
    images[0].save(output/'hjc-mat111-private-review.gif',save_all=True,append_images=images[1:],
        duration=100,loop=0,optimize=False,disposal=2)
    for picture in images:
        picture.close()
    assert hashlib.sha256((source/'raw/checkpoint.json').read_bytes()).hexdigest() == initial_hash
    assert 'jax' not in sys.modules
    receipt = dict(passed=True,scope='Private actual-state visual review; structural energy and time-step qualification pending.',
        numerical_run=report['child_archive']['run'],numerical_artifact=report['child_archive']['artifact'],
        full_raw_audit_commit=report['verifier_commit'],render_commit=os.environ.get('GITHUB_SHA'),
        camera_reference=str(reference_path.relative_to(ROOT)),
        camera_reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        actual_frames=61,particles=121104,physical_last_time_s=float(times[-1]),
        resolution=[400,400],geometry_scale=1,annotations=False,physical_steps_executed=0,
        structural_energy_certified=False,temporal_convergence_certified=False,
        published=False,public_visual_review_passed=False,
        gif_sha256=hashlib.sha256((output/'hjc-mat111-private-review.gif').read_bytes()).hexdigest())
    (output/'private-render-review.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('audit',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    render(args.audit.resolve(),args.output.resolve())
