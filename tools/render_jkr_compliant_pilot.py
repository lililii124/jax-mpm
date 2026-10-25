"""Private real-state rendering only; no publication or geometry manipulation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.review_jkr_compliant_pilot import descriptor, review


def render(source, output):
    from mpm_jax.export import export_series
    from mpm_jax.visualization import find_pvpython
    if output.exists():
        raise ValueError('Separate fresh private render directory required')
    source_hash = hashlib.sha256((source/'checkpoint.npz').read_bytes()).hexdigest()
    plan = json.loads((source/'plan.json').read_text())
    summary = json.loads((source/'summary.json').read_text())
    numerical = review(source, output.parent/(output.name+'-independent-numerical.json'))
    with np.load(source/'checkpoint.npz') as checkpoint:
        frame = int(checkpoint['frame'])
        tables = json.loads(str(checkpoint['tables']))
    times = np.asarray(tables['times'])
    arrays = {name:np.load(source/f'trajectory_{name}.npy', mmap_mode='r')[:frame+1]
              for name in ('x','damage','failed','speed','temperature')}
    with np.load(source/'particle_fields.npz') as labels:
        fields = {name:np.array(labels[name]) for name in labels.files}
    assert np.all(fields['phase'] == 1) and set(np.unique(fields['family_id'])) == {0,1}
    fields.update({name:arrays[name] for name in ('damage','failed','speed','temperature')})
    fields['body'] = np.zeros(plan['particles'], dtype=np.int32)
    output.mkdir(parents=True)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    export_series(arrays['x'], fields, times, output,
                  spacing=plan['geometry']['particle_spacing_m'], dim=3)
    camera = plan['camera']
    command = [find_pvpython(), '--force-offscreen-rendering', str(ROOT/'tools/paraview_jkr_compliant_scene.py'),
        str(output), '--field', 'family_id', '--resolution', str(camera['resolution']),
        '--focus-bounds', *map(str,camera['focus_bounds_m'])]
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    environment.pop('PYTHONHOME', None)
    with (output/'render.log').open('w') as log:
        subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    record = json.loads((output/'render.json').read_text())
    for key, expected in (('camera_position',camera['position']),
                          ('camera_focal_point',camera['focal_point']),
                          ('camera_parallel_scale',camera['parallel_scale'])):
        np.testing.assert_allclose(record[key], expected, rtol=0., atol=1e-12)
    assert record['resolution'] == [400,400] and record['frames_rendered'] == frame+1
    assert record['geometry_scale'] == 1 and not record['axes'] and not record['annotations']
    assert record['camera_bounds'] == camera['focus_bounds_m']
    from PIL import Image
    images = []
    raster = []
    for index in range(frame+1):
        path = output/'frames'/f'frame_{index:05d}.png'
        with Image.open(path) as picture:
            rgb = np.asarray(picture.convert('RGB')).astype(float)
            blue = (rgb[:,:,2] > 1.12*rgb[:,:,0])&(rgb[:,:,2] > 1.03*rgb[:,:,1])
            orange = (rgb[:,:,0] > 1.15*rgb[:,:,1])&(rgb[:,:,0] > 1.4*rgb[:,:,2])
            columns = slice(196,205)
            lower_rows = np.flatnonzero(np.any(blue[:,columns], axis=1))
            upper_rows = np.flatnonzero(np.any(orange[:,columns], axis=1))
            gap = int(np.min(np.abs(lower_rows[:,None]-upper_rows[None]))) if len(lower_rows) and len(upper_rows) else None
            raster.append(dict(frame=index,time_s=float(times[index]),
                central_strip_body_edge_separation_pixels=gap,
                blue_pixels=int(blue.sum()),orange_pixels=int(orange.sum())))
            images.append(picture.convert('RGB'))
    durations = np.maximum(20, np.rint(np.r_[np.diff(times), .015]*6000/.09)).astype(int)
    images[0].save(output/'private-preview.gif', save_all=True, append_images=images[1:],
                   duration=durations.tolist(), loop=0, optimize=False, disposal=2)
    for image in images:
        image.close()
    receipt = dict(actual_saved_frames=frame+1, actual_last_time_s=float(times[-1]),
        checkpoint_sha256=source_hash, camera=camera, numerical_review=numerical,
        source_plan_sha256=hashlib.sha256((source/'plan.json').read_bytes()).hexdigest(),
        renderer_controller_commit=os.environ.get('GITHUB_SHA'),
        renderer_source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ('tools/render_jkr_compliant_pilot.py','tools/paraview_jkr_compliant_scene.py',
                         'mpm_jax/paraview_scene.py')},
        actual_raster_measurements=raster, geometry_scale=1,
        disclosure=plan['public_comparison_disclosure'],
        public_visual_review_passed=False, production_eligible=False,
        scope='Private actual-state ParaView frames. Raster strip edges are image measurements, not surface gaps or a matched three-model comparison.')
    (output/'private-render-review.json').write_text(json.dumps(receipt, indent=2)+'\n')
    assert hashlib.sha256((source/'checkpoint.npz').read_bytes()).hexdigest() == source_hash


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    render(args.source.resolve(), args.output.resolve())
