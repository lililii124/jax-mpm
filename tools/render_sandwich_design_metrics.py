"""Plot saved sandwich design metrics in step with the material animation."""
from pathlib import Path
import argparse
import hashlib
import io
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def render(source, output):
    source, output = Path(source), Path(output)
    data = json.loads(source.read_text(encoding='utf-8'))
    states = data['frames']
    controls_path = source.parent/'controls.json'
    bundle_path = source.parent/'controls.npz'
    for path, key in ((controls_path, 'controls_json_sha256'),
                      (bundle_path, 'controls_artifact_sha256')):
        if hashlib.sha256(path.read_bytes()).hexdigest() != data[key]:
            raise ValueError('The material design inputs changed')
    controls = json.loads(controls_path.read_text(encoding='utf-8'))['states']
    with np.load(bundle_path, allow_pickle=False) as saved:
        snapshots = saved['controls']
    if not states or len(states) != len(controls) or len(states) != len(snapshots):
        raise ValueError('Expected one metric record per material frame')
    for state, reference, q in zip(states, controls, snapshots):
        if state['controls_sha256'] != reference['controls_sha256'] or \
                hashlib.sha256(q.tobytes()).hexdigest() != state['controls_sha256']:
            raise ValueError('Metrics do not match the material frame')
        cases = np.asarray(state['metrics']['cases'])
        if cases.shape != (3, 7) or not np.isfinite(cases).all():
            raise ValueError('Expected three complete load-case metric records')
        np.testing.assert_allclose(cases[:, 3].max(), state['metrics']['actual_peak_back_m'], rtol=1e-12)
        np.testing.assert_allclose(cases[:, 4].max(), state['metrics']['actual_peak_skin_Pa'], rtol=1e-12)
    values = np.array([[s['metrics']['actual_peak_back_m'] * 1000,
                        s['metrics']['actual_peak_skin_Pa'] / 1e6,
                        s['metrics']['mass_kg']] for s in states])
    if not np.isfinite(values).all() or np.any(values[:, :2] < 0):
        raise ValueError('Nonfinite or negative metrics')
    if not np.allclose(values[:, 2], 1.8468, rtol=1e-10, atol=0):
        raise ValueError('The fixed-mass condition changed')
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': '#344150', 'text.color': '#344150',
                         'xtick.color': '#627080', 'ytick.color': '#627080'})
    images = []
    steps = np.arange(len(states))
    for index in steps:
        fig, axes = plt.subplots(2, 1, figsize=(4, 4), dpi=100, sharex=True)
        fig.subplots_adjust(left=.16, right=.95, top=.88, bottom=.17, hspace=.42)
        fig.suptitle('Design metrics · reduced model', y=.97, fontsize=10)
        for column, (axis, label, color, unit) in enumerate(zip(
                axes, ['Peak deflection', 'Peak skin stress'], ['#236ba2', '#a35e25'], ['mm', 'MPa'])):
            axis.plot(steps[:index+1], values[:index+1, column], color=color,
                      linewidth=1.6, marker='o', markersize=3)
            axis.scatter([index], [values[index, column]], color=color, s=26, zorder=3)
            axis.set_title(f'{label} ({unit})', loc='left', fontsize=9, pad=7)
            axis.text(1, 1.06, f'{values[index, column]:.3f}', color=color,
                      transform=axis.transAxes, ha='right', fontsize=9)
            axis.set_xlim(-.3, len(states)-.7)
            axis.grid(axis='y', color='#e4e9ee', linewidth=.6)
            axis.set_axisbelow(True)
            axis.tick_params(length=3)
        low, high = values[:, 0].min(), values[:, 0].max()
        margin = max((high-low)*.17, .02)
        axes[0].set_ylim(max(0, low-margin), high+margin)
        axes[1].set_ylim(min(100, values[:, 1].min()*.9), 212)
        axes[1].axhline(200, linestyle='--', color='#a2a8ad', linewidth=.9)
        axes[1].text(len(states)-.8, 202, '200 MPa limit', ha='right', fontsize=7, color='#7a8289')
        ticks = np.unique(np.linspace(0, len(states)-1, 4, dtype=int))
        axes[1].set_xticks(ticks)
        axes[1].set_xlabel('Retained design step')
        fig.text(.5, .045, f'Fixed mass  1.8468 kg     ·     {index+1}/{len(states)}',
                 ha='center', fontsize=8)
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100, facecolor='white')
        plt.close(fig)
        buffer.seek(0)
        with Image.open(buffer) as image:
            images.append(image.convert('RGB').copy())
    atlas = Image.new('RGB', (400, 400*len(images)))
    for index, image in enumerate(images):
        atlas.paste(image, (0, index*400))
    palette = atlas.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    frames = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
    durations = [650]*len(images)
    durations[-1] = 2000
    gif = output/'design-metrics.gif'
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=False, disposal=2)
    images[-1].save(output/'design-metrics.png')
    with Image.open(gif) as saved:
        if saved.size != (400, 400) or saved.n_frames != len(states):
            raise ValueError('Metric animation changed dimensions or dropped a state')
    report = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  gif_sha256=hashlib.sha256(gif.read_bytes()).hexdigest(),
                  controls_sha256=[s['controls_sha256'] for s in states],
                  size=[400, 400], frames=len(states), duration_ms=durations,
                  quantity_source='frozen-ROM', full_mpm_verified=False)
    (output/'metrics-render.json').write_bytes((json.dumps(report, indent=2)+'\n').encode())
    print(json.dumps(dict(output=str(output), frames=len(states))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    render(args.source, args.output)
