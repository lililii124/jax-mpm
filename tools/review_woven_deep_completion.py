"""Independent whole-raw completion and actual ParaView publication admission.

Invoke on a fresh controller checkout, with a downloaded immutable producer run.
Expected producer commit and transport SHA must be frozen by the publisher.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import rht_impact_prefix as transport
from tools import review_woven_deep_final as final
from examples import woven_large_plate as base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--producer-commit', required=True)
    parser.add_argument('--transport-sha256', required=True)
    args = parser.parse_args()
    assert not args.destination.exists(), 'Fresh independent destination required'
    assert transport.digest(args.parent/'transport.json') == args.transport_sha256
    transport.verify(args.parent)
    manifest = json.loads((args.parent/'transport.json').read_text())
    args.destination.mkdir(parents=True)
    reader = transport.PartReader(args.parent/name for name in manifest['parts'])
    metadata = {}
    with tarfile.open(fileobj=reader, mode='r|') as archive:
        for member in archive:
            assert member.name in manifest['files'] and member.isfile()
            if member.name in ('source.json', 'receipt.json'):
                metadata[member.name] = json.load(archive.extractfile(member))
            elif member.name.startswith('output/'):
                target = args.destination/member.name
                assert target.resolve().is_relative_to((args.destination/'output').resolve())
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    shutil.copyfileobj(archive.extractfile(member), stream, 1024**2)
    assert metadata['source.json']['commit'] == args.producer_commit
    assert metadata['receipt.json']['commit'] == args.producer_commit
    assert metadata['receipt.json']['schema'] == 'woven-deep-f60-v1'
    folder = args.destination/'output/plain/deep-sphere-impact'
    before = {p.relative_to(folder).as_posix(): transport.digest(p) for p in folder.rglob('*') if p.is_file()}
    sample, plan = base.load_saved_case(folder)
    with np.load(folder/'checkpoint.npz', allow_pickle=False) as checkpoint:
        base.validate_checkpoint_control(checkpoint, plan)
        assert int(checkpoint['frame']) == 60
        arrays = {name: np.load(folder/f'trajectory_{name}.npy', mmap_mode='r', allow_pickle=False)
                  for name in ('x', 'speed', 'damage', 'temperature')}
        base.validate_saved_prefix(arrays, sample.positions, checkpoint['state_x'], checkpoint['state_v'],
                                   json.loads(str(checkpoint['tables'])), 60, plan['output_steps'], plan['dt_s'])
    assert transport.digest(ROOT/'tools/review_woven_deep_final.py') == '35c155ccc88737aac1ea4f219863e92b42c3c843ff5d40a2cd64cdf11e8b6125'
    result = final.review(folder, ROOT/'docs/results/completed--woven-plate-impact/observables.csv')
    transport.write(args.destination/'depth-admission.json', result)
    assert result['passed'], 'Full raw retained; >=5mm gate failed, publication forbidden'
    subprocess.run([sys.executable, '-m', 'mpm_jax.visualization', str(folder), '--gif', '--field',
                    'family_id', '--resolution', '1200', '--matrix-opacity', '.008', '--sphere-opacity', '.45'], check=True)
    checked = subprocess.run([os.environ['PARAVIEW_PYTHON'], '--force-offscreen-rendering',
                              str(ROOT/'tools/check_vtk.py'), str(folder)], check=True, capture_output=True, text=True)
    assert before == {name: transport.digest(folder/name) for name in before}
    transport.write(args.destination/'publication-admission.json', dict(passed=True,
        producer_commit=args.producer_commit, transport_sha256=args.transport_sha256,
        final_gate=result, actual_paraview_readback=True, vtk_stdout=checked.stdout,
        raw_unchanged=True, solver_executed=False,
        scope='Only this independent >=5mm depth and actual ParaView admission permits replacement of baseline media.'))


if __name__ == '__main__':
    main()
