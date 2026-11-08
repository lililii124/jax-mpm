"""Select a fixed-mass candidate only after complete binary blast replays."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from tools.compare_multicell_binary_blast import compare, read_run
from tools.reconstruct_multicell_blast import mask_digest, reconstruct


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('baseline_design', type=Path)
    parser.add_argument('baseline_geometry', type=Path)
    parser.add_argument('baseline_replay', type=Path)
    parser.add_argument('proposals', type=Path)
    parser.add_argument('--replays', type=Path, nargs='+', required=True,
                        help='Completed replay directories, in proposed-candidate order')
    parser.add_argument('--minimum-reduction', type=float, default=1e-6)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 <= args.minimum_reduction < 1:
        raise ValueError('Minimum reduction must lie in [0, 1)')
    inputs = json.loads(args.inputs.read_text(encoding='utf-8'))
    manifest = json.loads((args.proposals / 'manifest.json').read_text(encoding='utf-8'))
    if (hashlib.sha256(json.dumps(inputs['binding'], sort_keys=True).encode()).hexdigest() != inputs['signature'] or
            manifest['signature'] != inputs['signature'] or manifest['inputs_sha256'] != sha256(args.inputs)):
        raise ValueError('Proposal inputs do not match the optimizer binding')
    settings = inputs['binding']['settings']
    base_meta, _, _ = read_run(args.baseline_replay, .02)
    if base_meta['geometry_sha256'] != sha256(args.baseline_geometry):
        raise ValueError('Baseline replay geometry differs')
    with np.load(args.baseline_geometry, allow_pickle=False) as data:
        solid = data['solid_mask']
        if mask_digest(solid) != base_meta['solid_mask_sha256']:
            raise ValueError('Baseline solid mask differs from its replay')
        if not np.any(solid, axis=0).all():
            raise ValueError('Baseline does not cover the projected pressure footprint')
    with np.load(args.baseline_design, allow_pickle=False) as data:
        base_mask = reconstruct(data['rho_cell'], settings)[-1]['solid_mask_sha256']
    if (base_mask != manifest['baseline_solid_mask_sha256'] or
            base_mask != base_meta['solid_mask_sha256']):
        raise ValueError('Baseline design and replay differ')
    proposed = [row for row in manifest['steps'] if row['status'] == 'proposed']
    if len(proposed) != len(args.replays) or not proposed:
        raise ValueError('Supply one complete replay for every proposed candidate')
    comparisons = []
    for row, replay in zip(proposed, args.replays):
        design = args.proposals / row['design']
        geometry = args.proposals / row['geometry']
        if sha256(design) != row['design_sha256'] or sha256(geometry) != row['geometry_sha256']:
            raise ValueError('Proposal files do not match their manifest')
        meta, _, _ = read_run(replay, .02)
        if meta['geometry_sha256'] != row['geometry_sha256'] or meta['solid_mask_sha256'] != row['solid_mask_sha256']:
            raise ValueError('Replay does not match the proposed geometry')
        with np.load(geometry, allow_pickle=False) as data:
            solid = data['solid_mask']
            if mask_digest(solid) != row['solid_mask_sha256']:
                raise ValueError('Proposed solid mask differs from its manifest')
            if not np.any(solid, axis=0).all():
                raise ValueError('Candidate does not cover the projected pressure footprint')
        with np.load(design, allow_pickle=False) as data:
            report = reconstruct(data['rho_cell'], settings)[-1]
        if (report['solid_mask_sha256'] != row['solid_mask_sha256'] or report['components'] != 1 or
                not report['faces_connected'] or report['populated_column_fraction'] != 1.):
            raise ValueError('Candidate design and connected geometry differ')
        result = compare(args.baseline_replay, replay)
        comparisons.append({'step': row['step'], 'design': str(design),
                            'solid_mask_sha256': row['solid_mask_sha256'], 'comparison': result})
    best = min(comparisons, key=lambda row: row['comparison']['candidate_sampled_peak_m'])
    accepted = best['comparison']['sampled_peak_reduction_fraction'] > args.minimum_reduction
    selected = Path(best['design']) if accepted else args.baseline_design
    args.output.mkdir(parents=True)
    target = args.output / 'selected-design.npz'
    shutil.copyfile(selected, target)
    result = {'accepted': accepted, 'selected_step': best['step'] if accepted else 0.,
              'selected_design_sha256': sha256(target),
              'selected_solid_mask_sha256': best['solid_mask_sha256'] if accepted else base_mask,
              'minimum_reduction': args.minimum_reduction, 'comparisons': comparisons,
              'qualification': 'Sampled peak comparison for connected elastic solids under matched projected pressure; not experimental validation.'}
    temp = args.output / 'selection.json.tmp'
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(args.output / 'selection.json')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
