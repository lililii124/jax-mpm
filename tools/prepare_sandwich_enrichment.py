"""Build an explicitly partitioned ROM snapshot NPZ from full-MPM Gauss frames.

Run as ``python -m tools.prepare_sandwich_enrichment SPEC.json OUTPUT.npz``.
Paths in the specification are relative to the specification file.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.export_sandwich_fullmpm_face import recover_nodes


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def step_list(value):
    if isinstance(value, dict):
        value = range(value['start'], value['stop'] + 1, value['stride'])
    result = list(value)
    if not result or any(type(n) is not int or n < 0 for n in result):
        raise ValueError('Each supplied partition needs nonnegative integer steps')
    if result != sorted(set(result)):
        raise ValueError('Partition steps must be unique and increasing')
    return result


def prepare(spec_path, output, source_names=None):
    spec_path, output = Path(spec_path).resolve(), Path(output).resolve()
    receipt_path = output.with_suffix('.json')
    if output.exists() or receipt_path.exists():
        raise FileExistsError('Choose a fresh output and receipt path')
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    if spec.get('version') != 1:
        raise ValueError('Expected specification version1')
    spacing = np.asarray(spec['spacing_m'], dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('Expected three finite positive spacings')
    frozen = spec.get('frozen_model_receipt')
    frozen_sha = digest(spec_path.parent / frozen) if frozen else None
    fields, times, numbers, source_ids, partitions, frames = [], [], [], [], [], []
    sources = []
    common_reference = common_nodes = None
    names = set()
    available = {source['name'] for source in spec['sources']}
    if source_names and not set(source_names) <= available:
        raise ValueError('Requested source is absent from the specification')
    for source_id, source in enumerate(spec['sources']):
        name = source['name']
        if name in names:
            raise ValueError('Source names must be unique')
        names.add(name)
        if source_names and name not in source_names:
            continue
        directory = (spec_path.parent / source['directory']).resolve()
        status_path = directory / 'status.json'
        status = json.loads(status_path.read_text(encoding='utf-8'))
        if not status.get('valid', False):
            raise ValueError('Source full-MPM trajectory is not marked valid')
        binding = status['binding']
        if source.get('case', binding['case']) != binding['case']:
            raise ValueError('Full-MPM case differs from specification')
        if source.get('source_signature', binding.get('source_signature')) != binding.get('source_signature'):
            raise ValueError('Full-MPM physical source signature differs from specification')
        controls = np.asarray(binding['controls'], dtype=np.float64)
        controls_sha = hashlib.sha256(controls.tobytes()).hexdigest()
        if controls_sha != source['controls_sha256']:
            raise ValueError(f'{name}: full-MPM control hash differs from specification')
        groups = {}
        for partition in ('training', 'development', 'validation'):
            value = source.get(partition + '_steps', [])
            groups[partition] = step_list(value) if value else []
        assigned = [n for group in groups.values() for n in group]
        if not assigned or len(set(assigned)) != len(assigned):
            raise ValueError('Each source requires disjoint explicit partitions')
        if groups['validation'] and not frozen_sha:
            raise ValueError('Fresh validation extraction requires a frozen_model_receipt')
        if groups['training']:
            later = groups['development'] + groups['validation']
            if later and max(groups['training']) >= min(later):
                raise ValueError('Training must end before development/validation')
        reserved = source.get('reserved_validation_from_step')
        if reserved is not None:
            fitted = groups['training'] + groups['development']
            if fitted and max(fitted) >= reserved:
                raise ValueError('Training/development intersects the reserved future interval')
            if groups['validation'] and min(groups['validation']) < reserved:
                raise ValueError('Validation precedes its declared fresh interval')
        material_sha = None
        for number in sorted(assigned):
            partition = next(key for key, value in groups.items() if number in value)
            path = directory / f'frame_{number:07d}.npz'
            with np.load(path, allow_pickle=False) as saved:
                reference = saved['reference'].copy()
                displacement = saved['displacement'].copy()
                time = float(saved['time'])
                material = hashlib.sha256()
                for key in ('phi', 'young', 'density', 'volume', 'skin'):
                    material.update(np.ascontiguousarray(saved[key]).tobytes())
            current_material = material.hexdigest()
            if material_sha is None:
                material_sha = current_material
            elif current_material != material_sha:
                raise ValueError('A source changes its material field across frames')
            _, axes, u, recovery = recover_nodes(reference, displacement, spacing)
            # Match compact_tl's integer-grid construction bit for bit. The
            # inversion axes differ only by roundoff, not by geometry.
            origin = np.floor(reference / spacing - .5).astype(np.int64).min(axis=0)
            canonical = [(np.arange(len(axis)) + origin[a]) * spacing[a]
                         for a, axis in enumerate(axes)]
            for original, snapped in zip(axes, canonical):
                np.testing.assert_allclose(original, snapped, rtol=0, atol=1e-13)
            nodes = np.stack(np.meshgrid(*canonical, indexing='ij'), axis=-1).reshape(-1, 3)
            if common_reference is None:
                common_reference, common_nodes = reference, nodes
            else:
                np.testing.assert_array_equal(reference, common_reference)
                np.testing.assert_array_equal(nodes, common_nodes)
            if not np.isfinite(time) or time < 0:
                raise ValueError('Invalid saved physical time')
            fields.append(u.reshape(-1, 3)); times.append(time)
            numbers.append(number); source_ids.append(source_id); partitions.append(partition)
            frames.append(dict(source=name, step=number, partition=partition, time_s=time,
                               file=path.name, sha256=digest(path), recovery=recovery))
            print(json.dumps(dict(source=name, step=number, partition=partition)), flush=True)
        sources.append(dict(name=name, case=binding['case'], partitions=groups,
                            controls_sha256=controls_sha, material_sha256=material_sha,
                            status_sha256=digest(status_path),
                            binding_sha256=hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest(),
                            reserved_validation_from_step=reserved))
    masks = {key: np.asarray([p == key for p in partitions])
             for key in ('training', 'development', 'validation')}
    if not masks['training'].any():
        raise ValueError('At least one training frame is required')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, displacement=np.asarray(fields), nodes=common_nodes,
                        time=np.asarray(times), steps=np.asarray(numbers),
                        source_index=np.asarray(source_ids), **masks)
    temporary.replace(output)
    receipt = dict(schema_version=1, specification=spec, specification_sha256=digest(spec_path),
                   selected_source_names=source_names,
                   frozen_model_receipt_sha256=frozen_sha, sources=sources, frames=frames,
                   output_sha256=digest(output), training_count=int(masks['training'].sum()),
                   development_count=int(masks['development'].sum()), validation_count=int(masks['validation'].sum()),
                   nodes_sha256=hashlib.sha256(common_nodes.tobytes()).hexdigest(),
                   training_fields_sha256=hashlib.sha256(np.ascontiguousarray(np.asarray(fields)[masks['training']]).tobytes()).hexdigest(),
                   source_sha256={Path(__file__).name: digest(__file__),
                                  'export_sandwich_fullmpm_face.py': digest(Path(__file__).with_name('export_sandwich_fullmpm_face.py'))},
                   qualification='Dataset provenance and recovery checks only; no ROM accuracy or design certification.')
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    return dict(output=str(output), receipt=str(receipt_path), sha256=receipt['output_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('specification', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--source', action='append', help='Include only this named source; repeat for several')
    args = parser.parse_args()
    print(json.dumps(prepare(args.specification, args.output, args.source), indent=2))


if __name__ == '__main__':
    main()
