"""Compare a completed central-pressure candidate with the uniform baseline."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def compare(baseline_path, candidate_directory):
    baseline_path, candidate_directory = Path(baseline_path), Path(candidate_directory)
    baseline_bytes = baseline_path.read_bytes()
    baseline_record = json.loads(baseline_bytes)
    baseline_case = next(x for x in baseline_record['cases'] if x['case'] == 'central')
    baseline = baseline_case['full_mpm_summary']
    execution_path = candidate_directory / 'execution.json'
    status_path = candidate_directory / 'trajectory/status.json'
    summary_path = candidate_directory / 'trajectory/summary.json'
    execution_bytes, status_bytes, summary_bytes = (p.read_bytes() for p in
                                                     (execution_path, status_path, summary_path))
    inputs = json.loads(execution_bytes)['inputs']
    status = json.loads(status_bytes)
    candidate = json.loads(summary_bytes)
    if not (baseline['completed'] and candidate['completed'] and status['completed'] and status['valid']):
        raise ValueError('Both trajectories must be complete and valid')
    if baseline['case'] != 'central' or candidate['case'] != 'central' or inputs['case'] != 'central':
        raise ValueError('Both trajectories must use central pressure')
    if baseline['steps'] != candidate['steps'] or status['step'] != candidate['steps']:
        raise ValueError('Trajectory step counts differ')
    if not np.isclose(baseline['time_s'], candidate['time_s'], rtol=0, atol=1e-12):
        raise ValueError('Trajectory durations differ')
    if baseline['plan']['settings'] != candidate['plan']['settings']:
        raise ValueError('Physical model settings differ')
    if baseline['source_sha256'] != candidate['source_sha256'] or inputs['physical_sources'] != candidate['source_sha256']:
        raise ValueError('Physical source hashes differ')
    if inputs['physical_source_signature'] != status['binding']['source_signature']:
        raise ValueError('Candidate physical binding differs')
    q = np.asarray(status['binding']['controls'], dtype=np.float64)
    if hashlib.sha256(q.tobytes()).hexdigest() != inputs['controls_sha256']:
        raise ValueError('Candidate controls differ from execution binding')
    if not np.isclose(baseline['mass_kg'], candidate['mass_kg'], rtol=1e-10, atol=0):
        raise ValueError('Mass differs from the uniform baseline')
    peaks = [baseline['peak_back_displacement_m'], candidate['peak_back_displacement_m']]
    stresses = [baseline['peak_skin_stress_Pa'], candidate['peak_skin_stress_Pa']]
    if not np.isfinite(peaks + stresses).all() or min(peaks) <= 0:
        raise ValueError('Invalid peak metrics')
    digest = lambda data: hashlib.sha256(data).hexdigest()
    return dict(case='central', completed=True, baseline_peak_m=peaks[0],
        candidate_peak_m=peaks[1], peak_reduction_fraction=1-peaks[1]/peaks[0],
        baseline_peak_skin_Pa=stresses[0], candidate_peak_skin_Pa=stresses[1],
        mass_kg=candidate['mass_kg'], steps=candidate['steps'], time_s=candidate['time_s'],
        baseline_controls_sha256=baseline_case['controls_sha256'],
        candidate_controls_sha256=inputs['controls_sha256'],
        baseline_record_sha256=digest(baseline_bytes), candidate_execution_sha256=digest(execution_bytes),
        candidate_status_sha256=digest(status_bytes), candidate_summary_sha256=digest(summary_bytes),
        model_scope='elastic skins and effective elastic core', refinement_verified=False,
        nonlinear_blast_qualified=False, global_optimum_claim=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
