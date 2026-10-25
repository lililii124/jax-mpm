"""Admission for bounded persistent-grid smoke; never certifies a crack."""
import argparse
import json
from pathlib import Path
import numpy as np


def qualify(root):
    root = Path(root)
    records = []
    for cfl in ('.04', '.02', '.01'):
        folder = root / ('cfl' + cfl)
        summary = json.loads((folder / 'summary.json').read_text())
        history = np.genfromtxt(folder / 'history.csv', delimiter=',', names=True)
        assert summary['reached_requested_displacement'], 'Truncated physical interval'
        assert summary['maximum_kkt'] <= 1e-7
        assert summary['maximum_strain'] <= .1
        assert np.isfinite(np.array(history.tolist(), dtype=float)).all()
        assert abs(float(history['punch_displacement'][-1]) - .00005) < 1e-14
        audit = summary['audit']
        assert abs(audit['algebraic_closure']) < 1e-10
        assert abs(audit['force_drift_weak_identity_error']) < 1e-10
        with np.load(folder / f"frame_{summary['steps']:06d}.npz") as frame:
            assert all(np.isfinite(frame[key]).all() for key in frame.files)
            assert frame['damage'].min() >= -1e-12 and frame['damage'].max() <= 1+1e-12
            assert abs(frame['volume'].sum() - .098304) < 1e-14
        numerical = (audit['frozen_damage_strain_integration_defect'] +
                     audit['kick_drift_work_difference'])
        scale = max(abs(audit['actuator_work']), sum(summary['final_energy'].values()), 1e-20)
        records.append(dict(cfl=float(cfl), steps=summary['steps'], dt=summary['dt'],
            maximum_damage=summary['maximum_particle_damage'],
            signed_integration_defect=numerical, relative_integration_defect=numerical/scale,
            damage_split_potential_change=audit['damage_split_potential_change'],
            actuator_work=audit['actuator_work'], fracture_energy=summary['final_energy']['fracture']))
    # A contact transition can spoil monotone signed endpoint errors. Report it
    # explicitly and refuse promotion; do not hide it in algebraic closure.
    defects = [abs(record['signed_integration_defect']) for record in records]
    refined = all(b <= max(.8*a, 1e-13) for a, b in zip(defects, defects[1:]))
    result = dict(status='bounded_smoke_passed' if refined else 'refinement_not_admitted',
        temporal_defect_decreases=refined, records=records,
        crack_validated=False, medium_run_admitted=False)
    (root / 'refinement-admission.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    if not refined:
        raise RuntimeError('Temporal defect did not improve under dt refinement; inspect raw terms')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('root')
    print(json.dumps(qualify(parser.parse_args().root), indent=2))
