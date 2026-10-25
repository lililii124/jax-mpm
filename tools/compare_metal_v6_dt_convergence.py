"""Compare three temporal resolutions without claiming spatial convergence."""
import json
from pathlib import Path
import numpy as np

CHANNELS=('p2g','g2p','contact_allocation','viscosity','constitutive_defect',
          'numerical_increment','boundary','ledger_residual','identity_residual')


def compare(root):
    root=Path(root);proofs=[];traces=[]
    for divisor in (1,2,4):
        proof=json.loads((root/f'dt{divisor}/equivalence.json').read_text())
        if proof.get('admitted') is not True or proof['dt_divisor']!=divisor or proof['cfl']!=.1/divisor:
            raise ValueError('Rejected or wrong temporal cell')
        if set(proof['signed_J'])!=set(CHANNELS) or set(proof['absolute_J'])!=set(CHANNELS):raise ValueError('Nine channels required')
        if not all(v['bitwise_equal'] for v in proof['fields'].values()) or len(proof['fields'])!=35:raise ValueError('Strict35-field equivalence required')
        with np.load(root/f'dt{divisor}/adaptive-trace.npz',allow_pickle=False) as data:
            dt=data['dt_s'];values=data['channels_J']
            if tuple(data['channel_names'])!=CHANNELS or len(dt)!=proof['steps_probe'] or np.any(dt<=0):raise ValueError('Actual adaptive trace differs')
            if abs(dt.sum()-(proof['actual_stop_ns']-proof['actual_start_ns'])*1e-9)>1e-18:raise ValueError('Trace does not cover physical interval')
            if values.shape!=(len(dt),9) or not np.isfinite(values).all():raise ValueError('Invalid channel trace')
            if not np.allclose(values.sum(0),[proof['signed_J'][k] for k in CHANNELS],rtol=0,atol=proof['energy_identity_tolerance_J']):raise ValueError('Trace signed sums differ')
            if not np.allclose(abs(values).sum(0),[proof['absolute_J'][k] for k in CHANNELS],rtol=0,atol=proof['energy_identity_tolerance_J']):raise ValueError('Trace absolute sums differ')
            traces.append(dict(divisor=divisor,steps=len(dt),minimum_dt_s=float(dt.min()),maximum_dt_s=float(dt.max()),mean_dt_s=float(dt.mean()),covered_time_s=float(dt.sum())))
        proofs.append(proof)
    if len({p['parent_checkpoint_sha256'] for p in proofs})!=1 or len({p['actual_stop_ns'] for p in proofs})!=1:raise ValueError('Different parent or physical interval')
    channels={}
    for channel in CHANNELS:
        signed=np.array([p['signed_J'][channel] for p in proofs]);absolute=np.array([p['absolute_J'][channel] for p in proofs])
        difference=np.diff(signed)
        channels[channel]=dict(signed_J=signed.tolist(),absolute_J=absolute.tolist(),successive_signed_differences_J=difference.tolist(),
                               absolute_difference_contracts=bool(abs(difference[1])<abs(difference[0])),
                               difference_ratio=float(abs(difference[1]/difference[0])) if difference[0]!=0 else None)
    field_changes={}
    states=[]
    for divisor in (1,2,4):
        with np.load(root/f'dt{divisor}/probe-final.npz',allow_pickle=False) as data:
            states.append({k:data[k].copy() for k in ('x','v','C','F','damage','mass')})
    for a,b in zip(states,states[1:]):
        if not np.array_equal(a['mass'],b['mass']):raise ValueError('Mass/particle population differs')
    for name in ('x','v','C','F','damage'):
        field_changes[name]=[float(np.max(abs(states[i+1][name]-states[i][name]))) for i in range(2)]
    return dict(comparison_complete=True,dt_levels=traces,channels=channels,successive_max_abs_field_changes=field_changes,
                interpretation='Descriptive three-level temporal sensitivity only. Adaptive CFL division is not an exact halving of each dt. Shock/contact/topology can be nonsmooth; no Richardson order or limit inferred.',
                spatial_convergence_tested=False,particle_convergence_tested=False,repair_admitted=False,production_restart_allowed=False,validated_fragments=False)
