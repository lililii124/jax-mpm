"""Run the actual large-model CLI for20 steps and audit its saved prefix only."""
import argparse,json
from pathlib import Path
import numpy as np
from examples import woven_deep_impact as deep
from examples import woven_large_plate as base


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    deep.main(['--output',str(a.output),'--probe-steps','20','--persistence','controller'])
    folder=a.output/'plain/deep-sphere-impact'
    summary=json.loads((folder/'summary.json').read_text());review=json.loads((folder/'deep-review.json').read_text());sample,plan=base.load_saved_case(folder)
    assert summary['probe_complete'] and not summary['completed'] and summary['steps']==20 and plan['particles']==819200
    assert all(review[k] for k in ('stable','clamp_passed','material_energy_passed','energy_passed'))
    with np.load(folder/'checkpoint.npz',allow_pickle=False) as cp:
        base.validate_checkpoint_control(cp,plan)
        assert bool(cp['state_valid']) and float(cp['aggregate'][1])==1
        assert all(np.all(np.isfinite(cp[k])) for k in cp.files if np.issubdtype(cp[k].dtype,np.number))
        mass=base.validate_reference(sample,plan);F=cp['state_F'];minimum=float(np.linalg.det(F).min());assert minimum>0
        P=np.sum(mass[:,None]*cp['state_v'],axis=0)+float(cp['sphere_mass'])*cp['sphere_velocity']
        residual=P-float(cp['sphere_mass'])*np.asarray(plan['tool']['initial_velocity_m_s'])-cp['state_boundary_impulse']-cp['state_external_impulse']
        assert np.linalg.norm(residual)<=1e-10*max(np.linalg.norm(P),1.)
    result=dict(passed=True,current_entry_source=summary['source'],particles=819200,actual_steps=20,full_plan_steps=plan['full_steps'],probe_complete=True,full_protocol_completed=False,min_J=minimum,mass_kg=float(mass.sum()),system_momentum_residual_Ns=residual.tolist(),energy_review=review,scope='Current-checkout20-step prefix; free flight before impact. Entry/resource/initial validity check, not a revalidation of the frozen61-frame impact or its deformation response.')
    (a.output/'prefix-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
