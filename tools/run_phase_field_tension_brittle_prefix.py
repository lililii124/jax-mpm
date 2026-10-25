"""Fresh ideal-brittle material prefix; no scaled or resumed old fields."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import numpy as np
from run_phase_field_tension_localization import measured_run


def main(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    token=os.environ.pop('GH_TOKEN')
    request=urllib.request.Request('https://api.github.com/repos/lililii124/jax-mpm/actions/runs/40179674682',
        headers={'Authorization':'Bearer '+token})
    with urllib.request.urlopen(request,timeout=30) as response:admission=json.load(response)
    assert admission['conclusion']=='success' and admission['head_sha']=='7fab5ade036bd401898735a4e4e4e014bdd0a3e4'
    (output/'observer-run-admission.json').write_text(json.dumps(admission,indent=2))
    config=dict(case='new-independent-ideal-brittle-material',fresh_zero_mechanical_state=True, initial_damage='new minimum-energy discrete precrack solve; not zero and not scaled old data',
        E=1000.,nu=.2,density=1.,gc=.0000125,length=.0375,dx=.0125,
        speed=.005,displacement=.00045,ramp=.06,cfl=.04,stop_step=2400,
        old_gc=.0002,gc_ratio=.0625,load_amplitude_ratio=.25,
        transfer='persistent reference grid; material-point quadrature; no PIC reset',
        dynamic_loading=True,similarity='prediction only; actual new fields solved from zero',
        dimensional_unit_mapping=dict(length_m=.1,stress_Pa=30e6,density_kg_m3=2400.,time_s=.1*np.sqrt(2400./30e6)),
        physical_interpretation=dict(plate_m=[.06,.06,.0025],E_Pa=30e9,Gc_J_m2=37.5,ell_m=.00375,
            rho_kg_m3=2400.,speed_m_s=.5590169943749475,ramp_s=.00005366563145999495,travel_m=.000045,ideal_material_assumption=True),
        small_strain_note='nonseed <=.02; immutable initial unit seed only uses explicit geometry/residual-energy checks')
    (output/'case-configuration.json').write_text(json.dumps(config,indent=2))
    source=[Path(__file__),Path('examples/phase_field_tension_persistent_grid.py'),
        Path('examples/phase_field_tension_geometry.py'),Path('examples/phase_field_tension_qualification.py'),
        Path('tools/run_phase_field_tension_localization.py'),Path('.github/workflows/phase-field-tension-brittle-prefix.yml'),
        *sorted(Path('mpm_jax').rglob('*.py'))]
    (output/'source.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source},indent=2))
    command=[sys.executable,'-m','examples.phase_field_tension_persistent_grid','--gc','.0000125',
        '--speed','.005','--displacement','.00045','--steps','2400','--output',str(output/'raw')]
    measured_run(command,'fresh',output,timeout=2250)
    summary=json.loads((output/'raw/summary.json').read_text())
    assert summary['resumed_from_step']==0 and summary['steps']==2400
    assert summary['maximum_kkt']<=1e-7
    rows=[json.loads(line) for line in (output/'raw/strain-qualification.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==list(range(1,2401)) and all(r['passed'] for r in rows)
    with np.load(output/'raw/frame_000000.npz') as initial,np.load(output/'raw/frame_002400.npz') as final:
        assert not np.any(initial['nodal_displacement']) and not np.any(initial['nodal_velocity']) and not np.any(initial['history'])
        assert np.isfinite(final['nodal_damage']).all()
        initial_Gamma=float(np.sum(initial['volume']*initial['fracture_energy_density']))
    energy=summary['final_energy'];work=summary['audit']['actuator_work'];absolute=summary['audit']['absolute_step_work']
    residual=energy['nodal_kinetic']+energy['elastic']+energy['fracture']-initial_Gamma-work
    assert abs(residual)<=.05*max(absolute,1e-12)
    (output/'numerical-prefix-admission.json').write_text(json.dumps(dict(accepted_steps=2400,
        initial_fracture_energy=initial_Gamma,physical_energy_residual=residual,
        residual_over_absolute_work=residual/max(absolute,1e-30),
        physical_crack_localization_validated=False,quasistatic_claim=False),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True)
    main(p.parse_args().output)
