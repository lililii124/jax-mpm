"""Independent array/clock/mass/contact/energy audit of the actual candidate f4."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def audit(root,parent,seed106,bridge110,output):
    data=root/'result';manifest=json.loads((root/'raw-output-manifest.json').read_text())
    receipt=json.loads((root/'download-review.json').read_text())
    for name,item in receipt['downloaded_files'].items():assert digest(root/name)==item['sha256']
    progress=json.loads((data/'result.json').read_text());cp=json.loads((data/'checkpoint.json').read_text())
    proof=json.loads((data/'source-review.json').read_text());case=json.loads((data/'case.json').read_text())
    assert cp['step']==140 and cp['frame']==4 and progress['candidate_frame4_reached']
    assert progress['accepted_steps']==30 and [r['step'] for r in progress['steps']]==list(range(111,141))
    assert all(r['valid'] and all(r['body_valid']) and all(r['all_history_valid']) for r in progress['steps'])
    assert not cp['original_full_run_completed'] and not progress['original_full_run_completed']
    original=json.loads((parent/'raw/checkpoint.json').read_text())
    assert cp['signature']==original['signature']
    assert hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest()==cp['signature']
    for name,value in proof['runtime_source_manifest'].items():assert manifest['result/runtime-source/'+name]['sha256']==value
    assert digest(data/'runtime-source/mpm_jax/materials/rht.py')==proof['candidate_rht_sha256']
    dt=case['configs'][0]['dt'];h=case['configs'][0]['domain_size']/case['configs'][0]['n_grid']
    impulse=np.asarray(original['contact_impulse'])+np.asarray(seed106['contact_impulse'])
    for row in bridge110['steps']:impulse+=np.asarray(row['contact_impulse'])
    expected_cumulative={110:impulse.copy()}
    for row in progress['steps']:
        impulse+=np.asarray(row['contact_impulse'])
        if row['step']%5==0:expected_cumulative[row['step']]=impulse.copy()
    assert np.array_equal(impulse,np.asarray(cp['contact_impulse']))
    energy_rows={r['step']:r for r in progress['checkpoint_energy_records']}
    assert sorted(energy_rows)==list(range(110,141,5))
    snapshots=[];previous={};maximum_energy_error=0.;max_density_relative_error=0.
    for step in range(110,141,5):
        bodies=[];actual_times=[]
        assert np.array_equal(np.asarray(energy_rows[step]['contact_impulse_cumulative_on_body1']),expected_cumulative[step])
        for body in range(2):
            name=f'state{step}-body{body}.npz';path=data/name
            assert digest(path)==manifest['result/'+name]['sha256']
            with np.load(path,allow_pickle=False) as z,np.load(parent/'raw'/original['states'][body],allow_pickle=False) as initial:
                assert set(z.files)==set(initial.files) and len(z.files)==28
                for key in z.files:assert np.isfinite(z[key]).all(),(step,body,key)
                assert bool(z['valid']) and z['history_valid'].all()
                mass,volume=z['mass'],z['volume'];F=z['F'];J=np.linalg.det(F)
                assert np.array_equal(mass,initial['mass']) and np.array_equal(volume,initial['volume'])
                assert (mass>0).all() and (volume>0).all() and (J>0).all()
                assert abs(float(z['time'])-step*dt)<1e-15
                actual_times.append(float(z['time']))
                D,ep,failed=z['history_damage'],z['history_plastic_strain'],z['history_failed']
                assert (D>=0).all() and (D<=1).all() and (ep>=0).all()
                if body in previous:
                    old=previous[body];assert (D>=old[0]).all() and (ep>=old[1]).all() and (~old[2]|failed).all()
                previous[body]=(D.copy(),ep.copy(),failed.copy())
                v,C=z['v'],z['C'];K=float(.5*np.sum(mass*np.sum(v*v,axis=1)))
                Ka=float(h*h/8*np.sum(mass*np.sum(C*C,axis=(1,2))))
                U=float(np.sum(mass*z['history_energy']));work=float(np.sum(mass*z['history_work']))
                saved=energy_rows[step]['bodies'][body]
                for key,actual in [('translational_kinetic_J',K),('quadratic_APIC_affine_kinetic_diagnostic_J',Ka),('material_history_energy_J',U),('constitutive_work_J',work)]:
                    maximum_energy_error=max(maximum_energy_error,abs(actual-saved[key]));assert abs(actual-saved[key])<=1e-12*max(1.,abs(actual))
                stored=None
                if body==1:
                    # Use the declared E/nu from the physical definition, independent of material metadata formatting.
                    E=case['definition']['projectile_young_Pa'];nu=case['definition']['projectile_poisson'];mu=E/(2*(1+nu));lam=E*nu/((1+nu)*(1-2*nu))
                    stored=float(np.sum(volume*(.5*mu*(np.sum(F*F,axis=(1,2))-3)-mu*np.log(J)+.5*lam*np.log(J)**2)))
                    assert abs(stored-saved['stored_elastic_J'])<=1e-12*max(1.,abs(stored))
                else:
                    rho=mass/volume/J;err=float(np.max(abs(z['history_density']-rho)/rho));max_density_relative_error=max(max_density_relative_error,err)
                    assert err<=1e-12
                bodies.append(dict(body=body,particles=len(mass),mass_kg=float(mass.sum()),min_J=float(J.min()),max_J=float(J.max()),
                    max_damage=float(D.max()),failed_particles=int(failed.sum()),max_plastic_strain=float(ep.max()),
                    translational_K_J=K,affine_K_diagnostic_J=Ka,history_energy_J=U,constitutive_work_J=work,
                    stored_elastic_J=stored,momentum_Ns=np.sum(mass[:,None]*v,axis=0).tolist()))
        assert actual_times[0]==actual_times[1]
        snapshots.append(dict(step=step,time_s=actual_times[0],expected_time_s=step*dt,bodies=bodies,contact_impulse_cumulative=expected_cumulative[step].tolist()))
    initial_momenta=[np.asarray([r['momentum_'+axis] for axis in 'xyz']) for r in original['rows'][:2]]
    final_momenta=[np.asarray(b['momentum_Ns']) for b in snapshots[-1]['bodies']]
    projectile_impulse_residual=final_momenta[1]-initial_momenta[1]-impulse
    assert np.max(abs(projectile_impulse_residual))<=1e-12*max(1.,float(np.max(abs(initial_momenta[1]))))
    result=dict(passed=True,scope='Actual compact candidate reached frame4; original0to105 and checked106to110 bridge preserved. This is not original-run completion or a penetration result.',
        step=140,time_s=cp['time_s'],all14_full_raw_states_verified=True,full_state_fields_per_body=28,
        all_original_validity_flags_true=True,mass_and_volume_exact=True,irreversible_material_histories=True,
        all_runtime_source_hashes_match=True,cumulative_contact_impulse_exact=True,
        projectile_momentum_contact_impulse_residual_Ns=projectile_impulse_residual.tolist(),
        total_momentum_change_Ns=(sum(final_momenta)-sum(initial_momenta)).tolist(),
        maximum_recomputed_energy_record_error_J=maximum_energy_error,maximum_density_mass_volume_J_relative_error=max_density_relative_error,
        compile_seconds=progress['compile_seconds'],solve_seconds=sum(r['solve_seconds'] for r in progress['steps']),
        snapshots=snapshots,external_work_J=cp['external_work'],energy_scope='Actual translational/affine kinetic, RHT specific history energy and neo-Hookean strain energy only; no conserved global energy claim and no numerical loss assigned to heat.',
        original_full_run_completed=False,media_approved=False)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='snapshots'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','parent','seed106','bridge110','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();audit(a.root,a.parent,json.loads(a.seed106.read_text()),json.loads(a.bridge110.read_text()),a.output)
