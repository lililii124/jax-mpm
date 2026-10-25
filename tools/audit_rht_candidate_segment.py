"""Audit every complete saved candidate state, including a guarded partial segment."""
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


def audit(root,seed,output):
    data=root/'result';record=json.loads((data/'result.json').read_text());cp=json.loads((data/'checkpoint.json').read_text())
    initial=json.loads((seed/'checkpoint.json').read_text());proof=json.loads((data/'source-review.json').read_text())
    case=json.loads((data/'case.json').read_text());status=json.loads((root/'job/status.json').read_text())
    download=json.loads((root/'download-review.json').read_text());manifest=json.loads((root/'raw-output-manifest.json').read_text())
    for name,item in download['downloaded_files'].items():assert digest(root/name)==item['sha256']
    assert proof['seed_checkpoint_sha256']==digest(seed/'checkpoint.json')
    assert proof['seed_state_raw_files']==initial['raw_files']
    for name,item in initial['raw_files'].items():assert digest(seed/name)==item['sha256']
    assert cp['signature']==initial['signature']==hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest()
    assert cp['candidate_rht_sha256']==initial['candidate_rht_sha256']==proof['candidate_rht_sha256']
    for name,value in proof['runtime_source_manifest'].items():assert manifest['result/runtime-source/'+name]['sha256']==value
    assert not cp['original_full_run_completed'] and 'early numerical contact' in proof['contact_scope']
    complete=record['checkpoints'];assert [c['step'] for c in complete]==list(range(initial['step'],cp['step']+1,5))
    assert cp['step']==record['last_complete_step'] and cp['raw_files']==complete[-1]['raw_files']
    dt=case['configs'][0]['dt'];h=case['configs'][0]['domain_size']/case['configs'][0]['n_grid']
    impulse=np.asarray(initial['contact_impulse']);expected={initial['step']:impulse.copy()};physical_failure=False
    for row in record['steps']:
        if not row['valid']:physical_failure=True;break
        assert all(row['body_valid']) and all(row['all_history_valid'])
        impulse+=np.asarray(row['contact_impulse'])
        if row['step']%5==0:expected[row['step']]=impulse.copy()
    assert np.array_equal(np.asarray(cp['contact_impulse']),expected[cp['step']])
    previous={};snapshots=[];max_energy_error=0.;max_density_error=0.
    for saved in complete:
        step=saved['step'];bodies=[];times=[]
        point=json.loads((data/f'checkpoint-step{step}.json').read_text())
        assert point['step']==step and point['raw_files']==saved['raw_files']
        assert np.array_equal(np.asarray(saved['contact_impulse_cumulative_on_body1']),expected[step])
        for body,name in enumerate(saved['states']):
            path=data/name;assert digest(path)==saved['raw_files'][name]['sha256']
            with np.load(path,allow_pickle=False) as z,np.load(seed/initial['states'][body],allow_pickle=False) as iz:
                assert set(z.files)==set(iz.files) and len(z.files)==28
                for key in z.files:assert np.isfinite(z[key]).all(),(step,body,key)
                assert bool(z['valid']) and z['history_valid'].all()
                m,V,F=z['mass'],z['volume'],z['F'];J=np.linalg.det(F)
                assert np.array_equal(m,iz['mass']) and np.array_equal(V,iz['volume'])
                assert (m>0).all() and (V>0).all() and (J>0).all()
                times.append(float(z['time']));assert abs(times[-1]-step*dt)<1e-15
                D,ep,failed=z['history_damage'],z['history_plastic_strain'],z['history_failed']
                assert (D>=0).all() and (D<=1).all() and (ep>=0).all()
                if body not in previous:previous[body]=(iz['history_damage'],iz['history_plastic_strain'],iz['history_failed'])
                old=previous[body];assert (D>=old[0]).all() and (ep>=old[1]).all() and (~old[2]|failed).all()
                previous[body]=(D.copy(),ep.copy(),failed.copy())
                v,C=z['v'],z['C'];K=float(.5*np.sum(m*np.sum(v*v,axis=1)));Ka=float(h*h/8*np.sum(m*np.sum(C*C,axis=(1,2))))
                U=float(np.sum(m*z['history_energy']));W=float(np.sum(m*z['history_work']));energy=saved['bodies'][body]
                for key,actual in [('translational_kinetic_J',K),('quadratic_APIC_affine_kinetic_diagnostic_J',Ka),('material_history_energy_J',U),('constitutive_work_J',W)]:
                    err=abs(actual-energy[key]);max_energy_error=max(max_energy_error,err);assert err<=1e-12*max(1.,abs(actual))
                if body==0:
                    rho=m/V/J;err=float(np.max(abs(z['history_density']-rho)/rho));max_density_error=max(max_density_error,err);assert err<=1e-12
                else:
                    E=case['definition']['projectile_young_Pa'];nu=case['definition']['projectile_poisson'];mu=E/(2*(1+nu));lam=E*nu/((1+nu)*(1-2*nu))
                    stored=float(np.sum(V*(.5*mu*(np.sum(F*F,axis=(1,2))-3)-mu*np.log(J)+.5*lam*np.log(J)**2)))
                    assert abs(stored-energy['stored_elastic_J'])<=1e-12*max(1.,abs(stored))
                bodies.append(dict(body=body,particles=len(m),mass_kg=float(m.sum()),min_J=float(J.min()),max_J=float(J.max()),max_damage=float(D.max()),failed_particles=int(failed.sum()),
                    max_plastic_strain=float(ep.max()),Ktrans_J=K,Kaff_diagnostic_J=Ka,history_energy_J=U,constitutive_work_J=W,stored_elastic_J=energy['stored_elastic_J'],
                    momentum_Ns=np.sum(m[:,None]*v,axis=0).tolist(),particle_nose_y_m=float(z['x'][:,1].min()) if body==1 else None))
        assert times[0]==times[1]
        snapshots.append(dict(step=step,time_s=times[0],frame=step//35 if step%35==0 else None,bodies=bodies,contact_impulse_cumulative=expected[step].tolist()))
    initial_P=[np.asarray([r['momentum_'+axis] for axis in 'xyz']) for r in initial['rows'][:2]]
    final_P=[np.asarray(b['momentum_Ns']) for b in snapshots[-1]['bodies']]
    projectile_residual=final_P[1]-initial_P[1]-np.asarray(cp['contact_impulse'])
    assert np.max(abs(projectile_residual))<=1e-12*max(1.,float(np.max(abs(initial_P[1]))))
    result=dict(saved_prefix_verified=True,segment_target_reached=record['target_reached'],physical_gate_failure=physical_failure,
        start_step=initial['step'],last_complete_step=cp['step'],target_step=record['target_step'],complete_state_pairs=len(complete),all28_fields_verified=True,
        mass_volume_exact=True,finite_positive_orientation=True,irreversible_histories=True,cumulative_contact_exact=True,
        maximum_recomputed_energy_error_J=max_energy_error,maximum_density_relative_error=max_density_error,
        projectile_momentum_contact_residual_Ns=projectile_residual.tolist(),total_momentum_change_Ns=(sum(final_P)-sum(initial_P)).tolist(),
        compile_seconds=record['compile_seconds'],solve_seconds=sum(r['solve_seconds'] for r in record['steps']),terminal_state=status['state'],guard_reason=status['guard_reason'],
        last_observation=cp['observations'][-1],snapshots=snapshots,contact_scope=proof['contact_scope'],
        energy_scope='Actual stored component records only; not a global energy conservation certificate or numerical heat assignment.',
        original_full_run_completed=False,media_approved=False)
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='snapshots'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','seed','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();audit(a.root,a.seed,a.output)
