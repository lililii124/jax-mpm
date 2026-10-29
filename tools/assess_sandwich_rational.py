"""Assess a frozen rational basis without fitting or changing its directions.

Run ``python -m tools.assess_sandwich_rational CONFIG.json``. Use
``--check-frozen BASIS.npz RESPONSE.npz`` for a cheap frozen forward replay.
"""
from pathlib import Path
import argparse,hashlib,json,os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_key,'4')
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from tools.build_sandwich_rational import digest,physics_support,verify_sources,material_fields_digest


def finite_arrays(**arrays):
    for name,value in arrays.items():
        if not np.isfinite(value).all():raise ValueError('Nonfinite '+name)


def validate_operators(M,K,L,back):
    finite_arrays(mass=M,stiffness=K,loads=L,back=back)
    rank=len(M)
    if rank<1 or M.shape!=(rank,rank) or K.shape!=M.shape or L.shape!=(rank,3) or back.ndim!=2 or back.shape[0]<1 or back.shape[1]!=rank:
        raise ValueError('Operator shape mismatch')
    if np.any(np.linalg.norm(L,axis=0)<=0) or np.linalg.norm(back)<=0:raise ValueError('Zero load or observation normalization')
    for matrix in (M,K):
        np.testing.assert_allclose(matrix,matrix.T,rtol=1e-10,atol=1e-12)
        np.linalg.cholesky(matrix)


def integrate(mass,stiffness,loads,back,dt=1.25e-7,maximum_steps=None,count_override=None):
    """The v20 Newmark ordering, pressure samples and curve reconstruction."""
    validate_operators(mass,stiffness,loads,back)
    if not np.isfinite(dt) or dt<=0:raise ValueError('Invalid time step')
    count=int(np.ceil(.002/dt)) if count_override is None else int(count_override)
    if count<1:raise ValueError('Positive step count required')
    h=.002/count;t=np.arange(count+1)*h
    pressure=np.where(t<.0002,.5e6*np.sin(np.pi*t/.0002)**2,0.)
    rank=len(mass);factor=cho_factor(mass+.25*h*h*stiffness)
    u=np.zeros((rank,3));v=u.copy();a=u.copy();work=np.zeros(3)
    states=[u.copy()];velocities=[v.copy()];energy=[]
    stop=count if maximum_steps is None else min(count,maximum_steps)
    for n in range(stop):
        predicted=u+h*v+.25*h*h*a
        following_a=cho_solve(factor,pressure[n+1]*loads-stiffness@predicted,check_finite=False)
        following=predicted+.25*h*h*following_a;following_v=v+.5*h*(a+following_a)
        work+=.5*(pressure[n]+pressure[n+1])*np.sum(loads*(following-u),axis=0)
        kinetic=.5*np.sum(following_v*(mass@following_v),axis=0)
        elastic=.5*np.sum(following*(stiffness@following),axis=0)
        energy.append(np.stack((kinetic,elastic,work),axis=-1))
        u,v,a=following,following_v,following_a
        states.append(u.copy());velocities.append(v.copy())
    states=np.asarray(states);energy=np.asarray(energy)
    finite_arrays(displacement=states,energy=energy)
    curves=np.concatenate([np.max(np.abs(np.einsum('pr,trc->tpc',back,states[start:start+32])),axis=1)
        for start in range(0,len(states),32)])
    finite_arrays(curves=curves)
    if np.any(np.max(np.abs(curves),axis=0)<=0):raise ValueError('Zero case response normalization')
    scale=float(np.max(np.abs(energy)))
    if scale<=0:raise ValueError('Zero energy normalization')
    balance=float(np.max(np.abs(energy[:,:,0]+energy[:,:,1]-energy[:,:,2]))/scale)
    if balance>=1e-8:raise ValueError('Unchanged linear energy gate failed')
    return dict(time=t[:stop+1],displacement=states,velocity=np.asarray(velocities),energy=energy,curves=curves,energy_relative_error=balance)


def check_frozen(basis_path,response_path):
    """Replay 96 steps and reconstruct 67 full-horizon saved face fields."""
    with np.load(basis_path,allow_pickle=False) as saved:
        V=saved['basis'];M=saved['mass'];K=saved['stiffness'];L=saved['loads'];back=saved['back']
    with np.load(response_path,allow_pickle=False) as saved:
        t=saved['time'];states=saved['displacement'];curves=saved['curves'];energy=saved['energy']
    validate_operators(M,K,L,back)
    finite_arrays(basis=V,time=t,displacement=states,curves=curves,energy=energy)
    rank=len(M)
    if V.ndim!=3 or V.shape[1:]!=(3,rank) or len(t)<2 or states.shape!=(len(t),rank,3) or curves.shape!=(len(t),3) or energy.shape!=(len(t)-1,3,3):
        raise ValueError('Frozen trajectory shape mismatch')
    if t[0]!=0 or abs(t[-1]-.002)>1e-14 or not np.all(np.diff(t)>0):raise ValueError('Expected complete increasing2ms time grid')
    np.testing.assert_allclose(t,np.arange(len(t))*(.002/(len(t)-1)),rtol=0,atol=1e-16)
    if min(np.linalg.norm(states),np.max(np.abs(curves)),np.max(np.abs(energy)))<=0:raise ValueError('Zero frozen response normalization')
    # Reuse the saved integer count: dividing by rounded effective dt can
    # introduce an extra ceil step and would change the frozen time grid.
    replay=integrate(M,K,L,back,maximum_steps=96,count_override=len(t)-1)
    count=len(replay['time'])
    np.testing.assert_array_equal(replay['time'],t[:count])
    if np.linalg.norm(states[:count])<=0:raise ValueError('Zero prefix state normalization')
    state_error=float(np.linalg.norm(replay['displacement']-states[:count])/np.linalg.norm(states[:count]))
    sample=np.linspace(0,len(t)-1,67,dtype=int)
    rebuilt=np.max(np.abs(np.einsum('pr,trc->tpc',back,states[sample])),axis=1)
    face_error=float(np.max(np.abs(rebuilt-curves[sample]))/np.max(np.abs(curves)))
    if state_error>=1e-9 or face_error>=1e-9:raise ValueError('Frozen public replay mismatch')
    balance=float(np.max(np.abs(energy[:,:,0]+energy[:,:,1]-energy[:,:,2]))/np.max(np.abs(energy)))
    if balance>=1e-8:raise ValueError('Frozen full-horizon energy gate failed')
    return dict(scope='Public forward replay of 96 initial steps and 67 full-horizon saved face reconstructions; no new MPM simulation or qualification.',
        public_assessor_sha256=digest(__file__),
        basis_artifact_sha256=digest(basis_path),basis_sha256=hashlib.sha256(V.tobytes()).hexdigest(),
        response_artifact_sha256=digest(response_path),replay_steps=count-1,
        prefix_state_relative_l2=state_error,full_horizon_face_probe_relative_maximum_error=face_error,
        full_horizon_linear_energy_relative_error=balance,passed=True)


def omitted_energy(V,stiffness,support,young,skin,fields,steps):
    nu=np.where(skin,.33,.25);mu=young/(2*(1+nu));lam=young*nu/((1+nu)*(1-2*nu))
    flat=V.reshape(-1,V.shape[-1]);factor=cho_factor(stiffness);records=[]
    for number,u in zip(steps,fields):
        gradient=np.einsum('pki,pkj->pij',u[support.indices],support.gradients)
        strain=.5*(gradient+gradient.swapaxes(1,2));trace=np.trace(strain,axis1=1,axis2=2)
        stress=2*mu[:,None,None]*strain+lam[:,None,None]*trace[:,None,None]*np.eye(3)
        energy=float(np.sum(.5*support.volume*np.sum(stress*strain,axis=(1,2))))
        if not np.isfinite(energy) or energy<=0:raise ValueError('Nonpositive or nonfinite reference strain energy')
        local=-np.einsum('pij,pkj->pki',stress,support.gradients)*support.volume[:,None,None]
        force=np.zeros(V.shape[:2])
        for axis in range(3):np.add.at(force[:,axis],support.indices.ravel(),local[:,:,axis].ravel())
        g=-flat.T@force.reshape(-1)
        fraction=float(1-.5*g@cho_solve(factor,g)/energy)
        if not np.isfinite(fraction):raise ValueError('Nonfinite omitted energy fraction')
        if fraction< -1e-8:raise ValueError('Negative omitted energy beyond roundoff')
        records.append(dict(step=int(number),omitted_linear_strain_energy_fraction=fraction))
    return records


def assess(config_path):
    config_path=Path(config_path).resolve();config=json.loads(config_path.read_text(encoding='utf-8'))
    def path(key):return (config_path.parent/config[key]).resolve()
    expected=dict(response_gate=.05,omitted_energy_gate=.05,time_gate=.01,linear_energy_gate=1e-8)
    for key,value in expected.items():
        if config.get(key)!=value:raise ValueError('Frozen gate mismatch: '+key)
    freeze=json.loads(path('freeze_receipt').read_text(encoding='utf-8'))
    if freeze.get('schema_version')!=2:raise ValueError('Assessment requires a version2 input-bound freeze receipt')
    verify_sources(freeze['source_sha256'])
    if 'operators' in config:raise ValueError('Operator overrides are not allowed; freeze a separate operator variant')
    if config['role'] not in ('fresh-validation','development'):raise ValueError('Unknown assessment role')
    vertex=freeze['operator_vertex']
    if vertex not in ('initial','selected'):raise ValueError('Unknown frozen operator vertex')
    if digest(path('material'))!=freeze['inputs'][vertex+'_material']['sha256']:raise ValueError('Frozen material artifact hash mismatch')
    if material_fields_digest(path('material'))!=freeze['material_fields_sha256'][vertex]:raise ValueError('Frozen material field hash mismatch')
    snapshot_receipt=path('snapshots').with_suffix('.json')
    provenance=json.loads(snapshot_receipt.read_text(encoding='utf-8'))
    if provenance['output_sha256']!=digest(path('snapshots')):raise ValueError('Snapshot artifact changed after extraction')
    if len(provenance['sources'])!=1:raise ValueError('Assess one explicitly bound material vertex per dataset')
    source=provenance['sources'][0]
    controls=freeze['construction_configuration'][vertex+'_controls_sha256']
    if source['controls_sha256']!=controls or source['material_sha256']!=freeze['material_fields_sha256'][vertex]:raise ValueError('Snapshot material/control vertex mismatch')
    if config['case_index'] not in (0,1,2) or source['case']!=('uniform','central','eccentric')[config['case_index']]:raise ValueError('Snapshot load case mismatch')
    if provenance['specification']['sources'][0]['source_signature']!=freeze['full_mpm_source_signature']:raise ValueError('Full-MPM physical source mismatch')
    history_receipt=json.loads(path('history_receipt').read_text(encoding='utf-8'))
    if history_receipt['reference_history_sha256']!=digest(path('history')):raise ValueError('History artifact changed after extraction')
    if history_receipt['controls_sha256']!=controls or history_receipt['case']!=source['case'] or history_receipt['source_signature']!=freeze['full_mpm_source_signature']:
        raise ValueError('History physical/control/load binding mismatch')
    if digest(path('basis'))!=freeze['basis']['sha256']:raise ValueError('Frozen basis artifact hash mismatch')
    with np.load(path('basis'),allow_pickle=False) as saved:
        V=saved['basis'];nodes=saved['nodes'];M=saved['mass'];K=saved['stiffness'];L=saved['loads'];back=saved['back']
    if hashlib.sha256(V.tobytes()).hexdigest()!=freeze['basis']['numeric_basis_sha256']:raise ValueError('Frozen numeric basis hash mismatch')
    validate_operators(M,K,L,back)
    finite_arrays(basis=V,nodes=nodes)
    _,geo,support,_=physics_support();np.testing.assert_array_equal(nodes,support.nodes)
    with np.load(path('material'),allow_pickle=False) as saved:young=saved['young'];skin=saved['skin'].astype(bool)
    with np.load(path('snapshots'),allow_pickle=False) as saved:
        fields=saved['displacement'];steps=saved['steps'];np.testing.assert_array_equal(nodes,saved['nodes'])
    finite_arrays(displacement=fields,steps=steps)
    if fields.shape!=(len(steps),len(nodes),3) or not np.all(np.diff(steps)>0) or not np.all(steps==np.floor(steps)):raise ValueError('Invalid snapshot shape or ordering')
    if config['role']=='fresh-validation':
        allocation=freeze['fresh_validation']
        if config['windows']!=[allocation['history_interval_steps']]:raise ValueError('Fresh window changed after freezing')
        if config.get('energy_windows',config['windows'])!=config['windows']:raise ValueError('Fresh energy window changed after freezing')
        if provenance['frozen_model_receipt_sha256']!=digest(path('freeze_receipt')):raise ValueError('Fresh extraction did not bind this freeze')
        if history_receipt['freeze_sha256']!=digest(path('freeze_receipt')):raise ValueError('Fresh history did not bind this freeze')
        with np.load(path('snapshots'),allow_pickle=False) as saved:
            mask=saved['validation'];fields=fields[mask];steps=steps[mask]
        np.testing.assert_array_equal(steps,allocation['steps'])
    energy_windows=config.get('energy_windows',config['windows'])
    if len(energy_windows)!=len(config['windows']):raise ValueError('One explicit energy interval per response interval required')
    first=min(row[0] for row in energy_windows);last=max(row[1] for row in energy_windows)
    selected=(steps>=first)&(steps<=last)
    projection=omitted_energy(V,K,support,young,skin,fields[selected],steps[selected])
    fine=integrate(M,K,L,back,1.25e-7);coarse=integrate(M,K,L,back,2.5e-7)
    sensitivity=[]
    for case in range(3):
        finer=np.interp(coarse['time'],fine['time'],fine['curves'][:,case]);old=coarse['curves'][:,case]
        sensitivity.append(dict(case=case,peak_relative_difference=float(abs(fine['curves'][:,case].max()/old.max()-1)),
            curve_relative_l2=float(np.linalg.norm(finer-old)/np.linalg.norm(finer)),
            maximum_difference_over_finer_peak=float(np.max(np.abs(finer-old))/fine['curves'][:,case].max())))
    finite_arrays(time_sensitivity=np.asarray([[row[key] for key in ('peak_relative_difference','curve_relative_l2','maximum_difference_over_finer_peak')] for row in sensitivity]))
    history=np.genfromtxt(path('history'),delimiter=',',names=True);windows=[]
    finite_arrays(history=np.column_stack([history[name] for name in history.dtype.names]))
    if not np.all(history['step']==np.floor(history['step'])) or not np.all(np.diff(history['step'])==1) or not np.all(np.diff(history['time_s'])>0):raise ValueError('History must have unique consecutive increasing integer steps')
    if np.min(history['step'])<0 or np.min(history['time_s'])<0 or np.max(history['time_s'])>.002+1e-14:raise ValueError('History outside physical horizon')
    np.testing.assert_allclose(history['time_s'],history['step']*geo['plan']['dt_s'],rtol=0,atol=2e-14)
    for (start,stop),(energy_first,energy_last) in zip(config['windows'],energy_windows):
        if history['step'][0]>start or history['step'][-1]<stop:raise ValueError('Incomplete declared history interval')
        mask=(history['step']>=start)&(history['step']<=stop);reference=history['peak_back_displacement_m'][mask]
        if int(mask.sum())!=stop-start+1:raise ValueError('Incomplete declared history rows')
        if np.max(reference)<=0 or np.linalg.norm(reference)<=0:raise ValueError('Zero reference response normalization')
        prediction=np.interp(history['time_s'][mask],fine['time'],fine['curves'][:,config['case_index']])
        energy_rows=[row['omitted_linear_strain_energy_fraction'] for row in projection if energy_first<=row['step']<=energy_last]
        if not energy_rows:raise ValueError('No energy snapshots in declared window')
        response=dict(peak_relative_error=float(abs(prediction.max()/reference.max()-1)),curve_relative_l2=float(np.linalg.norm(prediction-reference)/np.linalg.norm(reference)),
            maximum_difference_over_reference_peak=float(np.max(np.abs(prediction-reference))/reference.max()))
        finite_arrays(response_metrics=np.asarray(list(response.values())))
        windows.append(dict(first_step=start,last_step=stop,energy_interval_steps=[energy_first,energy_last],response=response,maximum_omitted_energy_fraction=max(energy_rows),
            response_gate_passed=max(response.values())<.05,projection_gate_passed=max(energy_rows)<.05))
    time_pass=max(max(value for key,value in row.items() if key!='case') for row in sensitivity)<.01
    return dict(scope='Frozen rational-basis response and linear-strain-energy assessment; no fitting.',role=config['role'],windows=windows,projection=projection,
        time_sensitivity=sensitivity,time_gate_passed=time_pass,linear_energy_relative_error=fine['energy_relative_error'],
        passed=time_pass and all(row['response_gate_passed'] and row['projection_gate_passed'] for row in windows),
        source_sha256={key:digest(path(key)) for key in ('basis','material','snapshots','history','history_receipt','freeze_receipt')},
        complete2ms_three_case_mpm_qualification=False,global_optimum_claim=False)


def prepare_history(directory,freeze_path,output,development_stop=None):
    directory=Path(directory).resolve();freeze_path=Path(freeze_path).resolve();output=Path(output).resolve()
    if output.exists() or output.with_suffix('.json').exists():raise FileExistsError('Choose fresh history output paths')
    freeze=json.loads(freeze_path.read_text(encoding='utf-8'))
    if freeze.get('schema_version')!=2:raise ValueError('Expected input-bound version2 freeze')
    verify_sources(freeze['source_sha256'])
    status_path=directory/'status.json';status=json.loads(status_path.read_text(encoding='utf-8'))
    if not status.get('valid'):raise ValueError('Source trajectory is not marked valid')
    binding=status['binding'];controls=np.asarray(binding['controls'],dtype=np.float64)
    controls_sha=hashlib.sha256(controls.tobytes()).hexdigest();vertex=freeze['operator_vertex']
    if controls_sha!=freeze['construction_configuration'][vertex+'_controls_sha256'] or binding['source_signature']!=freeze['full_mpm_source_signature']:
        raise ValueError('Source trajectory differs from frozen control/physics binding')
    if freeze['fresh_validation'] is None:
        if development_stop is None or development_stop<1:raise ValueError('Initial development history requires --history-end-step')
        stop=development_stop
    else:
        stop=freeze['fresh_validation']['history_interval_steps'][1]
        if development_stop is not None and development_stop!=stop:raise ValueError('Cannot override fresh history allocation')
    lines=[]
    with (directory/'history.csv').open(encoding='utf-8') as stream:
        lines.append(next(stream))
        for line in stream:
            if not line.strip():continue
            number=int(float(line.split(',')[0]))
            if number>stop:raise ValueError('Declared final step missing from source history')
            lines.append(line)
            if number==stop:break
        else:raise ValueError('Source history has not reached declared final step')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(''.join(lines),encoding='utf-8')
    receipt=dict(freeze_sha256=digest(freeze_path),reference_history_sha256=digest(output),reference_history_maximum_step=stop,
        controls_sha256=controls_sha,case=binding['case'],source_signature=binding['source_signature'],source_status_sha256=digest(status_path))
    output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return dict(scope='History prefix extraction only; no response qualification.',**receipt,passed=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configuration',type=Path,nargs='?')
    parser.add_argument('--check-frozen',nargs=2,type=Path,metavar=('BASIS','RESPONSE'))
    parser.add_argument('--prepare-history',type=Path,metavar='TRAJECTORY_DIRECTORY')
    parser.add_argument('--freeze-receipt',type=Path)
    parser.add_argument('--history-end-step',type=int,help='Explicit stop for an initial-vertex development history only')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if sum(map(bool,(args.configuration,args.check_frozen,args.prepare_history)))!=1:parser.error('Choose one configuration, --check-frozen, or --prepare-history')
    if args.prepare_history:
        if not args.freeze_receipt:parser.error('--prepare-history requires --freeze-receipt')
        report=prepare_history(args.prepare_history,args.freeze_receipt,args.output,args.history_end_step)
        print(json.dumps(report,indent=2));return
    report=check_frozen(*args.check_frozen) if args.check_frozen else assess(args.configuration)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
