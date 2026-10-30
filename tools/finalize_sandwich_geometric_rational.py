"""Finalize a complete receipt from hash-bound second-order direction caches.

This does not alter or bless the incomplete v2 receipt. It checks all cached
physical inputs and direction provenance, reconstructs the reduced operators,
replays their response, and freezes a new receipt only before the allocated
71000--83000 MPM window is reached.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, shutil
import numpy as np
from scipy import sparse

Q='94934d1d5792131b22756f3e2634d25708de07ade1a0ff37610881f8bb1b6ec4'
PARENT_SHA='f3a30fffa434aa386e2e351fa0ed107a6291cd7e79f0bb7a8672f88aed2173b6'
V2_SHA='a6af7bde3faa80dedcdb1fa0f6a3b68360f1d7d75198fb78d8025a1763b0a323'
V2_ARCHIVE_SHA='64f043763fc91cc50597beebe70e77f2ac2db6d1f0b1c0a3a2719329836c29a2'
FRESH=list(range(71000,83001,1000));CASES=('uniform','central','eccentric')
V2_SOURCE='evidence/blast-sandwich-reduced/vertex184/second-order-v2-worker.py'
FINALIZER_SOURCE='tools/finalize_sandwich_geometric_rational.py'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')

def source_heads(paths,signature):
    rows=[]
    for case in CASES:
        p=Path(paths[case]);status=json.loads(p.read_text(encoding='utf-8'));binding=status['binding']
        controls=np.asarray(binding['controls'],dtype=np.float64)
        qsha=hashlib.sha256(controls.tobytes()).hexdigest()
        if not status.get('valid') or binding.get('case')!=case or binding.get('source_signature')!=signature or qsha!=Q:
            raise ValueError('Source status/control binding mismatch: '+case)
        step=int(status['step'])
        if step>=FRESH[0]:raise RuntimeError('Allocated fresh window was reached before new receipt freeze: '+case)
        rows.append(dict(case=case,step=step,time_s=float(status['time_s']),status_sha256=sha(p)))
    return rows

def run(config_path):
    cfg=json.loads(Path(config_path).read_text(encoding='utf-8'))
    extension_root=Path(cfg['extension_root']);frozen=Path(cfg['frozen_dir'])
    cache=Path(cfg['cache_root']);out=Path(cfg['output_dir']);mpm_status_paths=cfg['mpm_status_paths']
    if out.exists() and any(out.iterdir()):raise FileExistsError('Finalizer output must be new')
    out.mkdir(parents=True,exist_ok=True)
    source_manifest=json.loads(Path('/work/source-sha256.json').read_text(encoding='utf-8'))
    for name,value in source_manifest.items():
        if sha(Path('/work')/name)!=value:raise ValueError('Finalizer task source hash mismatch: '+name)
    if source_manifest.get('worker.py')!=sha(Path(__file__)):
        raise ValueError('Finalizer worker is not the source-pinned file')
    if source_manifest.get(FINALIZER_SOURCE)!=sha(Path('/work')/FINALIZER_SOURCE):
        raise ValueError('Finalizer source snapshot is not pinned')
    if source_manifest.get(V2_SOURCE)!=sha(Path('/work')/V2_SOURCE):
        raise ValueError('Cached construction worker snapshot is not pinned')
    import sys
    sys.path.insert(0,str(extension_root))
    from examples.blast_sandwich_design import Settings,source_signature
    from tools.build_sandwich_rational import physics_support,material_fields_digest
    from tools.assess_sandwich_rational import integrate

    oldfreeze_path=frozen/'model-freeze.json';parent=json.loads(oldfreeze_path.read_text())
    if sha(oldfreeze_path)!=PARENT_SHA or parent['basis']['rank']!=184:raise ValueError('Parent freeze mismatch')
    for name,value in parent['source_sha256'].items():
        if sha(extension_root/name)!=value:raise ValueError('Parent source pin mismatch: '+name)
    v2freeze_path=cache/'output/model-freeze.json';v2freeze=json.loads(v2freeze_path.read_text())
    if sha(v2freeze_path)!=V2_SHA or v2freeze['basis']['rank']!=191:
        raise ValueError('Superseded v2 candidate receipt changed')
    if v2freeze.get('construction_configuration',{}).get('selected_controls_sha256'):
        raise ValueError('Expected original v2 incomplete receipt; do not silently replace it')
    if v2freeze['parent_freeze_sha256']!=PARENT_SHA or v2freeze['selected_controls_sha256']!=Q:
        raise ValueError('v2 cache belongs to a different parent/control')
    if v2freeze['inputs']['selected_material']['sha256']!=parent['inputs']['selected_material']['sha256']:
        raise ValueError('Cached selected material differs from parent')
    if v2freeze['inputs']['selected_controls']['sha256']!=parent['inputs']['selected_controls']['sha256']:
        raise ValueError('Cached controls differ from parent')
    if v2freeze['inputs']['selected_tangent']['sha256']!=parent['inputs']['selected_tangent']['sha256']:
        raise ValueError('Cached tangent differs from parent')
    v2job=json.loads((cache/'job/status.json').read_text())
    if v2job.get('state')!='complete' or v2job.get('returncode')!=0 or v2job.get('guard_reason'):
        raise ValueError('v2 cache task was not a clean terminal completion')
    upload=json.loads((cache/'upload.json').read_text())
    if upload.get('archive_sha256')!=V2_ARCHIVE_SHA or sha(cache/'input.tar')!=V2_ARCHIVE_SHA:
        raise ValueError('Cached v2 task archive hash mismatch')
    source_manifest=json.loads((cache/'source-sha256.json').read_text())
    v2_tool_sha=v2freeze['source_sha256']['tools/extend_sandwich_geometric_rational.py']
    if source_manifest.get('worker.py')!=v2_tool_sha or sha(V2_SOURCE)!=v2_tool_sha:
        raise ValueError('Cached v2 physical construction source is not byte-bound')
    v2policy_path=cache/'output/construction-policy.json';v2policy=json.loads(v2policy_path.read_text())
    if v2policy['parent_freeze_sha256']!=PARENT_SHA or v2policy['controls_sha256']!=Q:
        raise ValueError('Cached construction policy input mismatch')
    if len(v2policy['static_solve_receipts'])!=3 or len(v2policy['second_order_solve_receipts'])!=6:
        raise ValueError('Cached physical solve receipts incomplete')
    if max(row['relative_residual'] for row in v2policy['static_solve_receipts']+
           v2policy['second_order_solve_receipts'])>=1e-8:
        raise ValueError('Cached static solve residual exceeds pinned tolerance')
    if max(row['relative_error_vs_negative_Kd'] for row in v2policy['linear_tangent_checks'])>=1e-8:
        raise ValueError('Cached internal-force tangent/sign check failed')
    if max(row['relative_error'] for row in v2policy['finite_difference_checks'])>=.02:
        raise ValueError('Cached central-difference JVP check failed')
    cache_basis_path=cache/'output/frozen-basis.npz';cache_response_path=cache/'output/response.npz'
    if sha(cache_basis_path)!=v2freeze['basis']['sha256'] or sha(cache_response_path)!=v2freeze['response']['sha256']:
        raise ValueError('Cached candidate artifacts differ from v2 receipt')

    settings,geo,support,namespace=physics_support()
    settings=Settings(xy_step=.01,z_step=.001,controls=(24,24,3),stage='fixed-mass')
    signature,_=source_signature(settings)
    if signature!=parent['full_mpm_source_signature']:raise ValueError('Full-MPM physical signature changed')
    preheads=source_heads(mpm_status_paths,signature)
    old_preheads=v2policy['source_heads_before_construction']
    if any(row['step']>=FRESH[0] for row in old_preheads):raise ValueError('Cached direction construction began after window')
    predecl=json.loads(Path('/work/candidate-predeclaration.json').read_text(encoding='utf-8'))
    if predecl['finalizer_source_sha256']!=sha(Path(__file__)):
        raise ValueError('Finalizer source does not match predeclaration')
    if predecl['superseded_incomplete_v2']['freeze_sha256']!=V2_SHA:
        raise ValueError('Superseded v2 receipt is not the declared source')
    predeclaration=dict(input=predecl,input_sha256=sha('/work/candidate-predeclaration.json'),
        predeclared_heads=preheads,cached_direction_task=cfg['cache_task'],
        cached_direction_archive_sha256=V2_ARCHIVE_SHA,cached_direction_freeze_sha256=V2_SHA)
    dump(out/'predeclaration.json',predeclaration)

    with np.load(frozen/'frozen-basis.npz',allow_pickle=False) as f:
        old=f['basis'].copy();nodes=f['nodes'].copy()
    with np.load(frozen/'material.npz',allow_pickle=False) as f:
        density=f['density'].copy();young=f['young'].copy();skin=f['skin'].astype(bool).copy()
    if sha(frozen/'material.npz')!=parent['inputs']['selected_material']['sha256'] or \
       material_fields_digest(frozen/'material.npz')!=parent['material_fields_sha256']['selected']:
        raise ValueError('Parent selected material hash/fields mismatch')
    if sha(frozen/'tangent.npz')!=parent['inputs']['selected_tangent']['sha256']:
        raise ValueError('Parent selected tangent hash mismatch')
    with np.load(frozen/'controls.npz',allow_pickle=False) as f:q=f['q'].copy()
    if hashlib.sha256(q.tobytes()).hexdigest()!=Q:raise ValueError('q94934 controls hash mismatch')
    with np.load(cache_basis_path,allow_pickle=False) as f:
        cached_basis=f['basis'].copy();cached_nodes=f['nodes'].copy()
        cached_M=f['mass'].copy();cached_K=f['stiffness'].copy();cached_L=f['loads'].copy();cached_back=f['back'].copy()
        static_dirs=f['static_directions'].copy();second_dirs=f['second_order_directions'].copy()
    np.testing.assert_array_equal(cached_nodes,nodes);np.testing.assert_array_equal(nodes,support.nodes)
    if cached_basis.shape!=(len(nodes),3,191) or static_dirs.shape!=(len(nodes),3,3) or second_dirs.shape!=(len(nodes),3,6):
        raise ValueError('Cached direction/basis dimensions mismatch')

    Kfull=sparse.load_npz(frozen/'tangent.npz').tocsr()
    _,_,fixed=namespace['clamped_basis'](nodes,settings,2)
    ids=np.flatnonzero(np.repeat(~fixed,3));free=np.repeat(~fixed,3)
    C=old.reshape(-1,old.shape[-1])[ids]
    rho0=np.where(geo['skin'],2700.,2700.*settings.target_phi)
    mass0=np.zeros(len(nodes));np.add.at(mass0,support.indices.ravel(),
        (support.weights*(support.volume*rho0)[:,None]).ravel())
    if abs(mass0.sum()/1.8468-1.)>1e-10:raise ValueError('Initial M0 mass mismatch')
    mf0=np.repeat(mass0,3)[ids]
    if np.max(np.abs(C.T@(mf0[:,None]*C)-np.eye(C.shape[1])))>1e-9:
        raise ValueError('Parent span is not M0-orthonormal')
    basis_free=cached_basis.reshape(-1,191)[ids]
    gram=basis_free.T@(mf0[:,None]*basis_free)
    gram_error=float(np.max(np.abs(gram-np.eye(191))))
    if gram_error>1e-10:raise ValueError('Cached candidate basis lost M0 orthogonality')
    lost=C-basis_free@(basis_free.T@(mf0[:,None]*C))
    parent_span_residual=float(np.linalg.norm(np.sqrt(mf0)[:,None]*lost))
    if parent_span_residual>1e-10:raise ValueError('Cached candidate did not retain the entire parent span')
    cache_directions=np.column_stack((static_dirs.reshape(-1,3)[ids],second_dirs.reshape(-1,6)[ids]))
    direction_residual=cache_directions-basis_free@(basis_free.T@(mf0[:,None]*cache_directions))
    direction_residual_norm=float(np.linalg.norm(np.sqrt(mf0)[:,None]*direction_residual))
    direction_norm=float(np.linalg.norm(np.sqrt(mf0)[:,None]*cache_directions))
    if direction_residual_norm/max(direction_norm,np.finfo(float).tiny)>1e-8:
        raise ValueError('Frozen basis does not span its cached static/second-order directions')

    massq=np.zeros(len(nodes));np.add.at(massq,support.indices.ravel(),
        (support.weights*(support.volume*density)[:,None]).ravel())
    if abs(massq.sum()/1.8468-1.)>1e-10:raise ValueError('Selected mass changed')
    mfq=np.repeat(massq,3)[ids];Kfree=Kfull[ids][:,ids].tocsr()
    Mr=basis_free.T@(mfq[:,None]*basis_free);Kr=basis_free.T@(Kfree@basis_free);Kr=.5*(Kr+Kr.T)
    top_ids,top_weights=namespace['surface_weights'](support,geo['top']);loads=np.zeros((len(nodes),3,3))
    for case in range(3):np.add.at(loads[:,2,case],top_ids.ravel(),
        (-geo['face_area'][:,None]*geo['footprints'][case,:,None]*top_weights).ravel())
    Lr=basis_free.T@loads.reshape(-1,3)[ids]
    basis_nodes=np.zeros((len(nodes)*3,191));basis_nodes[ids]=basis_free;basis_nodes=basis_nodes.reshape(len(nodes),3,191)
    bottom_ids,bottom_weights=namespace['surface_weights'](support,geo['bottom'])
    back=np.einsum('pk,pkr->pr',bottom_weights,basis_nodes[bottom_ids,2,:])
    matrix_errors=dict(M=float(np.max(np.abs(Mr-cached_M))/max(np.max(np.abs(Mr)),1.)),
        K=float(np.max(np.abs(Kr-cached_K))/max(np.max(np.abs(Kr)),1.)),
        loads=float(np.max(np.abs(Lr-cached_L))/max(np.max(np.abs(Lr)),1.)),
        back=float(np.max(np.abs(back-cached_back))/max(np.max(np.abs(back)),1.)))
    if max(matrix_errors.values())>1e-10:raise ValueError('Cached reduced operators do not reproduce from frozen inputs')
    fine=integrate(Mr,Kr,Lr,back,1.25e-7);coarse=integrate(Mr,Kr,Lr,back,2.5e-7)
    with np.load(cache_response_path,allow_pickle=False) as r:
        for name,value in [('time',fine['time']),('curves',fine['curves']),('displacement',fine['displacement']),('velocity',fine['velocity']),('energy',fine['energy'])]:
            if not np.allclose(value,r[name],rtol=1e-12,atol=1e-15):raise ValueError('Cached response mismatch: '+name)
    if fine['energy_relative_error']>=1e-8:raise ValueError('Linear response energy balance failed')
    sensitivity=[]
    for case in range(3):
        finer=np.interp(coarse['time'],fine['time'],fine['curves'][:,case]);older=coarse['curves'][:,case]
        err=dict(peak_relative_difference=float(abs(fine['curves'][:,case].max()/older.max()-1)),
            curve_relative_l2=float(np.linalg.norm(finer-older)/np.linalg.norm(finer)),
            maximum_difference_over_finer_peak=float(np.max(np.abs(finer-older))/fine['curves'][:,case].max()))
        if max(err.values())>=.01:raise ValueError('Reduced time sensitivity exceeded 1%')
        sensitivity.append(dict(case=CASES[case],**err))

    # Copy validated immutable artifacts; the original v2 receipt is untouched.
    shutil.copy2(cache_basis_path,out/'frozen-basis.npz');shutil.copy2(cache_response_path,out/'response.npz')
    policy=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),
        method='Rank-191 q94934 basis: retained rank-184 span plus three public-operator unit-load static modes and six Neo-Hookean second-order modes from hash-bound v2 direction caches.',
        parent_freeze_sha256=PARENT_SHA,controls_sha256=Q,fresh_steps=FRESH,
        response_gate=.05,omitted_energy_gate=.05,source_heads_before_construction=preheads,
        source_heads_before_freeze=source_heads(mpm_status_paths,signature),
        direction_cache=dict(task=cfg['cache_task'],archive_sha256=V2_ARCHIVE_SHA,
            incomplete_v2_freeze_sha256=V2_SHA,basis_sha256=sha(cache_basis_path),
            response_sha256=sha(cache_response_path),construction_policy_sha256=sha(v2policy_path),
            source_manifest_sha256=sha(cache/'source-sha256.json'),physical_builder_sha256=v2_tool_sha,
            cached_static_count=3,cached_second_order_count=6,
            static_solve_receipts=v2policy['static_solve_receipts'],
            second_order_solve_receipts=v2policy['second_order_solve_receipts'],
            linear_tangent_checks=v2policy['linear_tangent_checks'],
            finite_difference_checks=v2policy['finite_difference_checks']),
        M0_mass_kg=float(mass0.sum()),selected_mass_kg=float(massq.sum()),
        M0_gram_max_abs_error=gram_error,parent_span_residual_norm=parent_span_residual,
        cached_direction_span_residual_norm=direction_residual_norm,
        reduced_operator_relative_errors=matrix_errors,time_sensitivity=sensitivity,
        linear_energy_relative_error=fine['energy_relative_error'],no_mpm_displacement_data_read=True)
    dump(out/'construction-policy.json',policy)
    # Check all three source heads again after the actual new freeze is ready.
    postheads=source_heads(mpm_status_paths,signature)
    if any(row['step']>=FRESH[0] for row in postheads):
        dump(out/'no-freeze-window-consumed.json',dict(reason='At least one declared validation source reached step 71000 before final receipt creation.',
            source_heads=postheads,cache_freeze_sha256=V2_SHA,cached_basis_sha256=sha(cache_basis_path)))
        print(json.dumps(dict(stage='candidate-finalized-no-freeze-window-consumed',source_heads=postheads)),flush=True);return
    finalizer_sha=sha(Path(__file__))
    source_pins=dict(parent['source_sha256'])
    source_pins[V2_SOURCE]=sha(V2_SOURCE);source_pins[FINALIZER_SOURCE]=finalizer_sha
    receipt=dict(schema_version=2,scope=policy['method'],created_utc=datetime.now(timezone.utc).isoformat(),
        operator_vertex='selected',basis=dict(path='frozen-basis.npz',sha256=sha(out/'frozen-basis.npz'),
            numeric_basis_sha256=hashlib.sha256(basis_nodes.tobytes()).hexdigest(),rank=191),
        parent_basis=parent['basis'],parent_freeze_sha256=PARENT_SHA,
        response=dict(path='response.npz',sha256=sha(out/'response.npz')),selected_controls_sha256=Q,
        inputs={key:dict(path=name,sha256=sha(frozen/name)) for key,name in [
            ('selected_material','material.npz'),('selected_controls','controls.npz'),('selected_tangent','tangent.npz')]},
        material_fields_sha256=parent['material_fields_sha256'],source_sha256=source_pins,
        full_mpm_source_signature=signature,
        construction_configuration=dict(selected_controls_sha256=Q,nominal_dt_s=1.25e-7,horizon_s=.002,
            retained_parent_span=True,added_static_directions=3,added_second_order_directions=6,
            mpm_snapshot_directions=False,direction_cache_task=cfg['cache_task'],
            direction_cache_archive_sha256=V2_ARCHIVE_SHA,direction_cache_basis_sha256=sha(cache_basis_path),
            superseded_incomplete_v2_freeze_sha256=V2_SHA,
            construction_policy_sha256=sha(out/'construction-policy.json'),predeclaration_sha256=sha(out/'predeclaration.json')),
        fresh_validation=dict(steps=FRESH,history_interval_steps=[FRESH[0],FRESH[-1]],response_gate=.05,omitted_energy_gate=.05),
        actual_mpm_heads_before_fresh_window=postheads,time_sensitivity=sensitivity,
        linear_energy_relative_error=fine['energy_relative_error'],complete2ms_three_case_mpm_qualification=False,
        global_optimum_claim=False)
    required_configuration=dict(selected_controls_sha256=Q,nominal_dt_s=1.25e-7,horizon_s=.002)
    if any(receipt['construction_configuration'].get(key)!=value for key,value in required_configuration.items()):
        raise ValueError('Required optimizer configuration fields are missing or inconsistent')
    if receipt['fresh_validation']['steps']!=FRESH or receipt['fresh_validation']['response_gate']!=.05 or \
       receipt['fresh_validation']['omitted_energy_gate']!=.05:
        raise ValueError('Fresh allocation or original 5% gates changed')
    if receipt['basis']['rank']!=191 or receipt['parent_basis']['rank']!=184 or gram_error>1e-10 or parent_span_residual>1e-10:
        raise ValueError('Frozen rank/M0 invariants failed pre-write validation')
    dump(out/'model-freeze.json',receipt)
    print(json.dumps(dict(stage='frozen',rank=191,freeze_sha256=sha(out/'model-freeze.json'),source_heads=postheads,
        operator_reconstruction=matrix_errors,response_energy_error=fine['energy_relative_error'])),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('/work/run-config.json'))
    run(parser.parse_args().config)
