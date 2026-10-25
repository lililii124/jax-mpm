"""Authenticated bounded125--135ns dual-carry diagnostic; never public media."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
NEW_SOURCE=('examples/metal_fragment_v6_geometric125.py',
    'mpm_jax/fragment_v6_geometric_delta_candidate.py',
    'mpm_jax/fragment_v6_geometric_delta_diagnostic.py',
    'mpm_jax/fragment_v6_geometric_kinematics.py',
    'tools/prove_metal_geometric_delta.py','tools/run_metal_v6_geometric125.py',
    'tools/compare_metal_v6_geometric125.py','tools/metal_v6_process_guard.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_proof(path):
    proof=json.loads(path.read_text())
    if proof.get('passed') is not True or proof.get('impact_started') is not False:
        raise ValueError('Passing analytic-only proof required before impact')
    for name,digest in proof['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Analytically qualified source changed: '+name)
    required={'actual_full_step_force_free_affine','objective_rotation_material','affine_work_time_quadrature',
        'virtual_work_and_affine_acceleration','general_zero_force_two_cell_crossing',
        'actual_full_step_rigid_two_cell_crossing','analytic_contact','actual_full_step_contact','actual_nonzero_stress_work'}
    if not required<=set(proof['checks']):raise ValueError('Missing actual risk benchmark')
    return proof


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--admission',type=Path,required=True)
    p.add_argument('--analytic-proof',type=Path,required=True)
    p.add_argument('--prepare-source-admission',type=Path)
    p.add_argument('--source-admission',type=Path)
    p.add_argument('--audit',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--cfl',type=float,choices=(.1,.05,.025),default=.1)
    args=p.parse_args()
    admission=json.loads(args.admission.read_text());proof=verify_proof(args.analytic_proof)
    for name,digest in admission['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen original source changed: '+name)
    names=sorted(set(admission['source_sha256'])|set(NEW_SOURCE))
    frozen=dict(schema='geometric-delta125-135-prospective-v1',
        source_sha256={name:sha(ROOT/name) for name in names},
        parent_admission_sha256=sha(args.admission),analytic_proof_sha256=sha(args.analytic_proof),
        seed_initialization='All original fields unchanged; C_transfer aliases C; independent L initialized from original physical-gradient C bytes.',
        energy_roundoff_J=admission['energy_roundoff_J'],start_ns=125,stop_ns=135,
        cfl_levels=[.1,.05,.025],max_steps=512,production_restart_allowed=False,public_media_allowed=False)
    if args.prepare_source_admission is not None:
        if args.prepare_source_admission.exists():raise FileExistsError('Preserve earlier source freeze')
        args.prepare_source_admission.parent.mkdir(parents=True,exist_ok=True)
        args.prepare_source_admission.write_text(json.dumps(frozen,indent=2)+'\n')
        print('Prepared source-bound diagnostic admission; no impact started')
        return
    if args.source_admission is None or json.loads(args.source_admission.read_text())!=frozen:
        raise ValueError('Exact reviewed prospective source admission required')
    if args.audit is None or args.output is None:raise ValueError('Explicit audited raw and fresh output required')
    if os.name!='posix':raise RuntimeError('Impact requires the original Linux cloud runtime; analytic proofs may run locally')
    if args.output.exists():raise FileExistsError('Preserve previous raw output')
    args.output.mkdir(parents=True)
    for name,digest in admission['input_sha256'].items():
        if sha(args.audit/name)!=digest:raise ValueError('Original audited input differs: '+name)
    shutil.copyfile(args.admission,args.output/'original-admission.json')
    shutil.copyfile(args.source_admission,args.output/'source-admission.json')
    shutil.copyfile(args.analytic_proof,args.output/'analytic-proof.json')
    shutil.copyfile(args.audit/'result/dt1/epoch-initial.npz',args.output/'original-epoch-initial.npz')
    shutil.copyfile(args.audit/'result/dt1/grid-spec.json',args.output/'original-grid-spec.json')
    for name in frozen['source_sha256']:
        target=args.output/'runtime-source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    import jax,jaxlib
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as j
    from mpm_jax import fragment_multifield_v6_mpm as original
    from mpm_jax import fragment_v6_geometric_delta_candidate as candidate
    from mpm_jax.fragment_v6_geometric_delta_diagnostic import advance,replay
    from mpm_jax.materials.linear_metal import linear_metal
    from tools.audit_metal_onset_state import audit_state_pair
    expected=json.loads((args.audit/'result/dt1/equivalence.json').read_text())
    runtime=dict(python=platform.python_version(),numpy=np.__version__,jax=jax.__version__,jaxlib=jaxlib.__version__,backend=jax.default_backend(),x64=bool(jax.config.jax_enable_x64))
    if runtime!=expected['runtime'] or any(runtime[name]!=value for name,value in proof['runtime'].items()):
        raise ValueError('Exact original and analytically qualified runtime required')
    with np.load(args.audit/'result/dt1/epoch-initial.npz',allow_pickle=False) as cp:
        fields={name:cp[name].copy() for name in cp.files}
    state=original.ImpactState(*(j.asarray(fields[name]) for name in original.ImpactState._fields))
    if float(state.time)!=125000*1e-12 or not bool(state.valid):raise ValueError('Valid exact125ns seed required')
    x=fields['x'];labels=fields['body_id'];reference=fields['reference'];physical=fields['physical_body'];dx=.00025
    bodies=np.array([np.unique(physical[labels==i]).item() for i in range(int(labels.max())+1)])
    predicted=x+125000*1e-12*fields['v'];lower=np.full((len(bodies),3),np.inf);upper=np.full_like(lower,-np.inf)
    np.minimum.at(lower,labels,np.minimum(x,predicted));np.maximum.at(upper,labels,np.maximum(x,predicted))
    bounds=np.stack((lower-2*dx,upper+2*dx),axis=-1)
    grid=original.build_grid(x,labels,dx=dx,padding_cells=3,reference_bounds=bounds,max_nodes=2_000_000,
        field_physical_body=bodies,contact_sweeps=(x,predicted))
    spec=json.loads((args.audit/'result/dt1/grid-spec.json').read_text())
    if grid.layout.node_count!=spec['nodes'] or len(grid.pairs)!=spec['pairs'] or hashlib.sha256(np.asarray(grid.pairs).tobytes()).hexdigest()!=spec['pairs_ordered_bytes_sha256']:
        raise ValueError('Exact original grid/pair topology required')
    fixed=(physical==1)&((abs(reference[:,0])>=.009)|(abs(reference[:,2])>=.009))
    support=original.stencil(state.x,state.body_id,grid.layout);mask=np.zeros((grid.layout.node_count,3),bool)
    nodes=np.asarray(support.indices)[fixed].ravel();nodes=nodes[(nodes>=0)&(nodes<grid.layout.node_count)];mask[nodes]=True
    grid=replace(grid,clamped=j.asarray(mask));material=linear_metal(**admission['material'])
    config=original.ImpactConfig(cfl=args.cfl,contact_iterations=1024,contact_tolerance=.005)
    original.validate_state(state,material,grid)
    initial=candidate.initialize(state)
    candidate.save_checkpoint(args.output/'initial.npz',initial)
    (args.output/'source-runtime.json').write_text(json.dumps(dict(source_admission=frozen,runtime=runtime,
        cfl=args.cfl,original_seed_sha256=sha(args.audit/'result/dt1/epoch-initial.npz'),
        independent_L_initial_sha256=hashlib.sha256(np.asarray(initial.L_physical).tobytes()).hexdigest()),indent=2)+'\n')
    checkpoints=args.output/'all-step-checkpoints';checkpoints.mkdir()
    def save(index,carry,diagnostics,extra):candidate.save_checkpoint(checkpoints/f'step-{index:04d}.npz',carry)
    final,rows,failure=advance(initial,material,grid,config,135000*1e-12,
        energy_roundoff_J=admission['energy_roundoff_J'],max_steps=512,save=save)
    candidate.save_checkpoint(args.output/'candidate-final.npz',final)
    dt=np.array([row['dt_s'] for row in rows]);d=rows[0]['diagnostics'] if rows else None
    if not rows:raise RuntimeError('No real update; raw seed preserved')
    values={name:np.stack([np.asarray(getattr(row['diagnostics'],name)) for row in rows]) for name in d._fields}
    values.update({'geometry_'+name:np.stack([np.asarray(getattr(row['geometry'],name)) for row in rows]) for name in rows[0]['geometry']._fields})
    np.savez(args.output/'step-diagnostics.npz',dt_s=dt,**values)
    endpoint=float(final.state.time)==135000*1e-12
    result=dict(scope='Own-method temporal convergence diagnostic; old-APIC shape spread is not an accuracy definition.',
        cfl=args.cfl,steps=len(rows),failure=failure,endpoint=endpoint,all_step_gates=all(all(row['gates'].values()) for row in rows),
        production_restart_allowed=False,public_media_allowed=False)
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    if failure is not None:raise RuntimeError('First candidate rejection retained; do not continue invalid carry')
    replayed=replay(initial,material,grid,config,dt,energy_roundoff_J=admission['energy_roundoff_J'])
    candidate.save_checkpoint(args.output/'replay-final.npz',replayed)
    same=all(np.asarray(a).tobytes()==np.asarray(b).tobytes() for a,b in zip(final.state,replayed.state)) and np.asarray(final.L_physical).tobytes()==np.asarray(replayed.L_physical).tobytes()
    original.validate_state(final.state,material,grid)
    audit=audit_state_pair(state,final.state,reference,fields['edges'],fields['alive'],physical,dx)
    numerical=float(final.state.numerical_reservoir-state.numerical_reservoir)
    parts=values['p2g_dissipation']+values['g2p_dissipation']+values['contact_compatible_energy']+values['viscosity_work']+values['constitutive_work_defect']
    result.update(replay_all_state_and_L_bitwise=same,physical_audit=audit,
        numerical_increment_J=numerical,maximum_channel_identity_J=float(np.max(abs(values['numerical_energy_increment']-parts))),
        reservoir_sum_error_J=abs(numerical-float(values['numerical_energy_increment'].sum())),
        final_ledger_J=np.asarray(original.energy_ledger(final.state,grid)).tolist(),
        all_input_unchanged=all(sha(args.audit/name)==digest for name,digest in admission['input_sha256'].items()))
    result['bounded_diagnostic_passed']=bool(same and audit['mass_exact'] and result['maximum_channel_identity_J']<=admission['energy_roundoff_J'] and result['reservoir_sum_error_J']<=admission['energy_roundoff_J'] and result['all_input_unchanged'])
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if not result['bounded_diagnostic_passed']:raise RuntimeError('Bounded diagnostic failed; preserve whole raw closure')


if __name__=='__main__':main()
