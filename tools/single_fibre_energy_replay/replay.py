"""Isolated passive replay. Default f2; full requires its exact accepted receipt.

No rendering, material changes, residual correction or original-artifact writes.
"""
import argparse,csv,hashlib,importlib.metadata,json,os,platform,sys,time
from functools import partial
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
BASELINE_SHA='ad1a6c2fbe7470b416e5ec5b8ef93899a10fda2b38cd05e2a2908eca80e47f06'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(p)
def sources():
    paths=[Path(__file__),Path(__file__).with_name('energy_terms.py'),ROOT/'mpm_jax/fibre_mpm.py',ROOT/'mpm_jax/materials/isotropic_brittle.py',ROOT/'examples/single_fibre_tension.py']
    return {p.relative_to(ROOT).as_posix():digest(p) for p in paths}
def inventory(root):return {p.relative_to(root).as_posix():{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(root.rglob('*')) if p.is_file() and p.name!='receipt.json'}

def run(a):
    from control import bind
    control_context=bind()
    import numpy as np
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from energy_terms import energy_terms
    from examples.single_fibre_tension import plan_case,motion
    from mpm_jax.fibre_mpm import FibreState,FibreConfig,build_layout,make_state,reference_grips,step_with_diagnostics
    from mpm_jax.materials.isotropic_brittle import BrittleHistory,isotropic_brittle
    baseline=a.baseline.resolve();out=a.output.resolve()
    if baseline==out or baseline in out.parents or out in baseline.parents:raise ValueError('Original and replay directories must be disjoint')
    if digest(baseline/'receipt.json')!=BASELINE_SHA:raise ValueError('Exact original receipt required')
    original=json.loads((baseline/'receipt.json').read_text())
    if inventory(baseline)!=original['files']:raise ValueError('Original complete raw inventory mismatch')
    for n in ('mpm_jax/fibre_mpm.py','mpm_jax/materials/isotropic_brittle.py'):
        if digest(ROOT/n)!=original['source']['files'][n]:raise ValueError('Original numerical/material source required')
    runtime={'python':platform.python_version(),'packages':{n:importlib.metadata.version(n) for n in original['environment']['packages']}}
    if runtime!=original['environment']:raise ValueError('Exact original runtime required')
    data=baseline/'data';plan=json.loads((data/'plan.json').read_text());geometry,current=plan_case()
    for key in ('steps','dt','frame_steps','particles','geometry','grid','material','protocol','duration_s'):
        if current[key]!=plan[key]:raise ValueError('Original physical plan mismatch: '+key)
    raw={k:np.load(data/f'trajectory_{k}.npy',mmap_mode='r') for k in ('x','damage','failed','speed','temperature')}
    if not np.array_equal(geometry['x'],raw['x'][0]):raise ValueError('Original particle seed differs')
    endpoint=np.load(data/'final_state_0.npz')
    if not np.array_equal(geometry['mass'],endpoint['mass']) or not np.array_equal(geometry['volume'],endpoint['volume']):raise ValueError('Original mass/volume differs')
    material=isotropic_brittle(**plan['material']);grid=plan['grid'];layout=build_layout(geometry['x'],geometry['fibre_id'],dx=grid['dx_m'],padding_cells=6,reference_bounds=grid['bounds_m'],max_nodes=grid['max_nodes'])
    state=make_state(geometry['x'],geometry['volume'],geometry['fibre_id'],material,layout);config=FibreConfig(plan['dt'])
    def speed(t):return (motion(t,plan['protocol'],jnp)-motion(t-config.dt,plan['protocol'],jnp))/config.dt
    def targets(t):return jnp.stack((jnp.array([-speed(t)/2,0.,0.]),jnp.array([speed(t)/2,0.,0.])))
    boundary=reference_grips(geometry['masks'],(lambda t:targets(t)[0],lambda t:targets(t)[1]))
    source=sources();source['control_context']=control_context;step=0;start_frame=0;comparisons=[];sums={};parent=None
    if a.end_frame==60:
        if not a.parent_receipt or digest(out/'receipt.json')!=a.parent_receipt:raise ValueError('Full run requires exact accepted f2 receipt')
        parent=json.loads((out/'receipt.json').read_text())
        if parent['frame']!=2 or not parent['qualification_passed'] or parent['source']!=source or parent['runtime']!=runtime or parent['baseline_receipt']!=BASELINE_SHA:raise ValueError('Rejected or changed f2 cannot continue')
        if inventory(out)!=parent['files']:raise ValueError('f2 complete snapshot changed')
        cp=np.load(out/'checkpoint.npz');state=FibreState(**{k:jnp.asarray(cp[k]) for k in state._fields if k!='history'},history=BrittleHistory(*(jnp.asarray(cp['history_'+k]) for k in state.history._fields)))
        step=int(cp['step']);start_frame=2;comparisons=parent['comparisons'];sums=parent['sums']
        (out/'parent-receipt.json').write_bytes((out/'receipt.json').read_bytes())
    else:
        if out.exists():raise ValueError('Fresh f2 requires new output directory')
        out.mkdir(parents=True)
    save(out/'plan.json',plan);save(out/'runtime.json',runtime)
    fields={'x':np.float64,'damage':np.float64,'failed':np.uint8,'speed':np.float64,'temperature':np.float32}
    arrays={k:(np.load(out/f'trajectory_{k}.npy',mmap_mode='r+') if parent else np.lib.format.open_memmap(out/f'trajectory_{k}.npy',mode='w+',dtype=t,shape=(61,plan['particles'],3) if k=='x' else (61,plan['particles']))) for k,t in fields.items()}
    def record(frame):
        values={'x':np.asarray(state.x),'damage':np.asarray(state.history.damage),'failed':np.asarray(state.history.failed).astype(np.uint8),'speed':np.linalg.norm(np.asarray(state.v),axis=1),'temperature':np.full(plan['particles'],293.15,np.float32)}
        comparison={'frame':frame,'step':step,'bitwise':{},'max_abs_error':{}}
        from mpm_jax.fibre_mpm import stored_energy
        strains=np.linalg.svd(np.asarray(state.F),compute_uv=False)-1
        mu=material.young/(2*(1+material.poisson));lam=material.young*material.poisson/((1+material.poisson)*(1-2*material.poisson))
        independent_u=float(np.sum(np.asarray(state.volume)*(1-np.asarray(state.history.damage))*(.5*lam*strains.sum(1)**2+mu*np.sum(strains**2,axis=1))))
        comparison['independent_numpy_SVD_energy_error_J']=abs(independent_u-float(stored_energy(state,material)))
        if comparison['independent_numpy_SVD_energy_error_J']>1e-12:raise ValueError('Independent saved-frame SVD energy disagrees')
        for k,v in values.items():
            arrays[k][frame]=v;arrays[k].flush();comparison['bitwise'][k]=bool(np.array_equal(v,raw[k][frame]));comparison['max_abs_error'][k]=float(np.max(np.abs(v.astype(float)-raw[k][frame].astype(float))))
        comparisons.append(comparison)
        leaves={k:np.asarray(v) for k,v in state._asdict().items() if k!='history'};leaves.update({'history_'+k:np.asarray(v) for k,v in state.history._asdict().items()});leaves['step']=np.array(step)
        with (out/'checkpoint.tmp').open('wb') as f:np.savez_compressed(f,**leaves)
        (out/'checkpoint.tmp').replace(out/'checkpoint.npz');save(out/'progress.json',{'frame':frame,'step':step,'comparisons':comparisons,'sums':sums,'source':source})
    if not parent:record(0)
    @partial(jax.jit,static_argnums=(1,))
    def block(s,count):
        def body(s,_):
            new,diag=step_with_diagnostics(s,material,layout,config,boundary=boundary)
            terms=energy_terms(s,new,material,layout,config,boundary,targets(new.time))
            return new,terms
        return jax.lax.scan(body,s,None,length=count)
    # Static length keeps only a small 16-step scalar output; particle histories
    # are carried, not stacked. No per-step particle dump or debug callback.
    with (out/'step-energy.csv').open('a' if parent else 'w',newline='') as stream:
        writer=None
        for frame in range(start_frame+1,a.end_frame+1):
            target=plan['frame_steps'][frame]
            while step<target:
                count=min(16,target-step);state,terms=block(state,count);state.x.block_until_ready()
                if not bool(state.valid):raise FloatingPointError('Original solver state invalid')
                host={k:np.asarray(v) for k,v in terms.items()}
                if writer is None:
                    writer=csv.DictWriter(stream,fieldnames=['step','time_s']+list(host))
                    if not parent:writer.writeheader()
                for i in range(count):
                    row={k:float(v[i]) for k,v in host.items()};step+=1
                    if not all(np.isfinite(v) for v in row.values()):raise FloatingPointError('Nonfinite energy diagnostic')
                    scale=max(abs(row['fixture_work_J']),abs(row['split_sum_J']),1e-12)
                    if abs(row['identity_error_J'])>1e-10*scale+1e-18 or abs(row['projection_work_identity_error_J'])>1e-10*scale+1e-18:raise ValueError('Passive energy identity failed')
                    for k,v in row.items():sums[k]=sums.get(k,0.)+v
                    writer.writerow({'step':step,'time_s':step*config.dt,**row})
                stream.flush()
            record(frame);print(json.dumps({'frame':frame,'step':step,'bitwise':comparisons[-1]['bitwise']}),flush=True)
    if {**sources(),'control_context':bind()}!=source:raise ValueError('Source changed during replay')
    bitwise=all(all(c['bitwise'].values()) for c in comparisons)
    # This is diagnostic qualification, NEVER physical-energy acceptance.
    cumulative=[(0.,0.)];w=k=u=d=0.
    frame_steps=set(plan['frame_steps'])
    for row in csv.DictReader((out/'step-energy.csv').open()):
        w+=float(row['fixture_work_J']);d+=float(row['original_damage_loss_J'])
        if int(row['step']) in frame_steps:cumulative.append((w,w-float(row['kinetic_energy_J'])-float(row['elastic_energy_J'])-d))
    scale=max(max(abs(x[0]) for x in cumulative),1e-20);ratio=max(abs(x[1]) for x in cumulative)/scale
    receipt={'kind':'passive-energy-replay','frame':a.end_frame,'baseline_receipt':BASELINE_SHA,'source':source,'runtime':runtime,'sums':sums,'comparisons':comparisons,'qualification_passed':bitwise,'original_15percent_gate_passed':ratio<=.15,'original_residual_fraction_max':ratio,'public_media_authorized':False,'note':'No losses are subtracted from original energy gate. Full may diagnose failure while qualification passes.'}
    if a.end_frame==60:
        receipt['final_state_bitwise']={key:bool(np.array_equal(np.asarray(getattr(state,key)),endpoint[key])) for key in ('x','v','F','C','mass','volume','tau')}
        receipt['final_history_bitwise']={key:bool(np.array_equal(np.asarray(getattr(state.history,key)),endpoint['history_'+key])) for key in state.history._fields}
        receipt['qualification_passed']=bitwise and all(receipt['final_state_bitwise'].values()) and all(receipt['final_history_bitwise'].values())
    receipt['files']=inventory(out);save(out/'receipt.json',receipt)
    print(json.dumps({'receipt_sha256':digest(out/'receipt.json'),'qualification_passed':receipt['qualification_passed'],'original_15percent_gate_passed':receipt['original_15percent_gate_passed']}),flush=True)
    if not receipt['qualification_passed']:raise ValueError('Trajectory/final state differs bitwise: preserve diagnostic raw; no continuation')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--end-frame',type=int,choices=(2,60),default=2);p.add_argument('--parent-receipt',default='');a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():save(a.output/'failure.json',{'never_resumable':True,'error':type(error).__name__+': '+str(error)})
        raise
