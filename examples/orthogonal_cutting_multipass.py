"""Fresh or resumable three-pass Ti64 cutting, with actual event-aligned time.

Run as ``python -m examples.orthogonal_cutting_multipass --help``.
Shared cutting/contact CLI options are accepted by the single-pass driver.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

from . import orthogonal_cutting_ti64_2d as cutting
from .cutting_runtime import _atomic_json, _atomic_npz, _load_state, _state_arrays
from .cutting_path import next_window, clock_tolerance, velocity_clock_tolerance
from .cutting_dynamic_config import runtime_config_factory, validated_scalar

PATH_FILE=Path(__file__).with_name('cutting_three_pass_path.json')

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def valid(state):
    return (bool(state.valid) and bool(np.asarray(state.history.valid).all())
            and all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(state))
            and np.asarray(state.mass).min()>0 and np.asarray(state.volume).min()>0
            and np.linalg.det(np.asarray(state.F)).min()>0)

def aggregate(a,b):
    return a.at[0].add(b[0]).at[1:3].max(b[1:3]).at[3:9].add(b[3:9]).at[9:13].max(b[9:13]).at[13:21].add(b[13:21]).at[21:26].max(b[21:26]).at[26].add(b[26]).at[27:29].max(b[27:29])

def run_path(states,materials,configs,*,output,metadata,step_adapter,
             step_adapter_cache_factory,resume=False,**unused):
    options=run_path.options
    plan=run_path.plan
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    state=states[0];material=materials[0];base=configs[0]
    if step_adapter is None or step_adapter_cache_factory is None:
        raise ValueError('multipass requires particle-grid contact and warm-start cache')
    nominal_dt=float(base.dt)*options.dt_factor
    warm_period=500*float(base.dt)
    capacity=500//options.dt_factor
    tool=base.boundaries[-1]
    definition=json.loads(json.dumps(metadata))
    source_root=Path(__file__).resolve().parents[1]
    definition.setdefault('source_sha256',{}).update({name:hashlib.sha256((source_root/name).read_bytes()).hexdigest()
        for name in ('examples/orthogonal_cutting_multipass.py','examples/cutting_path.py','examples/cutting_dynamic_config.py')})
    definition['multipass']={
        'path':plan,'path_sha256':digest(plan),'nominal_dt_s':nominal_dt,
        'dt_factor':options.dt_factor,'warm_period_s':warm_period,
        'warm_protocol':'physical-time boundaries from t=0; cache preserved across output/segment/restart',
        'time_policy':'actual accumulated State.time; shortened event substeps; no clock snapping',
        'history_policy':'all particle and material history retained; no chip deletion or reset',
        'scope':'exploratory dt4 morphology; not baseline accuracy' if options.dt_factor==4 else 'baseline dt; physical validation pending',
    }
    # Output cadence and bounded stop are operational and do not invalidate a restart.
    for key in ('planned_steps','saved_frames','duration_s','substeps_per_saved_frame'):
        definition.pop(key,None)
    definition['duration_s']=plan['knots'][-1][0]
    definition['planned_tool_travel_m']=sum(max(0.,b[1]-a[1])
        for a,b in zip(plan['knots'],plan['knots'][1:]))
    definition['planned_tool_travel_definition']='sum of positive-x cutting strokes; excludes lifts and returns'
    definition['numerics']['baseline_dt_s']=float(base.dt)
    definition['numerics']['dt_s']=nominal_dt
    definition['numerics']['acoustic_CFL']*=options.dt_factor
    signature=digest(definition)
    frame=0;epoch=1;next_warm=warm_period;reset_pending=True
    cache=step_adapter_cache_factory(state)
    if resume:
        cp=json.loads((output/'checkpoint.json').read_text())
        if cp['schema']!='public-event-aligned-multipass-v1' or cp['signature']!=signature:
            raise ValueError('checkpoint physics/path/cache protocol mismatch; legacy checkpoints require explicit conversion')
        if digest(json.loads((output/'case.json').read_text()))!=signature:
            raise ValueError('saved case metadata signature mismatch')
        state=_load_state(output/cp['states'][0])
        with np.load(output/cp['states'][0]) as archive:cache=jnp.asarray(archive['contact_cache'])
        frame=cp['frame'];epoch=cp['next_warm_reset_epoch']
        next_warm=cp['next_warm_reset_time_s'];reset_pending=cp['warm_reset_pending']
        if not valid(state) or cache.shape!=(len(state.x),5) or not np.isfinite(np.asarray(cache)).all():
            raise ValueError('invalid checkpoint state or contact cache')
    else:
        if (output/'checkpoint.json').exists():raise ValueError('output already contains a checkpoint; use --resume')
        if float(state.time)!=0 or np.any(np.asarray(state.history.damage)) or np.any(np.asarray(state.history.failed)):
            raise ValueError('fresh run must start from virgin t=0 state')
        _atomic_json(output/'case.json',definition)
        _atomic_json(output/'path.json',plan)
    runtime_config=runtime_config_factory(base)
    @jax.jit
    def advance(current,initial_cache,dt,count,reset):
        initial_cache=jnp.where(reset,step_adapter_cache_factory(current),initial_cache)
        def body(carry,index):
            old,k,metrics,bad=carry
            def active(_):
                new,impulse,work,v,new_cache=step_adapter(old,material,runtime_config(dt),cache=k)
                invalid=(~new.valid)|(~jnp.all(new.history.valid))|(v[12]!=0)|(v[13]!=0)|(v[23]!=0)
                return (jax.tree.map(lambda a,b:jnp.where(invalid,a,b),old,new),
                        jnp.where(invalid,k,new_cache),aggregate(metrics,v),invalid)
            return jax.lax.cond((index<count)&~bad,active,lambda _:carry,None),None
        return jax.lax.scan(body,(current,initial_cache,jnp.zeros(29,dtype=current.x.dtype),jnp.array(False)),jnp.arange(capacity))[0]

    def save(segment,window,metrics):
        filename='checkpoint_%d_0.npz'%(frame%2)
        _atomic_npz(output/filename,**_state_arrays(state),contact_cache=np.asarray(cache))
        h=state.history
        _atomic_npz(output/('fields_%05d.npz'%frame),
            particle_id=np.arange(len(state.x)),x=np.asarray(state.x),F=np.asarray(state.F),
            reference_volume=np.asarray(state.volume),time=np.asarray(state.time),
            virtual_tip=np.asarray(tool.tip_position(state.time,state.x.dtype)),
            temperature_K=np.asarray(h.temperature),plastic_strain=np.asarray(h.plastic_strain),
            plastic_rate=np.asarray(h.plastic_rate),
            damage=np.asarray(h.damage),damage_softening=np.asarray(h.damage_softening),
            failed=np.asarray(h.failed))
        row={'frame':frame,'time':float(state.time),'segment':segment,'window':window,
             'virtual_tip_m':np.asarray(tool.tip_position(state.time,state.x.dtype)).tolist(),
             'contact_metrics':np.asarray(metrics).tolist()}
        _atomic_json(output/'checkpoint.json',{
            'schema':'public-event-aligned-multipass-v1','signature':signature,'frame':frame,
            'states':[filename],'rows':[row],'observations':[{'time':float(state.time),'segment':segment}],
            'path_sha256':digest(plan),'warm_reset_origin_time_s':0.,
            'next_warm_reset_epoch':epoch,'next_warm_reset_time_s':next_warm,
            'warm_reset_pending':reset_pending,'contact_cache_in_state':True})
        _atomic_json(output/'status.json',row)
        with (output/'windows.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')

    if not resume:save('initial',{'steps':0},jnp.zeros(29))
    windows=0
    ledger=output/'segment-ledger.json'
    events=json.loads(ledger.read_text()) if resume and ledger.exists() else []
    for i,event in enumerate(events):
        if i>=len(plan['segments']):raise ValueError('event ledger exceeds path')
        segment=plan['segments'][i]
        if (event['segment']!=segment['name'] or event['planned_end_s']!=segment['end_time_s']
                or abs(event['actual_end_s']-segment['end_time_s'])>velocity_clock_tolerance(segment['end_time_s'])
                or event['actual_end_s']>float(state.time)+clock_tolerance(float(state.time))):
            raise ValueError('saved event ledger is inconsistent with path/checkpoint')
    for segment_index,segment in enumerate(plan['segments']):
        if segment_index<len(events):continue
        end=segment['end_time_s'];tol=velocity_clock_tolerance(end)
        if float(state.time)>end+tol:continue
        while end-float(state.time)>tol:
            output_start=float(state.time)
            output_end=min(end,output_start+options.output_baseline_steps*float(base.dt))
            combined=jnp.zeros(29,dtype=state.x.dtype);count=0;chunks=[]
            while output_end-float(state.time)>(tol if output_end==end else clock_tolerance(output_end)):
                target=min(output_end,next_warm)
                velocity_event=abs(target-end)<=clock_tolerance(end)
                if velocity_event:target=end
                window=next_window(float(state.time),target,nominal_dt,capacity,
                    nominal_phase=abs(target-next_warm)<=clock_tolerance(target) and not velocity_event,
                    velocity_event=velocity_event)
                if not window['done']:
                    scalar=validated_scalar(base,window['dt_s'],state.x.dtype)
                    candidate,new_cache,metrics,bad=advance(state,cache,scalar,jnp.asarray(window['steps']),jnp.asarray(reset_pending))
                    jax.block_until_ready(candidate)
                    if bool(bad) or not valid(candidate):
                        _atomic_json(output/'guard-failure.json',{'prior_time_s':float(state.time),'window':window,'metrics':np.asarray(metrics).tolist()})
                        raise RuntimeError('candidate rejected; last accepted checkpoint preserved')
                    values=np.asarray(metrics)
                    if values[2]>.001*base.dx or max(values[21],values[24])>1e-9*base.dx or max(values[22],values[25])>.5*base.dx:
                        raise RuntimeError('contact penetration guard; last accepted checkpoint preserved')
                    state=candidate;cache=new_cache;reset_pending=False
                    combined=aggregate(combined,metrics);count+=window['steps'];chunks.append(window)
                if abs(float(state.time)-next_warm)<=clock_tolerance(next_warm):
                    epoch+=1;next_warm=float(state.time)+warm_period;reset_pending=True
            frame+=1;windows+=1
            save(segment['name'],{'steps':count,'actual_start_s':output_start,'actual_end_s':float(state.time),'chunks':chunks},combined)
            if options.max_output_windows and windows>=options.max_output_windows:return
        events.append({'segment':segment['name'],'planned_end_s':end,'actual_end_s':float(state.time),'endpoint_offset_s':float(state.time)-end})
        _atomic_json(output/'segment-ledger.json',events)
    _atomic_json(output/'stage-complete.json',{'time_s':float(state.time),'completed_segment':events[-1]['segment'],'complete':True})

def main():
    parser=argparse.ArgumentParser(description=__doc__,add_help=False,allow_abbrev=False)
    parser.add_argument('--dt-factor',type=int,choices=(1,4),default=1)
    parser.add_argument('--path',type=Path,default=PATH_FILE)
    parser.add_argument('--output-baseline-steps',type=int,default=5000)
    parser.add_argument('--max-output-windows',type=int)
    options,remaining=parser.parse_known_args()
    if options.output_baseline_steps<1:parser.error('output-baseline-steps must be positive')
    plan=json.loads(options.path.read_text())
    knots=np.asarray(plan['knots'],dtype=float)
    if (knots.ndim!=2 or knots.shape[1]!=3 or len(knots)<2
            or not np.isfinite(knots).all() or knots[0,0]!=0
            or not np.all(np.diff(knots[:,0])>0)
            or len(plan['segments'])!=len(knots)-1):
        parser.error('path must start at zero with finite increasing knots and one segment per interval')
    for i,segment in enumerate(plan['segments']):
        if segment['start_time_s']!=knots[i,0] or segment['end_time_s']!=knots[i+1,0]:
            parser.error('segment times must exactly match the velocity-path knots')
        for key,expected in (('start_virtual_tip_m',knots[i,1:]),('end_virtual_tip_m',knots[i+1,1:])):
            if key in segment and not np.array_equal(np.asarray(segment[key]),expected):
                parser.error('segment endpoints must match actual tool-path knots')
        if 'velocity_m_s' in segment and not np.allclose(segment['velocity_m_s'],(knots[i+1,1:]-knots[i,1:])/(knots[i+1,0]-knots[i,0]),rtol=1e-13,atol=0):
            parser.error('segment velocity must match actual tool-path knots')
    run_path.options=options;run_path.plan=plan
    defaults=['--grid','256','--cut-depth','1e-5','--tool-tip-radius','3e-6','--tool-tip-arc-segments','12',
        '--jc-damage','--fast-jc-return','--particle-grid-contact','--failed-wedge-swept-contact',
        '--particle-contact-max-sweeps','1024','--particle-contact-warm-start']
    sys.argv=[sys.argv[0],*defaults,*remaining]
    cutting.main(runner=run_path,translation_path=tuple(tuple(k) for k in plan['knots']))

if __name__=='__main__':main()
