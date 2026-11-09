"""Replay a saved DCB state with transactional time-step reduction."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import Config,MovingVelocityBox,VelocityBox,block,neo_hookean
from mpm_jax.adaptive_crack import retry_step
from mpm_jax.cohesive_crack import CohesiveLaw,elastic_energy,kinetic_energy
from mpm_jax.extrinsic_crack import initialize,insert_tip,segment_positions,surface_response


def load_saved(run,frame):
    info=json.loads((run/'input.json').read_text())
    rows=list(csv.DictReader((run/'history.csv').open()))
    row=rows[frame]
    if info['case'] not in ('Localized double-cantilever cohesive insertion','bending','tension','mixed') or row['valid']!='True':
        raise ValueError('Require a valid saved cohesive state')
    boundaries=[]
    for entry in info['boundary_conditions']:
        values={k:tuple(v) if isinstance(v,list) else v for k,v in entry.items()}
        boundary_type=values.pop('type',None)
        if values.get('end') is None:
            values['end']=float('inf')
        if boundary_type not in (None,'MovingVelocityBox','VelocityBox'):
            raise ValueError('Unsupported boundary type in saved state')
        boundaries.append((MovingVelocityBox if boundary_type=='MovingVelocityBox' or 'ramp_time' in values else VelocityBox)(**values))
    config=Config(dim=2,n_grid=info['grid'],dt=info['dt_s'],domain_size=info['grid']*info['dx_m'],
                  material='neo-hookean',gravity=(0.,0.),boundary_cells=0,boundaries=tuple(boundaries))
    state=block(config,info['lower_m'],info['upper_m'],particles_per_cell=2,
                density=info['density_kg_m3'],dtype=jnp.float64)
    source=run/f'frame-{frame:03d}.npz'
    with np.load(source,allow_pickle=False) as data:
        if len(data['x'])!=len(state.x) or not np.isclose(float(data['time']),float(row['time_s']),rtol=1e-13,atol=0):
            raise ValueError('Snapshot and history do not match')
        state=state._replace(**{k:jnp.asarray(data[k]) for k in ('x','v','F','C','time')})
        path=initialize(.5*(data['plus']+data['minus']),capacity=info['capacity'])
        updates={k:getattr(path,k).at[:len(data[k])].set(data[k]) for k in
                 ('plus','minus','area','birth_traction','maximum','bonded')}
        path=path._replace(**updates,extension=jnp.asarray(float(row['potential_extension_m'])))
    return state,path,config,info,hashlib.sha256(source.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--frame',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--end-time',type=float,default=.002)
    parser.add_argument('--max-steps',type=int,default=100000)
    parser.add_argument('--minimum-dt-factor',type=float,default=1/256)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be empty')
    if not 0<args.minimum_dt_factor<=1 or args.max_steps<1 or args.frame<0:
        parser.error('Invalid retry bounds')
    jax.config.update('jax_enable_x64',True)
    state,path,config,info,snapshot_hash=load_saved(args.run,args.frame)
    if not float(state.time)<args.end_time<=info['duration_s']:
        parser.error('End time must follow the saved state and remain within its loading schedule')
    material=neo_hookean(info['young_Pa'],info['poisson'])
    law=CohesiveLaw(**info['law'])
    start=float(state.time)
    minimum=config.dt*args.minimum_dt_factor

    def energy(s,p):
        response=surface_response(segment_positions(p.plus,p.minus),p.maximum,p.birth_traction,p.bonded,law)
        return kinetic_energy(s,config)+elastic_energy(s,material,config)+jnp.sum(p.area*(response.stored+response.dissipated))

    initial_energy=float(energy(state,path))

    @jax.jit
    def advance(state,path,ledger,controller,target):
        # controller: proposed dt, accepted count, rejected count, quiet count,
        # minimum accepted dt, maximum accepted residual, exhaustion flag.
        def pending(carry):
            s,p,w,c=carry
            return (s.time<target-1e-15)&(c[1]<args.max_steps)&(c[6]==0)

        def body(carry):
            s,p,w,c=carry
            requested=jnp.minimum(c[0],target-s.time)
            ns,npth,increment,report=retry_step(s,p,material,config,law,requested,
                                               consistent_gradients=info['consistent_gradients'],
                                               couple_tip_nodes=info.get('couple_tip_nodes',False),
                                               face_mass_fraction=info.get('face_mass_fraction',0.),
                                               particle_mediated_faces=info.get('particle_mediated_faces',False))
            accepted=report.accepted & ((report.accepted_dt>=minimum)|(requested<minimum))
            # Work has eight fields; the ninth ledger entry records insertion.
            padded=jnp.pad(jnp.stack(increment),(0,1))
            def accepted_commit(_):
                inserted,insert=insert_tip(ns,npth,material,config,law,info['insertion_spacing_m'],
                                           info['averaging_radius_m'],info['max_extension_m'])
                updated=(w+padded).at[8].add(insert.energy_jump)
                return ns._replace(valid=ns.valid & inserted.valid),inserted,updated
            ns,npth,w=jax.lax.cond(accepted,accepted_commit,lambda _:(s,p,w),None)
            accepted &= ns.valid & npth.valid
            ns,npth,w=jax.lax.cond(accepted,lambda _:(ns,npth,w),lambda _:(s,p,carry[2]),None)
            quiet=jnp.where(report.rejected>0,0.,c[3]+1)
            proposed=jnp.where(report.rejected>0,report.accepted_dt,c[0])
            proposed=jnp.where(quiet>=32,jnp.minimum(2*proposed,config.dt),proposed)
            quiet=jnp.where(quiet>=32,0.,quiet)
            updated=jnp.array([proposed,c[1]+accepted,c[2]+report.rejected,quiet,
                               jnp.minimum(c[4],jnp.where(accepted,report.accepted_dt,c[4])),
                               jnp.maximum(c[5],jnp.where(accepted,increment.residual,0.)),
                               jnp.asarray(~accepted | ~ns.valid | ~npth.valid,dtype=s.x.dtype)])
            return ns,npth,w,updated
        return jax.lax.while_loop(pending,body,(state,path,ledger,controller))

    args.output.mkdir(parents=True)
    ledger=jnp.zeros(9)
    controller=jnp.array([config.dt,0.,0.,0.,config.dt,0.,0.])
    begun=time.perf_counter();rows=[]
    for target in np.linspace(start,args.end_time,9)[1:]:
        state,path,ledger,controller=advance(state,path,ledger,controller,jnp.asarray(target))
        total_energy=float(energy(state,path));w=np.asarray(ledger);c=np.asarray(controller)
        remainder=total_energy-initial_energy+w[2]+w[3]-w[0]-w[1]-w[8]
        row=dict(time_s=float(state.time),accepted_steps=int(c[1]),rejected_attempts=int(c[2]),
                 minimum_accepted_dt_s=float(c[4]),max_accepted_residual=float(c[5]),
                 current_energy_J_per_m=total_energy,energy_remainder_J_per_m=float(remainder),
                 external_work_J_per_m=float(w[0]+w[1]),numerical_loss_J_per_m=float(w[2]+w[3]),
                 vertices=int(path.count),potential_extension_m=float(path.extension),
                 state_valid=bool(state.valid & path.valid),retry_exhausted=bool(c[6]),elapsed_s=time.perf_counter()-begun)
        rows.append(row);print(json.dumps(row),flush=True)
        if c[6] or c[1]>=args.max_steps:
            break
    count=int(path.count)
    np.savez_compressed(args.output/'last-accepted.npz',x=np.asarray(state.x),v=np.asarray(state.v),F=np.asarray(state.F),C=np.asarray(state.C),
                        plus=np.asarray(path.plus[:count]),minus=np.asarray(path.minus[:count]),area=np.asarray(path.area[:count-1]),
                        birth_traction=np.asarray(path.birth_traction[:count-1]),maximum=np.asarray(path.maximum[:count-1]),
                        bonded=np.asarray(path.bonded[:count-1]),time=float(state.time))
    sources=['tools/recover_extrinsic_dcb.py','mpm_jax/adaptive_crack.py','mpm_jax/extrinsic_crack.py']
    report=dict(completed=abs(float(state.time)-args.end_time)<1e-14 and bool(state.valid & path.valid),
                start_time_s=start,target_time_s=args.end_time,source_run=args.run.name,source_frame=args.frame,
                source_snapshot_sha256=snapshot_hash,minimum_dt_s=minimum,initial_energy_J_per_m=initial_energy,
                rows=rows,source_sha256={p:hashlib.sha256(Path(p).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sources},
                limitation='Recovery diagnostic only; no crack-path or time-discretization validation.')
    (args.output/'recovery.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
