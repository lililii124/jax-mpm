"""Identify 3D orthotropic MAT002 SVK stiffness from multi-load TL-MPM records.

Synthetic training signals are generated on a disclosed finer reference grid;
all data are produced by the same released MAT002 SVK constitutive energy.
The inverse fit uses lower-resolution 3D MPM with prescribed fixture histories.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
try:
    import resource
except ImportError:  # Windows CLI users can still run the same forward/inverse code.
    resource = None

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from mpm_jax.core import Config
from mpm_jax.materials.elastic import LinearMaterial, stvenant_kirchhoff_energy
from mpm_jax.total_lagrangian import NodalBoundary, initial_state, reference_grid, sample_particles, step

RHO = 1050.0
DOMAIN = 1.0
LENGTHS = np.asarray([0.40, 0.30, 0.25], dtype=np.float64)
HOLDOUT_LENGTHS = np.asarray([0.36, 0.28, 0.25], dtype=np.float64)
CLAMP_THICKNESS_M = 0.0475
SENSOR_SEED = 8031
L_SCALE = 850.0
SHEAR_SCALE = 1.0e6
DIAG_RANGE = (0.25, 2.50)
OFF_RANGE = 1.50
SHEAR_RANGE = (0.05, 3.50)
TRUE_ENGINEERING = {
    'E1_Pa': 1.20e6, 'E2_Pa': 8.0e5, 'E3_Pa': 6.0e5,
    'nu12': 0.22, 'nu13': 0.18, 'nu23': 0.25,
    'G12_Pa': 4.2e5, 'G13_Pa': 3.4e5, 'G23_Pa': 2.6e5,
}
INITIAL_ENGINEERING = {
    'E1_Pa': 9.0e5, 'E2_Pa': 1.0e6, 'E3_Pa': 8.0e5,
    'nu12': 0.18, 'nu13': 0.22, 'nu23': 0.18,
    'G12_Pa': 3.2e5, 'G13_Pa': 4.0e5, 'G23_Pa': 3.3e5,
}


def engineering_to_stiffness(p):
    E=np.asarray([p['E1_Pa'],p['E2_Pa'],p['E3_Pa']],dtype=np.float64)
    nu=np.asarray([p['nu12'],p['nu13'],p['nu23']],dtype=np.float64)
    G=np.asarray([p['G12_Pa'],p['G13_Pa'],p['G23_Pa']],dtype=np.float64)
    S=np.zeros((6,6),dtype=np.float64)
    S[0,0],S[1,1],S[2,2]=1/E
    S[0,1]=S[1,0]=-nu[0]/E[0]
    S[0,2]=S[2,0]=-nu[1]/E[0]
    S[1,2]=S[2,1]=-nu[2]/E[1]
    S[3,3],S[4,4],S[5,5]=1/G[2],1/G[1],1/G[0]
    if np.min(np.linalg.eigvalsh(S))<=0: raise ValueError('Engineering constants do not define positive compliance')
    return np.linalg.inv(S)


def stiffness_to_engineering(C):
    S=np.linalg.inv(np.asarray(C,dtype=np.float64))
    return {
        'E1_Pa':float(1/S[0,0]),'E2_Pa':float(1/S[1,1]),'E3_Pa':float(1/S[2,2]),
        'nu12':float(-S[0,1]/S[0,0]),'nu13':float(-S[0,2]/S[0,0]),'nu23':float(-S[1,2]/S[1,1]),
        'G12_Pa':float(1/S[5,5]),'G13_Pa':float(1/S[4,4]),'G23_Pa':float(1/S[3,3]),
    }


def encode_engineering(p):
    C=engineering_to_stiffness(p); L=np.linalg.cholesky(C[:3,:3])/L_SCALE
    d=(np.diag(L)-DIAG_RANGE[0])/(DIAG_RANGE[1]-DIAG_RANGE[0])
    if np.any((d<=0)|(d>=1)): raise ValueError('Normal stiffness Cholesky diagonal outside declared bounds')
    off=np.asarray([L[1,0],L[2,0],L[2,1]])/OFF_RANGE
    shear=np.asarray([C[5,5],C[4,4],C[3,3]])/SHEAR_SCALE
    f=(shear-SHEAR_RANGE[0])/(SHEAR_RANGE[1]-SHEAR_RANGE[0])
    if np.any(np.abs(off)>=1) or np.any((f<=0)|(f>=1)): raise ValueError('Initial stiffness outside parameter bounds')
    logit=lambda x:np.log(x/(1-x))
    return jnp.asarray([*logit(d),*np.arctanh(off),*logit(f)],dtype=jnp.float64)


def decode_stiffness(theta):
    t=jnp.asarray(theta,dtype=jnp.float64)
    diag=L_SCALE*(DIAG_RANGE[0]+(DIAG_RANGE[1]-DIAG_RANGE[0])*jax.nn.sigmoid(t[:3]))
    off=L_SCALE*OFF_RANGE*jax.numpy.tanh(t[3:6])
    L=jnp.zeros((3,3),dtype=t.dtype)
    L=L.at[0,0].set(diag[0]).at[1,1].set(diag[1]).at[2,2].set(diag[2])
    L=L.at[1,0].set(off[0]).at[2,0].set(off[1]).at[2,1].set(off[2])
    Cn=L@L.T
    sh=SHEAR_SCALE*(SHEAR_RANGE[0]+(SHEAR_RANGE[1]-SHEAR_RANGE[0])*jax.nn.sigmoid(t[6:9]))
    C=jnp.zeros((6,6),dtype=t.dtype).at[:3,:3].set(Cn)
    C=C.at[3,3].set(sh[2]).at[4,4].set(sh[1]).at[5,5].set(sh[0])
    return C


def make_material(theta):
    return LinearMaterial(decode_stiffness(theta),jnp.eye(3,dtype=jnp.float64))


def make_body(grid,ppc,dt,lengths=LENGTHS):
    lengths=np.asarray(lengths,dtype=np.float64)
    counts=np.maximum(4,np.rint(lengths*grid*ppc).astype(int))
    spacing=lengths/counts
    axes=[0.5*(DOMAIN-lengths[i])+0.5*spacing[i]+spacing[i]*np.arange(counts[i]) for i in range(3)]
    xyz=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
    vol=float(np.prod(spacing))*np.ones(len(xyz),dtype=np.float64)
    cfg=Config(n_grid=grid,dt=dt,domain_size=DOMAIN,dim=3,material='orthotropic-svk',boundary_cells=0,gravity=(0.,0.,0.))
    ref=reference_grid(jnp.asarray(xyz),jnp.asarray(vol),RHO,cfg)
    return cfg,ref,xyz,vol


def boundary_case(ref,cfg,kind,steps,period_steps,amplitude,reverse=False,lengths=LENGTHS):
    lengths=np.asarray(lengths,dtype=np.float64)
    body_lo=.5*(DOMAIN-lengths);body_hi=body_lo+lengths;body_center=.5*(body_lo+body_hi)
    nodes=np.asarray(ref.nodes); mass=np.asarray(ref.nodal_mass); occupied=mass>0
    fixed=np.zeros((len(nodes),3),dtype=bool); move=np.zeros(len(nodes),dtype=bool)
    support_mask=np.zeros(len(nodes),dtype=bool)
    pattern=np.zeros((len(nodes),3),dtype=np.float64)
    def face(axis,side):
        return occupied & (nodes[:,axis] <= body_lo[axis]+CLAMP_THICKNESS_M if side=='min' else nodes[:,axis] >= body_hi[axis]-CLAMP_THICKNESS_M)
    sign=-1.0 if reverse else 1.0
    if kind.startswith('tension_'):
        axis='xyz'.index(kind[-1]); lo,hi=face(axis,'min'),face(axis,'max')
        fixed[lo|hi,:]=True;support_mask=lo;move=hi;pattern[hi,axis]=sign
    elif kind in ('shear_xy','shear_xz','shear_yz'):
        a,b={'shear_xy':(0,1),'shear_xz':(0,2),'shear_yz':(1,2)}[kind]
        lo,hi=face(a,'min'),face(a,'max');fixed[lo|hi,:]=True;support_mask=lo;move=hi;pattern[hi,b]=sign
    elif kind=='bend_3point':
        xlo,xhi=body_lo[0],body_hi[0];span=xhi-xlo
        support=(occupied&(nodes[:,2]<=body_lo[2]+CLAMP_THICKNESS_M)&(((nodes[:,0]-xlo)<.18*span)|((xhi-nodes[:,0])<.18*span)))
        move=occupied&(nodes[:,2]>=body_hi[2]-CLAMP_THICKNESS_M)&(abs(nodes[:,0]-.5*(xlo+xhi))<=.12*span)
        support_mask=support;fixed[support,2]=True;fixed[move,2]=True;pattern[move,2]=-sign
    elif kind=='torsion_x':
        lo,hi=face(0,'min'),face(0,'max');fixed[lo|hi,:]=True;support_mask=lo;move=hi
        pattern[hi,1]=-(nodes[hi,2]-body_center[2])*sign
        pattern[hi,2]=(nodes[hi,1]-body_center[1])*sign
    elif kind=='holdout_mixed':
        lo,hi=face(0,'min'),face(0,'max');fixed[lo|hi,:]=True;support_mask=lo;move=hi
        pattern[hi,0]=sign
        radius=max(.5*float(np.hypot(lengths[1],lengths[2])),1e-12)
        pattern[hi,1]=-(nodes[hi,2]-body_center[2])/radius*sign
        pattern[hi,2]=(nodes[hi,1]-body_center[1])/radius*sign
    else: raise ValueError(f'unknown load case {kind}')
    if not move.any() or float(mass[move].sum())<=0: raise ValueError(f'{kind}: loaded material face has zero mass')
    if not support_mask.any() or float(mass[support_mask].sum())<=0: raise ValueError(f'{kind}: support region has zero mass')
    if np.any(support_mask&move): raise ValueError(f'{kind}: support and driven regions overlap')
    if not fixed.any(): raise ValueError(f'{kind}: no material supports')
    constrained_nodes=fixed.any(axis=1)
    center=body_center.copy()
    t=jnp.arange(steps+1,dtype=jnp.float64)*cfg.dt
    omega=2*jnp.pi/(period_steps*cfg.dt)
    phase=.5*(1-jnp.cos(omega*t)); phase_v=.5*omega*jnp.sin(omega*t); phase_a=.5*omega**2*jnp.cos(omega*t)
    # Translation amplitudes are metres; torsion converts the dimensionless angle to nodal displacements.
    if kind=='torsion_x': scale=float(amplitude)
    elif kind=='holdout_mixed': scale=jnp.asarray([amplitude,0.018,0.018])
    else: scale=float(amplitude)
    scaled_pattern=jnp.asarray(pattern)*jnp.asarray(scale)
    return (jnp.asarray(fixed),scaled_pattern,phase,phase_v,phase_a,
            jnp.asarray(move),jnp.asarray(center),{'kind':kind,'support_nodes':int(support_mask.sum()),
            'support_mass_kg':float(mass[support_mask].sum()),'support_volume_m3':float(mass[support_mask].sum()/RHO),
            'loaded_nodes':int(move.sum()),'loaded_mass_kg':float(mass[move].sum()),'loaded_volume_m3':float(mass[move].sum()/RHO),
            'support_loaded_overlap_nodes':int(np.sum(support_mask&move)),
            'constrained_nodes':int(constrained_nodes.sum()),'constrained_mass_kg':float(mass[constrained_nodes].sum()),
            'physical_body_extent_min_m':body_lo.tolist(),'physical_body_extent_max_m':body_hi.tolist(),
            'fixture_region_rule':'mass-bearing nodes selected relative to fixed physical specimen faces',
            'support_extent_min_m':nodes[support_mask].min(axis=0).tolist(),'support_extent_max_m':nodes[support_mask].max(axis=0).tolist(),
            'loaded_extent_min_m':nodes[move].min(axis=0).tolist(),'loaded_extent_max_m':nodes[move].max(axis=0).tolist(),
            'clamp_thickness_rule_m':CLAMP_THICKNESS_M,'amplitude':float(amplitude)})


def sensor_stencil(xyz,count,offset=0,lengths=LENGTHS):
    """Trilinearly sample fixed physical sensor coordinates on each structured particle grid."""
    rng=np.random.default_rng(SENSOR_SEED)
    lengths=np.asarray(lengths,dtype=np.float64)
    fraction_margin=(CLAMP_THICKNESS_M+.01)/lengths
    if np.any(fraction_margin>=.5):raise ValueError('clamp and sensor-clearance bands leave no interior sensor volume')
    fractions=rng.uniform(fraction_margin,1-fraction_margin,size=(32,3))
    lo=.5*(DOMAIN-lengths);targets=lo+fractions[offset:offset+count]*lengths
    axes=[np.unique(xyz[:,i]) for i in range(3)];shape=tuple(len(a) for a in axes)
    ids=[];weights=[]
    for target in targets:
        bracket=[];alpha=[]
        for axis,x in zip(axes,target):
            hi=int(np.searchsorted(axis,x,side='right'));hi=min(max(hi,1),len(axis)-1);low=hi-1
            bracket.append((low,hi));alpha.append(float((x-axis[low])/(axis[hi]-axis[low])))
        row_ids=[];row_w=[]
        for ix in (0,1):
            for iy in (0,1):
                for iz in (0,1):
                    bits=(ix,iy,iz);ind=[bracket[d][bits[d]] for d in range(3)]
                    w=np.prod([alpha[d] if bits[d] else 1-alpha[d] for d in range(3)])
                    row_ids.append((ind[0]*shape[1]+ind[1])*shape[2]+ind[2]);row_w.append(w)
        ids.append(row_ids);weights.append(row_w)
    return np.asarray(ids,dtype=np.int32),np.asarray(weights,dtype=np.float64),targets


def response_case(theta,cfg,ref,boundaries,sensor_ids,sensor_weights):
    material=make_material(theta);zero=jnp.zeros(3,dtype=jnp.float64)
    state=initial_state(ref,material,cfg,zero)
    fixed,pattern,phase,phase_v,phase_a,move,center=boundaries
    def one(s,i):
        b0=NodalBoundary(fixed,phase[i]*pattern,phase_v[i]*pattern,phase_a[i]*pattern)
        b1=NodalBoundary(fixed,phase[i+1]*pattern,phase_v[i+1]*pattern,phase_a[i+1]*pattern)
        ns,_=step(s,ref,material,cfg,zero,zero,b0,b1)
        sample=sample_particles(ns,ref)
        particle_u=sample.x-ref.positions
        sensor_u=jax.vmap(lambda ids,weights:jnp.sum(weights[:,None]*particle_u[ids],axis=0))(sensor_ids,sensor_weights)
        reaction=(ref.nodal_mass[:,None]*(phase_a[i+1]*pattern)-ns.force)*fixed
        RF=jnp.sum(reaction*move[:,None],axis=0)
        RM=jnp.sum(jnp.cross(ref.nodes+ns.displacement-center,reaction)*move[:,None],axis=0)
        return ns,(sensor_u,RF,RM,ns.valid)
    advance=jax.checkpoint(one)
    state,(u,force,moment,valid)=jax.lax.scan(advance,state,jnp.arange(phase.shape[0]-1))
    return u,force,moment,valid


def protocol_response(theta,cfg,ref,boundaries,sensor_data):
    sensor_ids,sensor_weights=sensor_data
    def one(inputs):
        b,si,sw=inputs
        return response_case(theta,cfg,ref,b,si,sw)
    return jax.lax.map(one,(boundaries,sensor_ids,sensor_weights))


def flatten_response(out):
    u,f,m,_=out
    return jnp.concatenate((u.reshape(-1),f.reshape(-1),m.reshape(-1)))


def write_json(path,obj):path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--grid',type=int,default=36);ap.add_argument('--ppc',type=float,default=1.6)
    ap.add_argument('--truth-grid',type=int,default=48);ap.add_argument('--truth-ppc',type=float,default=3.0)
    ap.add_argument('--steps',type=int,default=200);ap.add_argument('--iterations',type=int,default=35)
    ap.add_argument('--dt',type=float,default=4.0e-5);ap.add_argument('--noise-fraction',type=float,default=.003)
    ap.add_argument('--seed',type=int,default=1701);ap.add_argument('--sensors',type=int,default=12)
    ap.add_argument('--fd-epsilon',type=float,default=2e-4)
    ap.add_argument('--training-observations',type=Path)
    ap.add_argument('--observations-sha256')
    ap.add_argument('--training-result',type=Path)
    ap.add_argument('--training-result-sha256')
    ap.add_argument('--initial-checkpoint',type=Path)
    ap.add_argument('--checkpoint-sha256')
    ap.add_argument('--resume-diagnostics',type=Path)
    ap.add_argument('--resume-diagnostics-sha256')
    args=ap.parse_args(argv)
    if args.output.exists():ap.error(f'output already exists: {args.output}')
    if (args.steps<12 or args.iterations<1 or not 6<=args.sensors<=16 or args.noise_fraction<0
            or args.ppc<=0 or args.truth_ppc<=0):ap.error('invalid run controls')
    reuse=any((args.training_observations,args.observations_sha256,args.training_result,args.training_result_sha256))
    resume=any((args.initial_checkpoint,args.checkpoint_sha256))
    if reuse and not all((args.training_observations,args.observations_sha256,args.training_result,args.training_result_sha256)):
        ap.error('reuse requires observations/result files and their SHA-256 digests')
    if resume and not all((args.initial_checkpoint,args.checkpoint_sha256)):
        ap.error('resume requires a checkpoint and its SHA-256 digest')
    if resume and not reuse:ap.error('checkpoint continuation requires the exact frozen training observations')
    if bool(args.resume_diagnostics)!=bool(args.resume_diagnostics_sha256):ap.error('resume diagnostics require JSON and SHA-256')
    if args.resume_diagnostics and not resume:ap.error('resume diagnostics require checkpoint continuation')
    args.output.mkdir(parents=True)
    source_root=Path(__file__).resolve().parents[1]
    used=['examples/inverse_orthotropic_svk.py','mpm_jax/total_lagrangian.py','mpm_jax/materials/elastic.py','mpm_jax/core.py']
    start=time.monotonic();period=max(12,args.steps)
    train_kinds=('tension_x','tension_y','tension_z','shear_xy','shear_xz','shear_yz','bend_3point','torsion_x')
    amplitudes={'tension_x':.004,'tension_y':.003,'tension_z':.002,'shear_xy':.003,'shear_xz':.0025,'shear_yz':.0025,'bend_3point':.0025,'torsion_x':.018}
    truth_theta=encode_engineering(TRUE_ENGINEERING);initial_theta=encode_engineering(INITIAL_ENGINEERING)
    reused_diagnostics=None
    if args.resume_diagnostics:
        if sha(args.resume_diagnostics)!=args.resume_diagnostics_sha256:raise ValueError('resume diagnostics SHA-256 mismatch')
        reused_diagnostics=json.loads(args.resume_diagnostics.read_text(encoding='utf-8'))
        if reused_diagnostics.get('training_observation_sha256')!=args.observations_sha256:raise ValueError('resume diagnostics observations mismatch')
        if reused_diagnostics.get('source_result_sha256')!=args.training_result_sha256:raise ValueError('resume diagnostics source receipt mismatch')
        if reused_diagnostics.get('checkpoint_sha256')!=args.checkpoint_sha256:raise ValueError('resume diagnostics checkpoint mismatch')
        if (reused_diagnostics.get('grid')!=args.grid or not math.isclose(reused_diagnostics.get('ppc',-1),args.ppc,rel_tol=0,abs_tol=1e-15)
                or reused_diagnostics.get('steps')!=args.steps or not math.isclose(reused_diagnostics.get('dt_s',-1),args.dt,rel_tol=0,abs_tol=1e-15)):
            raise ValueError('resume diagnostics mesh/time mismatch')
    rng=np.random.default_rng(args.seed)
    truth_boundaries=[];truth_fixture_receipts=[];observation_data_record=None
    if reuse:
        if sha(args.training_observations)!=args.observations_sha256:raise ValueError('training observation SHA-256 mismatch')
        if sha(args.training_result)!=args.training_result_sha256:raise ValueError('training source result SHA-256 mismatch')
        source_result=json.loads(args.training_result.read_text(encoding='utf-8'))
        if source_result.get('status') not in ('completed','checkpoint_ready'):raise ValueError('source training result is not completed or checkpoint-ready')
        if reused_diagnostics:
            prior=reused_diagnostics.get('prior_source_sha256',{})
            if prior!=source_result.get('source_sha256'):raise ValueError('resume diagnostics source manifest differs from checkpoint receipt')
            for module_path in ('mpm_jax/core.py','mpm_jax/materials/elastic.py','mpm_jax/total_lagrangian.py'):
                if prior.get(module_path)!=sha(source_root/module_path):raise ValueError(f'resume diagnostics model module changed: {module_path}')
        result_observation_sha=source_result.get('training_observation_provenance',{}).get('sha256')
        if result_observation_sha is not None and result_observation_sha!=args.observations_sha256:
            raise ValueError('source result does not bind the supplied observation archive')
        source_train=source_result['training']
        if (source_train['steps']!=args.steps or not math.isclose(source_train['dt_s'],args.dt,rel_tol=0,abs_tol=1e-15)
                or source_train['truth_grid']!=args.truth_grid or source_train['truth_ppc']!=args.truth_ppc):
            raise ValueError('requested grid/time controls differ from the saved observation run')
        with np.load(args.training_observations,allow_pickle=False) as d:
            truth_u=d['truth_u'].copy();obs_u=d['observed_u'].copy();truth_f=d['truth_force'].copy();obs_f=d['observed_force'].copy()
            truth_m=d['truth_moment'].copy();obs_m=d['observed_moment'].copy();truth_sensor_coords=d['sensor_positions'].copy()
            truth_sensors=d['sensor_particle_stencil'].copy();truth_sensor_weights=d['sensor_weights'].copy()
            stored_cases=d['case_names'].astype(str).tolist();stored_time=d['time_s'].copy();noise_std=d['noise_std'].copy()
            stored_true=d['true_engineering'].copy()
        if stored_cases!=list(train_kinds):raise ValueError('saved observation load-case order mismatch')
        if len(truth_sensor_coords)!=args.sensors:raise ValueError('saved observation sensor count differs from requested count')
        if not np.array_equal(stored_time,np.arange(1,args.steps+1)*args.dt):raise ValueError('saved observation time axis mismatch')
        if not np.allclose(stored_true,np.asarray(list(TRUE_ENGINEERING.values())),rtol=0,atol=0):raise ValueError('saved truth constants mismatch')
        sig_u,sig_f,sig_m=map(float,noise_std)
        truth_particles=int(source_train['truth_particles'])
        truth_fixture_receipts=source_train['fixture_receipts_truth']
        if not np.all(np.isfinite(np.concatenate((truth_u.ravel(),obs_u.ravel(),truth_f.ravel(),obs_f.ravel(),truth_m.ravel(),obs_m.ravel())))):
            raise ValueError('saved observation arrays contain nonfinite values')
        observation_data_record={'mode':'reused_frozen_observations','sha256':args.observations_sha256,
            'source_result_sha256':args.training_result_sha256,'source_result_source_sha256':source_result['source_sha256'],
            'source_result_embedded_observation_sha256':result_observation_sha,
            'source_seed':source_train['seed'],'source_noise_fraction':source_train['noise_fraction']}
    else:
        # Synthetic observations use a finer 3D MAT002-SVK discretization, independent of inverse mesh.
        truth_cfg,truth_ref,truth_xyz,_=make_body(args.truth_grid,args.truth_ppc,args.dt)
        truth_particles=len(truth_xyz)
        truth_sensors,truth_sensor_weights,truth_sensor_coords=sensor_stencil(truth_xyz,args.sensors,offset=0)
        for kind in train_kinds:truth_boundaries.append(boundary_case(truth_ref,truth_cfg,kind,args.steps,period,amplitudes[kind]))
        truth_fixture_receipts=[b[-1] for b in truth_boundaries]
        tbd=tuple(jnp.stack([b[i] for b in truth_boundaries]) for i in range(7))
        truth_sensor_data=(jnp.broadcast_to(jnp.asarray(truth_sensors),(len(train_kinds),*truth_sensors.shape)),
            jnp.broadcast_to(jnp.asarray(truth_sensor_weights),(len(train_kinds),*truth_sensor_weights.shape)))
        synth_fn=jax.jit(lambda th,reference,boundary_data,sensor_ids,sensor_weights:
            protocol_response(th,truth_cfg,reference,boundary_data,(sensor_ids,sensor_weights)))
        truth_raw=synth_fn(truth_theta,truth_ref,tbd,*truth_sensor_data);jax.block_until_ready(truth_raw[0])
        if not np.asarray(truth_raw[3]).all():raise RuntimeError('fine-grid synthetic training trajectory became invalid')
        truth_u,truth_f,truth_m=(np.asarray(x) for x in truth_raw[:3])
        if not all(np.isfinite(x).all() for x in (truth_u,truth_f,truth_m)):raise RuntimeError('synthetic training response is nonfinite')
        # Fixed seeded noise with independent, documented channel standard deviations.
        sig_u=max(float(np.sqrt(np.mean(truth_u**2))*args.noise_fraction),1e-9)
        sig_f=max(float(np.sqrt(np.mean(truth_f**2))*args.noise_fraction),1e-5)
        sig_m=max(float(np.sqrt(np.mean(truth_m**2))*args.noise_fraction),1e-7)
        obs_u=truth_u+rng.normal(0,sig_u,truth_u.shape);obs_f=truth_f+rng.normal(0,sig_f,truth_f.shape);obs_m=truth_m+rng.normal(0,sig_m,truth_m.shape)
    training_archive=args.output/'training_synthetic.npz'
    if reuse:
        shutil.copy2(args.training_observations,training_archive)
    else:
        np.savez_compressed(training_archive,truth_u=truth_u,observed_u=obs_u,truth_force=truth_f,observed_force=obs_f,truth_moment=truth_m,observed_moment=obs_m,
            sensor_positions=truth_sensor_coords,sensor_particle_stencil=truth_sensors,sensor_weights=truth_sensor_weights,case_names=np.asarray(train_kinds),time_s=np.arange(1,args.steps+1)*args.dt,
            noise_std=np.asarray([sig_u,sig_f,sig_m]),true_stiffness=engineering_to_stiffness(TRUE_ENGINEERING),true_engineering=np.asarray(list(TRUE_ENGINEERING.values())))
    obs_u_j=jnp.asarray(obs_u);obs_f_j=jnp.asarray(obs_f);obs_m_j=jnp.asarray(obs_m)
    scales=jnp.asarray([sig_u,sig_f,sig_m])
    # Fit on a coarser particle discretization with the same physical fixture protocols.
    cfg,ref,xyz,_=make_body(args.grid,args.ppc,args.dt)
    sensors,sensor_weights,sensor_coords=sensor_stencil(xyz,args.sensors,offset=0)
    sensor_offset=np.linalg.norm(sensor_coords-truth_sensor_coords,axis=1)
    if float(np.max(sensor_offset))>1e-12:raise RuntimeError('training sensor coordinates differ between truth and inverse meshes')
    inverse_boundaries=[boundary_case(ref,cfg,k,args.steps,period,amplitudes[k]) for k in train_kinds]
    bbd=tuple(jnp.stack([b[i] for b in inverse_boundaries]) for i in range(7))
    sensor_data=(jnp.broadcast_to(jnp.asarray(sensors),(len(train_kinds),*sensors.shape)),
        jnp.broadcast_to(jnp.asarray(sensor_weights),(len(train_kinds),*sensor_weights.shape)))
    response_fit=jax.jit(lambda th,reference,boundary_data,sensor_ids,sensor_weights:
        protocol_response(th,cfg,reference,boundary_data,(sensor_ids,sensor_weights)))
    def objective(theta,reference,boundary_data,sensor_ids,sensor_weights,observed_u,observed_f,observed_m,noise_scales):
        # Reduce each load case to three scalars inside the sequential map. The
        # former protocol_response path stacked every time history for all eight
        # cases before the loss reduction, which needlessly raised reverse-mode
        # memory. Each case still executes the same full lax.scan trajectory,
        # with the same rematerialized MPM step, fixed sensors and observations.
        def one_case(inputs):
            boundary,ids,weights,target_u,target_f,target_m=inputs
            u,f,m,_=response_case(theta,cfg,reference,boundary,ids,weights)
            return jnp.stack((jnp.mean(((u-target_u)/noise_scales[0])**2),
                              jnp.mean(((f-target_f)/noise_scales[1])**2),
                              jnp.mean(((m-target_m)/noise_scales[2])**2)))
        per_case=jax.lax.map(jax.checkpoint(one_case),(boundary_data,sensor_ids,sensor_weights,observed_u,observed_f,observed_m))
        return jnp.mean(per_case)
    vg=jax.jit(jax.value_and_grad(objective,argnums=0))
    start_iteration=0;checkpoint_source_record=None
    if resume:
        if sha(args.initial_checkpoint)!=args.checkpoint_sha256:raise ValueError('initial checkpoint SHA-256 mismatch')
        with np.load(args.initial_checkpoint,allow_pickle=False) as cp:
            checkpoint_data={k:cp[k].copy() for k in ('theta','best_theta','adam_m','adam_v','iteration','best_loss')}
        if any(checkpoint_data[k].shape!=(9,) for k in ('theta','best_theta','adam_m','adam_v')):raise ValueError('resume checkpoint parameter shape mismatch')
        start_iteration=int(checkpoint_data['iteration']);q=jnp.asarray(checkpoint_data['theta'])
        if start_iteration>=args.iterations:raise ValueError('target iterations must exceed checkpoint iteration')
        source_best=float(source_result['loss']['best'])
        if source_result.get('status')=='completed' and not math.isclose(float(checkpoint_data['best_loss']),source_best,rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError('checkpoint best loss differs from its source result')
        checkpoint_source_record={'sha256':args.checkpoint_sha256,'iteration':start_iteration,
            'loss_history_sha256':sha(args.training_result.parent/'loss_history.csv')}
    else:q=initial_theta
    initial_pred=response_fit(q,ref,bbd,*sensor_data);jax.block_until_ready(initial_pred[0])
    if not np.asarray(initial_pred[3]).all():raise RuntimeError('inverse-grid initial guess produced an invalid MPM state')
    if reused_diagnostics:
        # Reuse only a SHA-bound receipt proving these exact frozen observations,
        # mesh/time protocol, prior source result, and checkpoint already passed
        # initial/true-theta losses and the pre-optimization directional FD gate.
        resume_start_loss=float(reused_diagnostics['optimization_start_loss'])
        initial_loss=float(reused_diagnostics['initial_loss'])
        true_parameter_loss=float(reused_diagnostics['true_parameters_on_inverse_mesh'])
        grad_check=reused_diagnostics['gradient_check']
        if not grad_check.get('passed_before_optimizer_iteration_1'):
            raise ValueError('resume receipt lacks a passed gradient-check gate')
        auto=grad_check.get('autodiff_directional');fd=grad_check.get('central_difference_directional')
        grad_rel=grad_check.get('relative_error');eps=float(grad_check['epsilon']);direction=grad_check.get('direction')
    else:
        resume_start_loss,initial_grad=vg(q,ref,bbd,*sensor_data,obs_u_j,obs_f_j,obs_m_j,scales);jax.block_until_ready(resume_start_loss)
        if not np.isfinite(float(resume_start_loss)) or not np.isfinite(np.asarray(initial_grad)).all():raise RuntimeError('initial inverse loss or gradient is nonfinite')
        initial_loss=float(resume_start_loss) if not resume else float(source_result['loss']['initial'])
        true_parameter_loss,_=vg(truth_theta,ref,bbd,*sensor_data,obs_u_j,obs_f_j,obs_m_j,scales)
        jax.block_until_ready(true_parameter_loss);true_parameter_loss=float(true_parameter_loss)
        direction=jnp.asarray(rng.normal(size=9));direction=direction/jnp.linalg.norm(direction)
        eps=args.fd_epsilon
        auto=float(jnp.dot(initial_grad,direction))
        fd=float((objective(q+eps*direction,ref,bbd,*sensor_data,obs_u_j,obs_f_j,obs_m_j,scales)-
                  objective(q-eps*direction,ref,bbd,*sensor_data,obs_u_j,obs_f_j,obs_m_j,scales))/(2*eps))
        grad_rel=abs(auto-fd)/max(abs(auto),abs(fd),1e-12)
        if not np.isfinite([auto,fd,grad_rel]).all() or grad_rel>5e-3:raise RuntimeError(f'directional derivative check failed: relative error {grad_rel}')
        grad_check={'autodiff_directional':auto,'central_difference_directional':fd,'relative_error':grad_rel,'epsilon':eps,'direction':np.asarray(direction).tolist(),'passed_before_optimizer_iteration_1':True}
    # Adam with bounded SPD parameters; every iterate remains a positive-definite stiffness.
    if resume:
        m1=jnp.asarray(checkpoint_data['adam_m']);v1=jnp.asarray(checkpoint_data['adam_v'])
        best_q=jnp.asarray(checkpoint_data['best_theta']);best=float(checkpoint_data['best_loss'])
    else:
        m1=jnp.zeros_like(q);v1=jnp.zeros_like(q);best_q=q;best=initial_loss
    losses=[];lr=.035
    loss_path=args.output/'loss_history.csv';checkpoint_path=args.output/'checkpoint.npz'
    if resume:shutil.copy2(args.training_result.parent/'loss_history.csv',loss_path)
    def save_checkpoint(iteration,current,best_iter,best_value,moment1,moment2):
        tmp=checkpoint_path.with_suffix('.npz.tmp')
        with tmp.open('wb') as stream:
            np.savez_compressed(stream,theta=np.asarray(current),best_theta=np.asarray(best_iter),stiffness=np.asarray(decode_stiffness(best_iter)),
                engineering=np.asarray(list(stiffness_to_engineering(np.asarray(decode_stiffness(best_iter))).values())),
                adam_m=np.asarray(moment1),adam_v=np.asarray(moment2),iteration=np.asarray(iteration),best_loss=np.asarray(best_value))
            stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,checkpoint_path)
    def save_checkpoint_receipt(iteration,best_value):
        receipt={'status':'checkpoint_ready','source_sha256':{p:sha(source_root/p) for p in used},
            'training_observation_provenance':observation_data_record or {'mode':'generated_in_this_run','sha256':sha(training_archive)},
            'training':{'geometry_lengths_m':LENGTHS.tolist(),'case_names':list(train_kinds),'steps':args.steps,'dt_s':args.dt,
                'grid':args.grid,'ppc':args.ppc,'particles':len(xyz),'truth_grid':args.truth_grid,'truth_ppc':args.truth_ppc,
                'truth_particles':truth_particles,'sensors':args.sensors,'seed':source_train['seed'] if reuse else args.seed,
                'noise_fraction':source_train['noise_fraction'] if reuse else args.noise_fraction,
                'noise_std_displacement_m':sig_u,'noise_std_force_N':sig_f,'noise_std_moment_Nm':sig_m,
                'fixture_receipts_truth':truth_fixture_receipts,'fixture_receipts_inverse':[b[-1] for b in inverse_boundaries]},
            'loss':{'initial':float(initial_loss),'optimization_start':float(resume_start_loss),
                'true_parameters_on_inverse_mesh':true_parameter_loss,'best':float(best_value)},
            'gradient_check':grad_check,
            'optimizer':{'algorithm':'Adam','iteration':iteration,'iteration_target':args.iterations,'learning_rate':lr}}
        tmp=args.output/'checkpoint_ready.json.tmp';write_json(tmp,receipt);os.replace(tmp,args.output/'checkpoint_ready.json')
    with loss_path.open('a' if resume else 'w',newline='',encoding='utf-8') as loss_stream:
        writer=csv.writer(loss_stream)
        if not resume:writer.writerow(['iteration','loss_at_gradient','gradient_norm','E1_Pa','E2_Pa','E3_Pa','nu12','nu13','nu23','G12_Pa','G13_Pa','G23_Pa'])
        loss_stream.flush();save_checkpoint(start_iteration,q,best_q,best,m1,v1);save_checkpoint_receipt(start_iteration,best)
        for k in range(start_iteration+1,args.iterations+1):
            theta_eval=q
            val,g=vg(theta_eval,ref,bbd,*sensor_data,obs_u_j,obs_f_j,obs_m_j,scales);jax.block_until_ready(val);cur=float(val)
            if not np.isfinite(cur) or not np.isfinite(np.asarray(g)).all():raise RuntimeError(f'nonfinite inverse iterate at iteration {k}')
            if cur<best:best=cur;best_q=theta_eval
            m1=.9*m1+.1*g;v1=.999*v1+.001*g*g
            q=theta_eval-lr*(m1/(1-.9**k))/(jnp.sqrt(v1/(1-.999**k))+1e-8)
            params=stiffness_to_engineering(np.asarray(decode_stiffness(theta_eval)))
            row=(k,cur,float(jnp.linalg.norm(g)),*params.values());writer.writerow(row);loss_stream.flush()
            save_checkpoint(k,q,best_q,best,m1,v1);save_checkpoint_receipt(k,best)
            print(json.dumps({'iteration':k,'loss':cur,'best_loss':best,'gradient_norm':float(jnp.linalg.norm(g))}),flush=True)
    theta_hat=best_q;C_hat=np.asarray(decode_stiffness(theta_hat));estimate=stiffness_to_engineering(C_hat)
    pred=response_fit(theta_hat,ref,bbd,*sensor_data);jax.block_until_ready(pred[0])
    if not np.asarray(pred[3]).all():raise RuntimeError('optimized inverse-grid trajectory became invalid')
    pred_u,pred_f,pred_m=(np.asarray(x) for x in pred[:3])
    # Identifiability: whiten response Jacobian by declared sensor noise, then inspect Fisher rank/correlations.
    def whitened(theta,reference,boundary_data,sensor_ids,sensor_weights,observed_u,observed_f,observed_m,noise_scales):
        u,f,m,_=protocol_response(theta,cfg,reference,boundary_data,(sensor_ids,sensor_weights))
        return jnp.concatenate((((u-observed_u)/noise_scales[0]).reshape(-1),((f-observed_f)/noise_scales[1]).reshape(-1),((m-observed_m)/noise_scales[2]).reshape(-1)))
    jac_fn=jax.jit(jax.jacfwd(whitened,argnums=0))
    jac=np.asarray(jac_fn(theta_hat,ref,bbd,*sensor_data,obs_u_j,obs_f_j,obs_m_j,scales)).reshape(-1,9)
    singular=np.linalg.svd(jac,compute_uv=False);tol=float(singular[0]*1e-7);rank=int(np.sum(singular>tol))
    fisher=jac.T@jac;covq=np.linalg.pinv(fisher,rcond=1e-12)
    def engineering_vec(theta):return jnp.asarray(list(stiffness_to_engineering(np.asarray(decode_stiffness(theta))).values()))
    # Differentiate engineering outputs with JAX directly from the SPD stiffness.
    def engineering_jax(theta):
        C=decode_stiffness(theta);S=jnp.linalg.inv(C)
        return jnp.asarray([1/S[0,0],1/S[1,1],1/S[2,2],-S[0,1]/S[0,0],-S[0,2]/S[0,0],-S[1,2]/S[1,1],1/S[5,5],1/S[4,4],1/S[3,3]])
    P=np.asarray(jax.jacfwd(engineering_jax)(theta_hat));covp=P@covq@P.T
    sd=np.sqrt(np.maximum(np.diag(covp),1e-30));corr=covp/np.outer(sd,sd)
    # Independent high-resolution mixed bending/torsion holdout, with distinct sensors and no fit data.
    hold_kind='holdout_mixed';holdout_dt=.8*args.dt;hsteps=int(round(args.steps*args.dt/holdout_dt))
    hcfg,href,hxyz,_=make_body(args.truth_grid,args.truth_ppc,holdout_dt,lengths=HOLDOUT_LENGTHS)
    hbound=boundary_case(href,hcfg,hold_kind,hsteps,max(12,hsteps),.0035,lengths=HOLDOUT_LENGTHS)
    hsensors,hsensor_weights,hsensor_coords=sensor_stencil(hxyz,min(args.sensors,12),offset=16,lengths=HOLDOUT_LENGTHS)
    holdout_sensor_surface_distance=float(np.min(np.minimum(hsensor_coords-.5*(DOMAIN-HOLDOUT_LENGTHS),.5*(DOMAIN+HOLDOUT_LENGTHS)-hsensor_coords)))
    holdout_unclamped_core_fraction=float(np.prod(HOLDOUT_LENGTHS-2*CLAMP_THICKNESS_M)/np.prod(HOLDOUT_LENGTHS))
    hb=tuple(hbound[i][None,...] for i in range(7))
    hdata=(jnp.asarray(hsensors)[None,...],jnp.asarray(hsensor_weights)[None,...])
    hold_fn=jax.jit(lambda th,reference,boundary_data,sensor_ids,sensor_weights:
        protocol_response(th,hcfg,reference,boundary_data,(sensor_ids,sensor_weights)))
    hold_truth=hold_fn(truth_theta,href,hb,*hdata);hold_fit=hold_fn(theta_hat,href,hb,*hdata);jax.block_until_ready(hold_fit[0])
    if not np.asarray(hold_truth[3]).all() or not np.asarray(hold_fit[3]).all():raise RuntimeError('independent holdout trajectory became invalid')
    hu_t,hf_t,hm_t=(np.asarray(x)[0] for x in hold_truth[:3]);hu_p,hf_p,hm_p=(np.asarray(x)[0] for x in hold_fit[:3])
    hold_u_rmse=float(np.sqrt(np.mean((hu_p-hu_t)**2)));hold_f_rmse=float(np.sqrt(np.mean((hf_p-hf_t)**2)));hold_m_rmse=float(np.sqrt(np.mean((hm_p-hm_t)**2)))
    u_rms=float(np.sqrt(np.mean(hu_t**2)));f_rms=float(np.sqrt(np.mean(hf_t**2)));m_rms=float(np.sqrt(np.mean(hm_t**2)))
    # Save the independent holdout histories after the fitted prediction completes.
    np.savez_compressed(args.output/'holdout_synthetic.npz',truth_u=hu_t,inferred_u=hu_p,truth_force=hf_t,inferred_force=hf_p,truth_moment=hm_t,inferred_moment=hm_p,
        sensor_positions=hsensor_coords,sensor_particle_stencil=hsensors,sensor_weights=hsensor_weights,time_s=np.arange(1,hsteps+1)*holdout_dt)
    save_checkpoint(args.iterations,q,theta_hat,best,m1,v1)
    if args.grid==args.truth_grid and not math.isclose(args.ppc,args.truth_ppc,rel_tol=0,abs_tol=1e-12):
        observation_scope='SYNTHETIC MAT002 orthotropic SVK; training truth uses a different particle quadrature on the same background grid with fixed-seed noise, not experiment'
        mesh_difference=(f'Same {args.grid}-cell background grid; truth nominal ppc={args.truth_ppc:g}, fit nominal ppc={args.ppc:g}.')
    elif args.truth_grid>args.grid:
        observation_scope='SYNTHETIC MAT002 orthotropic SVK; training truth generated on a finer MPM background grid with fixed-seed noise, not experiment'
        mesh_difference='Training observations use the disclosed finer 3D background grid; fitted on the coarser background grid.'
    elif args.truth_grid<args.grid:
        observation_scope='SYNTHETIC MAT002 orthotropic SVK; training truth generated on a coarser MPM background grid with fixed-seed noise, not experiment'
        mesh_difference='Training observations use the disclosed coarser background grid; fitted on the finer background grid.'
    else:
        observation_scope='SYNTHETIC MAT002 orthotropic SVK; training truth and fit use the same disclosed MPM discretization with fixed-seed noise, not experiment'
        mesh_difference='Training truth and fit use the same background grid and nominal particle quadrature.'
    result={'status':'completed','observation_scope':observation_scope,
        'model':'3D reference-grid TL-MPM velocity-Verlet using released MAT002 orthotropic St.Venant-Kirchhoff energy; no plasticity/damage/return mapping',
        'density_kg_m3':RHO,'true_engineering_constants':TRUE_ENGINEERING,'estimated_engineering_constants':estimate,'initial_engineering_constants':INITIAL_ENGINEERING,
        'positive_definiteness':{'parameterization':'normal 3x3 stiffness=L L^T with positive bounded Cholesky diagonal; yz/xz/xy shear moduli independently positive and bounded','estimated_normal_stiffness_eigenvalues_Pa':np.linalg.eigvalsh(C_hat[:3,:3]).tolist(),'estimated_shear_moduli_Pa':[float(C_hat[5,5]),float(C_hat[4,4]),float(C_hat[3,3])],'minimum_estimated_stiffness_eigenvalue_Pa':float(np.linalg.eigvalsh(C_hat).min())},
        'relative_parameter_error':{k:abs(estimate[k]/v-1) for k,v in TRUE_ENGINEERING.items()},'loss':{'initial':float(initial_loss),'optimization_start':float(resume_start_loss),'true_parameters_on_inverse_mesh':true_parameter_loss,'best':best},
        'optimizer':{'algorithm':'Adam','iteration_start':start_iteration,'iteration_target':args.iterations,'learning_rate':lr,'continued_checkpoint':checkpoint_source_record},
        'training_observation_provenance':observation_data_record or {'mode':'generated_in_this_run','sha256':sha(args.output/'training_synthetic.npz')},
        'gradient_check':grad_check,
        'identifiability':{'parameter_order':list(TRUE_ENGINEERING),'rank':rank,'parameter_count':9,'full_rank':rank==9,'singular_values':singular.tolist(),'rank_tolerance':tol,'condition_number':float(singular[0]/max(singular[-1],1e-300)),'engineering_constant_correlation':corr.tolist(),'engineering_constant_standard_deviation_local':sd.tolist(),'interpretation':'Local linearized Fisher covariance; correlations expose coupled combinations and are not a global confidence guarantee.'},
        'training':{'geometry_lengths_m':LENGTHS.tolist(),'case_names':list(train_kinds),'steps':args.steps,'dt_s':args.dt,'grid':args.grid,'ppc':args.ppc,'particles':len(xyz),'truth_grid':args.truth_grid,'truth_ppc':args.truth_ppc,'truth_particles':truth_particles,'sensors':args.sensors,'sensor_coordinates_are_common_reference_points':True,'maximum_truth_inverse_sensor_coordinate_offset_m':float(np.max(sensor_offset)),
            'background_cells_across_body_axes_inverse':(LENGTHS/(DOMAIN/args.grid)).tolist(),'background_cells_across_body_axes_truth':(LENGTHS/(DOMAIN/args.truth_grid)).tolist(),
            'minimum_sensor_distance_to_training_body_surface_m':float(np.min(np.minimum(truth_sensor_coords-.5*(DOMAIN-LENGTHS),.5*(DOMAIN+LENGTHS)-truth_sensor_coords))),
            'minimum_sensor_clearance_from_clamp_m':float(np.min(np.minimum(truth_sensor_coords-.5*(DOMAIN-LENGTHS),.5*(DOMAIN+LENGTHS)-truth_sensor_coords))-CLAMP_THICKNESS_M),
            'unclamped_core_volume_fraction':float(np.prod(LENGTHS-2*CLAMP_THICKNESS_M)/np.prod(LENGTHS)),
            'training_sensor_count':len(truth_sensor_coords),
            'noise_fraction':source_train['noise_fraction'] if reuse else args.noise_fraction,'noise_std_displacement_m':sig_u,'noise_std_force_N':sig_f,'noise_std_moment_Nm':sig_m,'seed':source_train['seed'] if reuse else args.seed,
            'fixture_receipts_truth':truth_fixture_receipts,'fixture_receipts_inverse':[b[-1] for b in inverse_boundaries],
            'mesh_difference':mesh_difference},
        'holdout':{'kind':hold_kind,'separate_from_fit':True,'geometry_lengths_m':HOLDOUT_LENGTHS.tolist(),'background_cells_across_body_axes':(HOLDOUT_LENGTHS/(DOMAIN/args.truth_grid)).tolist(),'sensor_coordinates_disjoint_from_training':True,'minimum_sensor_clearance_from_clamp_m':holdout_sensor_surface_distance-CLAMP_THICKNESS_M,'unclamped_core_volume_fraction':holdout_unclamped_core_fraction,'dt_s':holdout_dt,'grid':args.truth_grid,'ppc':args.truth_ppc,'particles':len(hxyz),'steps':hsteps,'sensors':len(hsensors),
            'displacement_rmse_m':hold_u_rmse,'displacement_truth_rms_m':u_rms,'normalized_displacement_rmse':hold_u_rmse/max(u_rms,1e-30),
            'reaction_force_rmse_N':hold_f_rmse,'reaction_force_truth_rms_N':f_rms,'normalized_force_rmse':hold_f_rmse/max(f_rms,1e-30),
            'reaction_moment_rmse_Nm':hold_m_rmse,'reaction_moment_truth_rms_Nm':m_rms,'normalized_moment_rmse':hold_m_rmse/max(m_rms,1e-30)},
        'resources':{'elapsed_s':time.monotonic()-start,'peak_rss_kib':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) if resource is not None else None,'jax':jax.__version__,'numpy':np.__version__},
        'source_sha256':{p:sha(source_root/p) for p in used}}
    write_json(args.output/'result.json',result)
    print(json.dumps({'status':result['status'],'initial_loss':result['loss']['initial'],'final_loss':result['loss']['best'],'rank':rank,'condition_number':result['identifiability']['condition_number'],'gradient_fd_relative_error':grad_rel,'holdout':result['holdout'],'elapsed_s':result['resources']['elapsed_s']},indent=2))

if __name__=='__main__':main()
