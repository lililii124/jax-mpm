"""Recoverable bounded line-search design loop for the full 3x3 blast model.

Each iteration stores its complete shared-cell logits and projected density.
All nine tile states remain in examples.multicell_blast_3d's global objective.
The density/stiffness relaxation represents soft material, not physical void.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from examples.multicell_blast_3d import build
from mpm_jax.shared_cell_design import project_fixed_volume
from mpm_jax.periodic_filter import filter_cell_logits
from tools.reconstruct_multicell_blast import reconstruct


def binary_feasibility(rho, settings):
    """Check the voxelized finite-domain material without repairing it."""
    report=reconstruct(rho,settings)[-1]
    ok=report['components']==1 and report['faces_connected']
    return ok,report


def atomic_json(path, obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)


def atomic_npz(path, **data):
    path=Path(path); tmp=path.with_name(path.stem+'.tmp.npz')
    np.savez_compressed(tmp,**data);os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cell',type=int,nargs=3,default=(16,16,20));p.add_argument('--cell-size',type=float,nargs=3,default=(.1,.1,.02))
    p.add_argument('--skin',type=float,default=.002);p.add_argument('--skin-layers',type=int,default=2);p.add_argument('--volume-fraction',type=float,default=.3)
    p.add_argument('--beta',type=float,default=5.);p.add_argument('--filter-passes',type=int,default=0);p.add_argument('--skin-young',type=float,default=70e9);p.add_argument('--core-young',type=float,default=20e9)
    p.add_argument('--require-connected-binary',action='store_true')
    p.add_argument('--solid-density',type=float,default=2700.);p.add_argument('--pressure',type=float,default=2e6);p.add_argument('--pulse',type=float,default=2e-5)
    p.add_argument('--blast-radius',type=float,default=.08);p.add_argument('--dt',type=float,default=1e-8);p.add_argument('--cfl-limit',type=float,default=.1)
    p.add_argument('--steps',type=int,default=20000);p.add_argument('--checkpoint-block',type=int,default=32);p.add_argument('--displacement-scale',type=float,default=.005)
    p.add_argument('--init',choices=('uniform','random'),default='uniform');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--iterations',type=int,default=20);p.add_argument('--initial-step',type=float,default=.2);p.add_argument('--min-step',type=float,default=1e-5);p.add_argument('--check-epsilon',type=float,default=1e-3)
    p.add_argument('--logit-bound',type=float,default=12.);p.add_argument('--output',type=Path,default=Path('outputs/multicell-blast-opt'))
    p.add_argument('--resume',action='store_true');p.add_argument('--check-only',action='store_true');a=p.parse_args()
    if a.iterations<1 or a.initial_step<=0 or a.min_step<=0 or a.logit_bound<=0 or a.filter_passes<0: raise ValueError('Invalid optimizer limits')
    if not a.check_only and a.steps*a.dt < a.pulse:
        raise ValueError('An optimization must include the entire pressure pulse')
    physics={k:v for k,v in vars(a).items() if k not in ('iterations','initial_step','min_step','output','resume','check_only','check_epsilon','init','seed')}
    relevant=('examples/multicell_blast_3d.py','mpm_jax/shared_cell_design.py','mpm_jax/periodic_filter.py','tools/reconstruct_multicell_blast.py',
        'mpm_jax/compact_tl.py','mpm_jax/total_lagrangian.py','mpm_jax/materials/elastic.py',
        'tools/optimize_multicell_blast.py')
    root=Path(__file__).resolve().parents[1]
    sources={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in relevant}
    import jax, jaxlib
    binding=dict(settings=physics,source_sha256=sources,jax=jax.__version__,jaxlib=jaxlib.__version__,numpy=np.__version__)
    signature=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()
    model_args=argparse.Namespace(**{k:v for k,v in vars(a).items() if k in ('cell','cell_size','skin','skin_layers','volume_fraction','beta','filter_passes','skin_young','core_young','solid_density','pressure','pulse','blast_radius','dt','cfl_limit','steps','checkpoint_block','displacement_scale')})
    objective,plan=build(model_args)
    import jax.numpy as jnp
    n=int(np.prod(a.cell)); out=a.output
    def density_for(q):
        return np.asarray(project_fixed_volume(filter_cell_logits(q.reshape(a.cell[2],a.cell[1],a.cell[0]),a.filter_passes),a.volume_fraction,beta=a.beta))
    if not a.resume and out.exists() and any(out.iterdir()):
        raise FileExistsError('Fresh optimization requires an empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    checkpoint=out/'checkpoint.npz'; history_path=out/'history.json'
    if a.resume:
        if not checkpoint.exists(): raise FileNotFoundError(f'No checkpoint to resume: {checkpoint}')
        with np.load(checkpoint,allow_pickle=False) as z:
            if str(z['signature'].item())!=signature: raise ValueError('Checkpoint physics or initialization settings differ')
            q=jnp.asarray(z['q']); iteration=int(z['iteration'])
        history=json.loads(history_path.read_text(encoding='utf-8'))
    else:
        rng=np.random.default_rng(a.seed); q=jnp.zeros(n) if a.init=='uniform' else jnp.asarray(rng.normal(0,.1,n))
        iteration=0;history=[]
        atomic_json(out/'inputs.json',dict(signature=signature,binding=binding,initialization=a.init,seed=a.seed,plan=plan))
    if q.shape!=(n,): raise ValueError('Checkpoint design shape does not match cell grid')
    if a.require_connected_binary and not binary_feasibility(density_for(q),vars(a))[0]:
        raise ValueError('Initial or resumed design does not produce a connected binary structure')
    if not a.resume and not a.check_only:
        rho0=density_for(q)
        atomic_npz(out/'design-00000.npz',logits=np.asarray(q),rho_cell=rho0,cell_shape=np.asarray(a.cell),
                   tile_repeats=np.asarray((3,3,1)),projected_mean=rho0.mean())
        atomic_json(history_path,[])
        atomic_npz(checkpoint,q=np.asarray(q),iteration=np.asarray(0),signature=np.asarray(signature))
    print(json.dumps(dict(stage='initializing-full-domain',plan=plan,iteration=iteration,signature=signature)),flush=True)
    vg=jax.jit(jax.value_and_grad(objective)); ev=jax.jit(objective)
    started=time.monotonic()
    print(json.dumps(dict(stage='initial-gradient',iteration=iteration)),flush=True)
    value,grad=vg(q); value=float(value)
    if not np.isfinite(value) or not np.isfinite(np.asarray(grad)).all(): raise FloatingPointError('Initial state is nonfinite/invalid')
    print(json.dumps(dict(stage='initial-gradient-complete',iteration=iteration,seconds=time.monotonic()-started)),flush=True)
    if a.check_only:
        rng=np.random.default_rng(a.seed+1);d=jnp.asarray(rng.normal(size=n));d/=jnp.linalg.norm(d);eps=a.check_epsilon
        print(json.dumps(dict(stage='gradient-finite-difference',iteration=iteration)),flush=True)
        fd=float((ev(q+eps*d)-ev(q-eps*d))/(2*eps)); ad=float(jnp.vdot(grad,d)); scale=max(abs(fd),abs(ad),1e-30)
        result=dict(plan=plan,objective=value,directional_fd=fd,directional_ad=ad,absolute_error=abs(fd-ad),relative_error=abs(fd-ad)/scale if scale>1e-15 else None,informative=scale>1e-15)
        atomic_json(out/'gradient-check.json',result);print(json.dumps(result,indent=2));return
    while iteration<a.iterations:
        g=np.asarray(grad);norm=float(np.linalg.norm(g))
        if norm==0: raise RuntimeError('Zero gradient; refusing to record a false optimization step')
        direction=-g/norm
        previous_value=value
        accepted=None; step=a.initial_step
        rejected_connectivity=0
        for ls in range(24):
            proposal=jnp.clip(q+step*jnp.asarray(direction),-a.logit_bound,a.logit_bound)
            if a.require_connected_binary and not binary_feasibility(density_for(proposal),vars(a))[0]:
                rejected_connectivity+=1;step*=.5
                if step<a.min_step:break
                continue
            print(json.dumps(dict(stage='line-search',iteration=iteration+1,trial=ls+1,step_size=step)),flush=True)
            candidate=float(ev(proposal))
            if np.isfinite(candidate) and candidate < value-1e-10*max(abs(value),1e-12):
                accepted=(proposal,candidate,step,ls+1);break
            step*=.5
            if step<a.min_step:break
        if accepted is None:
            raise RuntimeError(f'Bounded backtracking found no admissible descending design; {rejected_connectivity} proposals failed connectivity')
        q,new_value,step,nsearch=accepted; iteration+=1
        rho=density_for(q)
        # Store every actual candidate; tied tile identity is explicit in the file.
        design_file=f'design-{iteration:05d}.npz'
        atomic_npz(out/design_file,logits=np.asarray(q),rho_cell=rho,cell_shape=np.asarray(a.cell),
                   tile_repeats=np.asarray((3,3,1)),projected_mean=rho.mean())
        print(json.dumps(dict(stage='accepted-gradient',iteration=iteration)),flush=True)
        value,grad=vg(q);value=float(value)
        if not np.isfinite(value) or not np.isfinite(np.asarray(grad)).all(): raise FloatingPointError('Accepted design produced nonfinite objective/gradient')
        record=dict(iteration=iteration,objective=value,previous_objective=previous_value,
                    gradient_norm=float(jnp.linalg.norm(grad)),step_size=step,line_search_evaluations=nsearch,
                    connectivity_rejections=rejected_connectivity,binary_connectivity_required=a.require_connected_binary,
                    design_file=design_file,projected_mean=float(rho.mean()),projected_min=float(rho.min()),projected_max=float(rho.max()),
                    actual_core_mass_kg=float(rho.mean()*9*np.prod(a.cell_size)*a.solid_density),
                    fixed_volume_fraction=float(a.volume_fraction),nominal_total_mass_kg=plan['nominal_total_mass_kg'],
                    full_domain_tiles=9)
        history.append(record);atomic_json(history_path,history)
        atomic_npz(checkpoint,q=np.asarray(q),iteration=np.asarray(iteration),signature=np.asarray(signature))
        atomic_json(out/'summary.json',dict(plan=plan,signature=signature,iterations_completed=iteration,
                    objective=value,gradient_norm=record['gradient_norm'],last_design=design_file,
                    material='soft-material proxy; thresholded geometry needs explicit full-domain validation',history=str(history_path)))
        print(json.dumps(record),flush=True)

if __name__=='__main__':main()
