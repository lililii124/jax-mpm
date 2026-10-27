from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from examples.inverse_orthotropic_svk import (
    DOMAIN, LENGTHS, TRUE_ENGINEERING, boundary_case, make_body, response_case,
    sensor_stencil, stiffness_to_engineering, decode_stiffness,
)

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def main(argv=None):
    ap=argparse.ArgumentParser(description='One directional JAX AD/central-FD audit at a saved orthotropic MPM fit.')
    ap.add_argument('--fit-result',type=Path,required=True);ap.add_argument('--fit-result-sha256',required=True)
    ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--checkpoint-sha256',required=True)
    ap.add_argument('--training-observations',type=Path,required=True);ap.add_argument('--observations-sha256',required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--epsilon',type=float,default=2e-4)
    a=ap.parse_args(argv)
    for p,expected in ((a.fit_result,a.fit_result_sha256),(a.checkpoint,a.checkpoint_sha256),(a.training_observations,a.observations_sha256)):
        if sha(p)!=expected: raise ValueError(f'SHA-256 mismatch: {p}')
    result=json.loads(a.fit_result.read_text(encoding='utf-8'))
    if result.get('status')!='completed': raise ValueError('fit result is not completed')
    source=Path(__file__).resolve().parents[1]
    inverse_source=source/'examples/inverse_orthotropic_svk.py'
    inverse_sha=sha(inverse_source)
    if result.get('source_sha256',{}).get('examples/inverse_orthotropic_svk.py')!=inverse_sha:
        raise ValueError('inverse solver source differs from the fitted result')
    training=result['training'];grid=int(training['grid']);ppc=float(training['ppc'])
    steps=int(training['steps']);dt=float(training['dt_s']);kinds=tuple(training['case_names'])
    with np.load(a.checkpoint,allow_pickle=False) as cp:
        theta=jnp.asarray(cp['best_theta'],dtype=jnp.float64);iteration=int(cp['iteration']);best=float(cp['best_loss'])
    if not math.isclose(best,float(result['loss']['best']),rel_tol=1e-12,abs_tol=1e-12):
        raise ValueError('checkpoint best loss differs from result')
    estimate=stiffness_to_engineering(np.asarray(decode_stiffness(theta)))
    for key,value in result['estimated_engineering_constants'].items():
        if not math.isclose(estimate[key],float(value),rel_tol=2e-12,abs_tol=1e-10):raise ValueError(f'checkpoint estimate mismatch: {key}')
    with np.load(a.training_observations,allow_pickle=False) as data:
        obs_u=jnp.asarray(data['observed_u']);obs_f=jnp.asarray(data['observed_force']);obs_m=jnp.asarray(data['observed_moment'])
        sensor_positions=data['sensor_positions'].copy();noise=jnp.asarray(data['noise_std'])
    cfg,ref,xyz,_=make_body(grid,ppc,dt)
    ids,weights,coordinates=sensor_stencil(xyz,len(sensor_positions),offset=0)
    if not np.array_equal(coordinates,sensor_positions):raise ValueError('training sensor coordinates changed')
    amplitudes={'tension_x':.004,'tension_y':.003,'tension_z':.002,'shear_xy':.003,'shear_xz':.0025,'shear_yz':.0025,'bend_3point':.0025,'torsion_x':.018}
    period=max(12,steps)
    cases=[boundary_case(ref,cfg,k,steps,period,amplitudes[k]) for k in kinds]
    boundary_data=tuple(jnp.stack([b[i] for b in cases]) for i in range(7))
    sensor_data=(jnp.broadcast_to(jnp.asarray(ids),(len(kinds),*ids.shape)),jnp.broadcast_to(jnp.asarray(weights),(len(kinds),*weights.shape)))
    def objective(theta,reference,bd,sensor_ids,sensor_weights,observed_u,observed_f,observed_m,noise_scales):
        def one_case(inputs):
            boundary,si,sw,target_u,target_f,target_m=inputs
            u,f,m,_=response_case(theta,cfg,reference,boundary,si,sw)
            return jnp.stack((jnp.mean(((u-target_u)/noise_scales[0])**2),
                              jnp.mean(((f-target_f)/noise_scales[1])**2),
                              jnp.mean(((m-target_m)/noise_scales[2])**2)))
        return jnp.mean(jax.lax.map(jax.checkpoint(one_case),(bd,sensor_ids,sensor_weights,observed_u,observed_f,observed_m)))
    vg=jax.jit(jax.value_and_grad(objective,argnums=0));obj=jax.jit(objective)
    direction=np.random.default_rng(20261027).normal(size=9);direction/=np.linalg.norm(direction);direction=jnp.asarray(direction)
    start=time.monotonic()
    value,gradient=vg(theta,ref,boundary_data,*sensor_data,obs_u,obs_f,obs_m,noise);jax.block_until_ready(value)
    auto=float(jnp.dot(gradient,direction))
    plus=obj(theta+a.epsilon*direction,ref,boundary_data,*sensor_data,obs_u,obs_f,obs_m,noise)
    minus=obj(theta-a.epsilon*direction,ref,boundary_data,*sensor_data,obs_u,obs_f,obs_m,noise)
    jax.block_until_ready(plus);fd=float((plus-minus)/(2*a.epsilon))
    rel=abs(auto-fd)/max(abs(auto),abs(fd),1e-12)
    out={'status':'completed','objective':'same eight-case sequential lax.map/remat noise-whitened MSE used by inverse CLI',
         'grid':grid,'ppc':ppc,'particles':int(training['particles']),'steps':steps,'dt_s':dt,
         'parameter_source':'checkpoint.best_theta','checkpoint_iteration':iteration,'fit_best_loss':best,
         'fit_result_sha256':a.fit_result_sha256,'checkpoint_sha256':a.checkpoint_sha256,
         'training_observations_sha256':a.observations_sha256,
         'inverse_source_sha256':inverse_sha,'direction':np.asarray(direction).tolist(),'epsilon':a.epsilon,
         'objective_value':float(value),'autodiff_directional':auto,'central_difference_directional':fd,
         'relative_error':rel,'acceptance_threshold':5e-3,'passed':bool(np.isfinite(rel) and rel<=5e-3),
         'jax':jax.__version__,'numpy':np.__version__,'elapsed_s':time.monotonic()-start}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2,sort_keys=True))
    if not out['passed']:raise RuntimeError(f'directional gradient check failed: {rel}')

if __name__=='__main__':main()
