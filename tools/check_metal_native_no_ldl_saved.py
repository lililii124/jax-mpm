"""One bounded actual-input test of native column-scaled/deflated JAX QP.

Unlike the old diagonal-only incomplete125ns solve, this includes exact body
polynomial deflation. It first checks one saved physical contact impulse; a
failed P0 action stops before the contact QP. Iteration bounds are not raised.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from mpm_jax.fragment_v6_matrixfree_projection import quadratic_reconstruction,polynomial_modes,reconstructed_apply,reconstructed_transpose,consistent_project,physical_contact,physical_sample,physical_adjoint,cone_project,intersect_project


def main():
    begun=time.monotonic();out=Path('/work/result');out.mkdir();parent=Path('/original/result/step-22/accepted.npz');gold_path=Path('/gold/result/midpoint-raw.npz')
    parent_sha=hashlib.file_digest(parent.open('rb'),'sha256').hexdigest();assert parent_sha=='3579600dd7d78cfa728b3478f071d416d15e6226c5799012d37d346aa0530809'
    with np.load(parent,allow_pickle=False) as z:old={k:z[k] for k in z.files}
    with np.load(gold_path,allow_pickle=False) as z:gold={k:z[k] for k in ('trial','following','physical','reaction','multiplier','geometry_x','geometry_F')}
    dx=jnp.asarray(.00025);convert=lambda key:jnp.asarray(old[key])
    allowed=quadratic_reconstruction(*(convert(k) for k in ('x','mass','body_id','F','reference_root')),dx,convert('candidate_fixed_particles'))
    modes,_,mode_report=jax.jit(polynomial_modes)(allowed,dx);assert bool(allowed['valid']) and bool(mode_report['valid'])
    rows=len(allowed['column_norm'])
    def projector(y,p,T):
        return consistent_project(y,lambda z:reconstructed_apply(p,z),lambda z:reconstructed_transpose(p,z,rows),T,lambda z:z,lambda z:z,iterations=64)
    action=jax.jit(projector);probe=jnp.asarray(gold['physical']);t=time.monotonic();probe_result,probe_report=action(probe,allowed,modes);probe_result.block_until_ready();probe_seconds=time.monotonic()-t
    value=np.asarray(probe_result);expected=gold['physical']+gold['reaction'];error=value-expected
    normal=max(np.asarray(probe_report['physical_normal_relative_residual']));state_relative=float(np.linalg.norm(error)/np.linalg.norm(gold['trial']));work=float(np.sum(error*gold['following']))
    tol=1.414599609375e-8
    probe_valid=bool(normal<=1e-10 and state_relative<=1e-10 and abs(work)<=tol)
    initial=dict(stage='single-physical-impulse-projection',passed=probe_valid,source_checkpoint_sha256=parent_sha,iterations=np.asarray(probe_report['iterations']).tolist(),solver_flags=np.asarray(probe_report['converged']).tolist(),physical_normal_relative_residual=float(normal),combined_state_relative_difference=state_relative,actual_work_difference_J=work,compile_and_solve_seconds=probe_seconds,iteration_cap=64,preconditioner='Column equilibration and exact polynomial deflation; no LDL/regularization',scope='Saved actual impulse only; no force/material/trajectory replay. A failed probe is not retried with more iterations.')
    (out/'probe-review.json').write_text(json.dumps(initial,indent=2)+'\n');np.savez_compressed(out/'probe-raw.npz',input=gold['physical'],reference=expected,following=value)
    print(json.dumps(initial),flush=True)
    if not probe_valid:
        (out/'review.json').write_text(json.dumps(dict(status='failed-preconditioner-probe',probe=initial,qp_attempted=False,physical_steps_executed=0),indent=2)+'\n');raise SystemExit(2)
    config=dict(old,x=gold['geometry_x'],F=gold['geometry_F'])
    full=quadratic_reconstruction(jnp.asarray(config['x']),convert('mass'),convert('body_id'),jnp.asarray(config['F']),convert('reference_root'),dx,jnp.zeros(len(old['mass']),bool))
    T,poly,mode_report=jax.jit(polynomial_modes)(full,dx);assert bool(mode_report['valid'])
    contact=physical_contact(full,T,poly,dx,allowed['fixed_grid_keys']);assert bool(contact['valid'])
    parameters=dict(allowed=allowed,modes=modes,full=full,contact=contact)
    def kernel(y,p):
        project_range=lambda x:projector(x,p['allowed'],p['modes'])
        sample=lambda x:physical_sample(p['full'],p['contact'],x)
        adjoint=lambda x:physical_adjoint(p['full'],p['contact'],x)
        project_cone=lambda x:cone_project(x,sample,adjoint,p['contact']['mobility'],sweeps=512,tolerance=1e-7,physical_rows=p['contact']['pair_count'])
        return intersect_project(y,project_range,project_cone,sample,iterations=64)
    compiled=jax.jit(kernel);trial=jnp.asarray(gold['trial']);t=time.monotonic();executable=compiled.lower(trial,parameters).compile();compile_seconds=time.monotonic()-t
    t=time.monotonic();following,report=executable(trial,parameters);following.block_until_ready();solve_seconds=time.monotonic()-t
    array=np.asarray(following);raw={k:np.asarray(v) for k,v in report.items()};physical=raw.pop('physical_impulse');reaction=raw.pop('boundary_reaction');lam=raw.pop('multiplier');summary={k:v.item() for k,v in raw.items()}
    # Independent original physical sampler is constructed only after the native
    # solve, for audit; it is not used for preparation, iteration or callbacks.
    from tools.metal_affine_projection_numpy import ConsistentAffineProjection
    from tools.metal_affine_contact_numpy import CompatibleContact
    from tools.metal_affine_dynamic_numpy import moments
    original=ConsistentAffineProjection(config['x'],config['mass'],config['body_id'],.00025,config['F'],config['reference_root'],build_solver=False);original_contact=CompatibleContact(original);m=len(original_contact.pairs)
    native_values=original_contact.sample(array);kkt=float(np.max(np.where(lam[:m]>0,abs(native_values),np.maximum(native_values,0)),initial=0));P,L=moments(config,2*physical)
    relative=float(np.linalg.norm(array-gold['following'])/np.linalg.norm(gold['following']));boundary=float(np.sum(array*2*reaction));power=float(np.sum(array*2*physical))
    passed=bool(summary['converged'] and relative<=1e-10 and kkt<=.005 and abs(boundary)<=tol and power<=tol and np.linalg.norm(L)<=1.5782228774332438e-14 and np.all(lam[m:]==0))
    result=dict(status='passed' if passed else 'failed-qp',probe=initial,physical_steps_executed=0,source_checkpoint_sha256=parent_sha,golden_npz_sha256=hashlib.file_digest(gold_path.open('rb'),'sha256').hexdigest(),particles=len(old['mass']),prepare_and_probe_seconds=time.monotonic()-begun-compile_seconds-solve_seconds,compile_seconds=compile_seconds,solve_seconds=solve_seconds,relative_velocity_difference=relative,original_KKT_m_s=kkt,physical_momentum=P.tolist(),physical_torque=L.tolist(),boundary_work_J=boundary,contact_work_J=power,report=summary,scope='Full QP preparation and iterative solve JAX64; no LDL, SciPy numerical solve or JIT callbacks. Reference sampler is post-solve audit only. Not yet installed in an impact time integrator.')
    (out/'review.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out/'jax-raw.npz',trial=gold['trial'],following=array,physical=physical,reaction=reaction,multiplier=lam);print(json.dumps(result),flush=True)
    if not passed:raise SystemExit(3)


if __name__=='__main__':main()
