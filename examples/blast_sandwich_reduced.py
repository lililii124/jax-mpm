"""Fixed-basis linear transient design; complete MPM validation is still required."""
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import math
import time

import numpy as np

from .blast_sandwich_design import (Settings, build_model, atomic_json, atomic_npz,
    numeric, CASE_NAMES, source_signature as mpm_source_signature)


def build_reduced(settings, *, order=3, dt=2e-6, assembly_chunk=256, time_chunk=16,ks_relative_error=.005,
                  extra_basis=None):
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from jax import lax
    from mpm_jax.reduced_tl import projection,newmark,response
    if not np.isfinite(dt) or dt<=0 or time_chunk<1:raise ValueError('Positive ROM time step and chunk required')
    base=build_model(settings)
    projected=projection(base,order,assembly_chunk,extra_basis=extra_basis)
    baseline_mass=base['plan']['uniform_mass_kg']

    def responses(M,K,mass,time_step):
        def one(load):
            times,u,v,energy=newmark(M,K,load,settings,time_step)
            metrics=response(projected,u,settings,time_chunk,ks_relative_error)
            error=jnp.max(jnp.abs(energy[:,0]+energy[:,1]-energy[:,2]))
            relative=error/jnp.maximum(jnp.max(jnp.abs(energy)),1e-30)
            return jnp.concatenate((metrics,jnp.asarray([error,relative])))
        cases=lax.map(one,projected['loads'],batch_size=1)
        displacement_bound=jnp.max(cases[:,1]);stress_bound=jnp.max(cases[:,2])
        constraints=jnp.asarray([displacement_bound-1,stress_bound-1,mass/baseline_mass-1])
        objective=mass/baseline_mass if settings.stage=='min-mass' else displacement_bound
        merit=objective+100*jnp.sum(jax.nn.relu(constraints)**2)
        return merit,dict(cases=cases,mass_kg=mass,response=displacement_bound,
            actual_peak_back_m=jnp.max(cases[:,3]),actual_peak_skin_Pa=jnp.max(cases[:,4]),
            displacement_bound=displacement_bound,stress_bound=stress_bound,constraints=constraints)

    def metrics(q):
        M,K,mass=projected['operators'](q)
        return responses(M,K,mass,dt)
    def signature():
        root=Path(__file__).resolve().parents[1]
        mpm_digest,source=mpm_source_signature(settings)
        for name in ('examples/blast_sandwich_reduced.py','mpm_jax/reduced_tl.py'):
            source[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
        import jaxlib
        binding=dict(settings=asdict(settings),order=order,dt=dt,time_chunk=time_chunk,ks_relative_error=ks_relative_error,
            assembly_chunk=assembly_chunk,basis=projected['basis_sha256'],sources=source,
            jax=jax.__version__,jaxlib=jaxlib.__version__,numpy=np.__version__)
        return hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest(),binding
    return dict(base=base,projected=projected,settings=settings,dt=dt,order=order,
        metrics=metrics,responses=responses,signature=signature,q0=base['q0'],ks_relative_error=ks_relative_error,
        time_chunk=time_chunk)


def checked(record,settings,baseline_mass):
    cases=np.asarray(record['cases'])
    valid=bool(np.isfinite(cases).all() and np.max(cases[:,6])<1e-8)
    mass_ok=(abs(record['mass_kg']/baseline_mass-1)<=1e-10 if settings.stage=='fixed-mass'
        else record['mass_kg']<=baseline_mass*(1+1e-10))
    feasible=bool(valid and mass_ok
        and record['actual_peak_back_m']<=settings.displacement_limit
        and record['actual_peak_skin_Pa']<=settings.stress_limit)
    return valid,feasible


def forward(model,q,output,*,dt=None):
    import jax
    import jax.numpy as jnp
    from mpm_jax.reduced_tl import newmark,response
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    step=model['dt'] if dt is None else dt
    start=time.perf_counter()
    M,K,mass=jax.jit(model['projected']['operators'])(jnp.asarray(q))
    M,K,mass=map(jax.block_until_ready,(M,K,mass))
    matrix_seconds=time.perf_counter()-start
    matrix_M,matrix_K=np.asarray(M),np.asarray(K)
    np.linalg.cholesky(matrix_M);np.linalg.cholesky(matrix_K)
    # Assembly and field reconstruction need different large constants. Their
    # executable buffers need not coexist for this forward-only entry point.
    jax.clear_caches()
    solve=jax.jit(lambda M,K,load:newmark(M,K,load,model['settings'],step))
    measure=jax.jit(lambda u:response(model['projected'],u,model['settings'],model['time_chunk'],model['ks_relative_error']))
    cases=[]
    for k,name in enumerate(CASE_NAMES):
        times,u,v,energy=solve(M,K,model['projected']['loads'][k])
        metrics=measure(u)
        error=np.asarray(energy)[:,0]+np.asarray(energy)[:,1]-np.asarray(energy)[:,2]
        record=dict(name=name,steps=len(times)-1,time_s=float(times[-1]),
            response=float(metrics[0]),displacement_bound=float(metrics[1]),stress_bound=float(metrics[2]),
            peak_back_m=float(metrics[3]),peak_skin_Pa=float(metrics[4]),
            maximum_energy_residual_J=float(np.max(np.abs(error))),
            relative_energy_residual=float(np.max(np.abs(error))/max(np.max(np.abs(energy)),1e-30)))
        if not np.isfinite(list(v for key,v in record.items() if key!='name')).all() or record['relative_energy_residual']>1e-8:
            raise RuntimeError('ROM finite/linear-energy qualification failed')
        atomic_npz(output/(name+'.npz'),time=np.asarray(times),displacement=np.asarray(u),velocity=np.asarray(v),
            energy=np.asarray(energy),mass=matrix_M,stiffness=matrix_K,controls=np.asarray(q))
        cases.append(record)
    signature,binding=model['signature']()
    atomic_npz(output/'basis.npz',basis=model['projected']['basis_numpy'],
        reference_nodes=model['base']['support'].nodes,transform=model['projected']['transform'])
    result=dict(model='linear fixed-basis Ritz ROM of compact MPM tangent',
        enrichment=model['projected']['enrichment'],
        full_mpm_verified=False,nonlinear_blast_qualified=False,complete_rom_horizon=True,
        settings=asdict(model['settings']),rank=model['projected']['rank'],order=model['order'],
        actual_dt_s=model['settings'].horizon/(len(times)-1),mass_kg=float(mass),
        matrix_seconds=matrix_seconds,wall_seconds=time.perf_counter()-start,cases=cases,
        signature=signature,binding=binding,
        basis_families=['bending','x/y core zigzag shear','core thickness compression','x/y membrane'],
        pressure_impulse_per_area_Pa_s=model['settings'].peak_pressure*model['settings'].pulse_duration/2,
        nominal_resultant_impulse_Ns=model['settings'].length*model['settings'].width*model['settings'].peak_pressure*model['settings'].pulse_duration/2)
    result['ks_maximum_overestimate']=dict(relative_to_response_limit=model['ks_relative_error'],
        displacement_m=model['ks_relative_error']*model['settings'].displacement_limit,
        stress_Pa=model['ks_relative_error']*model['settings'].stress_limit,
        formula='rho=log(total valid space-time sample count)/relative_error')
    if model['projected']['enrichment']['retained_modes']:
        result['basis_families'].append('full-MPM snapshot residual POD')
    atomic_json(output/'summary.json',result)
    jax.clear_caches()
    return result


def derivative_check(model,q,step=1e-3):
    import jax
    import jax.numpy as jnp
    objective=jax.jit(lambda q:model['metrics'](q)[0])
    value,gradient=jax.jit(jax.value_and_grad(lambda q:model['metrics'](q)[0]))(jnp.asarray(q))
    gradient=np.asarray(gradient)
    size=np.linalg.norm(gradient)
    if not np.isfinite(gradient).all() or size==0:raise RuntimeError('Nonfinite or zero resolved ROM gradient')
    direction=gradient/size
    # The directional finite differences only need the primal executable.
    jax.clear_caches()
    fd=float((objective(q+step*direction)-objective(q-step*direction))/(2*step))
    half=float((objective(q+.5*step*direction)-objective(q-.5*step*direction))/step)
    scale=max(abs(size),abs(fd),abs(half),1e-30)
    error=max(abs(size-half),abs(fd-half))/scale
    resolved=scale>64*np.finfo(float).eps*max(abs(float(value)),1e-30)/step
    return dict(objective=float(value),gradient_size=len(q),gradient_l2=float(size),
        directional_ad=float(size),central_fd=fd,central_fd_half_step=half,
        relative_error=error,resolved=bool(resolved),passed=bool(resolved and error<2e-3),
        gradient_sha256=hashlib.sha256(gradient.tobytes()).hexdigest()),gradient


def bound_responses(model,output,*,dt=None):
    """Check the admissible design bounds, retaining the fixed perimeter core."""
    import jax
    import jax.numpy as jnp
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    jax.clear_caches()
    settings=model['settings'];geo=model['base']['geometry']
    step=model['dt'] if dt is None else dt
    assemble=jax.jit(model['projected']['operators_for_fraction'])
    evaluate=jax.jit(lambda M,K,mass:model['responses'](M,K,mass,step))
    records=[]
    for value in (settings.phi_min,settings.phi_max):
        phi=np.where(geo['skin'],1.,np.where(geo['design'],value,settings.target_phi))
        M,K,mass=assemble(jnp.asarray(phi))
        _,record=evaluate(M,K,mass);record=numeric(record)
        constraints_ok=bool(record['actual_peak_back_m']<=settings.displacement_limit
            and record['actual_peak_skin_Pa']<=settings.stress_limit)
        record.update(designable_core_phi=value,fixed_perimeter_core_phi=settings.target_phi,
            all_response_constraints_satisfied=constraints_ok,complete_rom_horizon=True,
            full_mpm_verified=False,nonlinear_blast_qualified=False)
        atomic_npz(output/f'phi-{value:g}.npz',mass=np.asarray(M),stiffness=np.asarray(K),fraction=phi)
        records.append(record)
    result=dict(bounds=records,rank=model['projected']['rank'],dt=step,settings=asdict(settings),
        minimum_mass_trivial_in_this_rom=records[0]['all_response_constraints_satisfied'],
        scope='If the all-lower admissible field satisfies both responses, minimum mass is the all-lower field; no complex distribution is justified by these constraints.')
    atomic_json(output/'bounds.json',result)
    return result


def optimize(model,output,*,iterations=1,resume=False,learning_rate=.05,initial_controls=None,initial_evaluation=None):
    import jax
    import jax.numpy as jnp
    if iterations<0 or learning_rate<=0:raise ValueError('Invalid optimization budget')
    jax.clear_caches()
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    signature,binding=model['signature']();checkpoint=output/'optimizer.npz'
    q=jnp.asarray(model['q0'] if initial_controls is None else initial_controls)
    if q.shape!=model['q0'].shape or not np.isfinite(q).all():raise ValueError('Invalid initial controls')
    first=jnp.zeros_like(q);second=jnp.zeros_like(q)
    start=0;history=[]
    if resume:
        with np.load(checkpoint,allow_pickle=False) as saved:
            if str(saved['signature'])!=signature:raise ValueError('ROM optimizer source/basis/runtime changed')
            q,first,second=(jnp.asarray(saved[name]) for name in ('q','first','second'))
            start=int(saved['iteration']);history=json.loads(str(saved['history']))
    elif any(output.iterdir()):raise ValueError('Fresh optimizer directory required')
    evaluate=jax.jit(model['metrics'])
    value_gradient=jax.jit(jax.value_and_grad(lambda q:model['metrics'](q)[0]))
    def save(iteration):
        atomic_npz(checkpoint,q=np.asarray(q),first=np.asarray(first),second=np.asarray(second),
            iteration=iteration,history=np.asarray(json.dumps(history)),signature=np.asarray(signature))
    save(start)
    for iteration in range(start,start+iterations):
        if iteration==0 and initial_evaluation is not None and not resume:
            if not np.array_equal(np.asarray(q),np.asarray(initial_evaluation['controls'])):
                raise ValueError('Cached derivative belongs to another design')
            value=jnp.asarray(initial_evaluation['value']);gradient=jnp.asarray(initial_evaluation['gradient'])
        else:value,gradient=value_gradient(q)
        if not np.isfinite(float(value)) or not np.isfinite(gradient).all():raise RuntimeError('Nonfinite ROM gradient')
        _,old=evaluate(q);old=numeric(old)
        _,feasible=checked(old,model['settings'],model['base']['plan']['uniform_mass_kg'])
        objective_key='mass_kg' if model['settings'].stage=='min-mass' else 'response'
        original_objective=old[objective_key]
        first=.9*first+.1*gradient;second=.999*second+.001*gradient**2
        direction=-(first/(1-.9**(iteration+1)))/(jnp.sqrt(second/(1-.999**(iteration+1)))+1e-8)
        accepted=False
        for backtrack in range(6):
            candidate=jnp.clip(q+learning_rate*(.5**backtrack)*direction,-12.,12.)
            candidate_value,record=evaluate(candidate);record=numeric(record)
            valid,candidate_feasible=checked(record,model['settings'],model['base']['plan']['uniform_mass_kg'])
            improvement=(record[objective_key]<original_objective if feasible else float(candidate_value)<float(value))
            if valid and improvement and (not feasible or candidate_feasible):
                q=candidate;value=candidate_value;old=record;accepted=True;break
        history.append(dict(iteration=iteration+1,accepted=accepted,merit=float(value),**old))
        save(iteration+1);atomic_json(output/'history.json',history)
    value,record=evaluate(q);record=numeric(record)
    valid,feasible=checked(record,model['settings'],model['base']['plan']['uniform_mass_kg'])
    result=dict(merit=float(value),metrics=record,valid=valid,feasible=feasible,controls=np.asarray(q).tolist(),
        iterations=start+iterations,convergence_claim=False,full_mpm_verified=False,
        nonlinear_blast_qualified=False,signature=signature,binding=binding)
    atomic_json(output/'summary.json',result)
    return result


def optimize_sqp(model,output,*,iterations=5,resume=False,initial_controls=None,tolerance=1e-7,
                 objective_scale=1.,evaluator='legacy'):
    """Constrained design with sequential VJPs and an independent feasible state."""
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize
    if iterations<1 or tolerance<=0 or not np.isfinite(objective_scale) or objective_scale<=0:
        raise ValueError('Positive SQP budget, tolerance and objective scale required')
    if evaluator not in ('legacy','case-pullback'):raise ValueError('Unknown SQP evaluator')
    casewise=None
    if evaluator!='legacy' or model.get('_sqp_evaluator_kind','legacy')!='legacy':
        from mpm_jax.reduced_case_pullback import configure_sqp_evaluator,FixedMassCasePullback
        configure_sqp_evaluator(model,evaluator)
        if evaluator=='case-pullback':casewise=FixedMassCasePullback(model)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    signature,binding=model['signature']();checkpoint=output/'sqp.npz'
    q=np.asarray(model['q0'] if initial_controls is None else initial_controls).copy()
    baseline=model['base']['plan']['uniform_mass_kg']
    history=[];last_feasible=None;best_feasible_objective=None;prior_iterations=0
    if resume:
        with np.load(checkpoint,allow_pickle=False) as saved:
            if str(saved['signature'])!=signature:raise ValueError('SQP source/basis/runtime changed')
            q=saved['q'].copy();history=json.loads(str(saved['history']))
            prior_iterations=int(saved['iterations'])
            if bool(saved['has_feasible']):
                last_feasible=saved['last_feasible'].copy()
                best_feasible_objective=float(saved['best_feasible_objective'])
    elif any(output.iterdir()):raise ValueError('Fresh SQP directory required')
    if q.shape!=model['q0'].shape or not np.isfinite(q).all():raise ValueError('Invalid SQP initial controls')
    old_history_size=len(history)
    fixed_mass=model['settings'].stage=='fixed-mass'
    output_count=4 if fixed_mass else 8

    def outputs(q):
        _,record=model['metrics'](q)
        primary=record['mass_kg']/baseline if model['settings'].stage=='min-mass' else record['response']
        # Fixed mass is enforced by the material-map custom root. The worst
        # displacement bound is both objective and constraint, so its VJP is
        # reused; only the three case stress rows need additional VJPs.
        values=(jnp.concatenate((jnp.atleast_1d(primary),1-record['cases'][:,2]))
            if fixed_mass else jnp.concatenate((jnp.atleast_1d(primary),1-record['cases'][:,1],
            1-record['cases'][:,2],jnp.atleast_1d(1+1e-10-record['mass_kg']/baseline))))
        return values,record
    @jax.jit
    def one_vjp(q,seed):
        value,pullback,record=jax.vjp(outputs,q,has_aux=True)
        return value,pullback(seed)[0],record
    cache={}
    def evaluation(x,derivatives=False):
        if 'q' not in cache or not np.array_equal(x,cache['q']):
            cache.clear();cache['q']=x.copy()
            seed=np.zeros(output_count);seed[0]=1.
            if casewise is None:values,row,record=one_vjp(jnp.asarray(x),jnp.asarray(seed))
            else:
                evaluated=casewise(x);values,row,record=evaluated['values'],evaluated['first_row'],evaluated['record']
            cache['values']=np.asarray(values);cache['first_row']=np.asarray(row)
            cache['record']=numeric(record)
            if not np.isfinite(cache['values']).all():raise RuntimeError('Nonfinite SQP response')
            print(json.dumps(dict(event='sqp-evaluation',objective=float(cache['values'][0]),
                minimum_constraint=float(np.min(cache['values'][1:])))),flush=True)
        if derivatives and 'jacobian' not in cache:
            if casewise is not None:
                cache['jacobian']=casewise(x)['jacobian']
                if not (output/'initial-jacobian.npz').exists():
                    atomic_npz(output/'initial-jacobian.npz',q=x,values=cache['values'],jacobian=cache['jacobian'])
                return cache
            rows=[cache['first_row']]
            for index,seed in enumerate(np.eye(output_count)[1:],1):
                values,row,_=one_vjp(jnp.asarray(x),jnp.asarray(seed))
                if not np.allclose(values,cache['values'],rtol=1e-10,atol=1e-13):raise RuntimeError('SQP VJP/primal mismatch')
                rows.append(np.asarray(row))
                print(json.dumps(dict(event='sqp-jacobian-row',row=index,of=output_count-1)),flush=True)
            cache['jacobian']=np.asarray(rows)
            if not np.isfinite(cache['jacobian']).all():raise RuntimeError('Nonfinite SQP constraint Jacobian')
            if not (output/'initial-jacobian.npz').exists():
                atomic_npz(output/'initial-jacobian.npz',q=x,values=cache['values'],jacobian=cache['jacobian'])
        return cache
    def constraints(x):
        values=evaluation(x)['values']
        return np.r_[1-values[0],values[1:]] if fixed_mass else values[1:]
    def constraint_jacobian(x):
        jacobian=evaluation(x,True)['jacobian']
        return np.vstack((-jacobian[0],jacobian[1:])) if fixed_mass else jacobian[1:]
    def save(x,iteration):
        atomic_npz(checkpoint,q=x,last_feasible=q if last_feasible is None else last_feasible,
            has_feasible=last_feasible is not None,iterations=iteration,
            best_feasible_objective=np.nan if best_feasible_objective is None else best_feasible_objective,
            history=np.asarray(json.dumps(history)),signature=np.asarray(signature))
    def callback(x):
        nonlocal last_feasible,best_feasible_objective
        evaluated=evaluation(x);values=evaluated['values'];record=evaluated['record']
        # SQP may explore infeasible iterates; retain a separate strict candidate.
        valid,actual_feasible=checked(record,model['settings'],baseline)
        feasible=bool(valid and actual_feasible and np.min(constraints(x))>=0)
        iteration=prior_iterations+len(history)-old_history_size+1
        improved=bool(feasible and (best_feasible_objective is None or values[0]<best_feasible_objective))
        if feasible:
            atomic_npz(output/f'feasible-{iteration:06d}.npz',q=x,signature=np.asarray(signature))
            atomic_json(output/f'feasible-{iteration:06d}.json',dict(iteration=iteration,
                metrics=record,mass_relative_change=record['mass_kg']/baseline-1))
        if improved:
            last_feasible=x.copy();best_feasible_objective=float(values[0])
            phi,density,young,_=model['base']['properties'](jnp.asarray(x))
            atomic_npz(output/'latest-feasible-material.npz',positions=model['base']['support'].positions,
                reference_volume=model['base']['support'].volume,fraction=np.asarray(phi),
                density=np.asarray(density),young=np.asarray(young),designable=model['base']['geometry']['design'],
                skin=model['base']['geometry']['skin'],controls=x,iteration=iteration,signature=np.asarray(signature))
        history.append(dict(iteration=prior_iterations+len(history)-old_history_size+1,objective=float(values[0]),
            constraints=constraints(x).tolist(),ks_feasible=feasible,selected_best_feasible=improved,metrics=record,
            controls_sha256=hashlib.sha256(x.tobytes()).hexdigest()))
        save(x,prior_iterations+len(history)-old_history_size)
        atomic_json(output/'history.json',history)
        print(json.dumps(dict(event='sqp-iterate',iteration=iteration,ks_feasible=feasible,
            objective=float(values[0]),mass_kg=record['mass_kg'],peak_back_m=record['actual_peak_back_m'],
            peak_skin_Pa=record['actual_peak_skin_Pa'])),flush=True)
    initial=evaluation(q)['values'].copy()
    atomic_json(output/'initial.json',dict(objective=float(initial[0]),constraints=constraints(q).tolist(),
        metrics=evaluation(q)['record'],controls_sha256=hashlib.sha256(q.tobytes()).hexdigest()))
    if np.min(constraints(q))>=0 and (best_feasible_objective is None or initial[0]<best_feasible_objective):
        last_feasible=q.copy();best_feasible_objective=float(initial[0])
    save(q,prior_iterations)
    # A positive objective scale changes SQP conditioning, not the physical
    # minimizer or the unscaled response constraints and saved metrics.
    result=minimize(lambda x:objective_scale*float(evaluation(x)['values'][0]),q,
        jac=lambda x:objective_scale*evaluation(x,True)['jacobian'][0],method='SLSQP',
        bounds=[(-12.,12.)]*len(q),
        constraints=[dict(type='ineq',fun=constraints,jac=constraint_jacobian)],callback=callback,
        options=dict(maxiter=iterations,ftol=tolerance,disp=True))
    final=np.asarray(result.x);callback(final)
    selected=last_feasible if last_feasible is not None else final
    record=evaluation(selected)['record']
    valid,feasible=checked(record,model['settings'],baseline)
    atomic_npz(output/'selected.npz',q=selected,signature=np.asarray(signature))
    phi,density,young,_=model['base']['properties'](jnp.asarray(selected))
    phi,density,young=map(np.asarray,(phi,density,young))
    design=model['base']['geometry']['design']
    atomic_npz(output/'selected-material.npz',positions=model['base']['support'].positions,
        reference_volume=model['base']['support'].volume,fraction=phi,density=density,young=young,
        designable=design,skin=model['base']['geometry']['skin'],controls=selected)
    report=dict(algorithm='SLSQP with sequential output VJPs',optimizer_success=bool(result.success),
        objective_scale=objective_scale,
        optimizer_status=int(result.status),optimizer_message=str(result.message),
        initial_objective=float(initial[0]),metrics=record,valid=valid,feasible=feasible,
        controls=selected.tolist(),has_ks_feasible_candidate=last_feasible is not None,
        best_feasible_objective=best_feasible_objective,
        designable_fraction_min=float(np.min(phi[design])),designable_fraction_max=float(np.max(phi[design])),
        convergence_claim=False,full_mpm_verified=False,nonlinear_blast_qualified=False,
        continuation='Control/history warm start; SLSQP internal BFGS matrix resets at each bounded invocation.',
        signature=signature,binding=binding)
    atomic_json(output/'summary.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('forward','check','optimize','bounds'))
    parser.add_argument('--xy-step',type=float,default=.01)
    parser.add_argument('--z-step',type=float,default=.001)
    parser.add_argument('--controls',type=int,nargs=3,default=(24,24,3))
    parser.add_argument('--order',type=int,default=3)
    parser.add_argument('--dt',type=float,default=2e-6)
    parser.add_argument('--stage',choices=('fixed-mass','min-mass'),default='fixed-mass')
    parser.add_argument('--output',type=Path,default=Path('outputs/blast-sandwich-reduced'))
    parser.add_argument('--iterations',type=int,default=1)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--algorithm',choices=('sqp','adam'),default='sqp')
    parser.add_argument('--objective-scale',type=float,default=1000.)
    parser.add_argument('--enrichment-snapshots',type=Path)
    parser.add_argument('--design-checkpoint',type=Path)
    args=parser.parse_args()
    settings=Settings(xy_step=args.xy_step,z_step=args.z_step,controls=tuple(args.controls),stage=args.stage)
    extra_basis=None;enrichment_nodes=None
    if args.enrichment_snapshots:
        with np.load(args.enrichment_snapshots,allow_pickle=False) as saved:
            fields=saved['displacement'];training=saved['training'];enrichment_nodes=saved['nodes']
        if training.dtype!=np.bool_ or training.shape!=(len(fields),) or not training.any():
            raise ValueError('Enrichment snapshots require an explicit nonempty Boolean training mask')
        extra_basis=fields[training].transpose(1,2,0)
    model=build_reduced(settings,order=args.order,dt=args.dt,extra_basis=extra_basis)
    if enrichment_nodes is not None:np.testing.assert_array_equal(enrichment_nodes,model['base']['support'].nodes)
    q=model['q0']
    if args.design_checkpoint:
        with np.load(args.design_checkpoint,allow_pickle=False) as saved:q=saved['q'].copy()
        if q.shape!=model['q0'].shape:raise ValueError('Control field shape mismatch')
    if args.action=='forward':result=forward(model,q,args.output)
    elif args.action=='check':
        result,_=derivative_check(model,q)
        args.output.mkdir(parents=True,exist_ok=True);atomic_json(args.output/'derivative.json',result)
        if not result['passed']:raise RuntimeError('ROM derivative failed')
    elif args.action=='bounds':result=bound_responses(model,args.output)
    else:
        solver=optimize_sqp if args.algorithm=='sqp' else optimize
        options=dict(iterations=args.iterations,resume=args.resume,initial_controls=q if args.design_checkpoint else None)
        if args.algorithm=='sqp':options['objective_scale']=args.objective_scale
        result=solver(model,args.output,**options)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
