"""Opt-in exact fixed-mass SQP evaluator with per-case operator pullbacks.

All original Newmark, observation, skin-stress and KS samples are retained.
Compiled functions are constructed once per model and accept dynamic controls.
"""
from pathlib import Path
import hashlib,json,time
import numpy as np
import jax
import jax.numpy as jnp

METHOD='case-pullback-v1'


def implementation_binding(model):
    return dict(method=METHOD,implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fixed_mass=True,time_chunk=model['time_chunk'],all_time_space_stress_samples_retained=True,
        max_tie_rule='Equal cotangents for exactly equal maxima, matching jnp.max.',
        mass_cotangent='Zero only for the scalar reported mass; M and K both retain their full material-map derivatives.')


def configure_sqp_evaluator(model,method='legacy'):
    """Bind the opt-in implementation into checkpoints; legacy is the default."""
    if method not in ('legacy','case-pullback'):raise ValueError('Unknown SQP evaluator')
    if method!='legacy' and model['settings'].stage!='fixed-mass':raise ValueError('Case pullback supports fixed-mass SQP only')
    base=model.setdefault('_sqp_base_signature',model['signature'])
    if method=='legacy':model['signature']=base
    else:
        descriptor=implementation_binding(model)
        def signature():
            _,binding=base();binding=dict(binding);binding['sqp_evaluator']=descriptor
            return hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest(),binding
        model['signature']=signature
    model['_sqp_evaluator_kind']=method


class FixedMassCasePullback:
    """Cached four-output/four-by-q Jacobian, with no per-q JIT construction."""
    def __init__(self,model):
        from .reduced_tl import newmark,response
        if model['settings'].stage!='fixed-mass':raise ValueError('Case pullback supports fixed-mass SQP only')
        if not jax.config.jax_enable_x64:raise ValueError('Case pullback requires JAX float64')
        self.model=model;self._cache=None;self.evaluations=0
        projected=model['projected'];settings=model['settings'];chunk=model['time_chunk']
        self._assemble=jax.jit(projected['operators'])
        @jax.jit
        def operator_pullback(point,gm,gk):
            values,pullback=jax.vjp(projected['operators'],point)
            return pullback((gm,gk,jnp.zeros_like(values[2])))[0]
        self._pullback=operator_pullback
        def stress_with_record(m,k,load):
            _,u,_,energy=newmark(m,k,load,settings,model['dt'])
            metrics=response(projected,u,settings,chunk,model['ks_relative_error'])
            error=jnp.max(jnp.abs(energy[:,0]+energy[:,1]-energy[:,2]))
            relative=error/jnp.maximum(jnp.max(jnp.abs(energy)),1e-30)
            return metrics[2],jnp.concatenate((metrics,jnp.asarray([error,relative])))
        self._stress=jax.jit(jax.value_and_grad(stress_with_record,argnums=(0,1),has_aux=True))
        def displacement_bound(m,k,load):
            _,u,_,_=newmark(m,k,load,settings,model['dt'])
            return response(projected,u,settings,chunk,model['ks_relative_error'])[1]
        self._displacement=jax.jit(jax.value_and_grad(displacement_bound,argnums=(0,1)))

    def cache_sizes(self):
        result={}
        for name in ('assemble','pullback','stress','displacement'):
            function=getattr(self,'_'+name);size=getattr(function,'_cache_size',None)
            result[name]=int(size()) if size is not None else None
        return result

    def __call__(self,point,derivatives=True):
        point=np.asarray(point)
        if point.shape!=np.shape(self.model['q0']) or point.dtype!=np.float64 or not np.isfinite(point).all():
            raise ValueError('Finite float64 controls with the configured shape required')
        if self._cache is not None and np.array_equal(point,self._cache['q']):return self._cache
        start=time.perf_counter();qj=jnp.asarray(point);M,K,mass=self._assemble(qj)
        records=[];stress_rows=[];timings=[]
        for case in range(3):
            before=time.perf_counter()
            (bound,aux),(gm,gk)=self._stress(M,K,self.model['projected']['loads'][case])
            row=np.asarray(self._pullback(qj,gm,gk))
            if not np.isfinite(row).all():raise RuntimeError('Nonfinite case stress gradient')
            records.append(np.asarray(aux));stress_rows.append(-row)
            timings.append(dict(case=case,stress_and_operator_pullback_seconds=time.perf_counter()-before))
        records=np.asarray(records)
        if not np.isfinite(records).all():raise RuntimeError('Nonfinite case response record')
        maximum=float(np.max(records[:,1]));ties=records[:,1]==maximum;weights=ties.astype(float)/ties.sum()
        objective_row=np.zeros_like(point)
        for case in np.flatnonzero(ties):
            bound,(gm,gk)=self._displacement(M,K,self.model['projected']['loads'][case])
            np.testing.assert_allclose(float(bound),records[case,1],rtol=1e-10,atol=1e-13)
            objective_row+=weights[case]*np.asarray(self._pullback(qj,gm,gk))
        values=np.r_[maximum,1-records[:,2]];jacobian=np.vstack((objective_row,stress_rows));mass=float(mass)
        if not np.isfinite(values).all() or not np.isfinite(jacobian).all():raise RuntimeError('Nonfinite SQP response/Jacobian')
        baseline=self.model['base']['plan']['uniform_mass_kg']
        record=dict(cases=records,mass_kg=mass,response=maximum,actual_peak_back_m=float(records[:,3].max()),
            actual_peak_skin_Pa=float(records[:,4].max()),displacement_bound=maximum,stress_bound=float(records[:,2].max()),
            constraints=np.asarray([maximum-1,float(records[:,2].max())-1,mass/baseline-1]))
        self.evaluations+=1
        self._cache=dict(q=point.copy(),values=values,first_row=jacobian[0],jacobian=jacobian,record=record,
            diagnostics=dict(method=METHOD,evaluations=self.evaluations,active_displacement_case_weights=weights.tolist(),
                wall_seconds=time.perf_counter()-start,case_timings=timings,compiled_cache_sizes=self.cache_sizes(),
                all_time_space_stress_samples_retained=True))
        return self._cache
