"""Conservation and refinement for the predeclared complete-map relaxation."""
import ast
from pathlib import Path
import numpy as np
from tools.metal_frozen_transfer import support
from tools.metal_relaxed_transfer_numpy import relaxed_remap


def fixture():
    rng=np.random.default_rng(137);x=rng.uniform(-.3,.3,(40,3));v=rng.normal(size=(40,3));C=rng.normal(size=(40,3,3));m=rng.uniform(.5,1.5,40)
    return x,v,C,m,support(x,np.zeros(40,int),m,.25)


def test_complete_blend_preserves_both_momenta_and_signed_energy():
    x,v,C,m,frozen=fixture();inc=np.random.default_rng(71).normal(size=(len(frozen[-1]),3))*.2
    _,_,d=relaxed_remap(x,v,C,m,.25,frozen,dt=.03,relaxation_time=.2,increment=inc)
    assert np.linalg.norm(d['momentum_defect'])<1e-13
    assert np.linalg.norm(d['angular_defect'])<1e-13
    assert abs(d['ledger_identity_residual_J'])<1e-12
    assert abs(d['convex_identity_residual_J'])<1e-12
    assert d['blending_variance_J']>=0


def test_frozen_zero_force_relaxation_has_refining_finite_time_limit():
    x,v,C,m,frozen=fixture();states=[]
    for steps in (16,32,64):
        nv,nC=v.copy(),C.copy();loss=0.
        for _ in range(steps):
            nv,nC,d=relaxed_remap(x,nv,nC,m,.25,frozen,dt=.1/steps,relaxation_time=.2)
            assert d['particle_delta_J']<=1e-12
            loss+=d['p2g_J']+d['g2p_signed_J']
        states.append((nv,nC,loss))
    error=lambda a,b:np.linalg.norm(a[0]-b[0])+np.linalg.norm(a[1]-b[1])
    assert error(states[1],states[2])<.6*error(states[0],states[1])
    assert abs(states[2][2]-states[1][2])<.6*abs(states[1][2]-states[0][2])


def test_relaxed_step_changes_only_transfer_and_additional_diagnostics():
    load=lambda p:next(n for n in ast.parse(Path(p).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_step')
    old=load('mpm_jax/fragment_multifield_v6_mpm.py');new=load('mpm_jax/fragment_v6_relaxed_candidate.py');new.args.kwonlyargs=[];new.args.kw_defaults=[]
    class Restore(ast.NodeTransformer):
        def visit_Assign(self,node):
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='transfer_relaxed':
                node.targets[0].elts=node.targets[0].elts[:3]
                node.value=ast.Call(func=ast.Name(id='transfer_to_particles',ctx=ast.Load()),args=[ast.Name(id=n,ctx=ast.Load()) for n in ('velocity','support','grid')],keywords=[])
            return self.generic_visit(node)
        def visit_Return(self,node):
            if isinstance(node.value,ast.Tuple) and len(node.value.elts)==3:node.value.elts=node.value.elts[:2]
            return node
    assert ast.dump(old)==ast.dump(Restore().visit(new))


def test_dynamic_host_uses_shared_cfl_and_stops_on_transfer_rejection():
    from collections import namedtuple
    from types import SimpleNamespace
    State=namedtuple('State','time valid x');Extra=namedtuple('Extra','alpha blending_variance_J convex_energy_identity_residual_J angular_defect_by_field angular_scale_by_field')
    calls=[];limits={}
    def step(name):
        def run(s,material,grid,config,dt,**kwargs):
            calls.append((name,float(s.time),float(dt)))
            out=State(s.time+dt,True,s.x);limits[id(out)]=.3 if name=='baseline' else .1
            if name=='baseline':return out,None
            angular=np.array([[0.,0.,0.]]) if s.time<.2 else np.array([[1.,0.,0.]])
            return out,None,Extra(.9,.01,0.,angular,np.array([1.]))
        return run
    first=State(0.,True,np.zeros((1,3),dtype=np.float64));limits[id(first)]=.2
    old=SimpleNamespace(stable_timestep=lambda s,*a:limits[id(s)],step=step('baseline'),energy_ledger=lambda *a:np.array([1.]))
    new=SimpleNamespace(step=step('candidate'))
    tree=ast.parse(Path('mpm_jax/fragment_v6_relaxed_diagnostic.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('transfer_gates','compare_shared')]
    ns=dict(jnp=np,original=old,candidate=new)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual-host-control>','exec'),ns)
    baseline,candidate,history,failure=ns['compare_shared'](first,None,None,None,.55,relaxation_time=.2,max_steps=8)
    assert len(history)==2 and failure['first_rejected_step_1based']==2
    assert failure['failed_gates']==['angular']
    np.testing.assert_allclose([row['dt_s'] for row in history],[.2,.1])
    assert baseline.time==candidate.time and len(calls)==4
