"""Non-affine remaps and finite grid increments, without initializing JAX."""
import numpy as np
import ast
from pathlib import Path
from tools.metal_frozen_transfer import support, remap


def fixture():
    rng=np.random.default_rng(731)
    x=rng.uniform(-.3,.3,(40,3));m=rng.uniform(.5,1.5,40)
    v=rng.normal(size=(40,3));C=rng.normal(size=(40,3,3))
    return m,v,C,support(x,np.zeros(40,int),m,.25)


def test_delta_zero_increment_is_identity_while_apic_projects():
    m,v,C,s=fixture()
    nv,nC,d=remap(v,C,m,.25,s,'DELTA')
    assert np.array_equal(nv,v) and np.array_equal(nC,C)
    assert abs(d['p2g_J']+d['g2p_signed_J'])<1e-12
    _,_,apic=remap(v,C,m,.25,s,'APIC')
    assert apic['p2g_J']+apic['g2p_signed_J']>1


def test_finite_increment_uses_signed_energy_and_preserves_grid_impulse():
    m,v,C,s=fixture();gm=s[-1]
    inc=np.random.default_rng(58).normal(size=(len(gm),3))*.1
    nv,nC,d=remap(v,C,m,.25,s,'DELTA',inc)
    np.testing.assert_allclose(np.sum(m[:,None]*(nv-v),axis=0),np.sum(gm[:,None]*inc,axis=0),atol=1e-13)
    assert abs(d['balance_residual_J'])<1e-12
    assert np.isfinite(nC).all()


def test_candidate_step_changes_only_transfer_operator():
    # Compare syntax trees without importing JAX or compiling a large graph.
    def step(path):
        tree=ast.parse(Path(path).read_text())
        return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_step')
    baseline=step('mpm_jax/fragment_multifield_v6_mpm.py')
    changed=step('mpm_jax/fragment_v6_delta_candidate.py')
    class RestoreTransfer(ast.NodeTransformer):
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name) and node.func.id=='transfer_delta':
                node.func.id='transfer_to_particles'
                node.args=[ast.Name(id='velocity',ctx=ast.Load()),node.args[-2],node.args[-1]]
            return self.generic_visit(node)
    restored=RestoreTransfer().visit(changed)
    assert ast.dump(baseline)==ast.dump(restored)


def test_shared_schedule_uses_both_limits_and_stops_at_first_rejection():
    # Execute the actual host orchestrator against tiny deterministic kernels.
    # This checks scheduling/control without initializing a JAX backend.
    from collections import namedtuple
    from types import SimpleNamespace
    State=namedtuple('State','time valid')
    Diagnostic=namedtuple('Diagnostic','valid')
    calls=[]
    def kernel(name,reject=False):
        def step(s,material,grid,config,dt):
            calls.append((name,float(s.time),float(dt)))
            valid=not (reject and s.time>=.2)
            following=State(s.time+dt,valid)
            limits[id(following)]=.3 if name=='original' else .1
            return following,Diagnostic(np.asarray(valid))
        return step
    def tree_map(fn,*values):
        return Diagnostic(*(fn(*(v[i] for v in values)) for i in range(len(values[0]))))
    fake_original=SimpleNamespace(stable_timestep=lambda *a:.3,step=kernel('original'))
    fake_candidate=SimpleNamespace(step=kernel('candidate',reject=True))
    # The CFL function is the common original helper applied to either state;
    # alter it after the candidate's first step by tracking the current object.
    first=State(0.,True)
    limits={id(first):.2}
    fake_original.stable_timestep=lambda state,*a:limits[id(state)]
    tree=ast.parse(Path('mpm_jax/fragment_v6_delta_diagnostic.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compare_shared_adaptive')
    namespace=dict(jnp=np,jax=SimpleNamespace(tree=SimpleNamespace(map=tree_map)),original=fake_original,candidate=fake_candidate)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual-host-orchestrator>','exec'),namespace)
    old,delta,count,dt,_,_=namespace['compare_shared_adaptive'](first,None,None,None,.55,max_steps=8)
    assert count==2 and not delta.valid
    np.testing.assert_allclose(dt,[.2,.1])
    assert len(calls)==4 and old.time==delta.time
    assert calls[0][2]==calls[1][2] and calls[2][2]==calls[3][2]


def test_report_serializes_numpy_scalar_rejection_without_changing_gate():
    import json
    import pytest
    from examples.metal_fragment_v6_delta125 import json_numpy_scalar
    result={'checks':{'shape':np.bool_(False)},'limit':np.float64(.2),'steps':np.int64(12)}
    decoded=json.loads(json.dumps(result,allow_nan=False,default=json_numpy_scalar))
    assert decoded=={'checks':{'shape':False},'limit':.2,'steps':12}
    with pytest.raises(ValueError):json.dumps({'invalid':np.float64(np.nan)},allow_nan=False,default=json_numpy_scalar)
    with pytest.raises(TypeError):json.dumps({'unsupported':object()},default=json_numpy_scalar)
