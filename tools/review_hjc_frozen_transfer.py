"""Authenticate frozen actual APIC records and recompute end-state invariants."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.audit_hjc_complete_prefix import official_archive,provenance_and_state,CHILD,save,load,sha

SPEC=dict(run=40179007877,artifact=12593520797,commit='e53dfd5f30503f056131088333edcc63dcdb11e9',bytes=26746746,
    sha256='3c41a2d15ac440b2a8a718b55033f2b19bf4f114844258cdd8977d60acaccf8d',frame=0)


def review(output):
    output.mkdir(parents=True,exist_ok=False)
    parent,parent_archive=official_archive('original-f40',CHILD,output)
    state=provenance_and_state(parent,CHILD)['final']
    folder,archive=official_archive('frozen',SPEC,output)
    report=load(folder/'probe/frozen-transfer-review.json')
    guard=load(folder/'resource-guard.json')
    assert report['passed'] and report['commit']==SPEC['commit']
    assert report['parent_checkpoint_sha256']==sha(parent/'raw/checkpoint_0_0.npz')
    assert report['physical_steps_executed']==report['material_updates']==0
    assert guard['guard_reason'] is None and guard['returncode']==0
    assert guard['peak_uss_bytes']<guard['uss_limit_bytes'] and guard['wall_seconds']<1800
    subprocess.run(['git','diff','--exit-code','d857cec8d682446a1f4821ebd0fb3b713ef761a2',SPEC['commit'],'--','mpm_jax'],cwd=ROOT,check=True)
    dx=.00625
    D=dx**2/4
    mass,x=state['mass'],state['x']
    def invariants(v,C):
        energy=float(.5*np.sum(mass*np.sum(v**2,axis=1))+.5*D*np.sum(mass[:,None,None]*C**2))
        momentum=np.sum(mass[:,None]*v,axis=0)
        spin=np.stack((C[:,2,1]-C[:,1,2],C[:,0,2]-C[:,2,0],C[:,1,0]-C[:,0,1]),axis=1)
        angular=np.sum(mass[:,None]*(np.cross(x,v)+D*spin),axis=0)
        return energy,momentum,angular
    initial_energy,initial_momentum,initial_angular=invariants(state['v'],state['C'])
    fractional=x/dx-np.floor(x/dx-.5)
    weights=np.stack((.5*(1.5-fractional)**2,.75-(fractional-1)**2,.5*(fractional-.5)**2),axis=1)
    offsets=np.arange(3)[None,:,None]
    delta=(offsets-fractional[:,None,:])*dx
    moment0=np.max(np.abs(weights.sum(1)-1))
    moment1=np.max(np.abs(np.sum(weights*delta,axis=1)))
    moment2=np.max(np.abs(np.sum(weights*delta**2,axis=1)-D))
    assert moment0<1e-14 and moment1<1e-16 and moment2<1e-18
    records=[]
    for case in report['cases']:
        label=case['label']
        ledger=folder/'probe'/(label+'-operator.npz')
        end=folder/'probe'/(label+'-end.npz')
        with np.load(ledger,allow_pickle=False) as raw:
            values,stages=raw['values'],raw['stages']
        assert values.shape==(128,12) and np.isfinite(values).all() and np.isfinite(stages).all()
        np.testing.assert_allclose(values[0,0],initial_energy,rtol=1e-14,atol=1e-12)
        np.testing.assert_allclose(values[1:,0],values[:-1,3],rtol=0,atol=1e-12)
        np.testing.assert_allclose(values[:,4],values[:,0]-values[:,1],rtol=0,atol=1e-12)
        np.testing.assert_allclose(values[:,5],values[:,2]-values[:,3],rtol=0,atol=1e-12)
        reconstructed=values[:,3]-values[:,0]-values[:,6]+values[:,4]+values[:,5]+values[:,7]
        assert np.max(np.abs(reconstructed))<1e-10
        assert np.min(values[:,4:6])>=-1e-10 and np.min(values[:,7])>=0
        assert np.max(np.abs(values[:,9:]))<1e-10
        if stages.shape[1]:
            np.testing.assert_allclose(stages[:,:,0].sum(1),values[:,6],rtol=0,atol=1e-12)
            np.testing.assert_allclose(stages[:,:,2].sum(1),values[:,7],rtol=0,atol=1e-12)
            assert np.max(np.abs(stages[:,:,0]-stages[:,:,1]-stages[:,:,2]))<1e-10
        with np.load(end,allow_pickle=False) as raw:
            v,C=raw['v'],raw['C']
        assert v.shape==(121104,3) and C.shape==(121104,3,3)
        assert np.isfinite(v).all() and np.isfinite(C).all()
        energy,momentum,angular=invariants(v,C)
        np.testing.assert_allclose(energy,values[-1,3],rtol=1e-14,atol=1e-12)
        impulse=stages[:,:,3:].sum(axis=(0,1)) if stages.shape[1] else np.zeros(3)
        momentum_error=float(np.max(np.abs(momentum-initial_momentum-impulse)))
        assert momentum_error<1e-10
        angular_error=float(np.max(np.abs(angular-initial_angular)))
        if label=='no-boundaries':
            assert not values[:,6:8].any() and angular_error<1e-10
            assert np.all(np.diff(values[:,3])<=1e-12)
        losses=values[:,4]+values[:,5]+values[:,7]
        records.append(dict(label=label,applications=128,initial_apic_energy_J=initial_energy,
            independently_recomputed_end_apic_energy_J=energy,momentum_error_Ns=momentum_error,
            angular_change_Nms=angular_error,angular_change_scope='No-boundary value must vanish; constrained cases may receive boundary torque.',
            cumulative_loss_64_J=float(losses[:64].sum()),cumulative_loss_128_J=float(losses.sum()),
            prescribed_work_J=float(values[:,6].sum()),ledger_sha256=sha(ledger),end_state_sha256=sha(end)))
    assert 'jax' not in sys.modules
    result=dict(passed=True,parent_official_archive=parent_archive,frozen_official_archive=archive,
        checks=records,independent_stencil_moments=dict(partition_error=float(moment0),first_moment_m=float(moment1),second_moment_error_m2=float(moment2)),
        guard=guard,physical_steps_executed=0,jax_initialized=False,
        conclusion='Force-free frozen APIC contraction depends on projection count at fixed geometry, without constitutive or time integration. Current D and inverse-D coefficients satisfy the actual quadratic stencil moments.',
        structural_energy_certified=False,temporal_convergence_certified=False)
    save(output/'independent-frozen-transfer-review.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    review(args.output.resolve())
