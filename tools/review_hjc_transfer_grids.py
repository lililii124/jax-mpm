"""NumPy-only authentication of same-particle frozen grid comparison."""
import argparse
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.audit_hjc_complete_prefix import official_archive,provenance_and_state,CHILD,save,load,sha

SPEC=dict(run=40179737527,artifact=12594431059,commit='24bd5850e11537899bc9476830fc28a5744a6271',bytes=53625655,
    sha256='08cb204c9eab8026002095f807bd0bdd1091adabd3fd64256a6e4c7a3e62b2f3',frame=0)


def review(output):
    output.mkdir(parents=True,exist_ok=False)
    parent,parent_archive=official_archive('original-f40',CHILD,output)
    state=provenance_and_state(parent,CHILD)['final']
    folder,archive=official_archive('grid-comparison',SPEC,output)
    report=load(folder/'probe/frozen-grid-comparison-review.json')
    guard=load(folder/'resource-guard.json')
    assert report['passed'] and report['commit']==SPEC['commit']
    assert report['source_checkpoint_sha256']==sha(parent/'raw/checkpoint_0_0.npz')
    assert report['physical_steps_executed']==report['material_updates']==0
    assert guard['returncode']==0 and guard['guard_reason'] is None
    assert guard['peak_uss_bytes']<guard['uss_limit_bytes'] and guard['wall_seconds']<1800
    subprocess.run(['git','diff','--exit-code','d857cec8d682446a1f4821ebd0fb3b713ef761a2',SPEC['commit'],'--','mpm_jax'],cwd=ROOT,check=True)
    definition=load(parent/'plan.json')['definition']
    mass,x=state['mass'],state['x']
    initial_momentum=np.sum(mass[:,None]*state['v'],axis=0)
    offsets=np.asarray(list(product(range(3),repeat=3)))
    records=[]
    for grid_report in report['grids']:
        grid=grid_report['grid']
        dx=.4/grid
        D=dx**2/4
        expected_dt=definition['time_step_s']*64/grid
        assert grid_report['dx_m']==dx and grid_report['nominal_dt_s']==expected_dt
        assert grid_report['particles']==len(mass)==121104 and grid_report['full_support']
        base=np.floor(x/dx-.5).astype(np.int64)
        fractional=x/dx-base
        weights1d=np.stack((.5*(1.5-fractional)**2,.75-(fractional-1)**2,.5*(fractional-.5)**2),axis=1)
        offsets1d=np.arange(3)[None,:,None]
        delta=(offsets1d-fractional[:,None,:])*dx
        assert np.max(np.abs(weights1d.sum(1)-1))<1e-14
        assert np.max(np.abs(np.sum(weights1d*delta,axis=1)))<1e-16
        assert np.max(np.abs(np.sum(weights1d*delta**2,axis=1)-D))<1e-18
        indices=np.zeros((len(x),27),dtype=np.int64)
        weights=np.ones((len(x),27))
        for axis in range(3):
            nodes=base[:,axis,None]+offsets[None,:,axis]
            assert nodes.min()>=0 and nodes.max()<=grid
            indices+=nodes*(grid+1)**(2-axis)
            weights*=weights1d[:,offsets[:,axis],axis]
        gm=np.bincount(indices.ravel(),weights=(weights*mass[:,None]).ravel(),minlength=(grid+1)**3)
        assert abs(gm.sum()-mass.sum())<1e-12
        np.testing.assert_allclose(grid_report['mass_kg'],mass.sum(),rtol=0,atol=1e-14)
        one_d=np.arange(grid+1)*dx
        node=np.arange(len(gm))
        coordinates=((node[:,None]//((grid+1)**np.arange(2,-1,-1)))%(grid+1))*dx
        for boundary in grid_report['actual_boundary_discretization']:
            original=definition['boundary_definitions'][definition['boundary_names'].index(boundary['name'])]
            assert boundary['nominal_lower_m']==original['lower'] and boundary['nominal_upper_m']==original['upper']
            displacement=np.zeros(3)
            target=np.asarray(original['velocity'])
            if 'loading_duration' in original:
                ramp=original['ramp_time']
                duration=original['loading_duration']
                elapsed=np.clip(report['boundary_time_common_s']-original['start'],0,duration)
                rising=min(elapsed,ramp)
                falling=np.clip(elapsed-(duration-ramp),0,ramp)
                distance=.5*(rising-ramp/np.pi*np.sin(np.pi*rising/ramp))+np.clip(elapsed-ramp,0,duration-2*ramp)+.5*(falling+ramp/np.pi*np.sin(np.pi*falling/ramp))
                displacement=target*distance
                speed=.5*(1-np.cos(np.pi*np.clip(elapsed/ramp,0,1)))*.5*(1+np.cos(np.pi*np.clip((elapsed-(duration-ramp))/ramp,0,1)))
                target=target*speed
            np.testing.assert_allclose(boundary['translated_lower_m'],np.asarray(original['lower'])+displacement,rtol=0,atol=1e-14)
            np.testing.assert_allclose(boundary['translated_upper_m'],np.asarray(original['upper'])+displacement,rtol=0,atol=1e-14)
            np.testing.assert_allclose(boundary['target_velocity_m_per_s'],target,rtol=0,atol=1e-12)
            lower=np.asarray(boundary['translated_lower_m'])
            upper=np.asarray(boundary['translated_upper_m'])
            for axis,name in enumerate(('x','y','z')):
                assert boundary['selected_grid_axis_indices'][name]==np.flatnonzero((one_d>=lower[axis])&(one_d<=upper[axis])).tolist()
            mask=np.all((coordinates>=lower)&(coordinates<=upper),axis=1)
            assert boundary['selected_grid_nodes']==int(mask.sum())
            assert boundary['occupied_selected_grid_nodes']==int(np.count_nonzero(mask&(gm>0)))
            np.testing.assert_allclose(boundary['selected_grid_mass_kg'],gm[mask].sum(),rtol=1e-13,atol=1e-14)
        del coordinates,node,indices,weights,weights1d
        def invariant(v,C):
            energy=float(.5*np.sum(mass*np.sum(v**2,axis=1))+.5*D*np.sum(mass[:,None,None]*C**2))
            momentum=np.sum(mass[:,None]*v,axis=0)
            spin=np.stack((C[:,2,1]-C[:,1,2],C[:,0,2]-C[:,2,0],C[:,1,0]-C[:,0,1]),axis=1)
            angular=np.sum(mass[:,None]*(np.cross(x,v)+D*spin),axis=0)
            return energy,momentum,angular
        initial_energy,_,initial_angular=invariant(state['v'],state['C'])
        for case in grid_report['cases']:
            prefix=f"grid{grid}-"+case['label']
            ledger=folder/'probe'/(prefix+'-operator.npz')
            end=folder/'probe'/(prefix+'-end.npz')
            with np.load(ledger,allow_pickle=False) as raw:
                values,stages=raw['values'],raw['stages']
            count=128*grid//64
            assert values.shape==(count,12) and np.isfinite(values).all() and np.isfinite(stages).all()
            np.testing.assert_allclose(values[0,0],initial_energy,rtol=1e-14,atol=1e-12)
            np.testing.assert_allclose(values[1:,0],values[:-1,3],rtol=0,atol=1e-12)
            np.testing.assert_allclose(values[:,4],values[:,0]-values[:,1],rtol=0,atol=1e-12)
            np.testing.assert_allclose(values[:,5],values[:,2]-values[:,3],rtol=0,atol=1e-12)
            if stages.shape[1]:
                np.testing.assert_allclose(stages[:,:,0].sum(1),values[:,6],rtol=0,atol=1e-12)
                np.testing.assert_allclose(stages[:,:,2].sum(1),values[:,7],rtol=0,atol=1e-12)
                assert np.max(np.abs(stages[:,:,0]-stages[:,:,1]-stages[:,:,2]))<1e-10
            assert np.max(np.abs(values[:,3]-values[:,0]-values[:,6]+values[:,4]+values[:,5]+values[:,7]))<1e-10
            assert np.min(values[:,4:6])>=-1e-10 and np.min(values[:,7])>=0
            with np.load(end,allow_pickle=False) as raw:
                v,C=raw['v'],raw['C']
            assert v.shape==(121104,3) and C.shape==(121104,3,3) and np.isfinite(v).all() and np.isfinite(C).all()
            energy,momentum,angular=invariant(v,C)
            np.testing.assert_allclose(energy,values[-1,3],rtol=1e-14,atol=1e-12)
            impulse=stages[:,:,3:].sum(axis=(0,1)) if stages.shape[1] else np.zeros(3)
            assert np.max(np.abs(momentum-initial_momentum-impulse))<1e-10
            if case['label']=='no-boundaries':
                assert not values[:,6:8].any() and np.max(np.abs(angular-initial_angular))<1e-10
            matching=64*grid//64
            records.append(dict(grid=grid,label=case['label'],initial_apic_energy_J=initial_energy,independent_end_apic_energy_J=energy,
                matched_applications=matching,matched_physical_window_label_s=matching*expected_dt,
                matched_transfer_loss_J=float(values[:matching,4:6].sum()),matched_boundary_projection_J=float(values[:matching,7].sum()),
                matched_boundary_work_J=float(values[:matching,6].sum()),ledger_sha256=sha(ledger),end_sha256=sha(end)))
    assert 'jax' not in sys.modules
    result=dict(passed=True,parent_official_archive=parent_archive,grid_official_archive=archive,checks=records,
        physical_steps_executed=0,jax_initialized=False,source_particle_count=121104,
        all_actual_boundary_node_selection_and_mass_recomputed=True,guard=guard,
        structural_energy_certified=False,temporal_convergence_certified=False,
        scope='Same true particle states with original physical boundary boxes; background-grid comparison of frozen transfer only. Changed discrete constrained mass is explicitly preserved, not corrected away.')
    save(output/'independent-frozen-grid-review.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    review(args.output.resolve())
