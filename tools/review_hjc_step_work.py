"""Zero-step NumPy-only review of authenticated actual boundary-work records."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.audit_hjc_complete_prefix import official_archive,sha,save,load
from tools.probe_hjc_step_work import numpy_work

SPEC=dict(run=40177152433,artifact=12593131521,commit='b2c74c97a9cfad65656ea8be516cb80fa7994d37',
    bytes=40550988,sha256='f803c9e9bf9290cbf89744944eef2f319fafca69211083480bd71b2970d5bf98',frame=0)


def review(output):
    output.mkdir(parents=True,exist_ok=False)
    restored,archive=official_archive('diagnostic',SPEC,output)
    report=load(restored/'probe/step-work-review.json')
    guard=load(restored/'resource-guard.json')
    assert report['passed'] and report['commit']==SPEC['commit']
    assert report['particles']==121104 and report['total_physical_steps_executed']==144
    assert guard['returncode']==0 and guard['guard_reason'] is None
    assert guard['seconds_limit']==1800 and guard['uss_limit_bytes']==int(5.5*1024**3)
    assert guard['wall_seconds']<1800 and guard['peak_uss_bytes']<guard['uss_limit_bytes']
    subprocess.run(['git','diff','--exit-code','d857cec8d682446a1f4821ebd0fb3b713ef761a2',SPEC['commit'],
        '--','mpm_jax','examples'],cwd=ROOT,check=True)
    records=[]
    for case in report['cases']:
        assert case['all_state_leaves_bitwise_equal'] and case['max_state_leaf_absolute_difference']==0
        label=case['label']
        path=restored/'probe'/(label+'-ledger.npz')
        with np.load(path,allow_pickle=False) as raw:
            values,stages=raw['values'],raw['stages']
            assert values.shape==(case['steps'],16) and stages.shape==(case['steps'],6,6)
            assert np.isfinite(values).all() and np.isfinite(stages).all()
            np.testing.assert_allclose(values[:,1]-values[:,0],report['config_dt_s'],rtol=0,atol=1e-18)
            assert np.max(np.abs(stages[:,:,0]-stages[:,:,1]-stages[:,:,2]))<1e-8
            assert np.min(stages[:,:,2])>=0
            assert not stages[:,0,:].any()
            np.testing.assert_allclose(values[:,11],stages[:,:,0].sum(1),rtol=0,atol=1e-12)
            np.testing.assert_allclose(values[:,12],stages[:,:,2].sum(1),rtol=0,atol=1e-12)
            reconstructed=values[:,6]+values[:,7]+values[:,8]+values[:,11]-values[:,12]+values[:,9]+values[:,4]
            assert np.max(np.abs(values[:,10]-reconstructed))<1e-8
            assert np.max(np.abs(values[:,14]))==0
            assert np.max(np.abs(values[:,4]-values[:,5]))<1e-8
            # All fixed supports/symmetry constraints have zero target speed.
            assert np.max(np.abs(stages[:,:5,0]))<1e-12
            t=values[:,1]
            ramp=.0003
            down=np.clip((t-(.0012-ramp))/ramp,0,1)
            up=np.clip(t/ramp,0,1)
            target=-13.333333333333334*.5*(1-np.cos(np.pi*up))*.5*(1+np.cos(np.pi*down))
            np.testing.assert_allclose(stages[:,5,0],stages[:,5,4]*target,rtol=2e-12,atol=1e-18)
            independent_loading_work=float(np.sum(stages[:,5,4]*target))
        material_path=restored/'probe'/(label+'-last-material-increment.npz')
        with np.load(material_path,allow_pickle=False) as raw:
            a={name:raw[name] for name in raw.files}
        assert all(np.isfinite(value).all() for value in a.values())
        before=SimpleNamespace(F=a['old_F'],mass=a['mass'],history=SimpleNamespace(stress=a['old_stress'],work=a['old_work']))
        following=SimpleNamespace(F=a['new_F'],history=SimpleNamespace(stress=a['new_stress'],work=a['new_work'],density=a['density']))
        checked=numpy_work(before,following)
        assert checked['max_specific_increment_error_J_per_kg']<1e-9
        np.testing.assert_allclose(checked['stored_increment_J'],values[-1,4],rtol=1e-12,atol=1e-14)
        records.append(dict(label=label,steps=case['steps'],ledger_sha256=sha(path),material_increment_sha256=sha(material_path),
            independent_loading_work_J=independent_loading_work,independent_material_increment=checked,
            max_stage_identity_error_J=float(np.max(np.abs(values[:,10]-reconstructed))),
            maximum_fixed_boundary_roundoff_work_J=float(np.max(np.abs(stages[:,:5,0]))),
            original_core_final_state_bitwise_equal=True))
    assert 'jax' not in sys.modules
    result=dict(passed=True,official_archive=archive,checks=records,guard=guard,physical_steps_executed=0,jax_initialized=False,
        structural_energy_certified=False,temporal_convergence_certified=False,
        scope='Actual per-step operator work/target-velocity/material-increment coverage and bitwise baseline equivalence; stage bookkeeping identity is not physical convergence.')
    save(output/'independent-step-work-review.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    review(args.output.resolve())
