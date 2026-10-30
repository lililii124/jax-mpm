"""Extend a frozen material-design basis using one more physical K/M vertex."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, os, shutil
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '4')
import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor
from tools.build_sandwich_rational import (
    digest, material_fields_digest, physics_support, source_receipt,
    check_recipe, tangent, build_selected)
from tools.assess_sandwich_rational import integrate
from examples.blast_sandwich_design import Settings, source_signature


def run(configuration):
    configuration = Path(configuration).resolve()
    config = json.loads(configuration.read_text())
    def path(key): return (configuration.parent/config[key]).resolve()
    parent = json.loads(path('parent_freeze').read_text())
    assert digest(path('parent_freeze')) == config['parent_freeze_sha256']
    assert digest(path('parent_basis')) == parent['basis']['sha256']
    assert digest(path('selected_material')) == parent['inputs']['selected_material']['sha256']
    assert material_fields_digest(path('selected_material')) == parent['material_fields_sha256']['selected']
    initial = json.loads(path('initial_freeze').read_text())
    assert digest(path('initial_material')) == initial['inputs']['initial_material']['sha256']
    with np.load(path('controls'), allow_pickle=False) as saved: q = saved['q'].copy()
    qsha = hashlib.sha256(q.tobytes()).hexdigest()
    assert qsha == config['controls_sha256'] == parent['selected_controls_sha256']
    settings = Settings(xy_step=.01, z_step=.001, controls=(24,24,3), stage='fixed-mass')
    physical_signature, sources = source_signature(settings)
    assert physical_signature == parent['full_mpm_source_signature']
    recipe = json.loads(path('base_recipe').read_text())
    check_recipe(recipe)
    out = path('output'); out.mkdir(parents=True, exist_ok=False)
    steps = config['fresh_steps']
    assert steps == list(range(71000,83001,1000))
    def heads():
        result = []
        for case, relative in config['mpm_status'].items():
            file = (configuration.parent/relative).resolve()
            status = json.loads(file.read_text()); binding = status['binding']
            assert status['valid'] and status['step'] < steps[0]
            assert binding['source_signature'] == physical_signature and binding['case'] == case
            assert hashlib.sha256(np.asarray(binding['controls'],dtype=np.float64).tobytes()).hexdigest() == qsha
            result.append(dict(case=case,step=status['step'],time_s=status['time_s'],status_sha256=digest(file)))
        assert len(result) == 3
        return result
    source_pins = dict(sources, **source_receipt(recipe))
    root = Path(__file__).resolve().parents[1]
    for name in ['tools/extend_sandwich_rational.py','tools/assess_sandwich_rational.py']:
        source_pins[name] = digest(root/name)
    policy = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        method='Old frozen span plus an independently constructed load-driven rational space at the new material vertex; no MPM displacement directions.',
        configuration_sha256=digest(configuration),source_sha256=source_pins,
        parent_freeze_sha256=digest(path('parent_freeze')),controls_sha256=qsha,
        fresh_steps=steps,response_gate=.05,omitted_energy_gate=.05,
        before_construction_heads=heads())
    (out/'construction-policy.json').write_bytes((json.dumps(policy,indent=2)+'\n').encode())
    for key,name in [('selected_material','material.npz'),('controls','controls.npz')]:
        shutil.copy2(path(key),out/name)
    recipe.update(_directory=str(out),selected_material='material.npz',selected_tangent='tangent.npz',
                  selected_controls='controls.npz',selected_controls_sha256=qsha,output='selected')
    print(json.dumps(dict(stage='tangent',fresh_steps=steps)),flush=True)
    tangent(recipe)
    print(json.dumps(dict(stage='rational-directions')),flush=True)
    build_selected(recipe)
    _,geo,support,namespace = physics_support()
    nodes = support.nodes
    _,_,fixed = namespace['clamped_basis'](nodes,settings,2)
    ids = np.flatnonzero(np.repeat(~fixed,3))
    def nodal_mass(file):
        with np.load(file,allow_pickle=False) as saved: density=saved['density']
        mass=np.zeros(len(nodes))
        np.add.at(mass,support.indices.ravel(),(support.weights*(support.volume*density)[:,None]).ravel())
        assert abs(mass.sum()/1.8468-1)<1e-10
        return mass
    mass0=nodal_mass(path('initial_material')); massq=nodal_mass(out/'material.npz')
    mf0=np.repeat(mass0,3)[ids]; mfq=np.repeat(massq,3)[ids]
    with np.load(path('parent_basis'),allow_pickle=False) as saved:
        old=saved['basis'];np.testing.assert_array_equal(saved['nodes'],nodes)
    assert hashlib.sha256(old.tobytes()).hexdigest()==parent['basis']['numeric_basis_sha256']
    C=old.reshape(-1,old.shape[-1])[ids]
    assert np.max(np.abs(C.T@(mf0[:,None]*C)-np.eye(C.shape[1])))<1e-10
    with np.load(out/'selected/rational-basis.npz',allow_pickle=False) as saved:
        vertex=saved['basis'];np.testing.assert_array_equal(saved['nodes'],nodes)
    raw=vertex.reshape(-1,vertex.shape[-1])[ids]
    for _ in range(2):raw-=C@(C.T@(mf0[:,None]*raw))
    U,singular,_=np.linalg.svd(np.sqrt(mf0)[:,None]*raw,full_matrices=False)
    keep=singular>singular[0]*1e-8
    Z=np.column_stack((C,U[:,keep]/np.sqrt(mf0)[:,None]))
    gram=Z.T@(mf0[:,None]*Z)
    Z=Z@np.linalg.solve(np.linalg.cholesky(.5*(gram+gram.T)).T,np.eye(len(gram)))
    assert np.max(np.abs(Z.T@(mf0[:,None]*Z)-np.eye(Z.shape[1])))<1e-10
    lost=C-Z@(Z.T@(mf0[:,None]*C))
    assert np.linalg.norm(np.sqrt(mf0)[:,None]*lost)<1e-10
    Kfull=sparse.load_npz(out/'tangent.npz');Kfree=Kfull[ids][:,ids]
    M=Z.T@(mfq[:,None]*Z);K=Z.T@(Kfree@Z);K=.5*(K+K.T)
    cho_factor(M);cho_factor(K)
    top_ids,top_weights=namespace['surface_weights'](support,geo['top'])
    loads=np.zeros((len(nodes),3,3))
    for case in range(3):
        np.add.at(loads[:,2,case],top_ids.ravel(),(-geo['face_area'][:,None]*geo['footprints'][case,:,None]*top_weights).ravel())
    L=Z.T@loads.reshape(-1,3)[ids]
    basis=np.zeros((len(nodes)*3,Z.shape[-1]));basis[ids]=Z;basis=basis.reshape(len(nodes),3,-1)
    bottom_ids,bottom_weights=namespace['surface_weights'](support,geo['bottom'])
    back=np.einsum('pk,pkr->pr',bottom_weights,basis[bottom_ids,2,:])
    np.savez_compressed(out/'frozen-basis.npz',basis=basis,nodes=nodes,mass=M,stiffness=K,loads=L,back=back)
    fine=integrate(M,K,L,back,1.25e-7);coarse=integrate(M,K,L,back,2.5e-7)
    sensitivity=[]
    for case in range(3):
        finer=np.interp(coarse['time'],fine['time'],fine['curves'][:,case]);older=coarse['curves'][:,case]
        errors=dict(peak_relative_difference=float(abs(fine['curves'][:,case].max()/older.max()-1)),
            curve_relative_l2=float(np.linalg.norm(finer-older)/np.linalg.norm(finer)),
            maximum_difference_over_finer_peak=float(np.max(np.abs(finer-older))/fine['curves'][:,case].max()))
        assert max(errors.values())<.01
        sensitivity.append(dict(case=case,**errors))
    assert fine['energy_relative_error']<1e-8
    np.savez_compressed(out/'response.npz',**{key:value for key,value in fine.items() if key!='energy_relative_error'})
    receipt=dict(schema_version=2,scope=policy['method'],created_utc=datetime.now(timezone.utc).isoformat(),
        operator_vertex='selected',basis=dict(path='frozen-basis.npz',sha256=digest(out/'frozen-basis.npz'),
            numeric_basis_sha256=hashlib.sha256(basis.tobytes()).hexdigest(),rank=basis.shape[-1]),
        parent_basis=parent['basis'],parent_freeze_sha256=digest(path('parent_freeze')),
        response=dict(path='response.npz',sha256=digest(out/'response.npz')),selected_controls_sha256=qsha,
        inputs={key:dict(path=name,sha256=digest(out/name)) for key,name in [('selected_material','material.npz'),('selected_controls','controls.npz'),('selected_tangent','tangent.npz')]},
        material_fields_sha256=dict(selected=material_fields_digest(out/'material.npz')),
        source_sha256=source_pins,full_mpm_source_signature=physical_signature,
        construction_configuration=dict(selected_controls_sha256=qsha,mass_kg=float(massq.sum()),nominal_dt_s=1.25e-7,horizon_s=.002,
            base_recipe_sha256=digest(path('base_recipe')),retained_parent_span=True,mpm_snapshot_directions=False),
        fresh_validation=dict(steps=steps,history_interval_steps=[steps[0],steps[-1]],response_gate=.05,omitted_energy_gate=.05),
        actual_mpm_heads_before_fresh_window=heads(),time_sensitivity=sensitivity,
        linear_energy_relative_error=fine['energy_relative_error'],construction_policy_sha256=digest(out/'construction-policy.json'),
        complete2ms_three_case_mpm_qualification=False,global_optimum_claim=False)
    (out/'model-freeze.json').write_bytes((json.dumps(receipt,indent=2)+'\n').encode())
    print(json.dumps(dict(stage='frozen',rank=basis.shape[-1],new_directions=int(keep.sum()),fresh_steps=steps)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configuration',type=Path)
    run(parser.parse_args().configuration)
