"""Compare compact107..110 against existing original callback-profiled evidence."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def compare(candidate,reference,original,parent,output):
    record=json.loads((candidate/'result.json').read_text())
    timing=json.loads((reference/'step-timing.json').read_text())
    proof=json.loads((reference/'source-review.json').read_text())
    assert record['accepted_steps']==4 and [r['step'] for r in record['steps']]==[107,108,109,110]
    assert all(r['valid'] and all(r['body_valid']) and all(r['all_history_valid']) for r in record['steps'])
    assert timing['start_step']==105 and timing['accepted_steps']==5 and timing['steps'][-1]['step']==110
    assert all(r['valid'] for r in timing['steps'])
    assert digest(original)==proof['original_kernel_sha256']
    assert digest(reference/'source/mpm_jax/materials/rht.py')==proof['profiled_kernel_sha256']
    module=importlib.util.spec_from_file_location('counter_transform',Path(__file__).with_name('profile_rht_steps.py'))
    transformed=importlib.util.module_from_spec(module);module.loader.exec_module(transformed)
    assert transformed.instrument(original.read_bytes())==(reference/'source/mpm_jax/materials/rht.py').read_bytes()
    review=json.loads((parent/'extraction-review.json').read_text())
    other_sources={name:digest(reference/name)==item['sha256'] for name,item in review['selected_members'].items()
        if name.startswith('source/') and name!='source/mpm_jax/materials/rht.py'}
    assert all(other_sources.values())
    assert json.loads((candidate/'case.json').read_text())==json.loads((reference/'case.json').read_text())
    fields={};raw={}
    for body in range(2):
        a_path=reference/f'last-state-{body}.npz';b_path=candidate/f'state110-body{body}.npz'
        raw[str(a_path)]=digest(a_path);raw[str(b_path)]=digest(b_path)
        assert digest(b_path)==record['steps'][-1]['raw_files'][b_path.name]['sha256']
        with np.load(a_path,allow_pickle=False) as az,np.load(b_path,allow_pickle=False) as bz:
            assert set(az.files)==set(bz.files)
            bodyfields={}
            for key in az.files:
                a,b=az[key],bz[key];assert a.shape==b.shape and a.dtype==b.dtype
                assert np.isfinite(a).all() and np.isfinite(b).all()
                exact=a.tobytes()==b.tobytes()
                if a.dtype.kind in 'biu':error=float(np.max(a!=b));scale=None;tol=0.;within=exact
                else:
                    error=float(np.max(abs(a-b))) if a.size else 0.
                    scale=max(1.,float(np.max(abs(a)))) if a.size else 1.;tol=1e-12*scale;within=error<=tol
                bodyfields[key]=dict(shape=list(a.shape),dtype=str(a.dtype),bitwise_equal=exact,
                    max_absolute_error=error,reference_scale=scale,existing_fixture_absolute_tolerance=tol,
                    within_existing_tolerance=bool(within),different_entries=int(np.count_nonzero(a!=b)),
                    reference_sha256=hashlib.sha256(a.tobytes()).hexdigest(),candidate_sha256=hashlib.sha256(b.tobytes()).hexdigest())
            fields[str(body)]=bodyfields
    impulses=[]
    for row in record['steps']:
        ref=next(r for r in timing['steps'] if r['step']==row['step'])
        a,b=np.asarray(ref['contact_impulse']),np.load(candidate/f"contact-impulse-step{row['step']}.npy")
        impulses.append(dict(step=row['step'],bitwise_equal=a.tobytes()==b.tobytes(),max_absolute_error=float(abs(a-b).max()),
            within_existing_tolerance=bool(np.max(abs(a-b))<=1e-12*max(1.,float(abs(a).max()))),
            candidate_solve_seconds=row['solve_seconds'],profiled_reference_solve_seconds=ref['seconds'],
            min_J_matches=row['target_min_J']==ref['target_min_J'],damage_matches=row['target_max_damage']==ref['target_max_damage']))
    result=dict(scope='Real compact106to110 with original step110 callback-profile reference; no full rerun and no production completion claim.',
        reference_scope=proof['scope'],counter_transform_exactly_verified=True,other_original_source_hash_checks=other_sources,
        original_iteration_limits_and_tolerances_unchanged=True,all_original_validity_passed=True,
        all_fields_within_existing_tolerance=all(v['within_existing_tolerance'] for fs in fields.values() for v in fs.values()),
        all_fields_bitwise_equal=all(v['bitwise_equal'] for fs in fields.values() for v in fs.values()),
        all_contact_impulses_within_existing_tolerance=all(v['within_existing_tolerance'] for v in impulses),
        candidate_compile_seconds=record['compile_seconds'],steps=impulses,raw_npz_sha256=raw,fields=fields,
        reference_contains_only_final_full_states=True,intermediate_comparison_scope='Per-step contact impulse, valid flag, minimum J and max damage; full arrays compared only at110.',
        production_continuation_complete=False)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('fields','other_original_source_hash_checks')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','reference','original','parent','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();compare(a.candidate,a.reference,a.original,a.parent,a.output)
