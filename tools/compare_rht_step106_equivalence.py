"""Compare every saved body field and the actual contact impulse after one step."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def compare(full, compact, output):
    reports=[json.loads((p/'result.json').read_text()) for p in (full,compact)]
    proofs=[json.loads((p/'source-review.json').read_text()) for p in (full,compact)]
    assert [r['mode'] for r in reports]==['full','compact']
    assert all(r['start_step']==105 and r['target_step']==106 and r['accepted_steps']==1 and r['valid'] for r in reports)
    assert proofs[0]['parent_checkpoint_sha256']==proofs[1]['parent_checkpoint_sha256']
    assert (full/'case.json').read_bytes()==(compact/'case.json').read_bytes()
    manifests=[json.loads((p/'runtime-source-manifest.json').read_text()) for p in (full,compact)]
    assert manifests[0].keys()==manifests[1].keys()
    changed=[k for k in manifests[0] if manifests[0][k]!=manifests[1][k]]
    assert changed==['mpm_jax/materials/rht.py'],changed
    records={}
    for name in ['state106-body0.npz','state106-body1.npz','contact-impulse.npy']:
        for folder,report in zip((full,compact),reports):
            h=hashlib.sha256()
            with (folder/name).open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
            assert h.hexdigest()==report['raw_files'][name]['sha256']
        files=[np.load(p/name,allow_pickle=False) for p in (full,compact)]
        if name.endswith('.npz'):
            assert set(files[0].files)==set(files[1].files)
            keys=files[0].files
        else:
            files=[{'contact_impulse':a} for a in files];keys=['contact_impulse']
        fields={}
        for key in keys:
            a,b=(z[key] for z in files)
            assert a.shape==b.shape and a.dtype==b.dtype
            assert np.isfinite(a).all() and np.isfinite(b).all()
            ha,hb=(hashlib.sha256(x.tobytes()).hexdigest() for x in (a,b))
            if a.dtype.kind in 'biu':
                error=float(np.max(a!=b));within=ha==hb;scale=None;tolerance=0.
            else:
                error=float(np.max(np.abs(a-b))) if a.size else 0.
                scale=max(1.,float(np.max(np.abs(a)))) if a.size else 1.
                tolerance=1e-12*scale;within=error<=tolerance
            fields[key]=dict(shape=list(a.shape),dtype=str(a.dtype),full_array_sha256=ha,
                compact_array_sha256=hb,bitwise_equal=ha==hb,max_absolute_error=error,
                numerically_different_entries=int(np.count_nonzero(a!=b)),
                reference_scale=scale,existing_fixture_absolute_tolerance=tolerance,
                within_existing_fixture_float_tolerance=bool(within))
        for file in files:
            if hasattr(file,'close'):file.close()
        records[name]=fields
    exact=all(r['bitwise_equal'] for fields in records.values() for r in fields.values())
    result=dict(scope='Exactly one real original105to106 step from the same parent, independent processes; not a completed production continuation.',
        same_initial_state=True,only_runtime_source_difference=changed,all_original_validity_passed=True,
        all_fields_and_contact_impulse_bitwise_equal=exact,
        all_within_existing_fixture_tolerance=all(r['within_existing_fixture_float_tolerance'] for fs in records.values() for r in fs.values()),
        compile_seconds=[r['compile_seconds'] for r in reports],solve_seconds=[r['solve_seconds'] for r in reports],
        parent_checkpoint_sha256=proofs[0]['parent_checkpoint_sha256'],fields=records,
        production_continuation_complete=False,step109_threshold_qualified=False)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='fields'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full',type=Path,required=True)
    parser.add_argument('--compact',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();compare(a.full,a.compact,a.output)
