"""Frozen-source, frozen-state production step800->1200 with verified observations AT2 bending prefix; no medium admission."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import numpy as np
from prove_phase_field_bending_restart import measured_run
from phase_field_bending_diagnostics import compare_restart_frame

PARENT_RUN = 40165005721
PARENT_ARTIFACT = 12588543164
PARENT_HEAD = '7c8261cf2a008638fae25a86a964775e9bc735b4'
PARENT_ZIP_SHA = 'ed5e913b527e3612e65d39dcfe2f33fd1c7b4f2407df3ca418e34814a32fefb4'
PARENT_BYTES = 125677147
CHECKPOINT_SHA = 'ee9ae000bf4da8324c5f28a6631e5528c8fbe59a6466143067bfabe2be6c5d16'


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch(destination):
    # Authenticated API request and uncredentialed signed-blob requests use
    # separate request objects. Remove the token before starting the solver.
    token = os.environ.pop('GH_TOKEN')
    request = urllib.request.Request(
        f'https://api.github.com/repos/lililii124/jax-mpm/actions/artifacts/{PARENT_ARTIFACT}/zip',
        headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json'})
    try:
        urllib.request.build_opener(NoRedirect).open(request,timeout=30)
        raise RuntimeError('Expected official artifact redirect')
    except urllib.error.HTTPError as response:
        if response.code != 302:
            response.close()
            raise
        location = response.headers['Location']
        response.close()
    parsed = urllib.parse.urlsplit(location)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
        raise RuntimeError('Invalid signed artifact URL')
    partial = destination.with_suffix('.partial')
    deadline = time.monotonic()+180
    digest = hashlib.sha256()
    with partial.open('wb') as stream:
        for start in range(0,PARENT_BYTES,1024*1024):
            end = min(PARENT_BYTES,start+1024*1024)-1
            for attempt in range(3):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Parent download exceeded180seconds')
                try:
                    request = urllib.request.Request(location,headers={'Range':f'bytes={start}-{end}'})
                    with urllib.request.urlopen(request,timeout=min(30,deadline-time.monotonic())) as response:
                        assert response.status == 206
                        assert response.headers['Content-Range'] == f'bytes {start}-{end}/{PARENT_BYTES}'
                        block = response.read(end-start+2)
                    assert len(block)==end-start+1
                    break
                except Exception:
                    if attempt==2:
                        raise
            digest.update(block)
            stream.write(block)
    assert partial.stat().st_size==PARENT_BYTES and digest.hexdigest()==PARENT_ZIP_SHA
    partial.replace(destination)


from run_phase_field_bending_c3_window import physics_ast
from phase_field_bending_connectivity import analyze

def telemetry_ast(source):
    nodes=[node for node in ast.walk(ast.parse(source))
        if isinstance(node,ast.If) and isinstance(node.test,ast.Name) and node.test.id=='telemetry']
    assert len(nodes)==1
    return ast.dump(nodes[0],include_attributes=False)


def prepare(output,parent_zip=None):
    output.mkdir(parents=True,exist_ok=False)
    archive=Path(parent_zip) if parent_zip else output.parent.parent/'at2-prefix1200-parent.zip'
    if not parent_zip:fetch(archive)
    assert archive.stat().st_size==PARENT_BYTES
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==PARENT_ZIP_SHA
    source_root=Path(__file__).resolve().parents[1]
    old_driver=subprocess.check_output(['git','show',PARENT_HEAD+':examples/phase_field_bending_persistent_grid.py'],cwd=source_root)
    assert old_driver==(source_root/'examples/phase_field_bending_persistent_grid.py').read_bytes()
    observed=(source_root/'examples/phase_field_bending_observed_production.py').read_bytes()
    assert physics_ast(old_driver)==physics_ast(observed),'Production physics/material/BC AST changed'
    qualified='1a04b4153d1920184c05f9583458ec7b0543bfbb'
    for name in ('tools/phase_field_bending_diagnostics.py','tools/run_phase_field_bending_c3_window.py'):
        assert subprocess.check_output(['git','show',qualified+':'+name],cwd=source_root)==(source_root/name).read_bytes()
    qualified_observer=subprocess.check_output(['git','show',qualified+':examples/phase_field_bending_cfl_diagnostic.py'],cwd=source_root)
    assert telemetry_ast(qualified_observer)==telemetry_ast(observed),'Qualified observational block changed'
    assert (source_root/'.github/workflows/engineering-previews.yml').read_bytes()==(source_root/'.github/workflows/phase-field-bending-prefix1200.yml').read_bytes()
    with zipfile.ZipFile(archive) as z:
        assert z.read('source-head.txt').decode().strip()==PARENT_HEAD
        for line in z.read('source-sha256.txt').decode().splitlines():
            expected,name=line.split(maxsplit=1)
            assert hashlib.sha256(subprocess.check_output(['git','show',PARENT_HEAD+':'+name],cwd=source_root)).hexdigest()==expected
        assert 'assert all(np.array_equal(parent[key],resumed[key]) for key in parent.files)' in z.read('run.log').decode()
        old_summary=json.loads(z.read('evidence/raw/summary.json'))
        assert old_summary['steps']==800 and old_summary['maximum_kkt']<=1e-7 and old_summary['maximum_strain']<=.1
        prior_resources=json.loads(z.read('evidence/prefix800.resources.json'))
        assert prior_resources['returncode']==0 and not prior_resources['timed_out']
        for name in ['source-head.txt','source-sha256.txt','evidence/raw/checkpoint_000800.npz',
                     'evidence/raw/frame_000800.npz','evidence/raw/history.csv','evidence/raw/summary.json']:
            path=output/'input'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(name))
    checkpoint=output/'input/evidence/raw/checkpoint_000800.npz'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==CHECKPOINT_SHA
    with np.load(checkpoint) as state:
        binding=json.loads(str(state['binding']))
        assert int(state['step'])==800 and 'diagnostic_window' not in binding
        for key,value in dict(dx=.025,length=.05,density=1.,gc=.005,speed=.12,
                              displacement=.035,cfl=.04,required_steps=9923,particles=53144).items():
            assert binding[key]==value,(key,binding[key])
        for name,digest in binding['source_hashes'].items():
            assert hashlib.sha256((source_root/name).read_bytes()).hexdigest()==digest,name
        assert all(np.isfinite(state[key]).all() for key in state.files if key!='binding')
        assert float(state['maximum_kkt'])<=1e-7 and float(state['maximum_strain'])<=.1
        assert state['history'].min()>=0 and state['nodal_damage'].min()>=0 and state['nodal_damage'].max()<=1
        assert abs(state['volume'].sum()-.098304)<1e-14 and state['nodal_mass'].min()>=0
    (output/'old-physics-driver.py').write_bytes(old_driver)
    (output/'observed-production-driver.py').write_bytes(observed)
    receipt=dict(parent_run=PARENT_RUN,parent_artifact=PARENT_ARTIFACT,parent_source=PARENT_HEAD,
        parent_zip_sha256=PARENT_ZIP_SHA,input_checkpoint800_sha256=CHECKPOINT_SHA,
        qualified_C3_run=40169060267,qualified_C3_source=qualified,
        physics_material_BC_ast_identical=True,verified_observer_ast_identical=True,
        source_physics_binding=binding,start_step=800,stop_step=1200,
        absolute_work_pre800_unknown=True,old_export_bitwise_failure_preserved=True,
        global_CFL_accuracy_not_certified=True,crack_validated=False)
    (output/'preflight.json').write_text(json.dumps(receipt,indent=2))
    return checkpoint


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--parent-zip')
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args();output=Path(args.output)
    checkpoint=prepare(output,args.parent_zip);os.environ.pop('GH_TOKEN',None)
    if args.preflight_only:
        print('Strict original production800 carry, old physics and qualified observer preflight passed')
        return
    command=[sys.executable,'-m','examples.phase_field_bending_observed_production',
        '--dx','.025','--length','.05','--density','1','--gc','.005','--speed','.12',
        '--cfl','.04','--displacement','.035','--steps','1200','--restart',str(checkpoint),
        '--output',str(output/'raw')]
    usage=measured_run(command,'prefix1200',output,timeout=2400)
    assert usage['samples']>0 and usage['peak_sampled_uss_bytes']>0
    assert max(usage['peak_sampled_uss_bytes'],usage['peak_sampled_rss_bytes'])<6*2**30
    assert usage['peak_sampled_uss_bytes']<=.8*6*2**30
    assert hashlib.sha256((output/'raw/checkpoint_000800.npz').read_bytes()).hexdigest()==CHECKPOINT_SHA
    with np.load(output/'input/evidence/raw/frame_000800.npz') as a,np.load(output/'raw/frame_000800.npz') as b:
        export_check=compare_restart_frame(a,b)
    (output/'restart-export-consistency.json').write_text(json.dumps(export_check,indent=2))
    summary=json.loads((output/'raw/summary.json').read_text())
    assert summary['steps']==1200 and summary['resumed_from_step']==800
    assert summary['maximum_kkt']<=1e-7 and summary['maximum_strain']<=.1
    rows=[json.loads(line) for line in (output/'raw/step-diagnostics.jsonl').read_text().splitlines()]
    assert len(rows)==400 and rows[0]['step']==801 and rows[-1]['step']==1200
    connectivity=[]
    for path in sorted((output/'raw').glob('frame_*.npz')):
        with np.load(path) as frame:
            assert all(np.isfinite(frame[key]).all() for key in frame.files)
            connectivity.append(dict(frame=path.name,time=float(frame['time']),
                thresholds=[analyze(frame,threshold,.025) for threshold in (.8,.9,.95)]))
    (output/'connectivity-history.json').write_text(json.dumps(connectivity,indent=2))
    total=lambda key:sum(row[key] for row in rows)
    projection=total('projection_loss')
    persistent=total('kinematically_persistent_normal_projection')+total('anchor_projection')
    other=total('other_normal_projection')
    assert abs(projection-persistent-other)<=1e-10*max(projection,1e-20)
    result=dict(status='bounded_observed_production1200_completed_unvalidated',start_step=800,stop_step=1200,
        resources=usage,maximum_particle_damage=summary['maximum_particle_damage'],
        particles_above_point9=summary['particles_above_point9'],
        maximum_kkt=summary['maximum_kkt'],maximum_strain=summary['maximum_strain'],
        absolute_step_work_since800=total('absolute_step_actuator_work'),
        absolute_nodal_work_since800=total('absolute_nodal_actuator_work'),
        signed_work_since800=total('actuator_work'),absolute_work_before800_unknown=True,
        persistent_projection=persistent,other_normal_projection=other,
        nominal_CFL_maximum=max(row['nominal_elastic_cfl'] for row in rows),
        wave_plus_velocity_Courant_maximum=max(row['elastic_plus_velocity_cfl'] for row in rows),
        actual_identity_relative_error_maximum=max(row['kick_projection_identity_relative_error'] for row in rows),
        production_checkpoint1200_sha256=hashlib.sha256((output/'raw/checkpoint_001200.npz').read_bytes()).hexdigest(),
        global_accuracy_validated=False,quasistatic_claim=False,crack_validated=False)
    (output/'prefix-result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
