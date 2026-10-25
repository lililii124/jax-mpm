"""Frozen-source, frozen-state step600 small-window CFL diagnostic AT2 bending prefix; no medium admission."""
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
CHECKPOINT_SHA = '0d3272f119cb394a5513edf201cdb004226c12b0ca1444ae87a2a2a1473847d5'


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


def physics_ast(source):
    tree=ast.parse(source)
    functions={'advance','strain_of','particle_state','potential'}
    assignments={'law','mass','active','denominator','left','right','punch','anchor','damping','ke'}
    selected=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.FunctionDef) and node.name in functions:
            selected.append(ast.dump(node,include_attributes=False))
        elif isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id in assignments for target in node.targets):
            selected.append(ast.dump(node,include_attributes=False))
    return selected


def prepare(output,parent_zip=None):
    output.mkdir(parents=True,exist_ok=False)
    archive=Path(parent_zip) if parent_zip else output.parent/'at2-c3-parent.zip'
    if not parent_zip:
        fetch(archive)
    assert archive.stat().st_size==PARENT_BYTES
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==PARENT_ZIP_SHA
    source_root=Path(__file__).resolve().parents[1]
    assert (source_root/'.github/workflows/engineering-previews.yml').read_bytes()==(source_root/'.github/workflows/phase-field-bending-c3-window.yml').read_bytes()
    old_driver=subprocess.check_output(['git','show',PARENT_HEAD+':examples/phase_field_bending_persistent_grid.py'],cwd=source_root)
    assert old_driver==(source_root/'examples/phase_field_bending_persistent_grid.py').read_bytes()
    diagnostic=(source_root/'examples/phase_field_bending_cfl_diagnostic.py').read_bytes()
    assert physics_ast(old_driver)==physics_ast(diagnostic),'Physics advance AST changed'
    with zipfile.ZipFile(archive) as z:
        assert z.read('source-head.txt').decode().strip()==PARENT_HEAD
        for line in z.read('source-sha256.txt').decode().splitlines():
            digest,name=line.split(maxsplit=1)
            frozen=subprocess.check_output(['git','show',PARENT_HEAD+':'+name],cwd=source_root)
            assert hashlib.sha256(frozen).hexdigest()==digest,name
        original_log=z.read('run.log').decode()
        assert 'assert all(np.array_equal(parent[key],resumed[key]) for key in parent.files)' in original_log
        assert 'AssertionError' in original_log
        prior=json.loads(z.read('evidence/raw/summary.json'))
        assert prior['steps']==800 and prior['maximum_kkt']<=1e-7 and prior['maximum_strain']<=.1
        prior_resources=json.loads(z.read('evidence/prefix800.resources.json'))
        assert prior_resources['returncode']==0 and not prior_resources['timed_out']
        for name in ['source-head.txt','source-sha256.txt','evidence/raw/checkpoint_000600.npz',
                     'evidence/raw/frame_000600.npz','evidence/raw/summary.json','evidence/prefix800.resources.json']:
            path=output/'input'/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(z.read(name))
    checkpoint=output/'input/evidence/raw/checkpoint_000600.npz'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==CHECKPOINT_SHA
    with np.load(checkpoint) as state:
        binding=json.loads(str(state['binding']))
        assert int(state['step'])==600 and binding['dx']==.025 and binding['particles']==53144
        assert binding['speed']==.12 and binding['displacement']==.035 and binding['cfl']==.04
        for name,digest in binding['source_hashes'].items():
            assert hashlib.sha256((source_root/name).read_bytes()).hexdigest()==digest,name
        assert all(np.isfinite(state[key]).all() for key in state.files if key!='binding')
    (output/'old-production-driver.py').write_bytes(old_driver)
    (output/'new-diagnostic-driver.py').write_bytes(diagnostic)
    (output/'old-prefix800-runner.py').write_bytes(subprocess.check_output(
        ['git','show',PARENT_HEAD+':tools/run_phase_field_bending_prefix800.py'],cwd=source_root))
    (output/'new-prefix800-export-check.py').write_bytes((source_root/'tools/run_phase_field_bending_prefix800.py').read_bytes())
    receipt=dict(parent_run=PARENT_RUN,parent_artifact=PARENT_ARTIFACT,parent_zip_sha256=PARENT_ZIP_SHA,
        input_checkpoint600_sha256=CHECKPOINT_SHA,input_binding=binding,
        original_run_status='failed only at recomputed elastic-export bitwise gate; not reclassified',
        physics_advance_and_boundary_material_ast_identical=True,production_driver_unchanged=True,
        physical_time_start=.018,physical_time_end=.01824,larger_loading_permitted=False)
    (output/'preflight.json').write_text(json.dumps(receipt,indent=2))
    return checkpoint


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--parent-zip')
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    output=Path(args.output)
    checkpoint=prepare(output,args.parent_zip)
    os.environ.pop('GH_TOKEN',None)
    if args.preflight_only:
        print('Frozen step600 carry and unchanged advance AST preflight passed')
        return
    resources={}
    cases=[('control_off','.04',608,True),('cfl04','.04',608,False),
           ('cfl02','.02',616,False),('cfl01','.01',632,False)]
    for name,cfl,stop,disabled in cases:
        command=[sys.executable,'-m','examples.phase_field_bending_cfl_diagnostic',
            '--dx','.025','--length','.05','--density','1','--gc','.005','--speed','.12',
            '--cfl','.04','--diagnostic-cfl',cfl,'--displacement','.035','--steps',str(stop),
            '--restart',str(checkpoint),'--output',str(output/name)]
        if disabled:command.append('--no-telemetry')
        resources[name]=measured_run(command,name,output,timeout=300)
        assert resources[name]['samples']>0 and resources[name]['peak_sampled_uss_bytes']>0
        assert resources[name]['peak_sampled_uss_bytes']<=.8*6*2**30
        with np.load(checkpoint) as original,np.load(output/name/'checkpoint_000600.npz') as loaded:
            for key in original.files:
                if key!='binding':assert np.array_equal(original[key],loaded[key]),(name,key)
        with np.load(output/'input/evidence/raw/frame_000600.npz') as parent,np.load(output/name/'frame_000600.npz') as loaded:
            check=compare_restart_frame(parent,loaded)
            (output/name/'initial-export-consistency.json').write_text(json.dumps(check,indent=2))
    with np.load(output/'control_off/checkpoint_000608.npz') as a,np.load(output/'cfl04/checkpoint_000608.npz') as b:
        assert set(a.files)==set(b.files)
        assert all(np.array_equal(a[key],b[key]) for key in a.files),'Telemetry changed solver carry'
    windows={}
    end_states={}
    for name,cfl,stop,disabled in cases:
        if disabled:continue
        rows=[json.loads(line) for line in (output/name/'step-diagnostics.jsonl').read_text().splitlines()]
        assert len(rows)==stop-600
        for row in rows:
            assert abs(row['nominal_elastic_cfl']-float(cfl))<1e-14
            assert row['maximum_kkt']<=1e-7 and row['maximum_strain']<=.1
        total=lambda key:sum(row[key] for row in rows)
        signed=total('actuator_work');projection=total('projection_loss')
        mechanical=total('frozen_strain_defect');kick=total('kick_drift_difference')
        persistent=total('kinematically_persistent_normal_projection')+total('anchor_projection')
        other=total('other_normal_projection')
        assert abs(projection-persistent-other)<=1e-10*max(projection,1e-20)
        assert abs(total('absolute_step_actuator_work')-rows[-1]['window_absolute_step_work'])<1e-20
        summary=json.loads((output/name/'summary.json').read_text())
        assert abs(summary['physical_end_time']-.01824)<1e-15
        windows[name]=dict(actual_cfl=float(cfl),steps=len(rows),physical_end_time=summary['physical_end_time'],
            signed_work=signed,absolute_step_work=total('absolute_step_actuator_work'),
            absolute_nodal_work=total('absolute_nodal_actuator_work'),projection_loss=projection,
            kick_drift_difference=kick,frozen_strain_defect=mechanical,
            kinematically_persistent_projection=persistent,other_normal_projection=other,
            gross_mechanical_plus_kick=mechanical+kick,
            combined_all_projection_counterterms=mechanical+kick-projection,
            persistent_projection_adjusted=mechanical+kick-persistent,
            maximum_projection_identity_relative_error=max(row['kick_projection_identity_relative_error'] for row in rows),
            maximum_actual_courant=max(row['elastic_plus_velocity_cfl'] for row in rows),
            maximum_kkt=summary['maximum_kkt'],maximum_strain=summary['maximum_strain'])
        with np.load(output/name/f'checkpoint_{stop:06d}.npz') as state:
            end_states[name]={key:state[key].copy() for key in ('nodal_displacement','nodal_velocity','history','nodal_damage')}
    convergence={}
    for key in end_states['cfl01']:
        coarse=float(np.linalg.norm(end_states['cfl04'][key]-end_states['cfl01'][key]))
        medium=float(np.linalg.norm(end_states['cfl02'][key]-end_states['cfl01'][key]))
        convergence[key]=dict(coarse_to_finest_l2=coarse,medium_to_finest_l2=medium,
            medium_error_reduced=medium<=.8*coarse+1e-12)
    result=dict(status='small_matched_time_diagnostic_completed',
        telemetry_control_terminal_checkpoint_bitwise_equal=True,windows=windows,
        terminal_carry_convergence=convergence,resources=resources,
        scope='64 total steps at t=.018 to .01824, below existing step800 loading',
        old_all_export_bitwise_gate_failed=True,old_800_step_not_recomputed=True,
        global_800_time_accuracy_proven=False,larger_loading_admitted=False,crack_validated=False)
    (output/'diagnostic-result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
