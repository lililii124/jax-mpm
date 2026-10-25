"""Read-only HJC40 whole-parent/child audit and byte-preserving delivery subsets.

No JAX import, physical integration, rendering, or structural-energy certification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARENT = dict(run=40169284307, artifact=12589003513,
    commit='52cea038373934b8c80596f5c42518bb37a0e9fc', bytes=180104537,
    sha256='61d9c91b823e92a7897359fe5643eb168df50e97f23380a687378c4e6c27af0e', frame=20)
CHILD = dict(run=40170774252, artifact=12590851516,
    commit='d857cec8d682446a1f4821ebd0fb3b713ef761a2', bytes=323177695,
    sha256='301cc2a68469363886e6d0ce0e0c2e2580f022dfb932d9ae3a8054994b6cc3ec', frame=40)
PARENT_REVIEW = '93a3cfe44126c24fb342c06da7550987d2dc5e870692f885f59ee522060523ed'
PART_BYTES = 90_000_000


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def load(path):
    return json.loads(path.read_text())


def official_archive(label, specification, destination):
    target = destination/label
    target.mkdir()
    base = 'https://api.github.com/repos/'+os.environ['GITHUB_REPOSITORY']
    headers = {'Authorization':'Bearer '+os.environ['GH_TOKEN'],
               'Accept':'application/vnd.github+json','User-Agent':'hjc-read-only-prefix-audit'}
    def api(path):
        with urllib.request.urlopen(urllib.request.Request(base+path,headers=headers),timeout=30) as response:
            return json.load(response)
    run = api('/actions/runs/'+str(specification['run']))
    meta = api('/actions/artifacts/'+str(specification['artifact']))
    assert run['head_sha'] == specification['commit'] and run['conclusion'] == 'success'
    assert meta['workflow_run']['id'] == specification['run']
    assert meta['workflow_run']['head_sha'] == specification['commit']
    assert meta['size_in_bytes'] == specification['bytes']
    assert meta['digest'] == 'sha256:'+specification['sha256'] and not meta['expired']
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, request, response, code, message, response_headers, newurl):
            return None
    try:
        urllib.request.build_opener(NoRedirect).open(urllib.request.Request(
            base+'/actions/artifacts/'+str(specification['artifact'])+'/zip',headers=headers),timeout=30)
        raise RuntimeError('Expected authenticated archive redirect')
    except urllib.error.HTTPError as response:
        if response.code != 302:
            raise
        signed_url = response.headers['Location']
    archive = target/'official-artifact.zip'
    # Do not forward the GitHub Authorization header to the blob service.
    with urllib.request.urlopen(signed_url,timeout=90) as response, archive.open('wb') as stream:
        for block in iter(lambda:response.read(1024*1024), b''):
            stream.write(block)
    assert archive.stat().st_size == specification['bytes'] and sha(archive) == specification['sha256']
    restored = target/'restored'
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        assert len(infos) == len({info.filename for info in infos})
        total = sum(info.file_size for info in infos)
        assert total < 4*1024**3
        assert shutil.disk_usage(destination).free > total+2*1024**3
        for info in infos:
            name = PurePosixPath(info.filename)
            assert not name.is_absolute() and '..' not in name.parts
            assert (info.external_attr >> 16)&0o170000 != 0o120000
        assert bundle.testzip() is None
        bundle.extractall(restored)
    receipt = dict(**specification, official_digest_matched=True,
        all_member_crc_checked=True, zip_members=len(infos), uncompressed_bytes=total)
    save(target/'archive-review.json', receipt)
    print(json.dumps(dict(stage=label+'-whole-archive', **receipt)),flush=True)
    return restored, receipt


def npz(path):
    with np.load(path,allow_pickle=False) as values:
        return {name:np.array(values[name]) for name in values.files}


def checked_state(values, initial):
    assert all(np.isfinite(value).all() for value in values.values())
    assert len(values['x']) == 121104 and bool(values['valid']) and np.all(values['history_valid'])
    assert all(np.array_equal(values[name],initial[name]) for name in ('mass','volume'))
    determinant = np.linalg.det(values['F'])
    assert determinant.min() > 0
    np.testing.assert_allclose(values['history_density'],2440/determinant,rtol=2e-14,atol=0)
    assert np.all((values['history_damage'] >= 0)&(values['history_damage'] <= 1))
    return determinant


def provenance_and_state(folder, specification):
    producer = load(folder/'prefix-review.json')
    assert producer['passed'] and all(producer['checks'].values())
    actual = {path.relative_to(folder).as_posix() for path in folder.rglob('*')
              if path.is_file() and path.name != 'prefix-review.json'}
    assert actual == set(producer['files'])
    for name, expected in producer['files'].items():
        path = (folder/name).resolve()
        assert path.is_relative_to(folder.resolve()) and sha(path) == expected, name
    source = load(folder/'source.json')
    assert source['commit'] == specification['commit']
    for name, expected in source['files'].items():
        raw = subprocess.check_output(['git','show',specification['commit']+':'+name],cwd=ROOT)
        assert hashlib.sha256(raw).hexdigest() == expected, name
    plan = load(folder/'plan.json')
    cp = load(folder/'raw/checkpoint.json')
    status = load(folder/'raw/status.json')
    frame = specification['frame']
    assert plan['stop_frame'] == cp['frame'] == status['frame'] == frame
    full = frame == 60
    assert status['phase'] == ('completed' if full else 'checkpoint-ready')
    assert status['segment_complete'] is (not full) and status['completed'] is full
    assert status['signature'] == cp['signature']
    assert plan['frames'] == status['frames'] == 60 and plan['substeps'] == 175
    assert plan['particles'] == 121104 and status['steps_completed'] == frame*175
    assert len(cp['rows']) == len(cp['observations']) == frame+1
    definition = plan['definition']
    assert definition['model'] == 'hjc' and definition['failure_mode'] == 'retain'
    assert definition['material_parameters_SI']['density'] == 2440
    assert definition['material_parameters_SI']['mat111_compatible']
    assert definition['material_parameters_SI']['mat111_compacted_tension']
    assert definition['material_parameters_SI']['legacy_v3']
    assert definition['acoustic_CFL'] < .15
    grid = round(.4/definition['grid_spacing_m'])
    selected = definition['support_selected_grid_x_indices']
    left, right = np.asarray(selected['left']), np.asarray(selected['right'])
    assert len(left) and np.array_equal(left,grid-right[::-1])
    assert definition['symmetry_plane_selected_indices'] == {'x':[grid//2],'z':[grid//2]}
    initial = npz(folder/'initial_state.npz')
    assert all(np.isfinite(value).all() for value in initial.values())
    assert bool(initial['valid']) and np.all(initial['history_valid'])
    assert not initial['history_damage'].any()
    np.testing.assert_array_equal(initial['F'],np.broadcast_to(np.eye(3),initial['F'].shape))
    assert abs(initial['mass'].sum()-2440*.3*.06**2) < 1e-12
    assert abs(initial['volume'].sum()-.3*.06**2) < 1e-15
    np.testing.assert_allclose(initial['mass'],2440*initial['volume'],rtol=2e-15,atol=0)
    final = npz(folder/'raw'/cp['states'][0])
    determinant = checked_state(final,initial)
    assert abs(float(final['time'])-frame*25e-6) < 1e-15
    guard = load(folder/'resource-guard.json')
    assert guard['returncode'] == 0 and guard['guard_reason'] is None
    assert guard['peak_uss_bytes'] < guard['uss_limit_bytes']
    return dict(producer=producer, source=source, plan=plan, checkpoint=cp,
                initial=initial, final=final, min_J=float(determinant.min()),guard=guard)


def frames_and_mirrors(folder, state, expected_frame):
    initial, final, cp = state['initial'],state['final'],state['checkpoint']
    reference = initial['x']
    axes = [np.unique(reference[:,axis]) for axis in range(3)]
    shape = tuple(map(len,axes))
    labels = np.array([np.searchsorted(axes[axis],reference[:,axis]) for axis in range(3)])
    assert np.prod(shape) == len(reference)
    index = np.ravel_multi_index(labels,shape)
    lookup = np.empty(len(reference),dtype=np.int64)
    lookup[index] = np.arange(len(reference))
    pairs = {}
    for axis,name in ((0,'x'),(2,'z')):
        mirrored = labels.copy()
        mirrored[axis] = shape[axis]-1-mirrored[axis]
        pairs[name] = lookup[np.ravel_multi_index(mirrored,shape)]
        target = reference.copy()
        target[:,axis] = .4-target[:,axis]
        np.testing.assert_allclose(reference[pairs[name]],target,rtol=0,atol=1e-15)
    files = list((folder/'raw/frames_data').glob('frame_*.npz'))
    assert len(files) == expected_frame+1
    records, previous = [], np.zeros(len(reference))
    volume = initial['volume']
    for frame in range(expected_frame+1):
        value = npz(folder/'raw/frames_data'/f'frame_{frame:05d}.npz')
        assert all(np.isfinite(field).all() for field in value.values())
        assert value['x'].shape == (121104,3)
        for name in ('damage','temperature','speed'):
            assert value[name].shape == (121104,)
        damage = value['damage']
        assert np.all(damage >= previous) and np.all((damage >= 0)&(damage <= 1))
        assert abs(float(value['time'])-frame*25e-6) < 1e-15
        observation = cp['observations'][frame]
        row = cp['rows'][frame]
        kinetic = .5*np.sum(initial['mass']*value['speed']**2)
        np.testing.assert_allclose(row['kinetic_energy'],kinetic,rtol=2e-14,atol=1e-10)
        for axis,name in enumerate(('x','y','z')):
            np.testing.assert_allclose(row['com_'+name],np.average(value['x'][:,axis],weights=initial['mass']),rtol=2e-14,atol=1e-14)
        rms = {}
        displacement = value['x']-reference
        for axis,name in ((0,'x'),(2,'z')):
            difference = damage-damage[pairs[name]]
            for metric, computed in (('mae',np.average(np.abs(difference),weights=volume)),
                ('rms',np.sqrt(np.average(difference**2,weights=volume))),
                ('max',np.max(np.abs(difference)))):
                assert abs(float(computed)-observation[f'damage_mirror_{name}_{metric}']) < 1e-15
                if metric == 'rms':
                    rms[name] = float(computed)
                    assert computed < 1e-2
            reflected = displacement[pairs[name]].copy()
            reflected[:,axis] *= -1
            du_rms = np.sqrt(np.average(np.sum((displacement-reflected)**2,axis=1),weights=volume))
            assert abs(float(du_rms)-observation[f'displacement_mirror_{name}_rms_m']) < 1e-15
        assert abs(float(np.max(damage))-observation['max_damage']) < 1e-15
        assert abs(float(np.average(damage,weights=volume))-observation['mean_damage_reference_volume']) < 1e-15
        if frame == 0:
            assert np.array_equal(value['x'],reference)
        if frame == expected_frame:
            for name, leaf in (('x','x'),('damage','history_damage'),('temperature','history_temperature')):
                assert np.array_equal(value[name],final[leaf])
            np.testing.assert_allclose(value['speed'],np.linalg.norm(final['v'],axis=1),rtol=2e-14,atol=1e-13)
            np.testing.assert_allclose(row['constitutive_work'],np.sum(final['mass']*final['history_work']),rtol=2e-14,atol=1e-10)
            np.testing.assert_allclose(row['current_volume'],np.sum(final['volume']*np.linalg.det(final['F'])),rtol=2e-14,atol=1e-15)
        records.append(dict(frame=frame,time_s=float(value['time']),damage_max=float(damage.max()),
            damage_mean_reference_volume=float(np.average(damage,weights=volume)),damage_mirror_rms=rms))
        previous = damage
    return records


def parent_continuity(parent, child, pstate, cstate):
    embedded = child/'parents'/f"frame-{PARENT['frame']:05d}"
    assert sha(parent/'prefix-review.json') == PARENT_REVIEW
    for name in ('source.json','plan.json','prefix-review.json','resource-guard.json'):
        assert sha(parent/name) == sha(embedded/name), name
    assert sha(parent/'raw/checkpoint.json') == sha(embedded/'checkpoint.json')
    for name in pstate['checkpoint']['states']:
        assert sha(parent/'raw'/name) == sha(embedded/name)
    assert sha(parent/'initial_state.npz') == sha(child/'initial_state.npz')
    for frame in range(PARENT['frame']+1):
        name = f'raw/frames_data/frame_{frame:05d}.npz'
        assert sha(parent/name) == sha(child/name)
    for key in ('rows','observations'):
        assert cstate['checkpoint'][key][:PARENT['frame']+1] == pstate['checkpoint'][key]
    assert cstate['checkpoint']['signature'] == pstate['checkpoint']['signature']
    assert cstate['plan']['definition'] == pstate['plan']['definition']
    allowed = {'.github/workflows/engineering-previews.yml','.github/workflows/hjc-mat111-beam-continue.yml'}
    pfiles, cfiles = pstate['source']['files'], cstate['source']['files']
    assert set(pfiles) == set(cfiles)
    for name in pfiles:
        if name not in allowed:
            assert pfiles[name] == cfiles[name], name
    assert np.all(cstate['final']['history_damage'] >= pstate['final']['history_damage'])
    return dict(parent_review_sha256=PARENT_REVIEW,preserved_raw_frames=PARENT['frame']+1,
        parent_checkpoint_sha256=sha(parent/'raw'/pstate['checkpoint']['states'][0]),
        numerical_and_noncontroller_source_unchanged=True,
        allowed_controller_differences=sorted(allowed))


def delivery(child, state, output):
    output.mkdir()
    checkpoint_names = ['source.json','plan.json','prefix-review.json','resource-guard.json',
        'initial_state.npz','raw/checkpoint.json','raw/status.json','raw/summary.json','raw/trajectory.npz']
    checkpoint_names += ['raw/'+name for name in state['checkpoint']['states']]
    checkpoint_names = [name for name in checkpoint_names if (child/name).is_file()]
    render_names = ['source.json','plan.json','prefix-review.json','raw/summary.json','raw/status.json','raw/trajectory.npz']
    render_names = [name for name in render_names if (child/name).is_file()]
    render_names += [f'raw/frames_data/frame_{frame:05d}.npz' for frame in range(CHILD['frame']+1)]
    def pack(name, files):
        path = output/name
        with tarfile.open(path,'w:gz',compresslevel=6) as bundle:
            for relative in files:
                bundle.add(child/relative,arcname=relative,recursive=False)
        return path
    checkpoint = pack('final-checkpoint.tar.gz',checkpoint_names)
    assert checkpoint.stat().st_size < PART_BYTES
    render = pack('render-data.tar.gz',render_names)
    parts = []
    with render.open('rb') as stream:
        index = 0
        while True:
            block = stream.read(PART_BYTES)
            if not block:
                break
            part = output/f'render-data.tar.gz.part-{index:04d}'
            part.write_bytes(block)
            parts.append(dict(file=part.name,bytes=part.stat().st_size,sha256=sha(part)))
            index += 1
    assert len(parts) <= 4
    selected = sorted(set(checkpoint_names+render_names))
    report = dict(scope='Byte-preserving subsets of completely verified official child artifact; not a full restart capsule.',
        original_artifact=CHILD, checkpoint=dict(file=checkpoint.name,bytes=checkpoint.stat().st_size,sha256=sha(checkpoint)),
        rendering=dict(concatenated_file=render.name,bytes=render.stat().st_size,sha256=sha(render),parts=parts),
        original_member_files={name:dict(bytes=(child/name).stat().st_size,sha256=sha(child/name)) for name in selected},
        frames=CHILD['frame']+1,particles=121104,physical_steps_executed=0,all_parts_below100MB=True)
    save(output/'delivery-manifest.json',report)
    return report


def main():
    parser = argparse.ArgumentParser()
    global PARENT, CHILD, PARENT_REVIEW
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--specification',type=Path)
    args = parser.parse_args()
    if args.specification:
        specification = load(args.specification)
        PARENT, CHILD, PARENT_REVIEW = specification['parent'],specification['child'],specification['parent_review']
        assert 0 < PARENT['frame'] < CHILD['frame'] <= 60
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Fresh audit directory required')
    output.mkdir(parents=True)
    parent, pzip = official_archive('parent',PARENT,output)
    child, czip = official_archive('child',CHILD,output)
    pstate = provenance_and_state(parent,PARENT)
    cstate = provenance_and_state(child,CHILD)
    precords = frames_and_mirrors(parent,pstate,PARENT['frame'])
    records = frames_and_mirrors(child,cstate,CHILD['frame'])
    continuity = parent_continuity(parent,child,pstate,cstate)
    spare = []
    for path in sorted((child/'raw').glob('checkpoint_*.npz')):
        values = npz(path)
        determinant = checked_state(values,cstate['initial'])
        frame = round(float(values['time'])/25e-6)
        assert 0 <= frame <= CHILD['frame']
        snapshot = npz(child/'raw/frames_data'/f'frame_{frame:05d}.npz')
        assert np.array_equal(values['x'],snapshot['x'])
        assert np.array_equal(values['history_damage'],snapshot['damage'])
        spare.append(dict(file=path.name,frame=frame,valid=True,min_J=float(determinant.min())))
    checkpoint_delivery = delivery(child,cstate,output/'delivery')
    current_wall = cstate['guard']['wall_seconds']
    max_uss = cstate['guard']['peak_uss_bytes']
    solver_delta = cstate['checkpoint']['solver_seconds']-pstate['checkpoint']['solver_seconds']
    projected_wall = 1.25*current_wall
    resource_ok = projected_wall < cstate['guard']['seconds_limit'] and 1.25*max_uss < cstate['guard']['uss_limit_bytes']
    assert 'jax' not in sys.modules
    report = dict(passed=True,scope='Whole raw/source/state/symmetry/continuation integrity; not structural energy or time/mesh convergence certification.',
        parent_archive=pzip,child_archive=czip,parent_continuity=continuity,
        numerical_commit=CHILD['commit'],verifier_commit=os.environ.get('GITHUB_SHA'),
        parent_verified_sources=len(pstate['source']['files']),child_verified_sources=len(cstate['source']['files']),
        child_review_sha256=sha(child/'prefix-review.json'),final_checkpoint_sha256=sha(child/'raw'/cstate['checkpoint']['states'][0]),
        frames=CHILD['frame']+1,steps=CHILD['frame']*175,particles=121104,final_time_s=float(cstate['final']['time']),
        min_J=cstate['min_J'],mass_kg=float(cstate['final']['mass'].sum()),
        final_damage_max=float(cstate['final']['history_damage'].max()),all_saved_damage_irreversible=True,
        parent_frame_observations=precords,frame_observations=records,all_raw_checkpoint_states=spare,
        guard=cstate['guard'],integration_solver_seconds_this_segment=solver_delta,
        continuation_to60=dict(integrity_admitted=True,resource_screen_passed=bool(resource_ok),
            allowed_under_existing_guard=bool(resource_ok),additional_steps=(60-CHILD['frame'])*175,
            projected_wall_seconds_with25percent_margin=projected_wall,
            scope='Same frozen numerical equations/loading; retain runtime/source/state and existing wall/USS guards. No automatic dispatch.'),
        delivery=checkpoint_delivery,structural_energy_certified=False,temporal_convergence_certified=False,
        physical_steps_executed=0,jax_initialized=False,public_visual_review_passed=False)
    observations = cstate['checkpoint']['observations']
    sample_times = np.asarray([row['boundary_sample_time_s'] for row in observations])
    supplied_power = -np.asarray([row['loading_reaction_y_N']*row['prescribed_speed_y_m_per_s'] for row in observations])
    sampled_work = float(np.sum(.5*(supplied_power[1:]+supplied_power[:-1])*np.diff(sample_times)))
    final_row = cstate['checkpoint']['rows'][-1]
    kinetic_plus_work = final_row['kinetic_energy']+final_row['constitutive_work']
    report['saved_sample_energy_diagnostic'] = dict(
        external_work_ledger_J=cstate['checkpoint']['external_work'],
        loading_reaction_saved_sample_work_J=sampled_work,
        final_kinetic_plus_constitutive_work_J=kinetic_plus_work,
        signed_difference_J=sampled_work-kinetic_plus_work,
        normalized_difference=(sampled_work-kinetic_plus_work)/max(abs(sampled_work),abs(kinetic_plus_work),1e-30),
        qualified=False,
        limitation='Saved reaction is a hypothetical next-step boundary probe at time+dt, velocity is sampled at time, and 25us saved cadence is not a step-level work ledger. Constitutive work is not a separately validated stored/dissipated energy split.')
    save(output/f"independent-hjc{CHILD['frame']}-review.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('frame_observations','parent_frame_observations','delivery')},indent=2))


if __name__ == '__main__':
    main()
