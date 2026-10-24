"""Verify a complete original pull-off cache, render a detail view and archive all raw bytes."""

import argparse
import csv
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PRODUCER = '81c57439c71cb3f48c575c4d55ee9fcc89b3aa39'
FOCUS = [-.012, .012, -.003, .0085, -.012, .012]
PART_BYTES = 192*1024**2


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def file_record(path):
    path = Path(path)
    before = path.stat()
    result = {'bytes': before.st_size, 'sha256': sha(path)}
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('File changed while reading: '+str(path))
    return result


def git(root, *arguments):
    return subprocess.check_output(['git', '-C', str(root), *arguments])


def source_binding(checkout, commit, expected=None, names=None):
    checkout = Path(checkout).resolve()
    if not re.fullmatch('[0-9a-f]{40}', commit) or git(checkout, 'rev-parse', 'HEAD').decode().strip() != commit:
        raise ValueError('The source checkout is not the requested complete Git commit')
    if expected is not None and (expected['commit'] != commit or expected['tree'] != git(checkout, 'rev-parse', commit+'^{tree}').decode().strip()):
        raise ValueError('Original producer commit/tree differs from receipt')
    if expected is not None and (not expected['files']
            or set(expected['files']) != set(expected['git_blob_sha'])
            or set(expected['files']) != set(expected['lf_sha256'])
            or expected['fingerprint'] != hashlib.sha256(json.dumps(expected['files'],sort_keys=True).encode()).hexdigest()):
        raise ValueError('Original producer source inventory/fingerprint differs')
    names = sorted(expected['files'] if expected is not None else names)
    records = {}
    for name in names:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or (checkout/path).is_symlink():
            raise ValueError('Unsafe source path')
        blob = git(checkout, 'show', f'{commit}:{name}')
        raw = (checkout/path).read_bytes()
        oid = hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()
        digest = hashlib.sha256(blob).hexdigest()
        normalized = hashlib.sha256(blob.decode().replace('\r\n','\n').encode()).hexdigest()
        if raw != blob or (expected is not None and (
            expected['files'][name] != digest or expected['git_blob_sha'][name] != oid or expected['lf_sha256'][name] != normalized)):
            raise ValueError('Source file differs from the original Git blob: '+name)
        records[name] = {'sha256':digest,'git_blob_sha':oid,'lf_sha256':normalized}
    return {'commit':commit, 'tree':git(checkout,'rev-parse',commit+'^{tree}').decode().strip(), 'files':records}


def load_producer(checkout):
    sys.path.insert(0, str(Path(checkout).resolve()))
    module = importlib.import_module('tools.run_fibre_pull_off')
    if Path(module.__file__).resolve() != Path(checkout).resolve()/'tools/run_fibre_pull_off.py':
        raise ValueError('The original producer auditor was shadowed by another checkout')
    return module


def verify_cache(cache, receipt_sha, producer_root, commit):
    cache = Path(cache).resolve()
    if sha(cache/'receipt.json') != receipt_sha:
        raise ValueError('The original complete receipt hash differs')
    receipt = read(cache/'receipt.json')
    if (receipt.get('schema') != 'fibre-pull-off-cache-v1' or receipt.get('kind') != 'accepted-prefix'
            or receipt.get('never_resumable') is not False or receipt.get('numerical_all_checks') is not True
            or receipt.get('prefix_verified') is not True or receipt.get('frame') != 60
            or receipt.get('production_eligible') is not False or receipt['source']['commit'] != commit):
        raise ValueError('Only the complete accepted frame-60 cache can be finalized')
    binding = source_binding(producer_root,commit,receipt['source'])
    original = load_producer(producer_root)
    req = receipt['request']
    if original.request(req['model']) != req or original.payload_inventory(cache,req) != receipt['files']:
        raise ValueError('Original raw file inventory or complete physical request differs')
    current, ancestry = receipt, set()
    while True:
        if (current.get('kind') != 'accepted-prefix' or current.get('never_resumable') is not False
                or current.get('numerical_all_checks') is not True or current['request'] != req
                or original.numerical_identity(current['source']) != original.numerical_identity(receipt['source'])
                or current['cache_key'] != original.cache_key(current['source'],req['model'],current['frame'])
                or current['prefix_review']['frame'] != current['frame'] or not current['prefix_review']['prefix_verified']):
            raise ValueError('Original accepted receipt ancestry is inconsistent')
        parent = current.get('parent_receipt_sha256','')
        if not parent:
            if current['frame'] != 2 or current.get('parent_cache_key'):
                raise ValueError('The original lineage must start with its frame-2 resource prefix')
            break
        if not re.fullmatch('[0-9a-f]{64}',parent) or parent in ancestry:
            raise ValueError('Invalid receipt ancestry')
        path = cache/'receipts'/f'{parent}.json'
        if sha(path) != parent:
            raise ValueError('Changed original parent receipt')
        old = read(path)
        if old['cache_key'] != current['parent_cache_key'] or old['frame'] >= current['frame']:
            raise ValueError('Receipt ancestry does not advance saved frames')
        ancestry.add(parent); current = old
    if {p.stem for p in (cache/'receipts').glob('*.json')} != ancestry:
        raise ValueError('Unbound receipt files')
    runtime = read(cache/'runtime/environment.json')
    if (cache/'runtime/requirements.txt').read_text() != original.requirements_text(runtime):
        raise ValueError('Original environment lock differs from its requirements')
    if read(cache/'runtime/request.json') != req or original.numerical_identity(read(cache/'runtime/source.json')) != original.numerical_identity(receipt['source']):
        raise ValueError('Original runtime source/request differs')
    return receipt, binding, original


def independent_material_review(folder):
    plan = read(folder/'plan.json')
    p = plan['material']; E,nu = p['young'],p['poisson']
    mu,lam = E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu))
    with (folder/'observables.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    with np.load(folder/'final_state_0.npz',allow_pickle=False) as final:
        F,D,kappa = final['F'],final['history_damage'],final['history_kappa']
        tau,V = final['tau'],final['volume']
        stress_error = stress_scale = lower_error = energy = maximum_strain = 0.
        for first in range(0,len(F),8192):
            f = F[first:first+8192]
            left,s,right = np.linalg.svd(f)
            rotation = left@right
            strain = s-1; tr = strain.sum(1)
            P = 2*mu*f+(lam*tr-2*mu)[:,None,None]*rotation
            expected_tau = (1-D[first:first+8192,None,None])*P@f.transpose(0,2,1)
            psi = (1-D[first:first+8192])*(.5*lam*tr**2+mu*np.sum(strain**2,axis=1))
            equivalent = np.maximum(np.max(lam*tr[:,None]+2*mu*strain,axis=1),0)/E
            stress_error = max(stress_error,float(np.max(np.abs(expected_tau-tau[first:first+8192]))))
            stress_scale = max(stress_scale,float(np.max(np.abs(expected_tau))))
            lower_error = max(lower_error,float(np.max(equivalent-kappa[first:first+8192])))
            maximum_strain = max(maximum_strain,float(np.max(np.linalg.norm(strain,axis=1))))
            energy += float(np.sum(V[first:first+8192]*psi))
        eps = np.finfo(float).eps
        stress_budget = 256*eps*E+1e-9*stress_scale
        energy_budget = 256*eps*E*float(V.sum())+1e-9*abs(energy)
        recorded = float(rows[-1]['elastic_energy_J'])
        checks = {'positive_orientation':bool(np.all(np.linalg.det(F)>0)),
            'undamaged_final_state':bool(np.all(D==0) and not final['history_failed'].any()),
            'maximum_tensile_history_contains_current_state':lower_error <= 256*eps,
            'history_below_damage_onset':bool(np.all(kappa <= p['tensile_strength']/E+256*eps)),
            'independent_final_kirchhoff_stress':stress_error <= stress_budget,
            'independent_final_stored_energy':np.isfinite(energy) and energy >= 0 and abs(energy-recorded) <= energy_budget}
    initial_mechanical = float(rows[0]['kinetic_energy_J'])+float(rows[0]['elastic_energy_J'])
    residual = [float(row['kinetic_energy_J'])+float(row['elastic_energy_J'])-initial_mechanical
                -float(row['total_clamp_work_J'])-float(row['total_contact_work_J']) for row in rows]
    checks = {name:bool(value) for name,value in checks.items()}
    return {'passed':all(checks.values()),'checks':checks,'production_eligible':False,
        'final_elastic_energy_J':energy,'recorded_final_elastic_energy_J':recorded,'energy_budget_J':energy_budget,
        'stress_Linf_error_Pa':stress_error,'stress_Linf_budget_Pa':stress_budget,
        'final_maximum_Biot_strain_norm':maximum_strain,'kappa_lower_bound_error':lower_error,
        'maximum_abs_dynamic_work_residual_J':max(map(abs,residual)), 'dynamic_work_residual_J':residual,
        'maximum_step_contact_force_N':float(rows[-1]['maximum_step_contact_force_N']),
        'minimum_step_contact_force_N':float(rows[-1]['minimum_step_contact_force_N']),
        'contact_exercised':max(abs(float(rows[-1][name])) for name in ('maximum_step_contact_force_N','minimum_step_contact_force_N')) > 0,
        'minimum_saved_contact_gap_m':min(float(row['contact_gap_m']) for row in rows),
        'final_contact_active':rows[-1]['contact_active']=='True',
        'final_contact_gap_m':float(rows[-1]['contact_gap_m']),
        'maximum_absolute_centre_displacement_y_m':{body:max(abs(float(row[body+'_centre_displacement_y_m'])) for row in rows) for body in ('lower','upper')},
        'scope':'Independent final F/SVD reconstruction at fixed damage. Saved K+U change minus actual fixture/contact work is reported, not assumed conserved. APIC losses, changing contact curvature and adhesive snap preclude a global conservative-contact claim. Full-time F and local contact equilibria are not independently reconstructed.'}


class ChunkWriter:
    def __init__(self, folder, limit=PART_BYTES):
        self.folder,self.limit = Path(folder),limit
        self.folder.mkdir(parents=True)
        self.parts,self.stream,self.part_hash,self.whole = [],None,None,hashlib.sha256()
        self.count,self.part_size = 0,0

    def write(self,data):
        size = len(data); self.whole.update(data); self.count += size
        while data:
            if self.stream is None:
                self.path = self.folder/f'raw.tar.gz.part-{len(self.parts):04d}'
                self.stream = self.path.open('xb'); self.part_hash=hashlib.sha256(); self.part_size=0
            take=min(len(data),self.limit-self.part_size)
            part,data=data[:take],data[take:]
            self.stream.write(part); self.part_hash.update(part); self.part_size+=take
            if self.part_size == self.limit: self.close_part()
        return size

    def flush(self):
        if self.stream: self.stream.flush()

    def close_part(self):
        if self.stream:
            self.stream.close()
            self.parts.append({'file':self.path.name,'bytes':self.part_size,'sha256':self.part_hash.hexdigest()})
            self.stream=None


def archive_complete_cache(cache, receipt, destination, receipt_sha, controller):
    records = dict(receipt['files']); records['receipt.json']=file_record(cache/'receipt.json')
    writer=ChunkWriter(destination)
    with gzip.GzipFile(filename='',mode='wb',fileobj=writer,compresslevel=1,mtime=0) as compressed:
        with tarfile.open(fileobj=compressed,mode='w|') as archive:
            for relative,record in sorted(records.items()):
                path=cache/relative
                if file_record(path) != record:
                    raise ValueError('Raw cache changed before archival: '+relative)
                info=tarfile.TarInfo('fibre-pull-off-cache/'+relative)
                info.size=record['bytes']; info.mtime=0; info.mode=0o644
                with path.open('rb') as stream: archive.addfile(info,stream)
                if file_record(path) != record:
                    raise ValueError('Raw cache changed during archival: '+relative)
    writer.close_part()
    report={'schema':'fibre-pull-off-complete-raw-archive-v1','format':'tar.gz split into ordered raw byte chunks',
        'producer':receipt['source'],'controller':controller,'cache_key':receipt['cache_key'],
        'receipt_sha256':receipt_sha,'compressed_bytes':writer.count,'sha256':writer.whole.hexdigest(),
        'parts':writer.parts,'members':{'fibre-pull-off-cache/'+k:v for k,v in records.items()},
        'all_original_raw_files_archived':True,'uncompressed_file_bytes':sum(r['bytes'] for r in records.values()),
        'consumer_backup_verified':False,
        'reconstruction':'Concatenate parts in listed order, verify the complete compressed SHA256, then verify every extracted member against members. No mmap or checkpoint file was omitted.'}
    save(destination/'archive-manifest.json',report)
    return report


def audit_worker(args):
    receipt,_,original=verify_cache(args.cache,args.receipt_sha,args.producer_root,args.producer_commit)
    if original.environment() != read(args.cache/'runtime/environment.json'):
        raise ValueError('Install the original complete runtime lock before rechecking the numerical prefix')
    report=original.audit_prefix(original.result_folder(args.cache,receipt['request']),receipt['request'],60)
    if (report.get('prefix_verified') is not True or report.get('full_protocol_completed') is not True
            or report.get('saved_states_checked') != 61
            or report['checkpoint_leaves'] != receipt['prefix_review']['checkpoint_leaves']
            or report['frame_records'] != receipt['prefix_review']['frame_records']):
        raise ValueError('Independent original-auditor complete-history/leaf check failed')
    save(args.output,report)


def finalize(args):
    cache,output=Path(args.cache).resolve(),Path(args.output).resolve()
    if output.exists() or cache == output or cache in output.parents or output in cache.parents:
        raise ValueError('Finalization needs a separate new output tree')
    receipt,binding,original=verify_cache(cache,args.receipt_sha,args.producer_root,args.producer_commit)
    commit=git(ROOT,'rev-parse','HEAD').decode().strip()
    controller=source_binding(ROOT,commit,names=['tools/finish_fibre_pull_off.py','mpm_jax/visualization.py',
        'mpm_jax/paraview_scene.py','mpm_jax/particle_fields.py','mpm_jax/export.py','tools/check_vtk.py','tools/encode_video.py',
        'tools/package_fibre_final_media.py','tools/run_woven_preview.py','.github/workflows/engineering-previews.yml'])
    raw_bytes=sum(item['bytes'] for item in receipt['files'].values())
    disk_needed=2*raw_bytes+61*receipt['request']['particles']*96+2*1024**3
    ancestor=output.parent
    while not ancestor.exists(): ancestor=ancestor.parent
    if shutil.disk_usage(ancestor).free < disk_needed:
        raise ValueError('Insufficient space for original-result copy, complete raw archive and VTK/media')
    output.mkdir(parents=True)
    logs=output/'logs'; logs.mkdir()
    original.engine.LOGS=logs
    def command(name,arguments,minutes):
        return original.engine.command(name,['timeout','--signal=TERM','--kill-after=30s',f'{minutes}m',*map(str,arguments)],monitor=True)
    command('independent-full-prefix',[sys.executable,__file__,'--audit-worker','--cache',cache,
        '--receipt-sha',args.receipt_sha,'--producer-root',args.producer_root,
        '--producer-commit',args.producer_commit,'--output',output/'prefix-review.json'],20)
    folder=original.result_folder(cache,receipt['request'])
    report=independent_material_review(folder)
    report.update(producer_source=binding,controller_source=controller,
        input_sha256={name:sha(folder/name) for name in ('final_state_0.npz','observables.csv','checkpoint.npz')})
    save(output/'material-review.json',report)
    if not report['passed']: raise ValueError('Independent final material-state review failed')
    archive=archive_complete_cache(cache,receipt,output/'raw-archive',args.receipt_sha,controller)
    rendered=output/'result'; shutil.copytree(folder,rendered)
    shim=('import sys,types,importlib; from pathlib import Path; root=Path(sys.argv.pop(1)); '
          'p=types.ModuleType("mpm_jax"); p.__path__=[str(root/"mpm_jax")]; sys.modules["mpm_jax"]=p; '
          'importlib.import_module("mpm_jax.visualization").main()')
    command('paraview-pull-off',[sys.executable,'-c',shim,ROOT,rendered,'--gif','--field','family_id',
        '--resolution','1200','--focus-bounds',*FOCUS],40)
    command('video-pull-off',[sys.executable,ROOT/'tools/encode_video.py',rendered],10)
    command('vtk-check-pull-off',[os.environ['PARAVIEW_PYTHON'],'--force-offscreen-rendering',ROOT/'tools/check_vtk.py',rendered],15)
    render,video,metadata=(read(rendered/name) for name in ('render.json','video.json','vtk_metadata.json'))
    png=sorted((rendered/'frames').glob('frame_*.png'))
    from PIL import Image
    for index,path in enumerate(png):
        if path.name != f'frame_{index:05d}.png': raise ValueError('Missing or extra ParaView image')
        with Image.open(path) as picture:
            if picture.size != (1200,1200): raise ValueError('Unexpected ParaView image size')
            picture.verify()
    animation=original.engine.gif_timing(rendered/'simulation.gif',61)
    with np.load(rendered/'trajectory.npz',allow_pickle=False) as trajectory:
        times=trajectory['time']
    pvd_times=[float(item.attrib['timestep']) for item in ET.parse(rendered/'particles.pvd').getroot().findall('.//DataSet')]
    checks={'frame_count':len(png)==metadata['frames']==render['frames_rendered']==video['frames']==61,
        'particle_count':metadata['particles']==receipt['request']['particles'],
        'actual_paraview':render['renderer']=='ParaView' and render['field']=='family_id' and render['animation']=='simulation.gif',
        'saved_output_times':np.array_equal(times,pvd_times), 'no_axes':render['axes'] is False,
        'no_annotations':render['annotations'] is False,'true_geometry':render.get('geometry_scale',1)==1,
        'fixed_detail_camera':render['camera_bounds']==FOCUS,'video_render_binding':video['render_sha256']==sha(rendered/'render.json'),
        'video_bytes':video['sha256']==sha(rendered/'simulation.mp4') and video['bytes']==(rendered/'simulation.mp4').stat().st_size,
        'original_cache_unchanged':original.payload_inventory(cache,receipt['request'])==receipt['files'] and sha(cache/'receipt.json')==args.receipt_sha}
    if not all(checks.values()): raise ValueError('Final media binding/check failed: '+str(checks))
    save(rendered/'vtk_check.json',{'passed':True,'sampled_frames':[0,60],
        'particles_per_frame':receipt['request']['particles'],'checker_sha256':controller['files']['tools/check_vtk.py']['sha256'],
        'rtol':1e-6,'atol':1e-7,'scope':'Actual ParaView first/last coordinate, scalar and integer-label readback.'})
    save(rendered/'media-check.json',{'passed':True,'checks':checks,'gif':animation,
        'png_sha256':{path.name:sha(path) for path in png},'geometry_scale':1,'axes':False,'annotations':False})
    files={}
    for p in sorted(rendered.rglob('*')):
        if p.is_file() and (p.parent.name=='frames' or p.suffix in ('.json','.png','.gif','.mp4','.csv','.pvd')):
            files[p.relative_to(output).as_posix()]=file_record(p)
    for name in ('prefix-review.json','material-review.json','raw-archive/archive-manifest.json'):
        files[name]=file_record(output/name)
    save(output/'delivery-manifest.json',{'schema':'fibre-pull-off-finalization-v1','passed':True,'checks':checks,
        'producer':receipt['source'],'controller':controller,'original_receipt_sha256':args.receipt_sha,
        'files':files,'complete_raw_archive_manifest':'raw-archive/archive-manifest.json',
        'complete_raw_archive_sha256':archive['sha256'],'production_eligible':False,'public_visual_review_passed':False,
        'consumer_backup_verified':False,'view_scope':'Shared central24mm detail view at true scale; rod ends may be outside the view, while complete points remain in raw data/VTK.'})
    print(json.dumps({'completed':True,'output':str(output),'raw_parts':len(archive['parts']),
                      'raw_archive_sha256':archive['sha256'],'public_visual_review_passed':False}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--receipt-sha',required=True)
    parser.add_argument('--producer-root',type=Path,required=True)
    parser.add_argument('--producer-commit',default=PRODUCER)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--audit-worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not re.fullmatch('[0-9a-f]{64}',args.receipt_sha): parser.error('A complete original receipt SHA256 is required')
    if args.audit_worker: audit_worker(args)
    else: finalize(args)


if __name__=='__main__': main()
