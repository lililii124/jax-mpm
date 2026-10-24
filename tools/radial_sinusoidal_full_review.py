"""Independent full-raw radial-contact review and matched read-only ParaView media."""
import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
PRODUCER=ROOT/'.local/radial-immutable-producer'
NUMERICAL=ROOT/'.local/radial-numerical-controller'
WORK=PRODUCER/'.local/radial-sinusoidal-full'
RAW=WORK/'result'
OUT=ROOT/'.local/radial-full-review-media'
PROOF=ROOT/'.local/radial-independent-proof/independent-admission.json'
API=ROOT/'.local/radial-full-run-identities.json'
PRODUCER_SHA='de2bfea446e66e6245e33329a663996c5d5b9e84'
NUMERICAL_SHA='93becdf685107c35193b610af231915a58249c52'
RUN='40094841691'
ATTEMPT='1'
RECEIPT_SHA='5719e19eae3efe4cfc9a9a0700a4f68f2c2df6271582f35e039b36f262778219'
CACHE_KEY='radial-dual-continuation-v1-93becdf685107c35193b610af231915a58249c52-40094841691-1'
MANIFEST=ROOT/'.local/radial-full-transport/transport.json'
MANIFEST_SHA='3b705b435eb847e6aa65c4f52d610a6cc00817e0943850c47885be5b5163b92d'
PREFIX_RECEIPT='d0af82b261b0433c024b80472cb5da67ee5d0113f11d2876f0510ee8f631c591'
PROOF_SHA='74840b9e9bf12f479787bdfb359bd34098354402fdcf3c13deccee698043836f'
TEMPLATE='.github/workflows/radial-sinusoidal-full-review-media.yml'
CONTROLLER='tools/radial_sinusoidal_full_review.py'
REGISTERED='.github/workflows/engineering-previews.yml'
REF='refs/heads/work/radial-sinusoidal-full'
REPOSITORY='lililii124/jax-mpm'
FOCUS=[.22,.78,.13,.82,.22,.78]
COLOR_RANGE=[0.,.07772661828919937]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')


def inventory(folder):
    files={}
    for path in sorted(Path(folder).rglob('*')):
        if path.is_symlink():raise ValueError('Evidence symlink rejected')
        if path.is_file():files[path.relative_to(folder).as_posix()]={'bytes':path.stat().st_size,'sha256':digest(path)}
    return files


def git(root,*args):return subprocess.check_output(['git',*args],cwd=root)


def head(root):return git(root,'rev-parse','HEAD').decode().strip()


def private_workflow_bytes(root=None):
    root=ROOT if root is None else Path(root)
    first,rest=(root/TEMPLATE).read_bytes().split(b'\n',1)
    ending=b'\r\n' if first.endswith(b'\r') else b'\n'
    if first.rstrip(b'\r')!=b'name: feature radial sinusoidal full review media':raise ValueError('Unexpected review template')
    return b'name: private radial sinusoidal full review media'+ending+rest


def controller_files():
    names=[CONTROLLER,TEMPLATE,REGISTERED]
    names += [f'docs/results/adhesion-visible--{name}-pull-off/{file}'
        for name in ('nonadhesive','adhesive') for file in ('summary.json','render.json')]
    names += [f'docs/images/home/adhesion-visible--{name}-pull-off.gif' for name in ('nonadhesive','adhesive')]
    return {name:digest(ROOT/name) for name in names}


def authenticate():
    commit=head(ROOT)
    if os.environ.get('GITHUB_ACTIONS')=='true':
        allowed={f'{REPOSITORY}/{path}@{REF}' for path in (REGISTERED,TEMPLATE)}
        if not (os.environ.get('GITHUB_REF')==REF and os.environ.get('GITHUB_REPOSITORY')==REPOSITORY
            and os.environ.get('GITHUB_WORKFLOW_REF') in allowed and
            commit==os.environ.get('GITHUB_SHA')==os.environ.get('GITHUB_WORKFLOW_SHA')):
            raise ValueError('Actual read-only controller/workflow identity differs')
        if os.environ['GITHUB_WORKFLOW_REF']==f'{REPOSITORY}/{REGISTERED}@{REF}' and (ROOT/REGISTERED).read_bytes()!=private_workflow_bytes():
            raise ValueError('Actual registered review workflow bytes differ')
    files=controller_files()
    for name in files:
        if (ROOT/name).read_bytes()!=git(ROOT,'show',f'HEAD:{name}'):raise ValueError('Review source differs from committed HEAD')
    if head(PRODUCER)!=PRODUCER_SHA or head(NUMERICAL)!=NUMERICAL_SHA:raise ValueError('Immutable checkout commit differs')
    if digest(PROOF)!=PROOF_SHA or digest(WORK/'continuation-receipt.json')!=RECEIPT_SHA:
        raise ValueError('Exact prefix proof/full numerical receipt differs')
    proof=read(PROOF);receipt=read(WORK/'continuation-receipt.json')
    if digest(MANIFEST)!=MANIFEST_SHA:raise ValueError('Exact complete raw transport manifest differs')
    transport=read(MANIFEST)
    if transport['status']!='complete-raw-packaged' or inventory(WORK)!=transport['files']:
        raise ValueError('Whole restored numerical evidence differs from complete transport')
    if {name.removeprefix('result/'):value for name,value in transport['files'].items() if name.startswith('result/')}!=receipt['raw_files']:
        raise ValueError('Full raw receipt and transport file lists differ')
    if os.environ.get('GITHUB_ACTIONS')=='true' and (os.environ.get('CACHE_HIT')!='true' or os.environ.get('CACHE_MATCH')!=CACHE_KEY):
        raise ValueError('Exact full raw cache restoration differs')
    if not (receipt['status']=='complete' and receipt['controller_commit']==NUMERICAL_SHA
        and receipt['producer_commit']==PRODUCER_SHA and str(receipt['run_id'])==RUN
        and str(receipt['run_attempt'])==ATTEMPT and receipt['cache_key']==CACHE_KEY
        and receipt['parent_receipt_sha256']==PREFIX_RECEIPT
        and receipt['independent_proof_sha256']==PROOF_SHA and receipt['parent_raw_unchanged'] is True
        and receipt['producer_source_files']==proof['source_files'] and receipt['runtime']==proof['runtime']):
        raise ValueError('Completed immutable dual-producer provenance differs')
    for root,source in ((NUMERICAL,receipt['controller_files']),(PRODUCER,proof['source_files'])):
        for name,expected in source.items():
            if digest(root/name)!=expected or (root/name).read_bytes()!=git(root,'show',f'HEAD:{name}'):
                raise ValueError('Frozen numerical/controller source bytes differ: '+name)
    if {name:importlib.metadata.version(name) for name in proof['runtime']}!=proof['runtime']:
        raise ValueError('Exact numerical/renderer runtime differs')
    identities=read(API)
    if not (str(identities['run_id'])==RUN and identities['head_sha']==NUMERICAL_SHA
        and identities['conclusion']=='success' and identities['workflow_path']==REGISTERED):
        raise ValueError('Actual full numerical run identity differs')
    prefix=f'radial-cont-raw-{RUN}-{ATTEMPT}-'
    if {value['name'] for value in identities['artifacts']}!={prefix+'manifest'}|{prefix+Path(name).stem for name in transport['parts']}:
        raise ValueError('Exact complete raw artifact identity set differs')
    if digest(WORK/'receipt.json')!=PREFIX_RECEIPT:
        raise ValueError('Complete raw or original prefix receipt differs')
    for name,expected in proof['all_raw_files'].items():
        if name.startswith('frames_data/') and digest(RAW/name)!=expected['sha256']:
            raise ValueError('Admitted original prefix frame changed')
    numeric_audit=read(WORK/'full-numerical-audit.json')
    if not (numeric_audit['passed'] and numeric_audit['saved_states']==51 and numeric_audit['steps']==3700
        and numeric_audit['particles']==367744 and numeric_audit['all_checkpoint_history_checked'] and numeric_audit['prefix_frames_unchanged']):
        raise ValueError('Original complete numerical audit differs')
    return {'controller_commit':commit,'controller_files':files,'numerical_receipt':receipt,
        'actual_numerical_run':identities,'runtime':proof['runtime'],'proof':proof}


def old_helpers():
    sys.path.insert(0,str(PRODUCER));sys.path.insert(0,str(PRODUCER/'tools'))
    spec=importlib.util.spec_from_file_location('immutable_radial_review_helpers',PRODUCER/'tools/radial_sinusoidal_cloud.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def rows(path):
    with Path(path).open(newline='') as stream:
        return [{name:None if value in ('','None') else float(value) for name,value in row.items()} for row in csv.DictReader(stream)]


def motion(path,time):
    import numpy as np
    times=np.asarray(path['times']);centers=np.asarray(path['centers']);duration=np.diff(times)
    phase=np.clip((time-times[:-1])/duration,0,1)
    rate=np.pi/(2*duration)*np.sin(np.pi*phase)
    rate[(time<=times[:-1])|(time>=times[1:])]=0
    change=np.diff(centers,axis=0)
    return centers[0]+((1-np.cos(np.pi*phase))[:,None]/2*change).sum(0),(rate[:,None]*change).sum(0)


def gap_gradient(points,center,radius,profile):
    import numpy as np
    radial=points-center;rho=np.linalg.norm(radial[:,(0,2)],axis=1)
    k=2*np.pi/profile['wavelength_m'];amplitude=profile['amplitude_m']
    height=amplitude*(np.cos(k*rho)-1)
    warped=radial.copy();warped[:,1]-=height
    distance=np.linalg.norm(warped,axis=1)
    if np.any(distance==0):raise ValueError('Surface point reached warped sphere origin')
    unit=warped/distance[:,None]
    factor=-amplitude*k*k*np.sinc(k*rho/np.pi)
    gradient=unit.copy()
    gradient[:,0]-=unit[:,1]*factor*radial[:,0]
    gradient[:,2]-=unit[:,1]*factor*radial[:,2]
    return distance-radius,gradient


def interaction(points,area,center,radius,law,profile):
    import numpy as np
    gap,gradient=gap_gradient(points,center,radius,profile)
    work=law['work_of_adhesion'];reach=law['interaction_range'];stiffness=law['penalty_stiffness']
    compressed=gap<0;adhesive=(gap>=0)&(gap<reach)&(work>0)
    traction=np.where(compressed,-stiffness*gap,np.where(adhesive,-work/reach,0))
    force=(area*traction)[:,None]*gradient
    penalty=float((area*.5*stiffness*np.minimum(gap,0)**2).sum())
    attraction=float((area*work*(np.clip(gap/reach,0,1)-1)).sum())
    return {'forces':force,'reaction':-force.sum(0),'moment':-np.cross(points-center,force).sum(0),
        'compressed_points':int(compressed.sum()),'adhesive_points':int(adhesive.sum()),
        'contact_reference_area_m2':int(compressed.sum())*area,'adhesive_reference_area_m2':int(adhesive.sum())*area,
        'min_gap_m':float(gap.min()),'max_penetration_m':float(max(0,-gap.min())),
        'potential_J':penalty+attraction,'penalty_potential_J':penalty,'adhesive_potential_J':attraction}


def audit_contact():
    import numpy as np
    summary=read(RAW/'summary.json');status=read(RAW/'status.json')
    definition=summary['definition'];profile=definition['radial_sinusoid'];law=definition['adhesion']
    path=definition['rigid_spheres'][0];config=summary['configs'][0]
    observed=rows(RAW/'observables.csv');diagnostics=rows(RAW/'diagnostics.csv')
    if not (summary['completed'] and summary['particles']==[367744] and summary['steps']==3700
        and abs(summary['simulation_time']-1.3)<1e-12 and status['completed']
        and len(observed)==len(diagnostics)==51 and profile['amplitude_m']==.008
        and profile['wavelength_m']==.08 and profile['axis']=='y'):
        raise ValueError('Complete51-state radial protocol metadata differs')
    with np.load(RAW/'frames_data/frame_00000.npz',allow_pickle=False) as initial:reference=initial['x']
    surface=np.flatnonzero(reference[:,1]==reference[:,1].max());area=definition['reference_area_per_surface_point_m2']
    checks={};errors={};history=[]
    def compare(name,actual,expected):
        error=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
        errors[name]=max(errors.get(name,0),error)
        checks[name]=checks.get(name,True) and bool(np.allclose(actual,expected,rtol=2e-8,atol=2e-8))
    baseline_checks={};baseline_media={}
    for scene in ('nonadhesive','adhesive'):
        base=read(ROOT/f'docs/results/adhesion-visible--{scene}-pull-off/summary.json')
        fixed=base['definition']
        from PIL import Image
        gif_path=ROOT/f'docs/images/home/adhesion-visible--{scene}-pull-off.gif'
        with Image.open(gif_path) as gif:
            if gif.size!=(400,400) or gif.n_frames!=51:raise ValueError('Existing baseline GIF dimensions/states differ')
        baseline_media[scene]={'resolution':[400,400],'frames':51,'sha256':digest(gif_path)}
        baseline_checks[scene]=bool(base['steps']==3700 and base['particles']==[367744]
            and base['simulation_time']==1.3 and base['configs']==summary['configs']
            and base['materials']==summary['materials'] and fixed['rigid_spheres']==definition['rigid_spheres']
            and all(fixed[key]==definition[key] for key in ('lower','upper','particle_counts_per_axis',
                'particle_spacings_m','reference_area_per_surface_point_m2','young_modulus_Pa','poisson_ratio','density_kg_per_m3'))
            and fixed['adhesion']['interaction_range']==law['interaction_range']
            and fixed['adhesion']['penalty_stiffness']==law['penalty_stiffness'])
    for i in range(51):
        time=i*1.3/50;row=observed[i]
        with np.load(RAW/'frames_data'/f'frame_{i:05d}.npz',allow_pickle=False) as state:
            points=state['x']
            if points.shape!=(367744,3) or any(not np.isfinite(state[key]).all() for key in state.files):raise ValueError('Invalid actual saved frame')
            compare('saved_time',float(state['time']),time)
        center,velocity=motion(path,time)
        load=interaction(points[surface],area,center,path['radius'],law,profile)
        initial_load=interaction(reference[surface],area,center,path['radius'],law,profile)
        gap,_=gap_gradient(points,center,path['radius'],profile)
        load['max_bulk_penetration_m']=float(max(0,-gap.min()))
        load['actuator_power_W']=float(-load['reaction']@velocity)
        prefix='sphere_0_'
        for axis,name in enumerate('xyz'):
            for suffix,value in ((f'center_{name}_m',center[axis]),(f'velocity_{name}_m_per_s',velocity[axis]),
                (f'reaction_{name}_N',load['reaction'][axis]),(f'reaction_moment_{name}_Nm',load['moment'][axis]),
                (f'undeformed_surface_reaction_{name}_N',initial_load['reaction'][axis])):
                compare(prefix+suffix,row[prefix+suffix],value)
            compare('sphere_total_reaction_'+name+'_N',row['sphere_total_reaction_'+name+'_N'],load['reaction'][axis])
        for key in ('compressed_points','adhesive_points','contact_reference_area_m2','adhesive_reference_area_m2',
                    'min_gap_m','max_penetration_m','max_bulk_penetration_m','potential_J','actuator_power_W'):
            compare(prefix+key,row[prefix+key],load[key])
        compare('contact_potential_J',row['contact_potential_J'],load['potential_J'])
        for key in ('penalty_potential_J','adhesive_potential_J','actuator_power_W'):compare(key,row[key],load[key])
        compare('deformation_feedback',row[prefix+'deformation_feedback_force_difference_N'],np.linalg.norm(load['reaction']-initial_load['reaction']))
        uplift=points[surface,1]-reference[surface,1]
        compare('surface_displacement_y_min_m',row['surface_displacement_y_min_m'],uplift.min())
        compare('surface_displacement_y_max_m',row['surface_displacement_y_max_m'],uplift.max())
        compare('observation_time',row['time'],time);compare('diagnostic_time',diagnostics[i]['time'],time)
        if diagnostics[i]['min_J']<=0 or any(value is not None and not np.isfinite(value) for value in row.values()):raise ValueError('Invalid diagnostics or observables')
        origin=np.cross(points[surface],load['forces']).sum(0)+np.cross(center,load['reaction'])+load['moment']
        compare('moment_reciprocity',np.linalg.norm(origin),0)
        force_scale=max(1,float(np.linalg.norm(load['reaction'])))
        if row['force_balance_error_N']>1e-9*force_scale:raise ValueError('Force reciprocity failed')
        history.append({'frame':i,'time_s':time,'normal_reaction_N':float(load['reaction'][1]),
            'compressed_points':load['compressed_points'],'adhesive_points':load['adhesive_points'],
            'minimum_gap_m':load['min_gap_m'],'max_bulk_penetration_m':load['max_bulk_penetration_m'],'max_surface_penetration_m':load['max_penetration_m'],
            'surface_uplift_m':float(uplift.max()),'surface_lowering_m':float(uplift.min()),
            'cohesive_potential_J':load['adhesive_potential_J'],'rigid_vertical_velocity_m_per_s':float(velocity[1])})
    with np.load(RAW/'final_state_0.npz',allow_pickle=False) as final:
        valid=bool(final['valid'] and final['history_valid'].all())
        positive=bool((final['mass']>0).all() and (final['volume']>0).all() and (np.linalg.det(final['F'])>0).all())
        if not valid or not positive or any(not np.isfinite(final[key]).all() for key in final.files):raise ValueError('Final numerical state invalid')
    required={'all_independent_observables_match':all(checks.values()),'same_baseline_physical_protocol':all(baseline_checks.values()),
        'actual_compression':any(row['compressed_points']>0 for row in history),
        'actual_tensile_adhesion_during_retraction':any(row['normal_reaction_N']< -1e-8 and row['rigid_vertical_velocity_m_per_s']>0 for row in history),
        'fully_separated_at_end':history[-1]['minimum_gap_m']>law['interaction_range'] and history[-1]['compressed_points']==history[-1]['adhesive_points']==0,
        'positive_J_all_saved_diagnostics':min(row['min_J'] for row in diagnostics)>0,
        'finite_valid_final_state':valid and positive,
        'limited_bulk_gap_penetration':max(row['max_bulk_penetration_m'] for row in history)<=min(config['domain_size']/config['n_grid'],law['interaction_range'])/4,
        'no_deeper_unsampled_contact':max(row['max_bulk_penetration_m']-row['max_surface_penetration_m'] for row in history)<=.05*definition['particle_spacings_m'][1]}

    result={'schema':'radial-full-independent-contact-review-v1','passed':all(required.values()),
        'checks':required,'observable_checks':checks,'maximum_absolute_errors':errors,'baseline_protocol_checks':baseline_checks,'baseline_media':baseline_media,
        'frames_checked':51,'particles_per_frame':367744,'steps':3700,'duration_s':1.3,
        'minimum_J':min(row['min_J'] for row in diagnostics),'maximum_bulk_gap_penetration_m':max(row['max_bulk_penetration_m'] for row in history),
        'maximum_surface_uplift_m':max(row['surface_uplift_m'] for row in history),
        'most_tensile_reaction_N':min(row['normal_reaction_N'] for row in history),'history':history,
        'geometry':profile,'gap_scope':'warped-coordinate sphere gap; not Euclidean signed distance',
        'force_scope':'reference-area Dugdale traction multiplied by full gap gradient; normal follows actual concentric surface',
        'comparison_scope':'same v4 normal-loading protocol and adhesion; rigid radial sinusoidal geometry changes; no performance improvement claimed',
        'solver_executed':False,'public_eligible':False,'visual_review_pending':True}
    save(OUT/'full-contact-review.json',result)
    if not result['passed']:raise ValueError('Independent full-contact review rejected one or more checks')
    return result


def encode_gif(folder):
    from PIL import Image
    atlas=Image.new('RGB',(8*96,7*96),(246,249,251))
    for i in range(51):
        with Image.open(folder/'frames'/f'frame_{i:05d}.png') as frame:
            thumb=frame.convert('RGB');thumb.thumbnail((96,96));atlas.paste(thumb,((i%8)*96,(i//8)*96));thumb.close()
    palette=atlas.quantize(colors=256);atlas.close();header=None
    with (folder/'simulation.gif').open('wb') as destination:
        for i in range(51):
            with Image.open(folder/'frames'/f'frame_{i:05d}.png') as image:
                image=image.convert('RGB').resize((400,400),Image.Resampling.LANCZOS).quantize(palette=palette)
                stream=io.BytesIO();image.save(stream,format='GIF',duration=60,optimize=False,disposal=2,**({'loop':0} if i==0 else {}))
                payload=stream.getvalue();length=13+(3*2**((payload[10]&7)+1) if payload[10]&128 else 0)
                if i==0:header=payload[:length];destination.write(header)
                elif payload[:length]!=header:raise ValueError('GIF fixed history-wide palette changed')
                destination.write(payload[length:-1]);image.close()
        destination.write(b';')
    palette.close()
    with Image.open(folder/'simulation.gif') as image:
        if image.n_frames!=51 or image.size!=(400,400):raise ValueError('Exact51-frame400px GIF differs')
        for i in range(51):
            image.seek(i);image.load()
            if image.info.get('duration')!=60:raise ValueError('GIF timing differs')
    if (folder/'simulation.gif').stat().st_size>=9_000_000:raise ValueError('Public GIF is not below9MB')


def render_media():
    old=old_helpers();render=PRODUCER/'.local/radial-independent-render'
    if render.exists():raise ValueError('Fresh read-only render copy required')
    shutil.copytree(RAW,render)
    command=[sys.executable,'-m','mpm_jax.visualization',str(render),'--gif','--field','displacement',
        '--resolution','1200','--color-range',*[str(x) for x in COLOR_RANGE],'--focus-bounds',*[str(x) for x in FOCUS]]
    old.guard(command,OUT/'render.log',75*60,render)
    subprocess.run([os.environ['PARAVIEW_PYTHON'],'--force-offscreen-rendering',str(Path(__file__).resolve()),'vtk',
        '--folder',str(render)],cwd=PRODUCER,check=True,timeout=600)
    encode_gif(render)
    record=read(render/'render.json')
    if not (record['renderer']=='ParaView' and record['frames_rendered']==51 and record['particles']==367744
        and record['axes'] is False and record['annotations'] is False and record['geometry_scale']==1
        and record['field']=='displacement' and record['color_range']==COLOR_RANGE and record['camera_bounds']==FOCUS):
        raise ValueError('Matched actual ParaView rendering evidence differs')
    geometry=[value for value in record['auxiliary_geometry'] if value['type']=='prescribed_radial_sinusoidal_rigid_sphere']
    if len(geometry)!=1 or geometry[0]['radial_sinusoid']!=read(RAW/'summary.json')['definition']['radial_sinusoid']:
        raise ValueError('Actual visible corrugated indenter differs')
    from PIL import Image
    for i in range(51):
        with Image.open(render/'frames'/f'frame_{i:05d}.png') as image:
            image.load()
            if image.size!=(1200,1200) or all(lo==hi for lo,hi in image.convert('RGB').getextrema()):raise ValueError('PNG missing/blank')
    for name in ('simulation.gif','render.json','summary.json','vtk-full-mesh-audit.json'):shutil.copyfile(render/name,OUT/name)
    for i in (10,16,30,43,50):shutil.copyfile(render/'frames'/f'frame_{i:05d}.png',OUT/f'frame_{i:05d}.png')
    for name,expected in read(WORK/'continuation-receipt.json')['raw_files'].items():
        if digest(render/name)!=expected['sha256']:raise ValueError('Numerical render-copy bytes changed')
    save(OUT/'media-audit.json',{'passed':True,'gif_frames':51,'gif_resolution':[400,400],
        'gif_bytes':(OUT/'simulation.gif').stat().st_size,'axes':False,'text':False,
        'same_camera_and_displacement_color_scale_as_original_pair':True,
        'actual_radial_mesh_checked':True,'solver_executed':False,'visual_review_pending':True})


def vtk_mesh_audit(folder):
    import numpy as np
    from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
    from vtkmodules.util.numpy_support import vtk_to_numpy
    for i in range(51):
        reader=vtkXMLPolyDataReader();reader.SetFileName(str(folder/'vtk'/f'particles_{i:05d}.vtp'));reader.Update()
        data=reader.GetOutput()
        if data.GetNumberOfPoints()!=367744:raise ValueError('Actual exported VTK particle count differs')
        with np.load(folder/'frames_data'/f'frame_{i:05d}.npz',allow_pickle=False) as raw:
            np.testing.assert_allclose(vtk_to_numpy(data.GetPoints().GetData()),raw['x'],rtol=1e-6,atol=1e-7)
    spec=importlib.util.spec_from_file_location('immutable_radial_scene',PRODUCER/'mpm_jax/paraview_scene.py')
    scene=importlib.util.module_from_spec(spec);spec.loader.exec_module(scene)
    from paraview.simple import Sphere,Delete
    from paraview import servermanager
    summary=read(folder/'summary.json');profile=summary['definition']['radial_sinusoid'];path=summary['definition']['rigid_spheres'][0]
    sphere=Sphere();sphere.Radius=path['radius'];sphere.ThetaResolution=256;sphere.PhiResolution=192
    warped=scene.radial_sinusoidal_mesh(sphere,profile);evidence=[]
    for index in (0,10,16,30,43,50):
        center,_=motion(path,index*1.3/50);sphere.Center=center;warped.UpdatePipeline()
        points=vtk_to_numpy(servermanager.Fetch(warped).GetPoints().GetData())
        gap,_=gap_gradient(points,center,path['radius'],profile)
        maximum=float(np.max(np.abs(gap)))
        if maximum>=1e-7:raise ValueError('Actual rendered rigid mesh does not match force-gap zero set')
        evidence.append({'frame':index,'mesh_vertices':len(points),'maximum_gap_zero_set_error_m':maximum})
    Delete(warped);Delete(sphere)
    save(folder/'vtk-full-mesh-audit.json',{'passed':True,'actual_particle_frames_checked':51,
        'actual_warped_rigid_mesh_checks':evidence,'geometry':'concentric cosine corrugation centered on moving y axis'})


def run(media=False):
    if OUT.exists():raise ValueError('Fresh independent review destination required')
    OUT.mkdir(parents=True)
    save(OUT/'progress.json',{'phase':'admission','numerical_run':RUN,'solver_executed':False})
    try:
        provenance=authenticate();sealed=inventory(WORK)
        save(OUT/'admission.json',{'controller_commit':provenance['controller_commit'],
            'controller_files':provenance['controller_files'],'producer_commit':PRODUCER_SHA,
            'numerical_controller_commit':NUMERICAL_SHA,'numerical_run':RUN,'numerical_receipt_sha256':RECEIPT_SHA,
            'runtime':provenance['runtime'],'actual_numerical_run':provenance['actual_numerical_run'],
            'complete_raw_files':provenance['numerical_receipt']['raw_files'],'transport_sha256':MANIFEST_SHA,'solver_executed':False})
        old_helpers().guard([sys.executable,str(Path(__file__).resolve()),'contact'],
            OUT/'contact-review.log',600,WORK)
        review=read(OUT/'full-contact-review.json')
        if not review['passed']:raise ValueError('Independent contact review did not pass')
        if media:render_media()
        if inventory(WORK)!=sealed:raise ValueError('Sealed full numerical evidence mutated during read-only review/media')
        save(OUT/'read-only-receipt.json',{'passed':True,'raw_unchanged':True,'solver_executed':False,
            'full_review_passed':review['passed'],'media_rendered':media,'numerical_receipt_sha256':RECEIPT_SHA,
            'controller_commit':provenance['controller_commit'],'producer_commit':PRODUCER_SHA})
        save(OUT/'progress.json',{'phase':'complete','numerical_run':RUN,'solver_executed':False})
        save(OUT/'delivery.json',inventory(OUT))
        if sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())>100_000_000:raise ValueError('Independent delivery exceeds100MB')
    except BaseException as error:
        save(OUT/'review-failure.json',{'status':'read-only-review-rejected','error':f'{type(error).__name__}: {error}',
            'numerical_run':RUN,'solver_executed':False,'public_eligible':False})
        raise


def install():
    authenticate();old_helpers().install()


def verify_delivery(folder):
    actual=inventory(folder);actual.pop('delivery.json',None)
    if actual!=read(folder/'delivery.json'):raise ValueError('Same-run retrieved independent delivery differs')
    save(ROOT/'.local/radial-full-review-retrieval.json',{'passed':True,'artifact_id':os.environ.get('MEDIA_ARTIFACT_ID'),
        'files':actual,'numerical_run':RUN})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['audit','media','vtk','install','verify-delivery','contact'])
    parser.add_argument('--folder',type=Path)
    args=parser.parse_args()
    if args.mode=='audit':run()
    elif args.mode=='media':run(media=True)
    elif args.mode=='vtk':vtk_mesh_audit(args.folder)
    elif args.mode=='install':install()
    elif args.mode=='verify-delivery':verify_delivery(args.folder)
    elif args.mode=='contact':audit_contact()


if __name__=='__main__':main()
