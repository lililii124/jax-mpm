"""Render public cutting output with ParaView and Pillow.

Example: pvpython --force-offscreen-rendering tools/render_cutting_paraview.py
  --input outputs/cutting-three-pass --output outputs/cutting-three-pass-gifs

Use --validate-only with ordinary Python to inspect input without ParaView.
Saved constitutive plastic_rate is distinct from a saved-frame PEEQ increment.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

FIELDS = {'plastic_strain':'equivalent-plastic-strain',
          'temperature_K':'temperature','damage':'damage','plastic_rate':'constitutive-plastic-strain-rate'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fail(message):
    raise ValueError(message)

def load_inputs(args):
    folder=args.input.resolve()
    case_path=folder/'case.json';checkpoint_path=folder/'checkpoint.json'
    case=json.loads(case_path.read_text());checkpoint=json.loads(checkpoint_path.read_text())
    observations=checkpoint.get('observations',[])
    legacy={}
    # Public single-pass runtime explicitly saves one observation per frame.
    if len(observations)==int(checkpoint['frame'])+1:
        for index,row in enumerate(observations):
            if all(key in row for key in ('tool_tip_x_m','tool_tip_y_m')):
                stamp=row.get('time_s',row.get('time'))
                if stamp is not None:legacy[index]=(float(stamp),[row['tool_tip_x_m'],row['tool_tip_y_m']])
    records=[]
    for path in sorted(folder.glob('fields_*.npz')):
        frame=int(path.stem.split('_')[1])
        if frame>checkpoint['frame']:continue
        with np.load(path,allow_pickle=False) as archive:
            if 'time' in archive.files and 'virtual_tip' in archive.files:
                stamp=float(archive['time']);tip=np.asarray(archive['virtual_tip'],dtype=float)
                pose_source='saved time/virtual_tip arrays'
            elif frame in legacy:
                stamp,tip=legacy[frame];tip=np.asarray(tip,dtype=float)
                pose_source='saved public checkpoint observation indexed by frame'
            else:fail(f'{path}: missing actual saved time/virtual_tip and matching checkpoint observation')
        if tip.shape!=(2,) or not np.isfinite(tip).all() or not np.isfinite(stamp):fail(f'{path}: invalid actual time/pose')
        records.append({'frame':frame,'time_s':stamp,'virtual_tip_m':tip.tolist(),
                        'path':str(path),'pose_source':pose_source})
    if not records:fail('no accepted fields_*.npz archives')
    times=np.array([r['time_s'] for r in records])
    if not np.all(np.diff(times)>0):fail('saved physical times must be strictly increasing')
    start=times[0] if args.start is None else args.start
    end=times[-1] if args.end is None else args.end
    eligible=np.flatnonzero((times>=start)&(times<=end))
    if len(eligible)<2:fail('requested interval must contain at least two actual saved states')
    targets=np.linspace(times[eligible[0]],times[eligible[-1]],min(args.frames,len(eligible)))
    selected={int(eligible[np.argmin(abs(times[eligible]-t))]) for t in targets}
    # Preserve sampled states next to every actual prescribed velocity event.
    for knot in case.get('multipass',{}).get('path',{}).get('knots',[]):
        if start<=knot[0]<=end:selected.add(int(eligible[np.argmin(abs(times[eligible]-knot[0]))]))
    frames=[]
    for index in sorted(selected):
        record=records[index];path=Path(record['path'])
        with np.load(path,allow_pickle=False) as archive:
            required={'x','particle_id','failed',*FIELDS}
            if args.geometry=='affine':required|={'F','reference_volume'}
            missing=required-set(archive.files)
            if missing:fail(f'{path}: required saved fields missing: {sorted(missing)}')
            data={key:np.asarray(archive[key]) for key in required}
        count=len(data['x'])
        if data['x'].shape!=(count,2):fail(f'{path}: expected x shape(N,2)')
        if len(np.unique(data['particle_id']))!=count:fail(f'{path}: duplicate particle IDs')
        for key,array in data.items():
            if array.shape[0]!=count or not np.isfinite(array).all():fail(f'{path}: nonfinite/misaligned {key}')
        for key in (*FIELDS,'failed','particle_id'):
            if data[key].shape!=(count,):fail(f'{path}: {key} must be one scalar per particle')
        if not np.isin(data['failed'],[0,1]).all():fail(f'{path}: failed must be boolean')
        active=~data['failed'].astype(bool)
        if not active.any():fail(f'{path}: no active particles to display')
        if np.any(data['damage']<0) or np.any(data['damage']>1+1e-12):fail(f'{path}: damage outside [0,1]')
        data.update(record=record,active=active,source_sha256=sha(path))
        if args.geometry=='affine':
            if data['F'].shape!=(count,3,3) or data['reference_volume'].shape!=(count,):fail(f'{path}: invalid saved F/volume shapes')
            if np.any(data['reference_volume']<=0) or np.any(np.linalg.det(data['F'][active])<=0):fail(f'{path}: nonpositive saved volume/J')
            # Reproduce the documented public block's actual reference spacing;
            # ceil(counts) can make the two spacings slightly different.
            g=case['geometry'];n=case['numerics']
            try:
                extent=np.asarray(g['workpiece_upper_m'])-np.asarray(g['workpiece_lower_m'])
                counts=np.maximum(1,np.ceil(extent/n['dx_m']*n['particles_per_cell_per_axis']).astype(int))
                spacing=extent/counts
            except KeyError as error:fail(f'affine view needs saved block geometry/numerics: {error}')
            if np.prod(counts)!=count or not np.allclose(data['reference_volume'],np.prod(spacing),rtol=1e-12,atol=0):
                fail(f'{path}: saved reference volumes do not match public block spacing')
            reference=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])[None,:,:]*.5*spacing[None,None,:]
            data['corners']=data['x'][:,None,:]+np.einsum('pij,pkj->pki',data['F'][:,:2,:2],reference)
            q=data['corners'];area=.5*np.sum(q[:,:,0]*np.roll(q[:,:,1],-1,axis=1)-q[:,:,1]*np.roll(q[:,:,0],-1,axis=1),axis=1)
            if not np.allclose(area[active],data['reference_volume'][active]*np.linalg.det(data['F'][active,:2,:2]),rtol=1e-8,atol=1e-24):
                fail(f'{path}: affine domain areas do not match saved F/reference volume')
        frames.append(data)
    geometry=case['geometry']
    for key in ('rake_angle_deg','clearance_angle_deg','tool_tip_radius_m','tool_tip_arc_segments'):
        if key not in geometry:fail(f'case.json missing tool geometry: {key}')
    ranges={key:[float(min(frame[key][frame['active']].min() for frame in frames)),
                  float(max(frame[key][frame['active']].max() for frame in frames))] for key in FIELDS}
    ranges['plastic_strain'][0]=0.;ranges['plastic_rate'][0]=0.;ranges['damage']=[0.,1.]
    if args.ranges:
        overrides=json.loads(args.ranges.read_text())
        if set(overrides)-set(FIELDS):fail('range keys must be saved field names: '+', '.join(FIELDS))
        ranges.update(overrides)
    for key,bounds in ranges.items():
        if len(bounds)!=2 or not np.isfinite(bounds).all() or bounds[1]<bounds[0]:fail('invalid range: '+key)
    return case,checkpoint,frames,ranges,{'case_sha256':sha(case_path),'checkpoint_sha256':sha(checkpoint_path)}

def tool_polygon(tip,geometry,length):
    rake=np.deg2rad(geometry['rake_angle_deg']);clearance=np.deg2rad(geometry['clearance_angle_deg'])
    rd=np.array([-np.sin(rake),np.cos(rake)]);cd=np.array([-np.cos(clearance),np.sin(clearance)])
    radius=geometry['tool_tip_radius_m']
    if radius==0:return np.array([tip,tip+length*rd,tip+length*cd])
    half=.5*np.arccos(np.clip(np.dot(rd,cd),-1,1))
    center=tip+radius/np.sin(half)*(rd+cd)/np.linalg.norm(rd+cd)
    upper=tip+radius/np.tan(half)*rd;lower=tip+radius/np.tan(half)*cd
    pu=np.arctan2(*(upper-center)[::-1]);pl=np.arctan2(*(lower-center)[::-1])
    count=geometry['tool_tip_arc_segments']
    if count<2:fail('rounded tool requires at least two saved arc segments')
    angles=pl+np.arange(count)/count*((pu-pl)%(2*np.pi))
    arc=center+radius*np.column_stack((np.cos(angles),np.sin(angles)))
    return np.vstack((upper,tip+length*rd,tip+length*cd,arc))

def render(args,case,frames,ranges,receipt):
    from PIL import Image,ImageSequence
    from vtkmodules.vtkCommonCore import vtkIdList,vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray,vtkPolyData
    from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter
    from vtkmodules.util.numpy_support import numpy_to_vtk,vtk_to_numpy
    from paraview import simple as pv,servermanager
    from paraview.simple import _DisableFirstRenderCameraReset
    def write(path,xyz,field=None,values=None,quads=False,polygon=False):
        points=vtkPoints();points.SetData(numpy_to_vtk(xyz,deep=True))
        data=vtkPolyData();data.SetPoints(points);cells=vtkCellArray()
        groups=[range(len(xyz))] if polygon else (np.arange(len(xyz)).reshape(-1,4) if quads else [[i] for i in range(len(xyz))])
        for group in groups:
            ids=vtkIdList()
            for i in group:ids.InsertNextId(int(i))
            cells.InsertNextCell(ids)
        if polygon or quads:data.SetPolys(cells)
        else:data.SetVerts(cells)
        if field:
            array=numpy_to_vtk(values,deep=True);array.SetName(field);data.GetPointData().AddArray(array)
        writer=vtkXMLPolyDataWriter();writer.SetFileName(str(path));writer.SetInputData(data);writer.Write()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    png=out/'png';png.mkdir(exist_ok=True)
    _DisableFirstRenderCameraReset();view=pv.GetActiveViewOrCreate('RenderView')
    size=400*args.supersample;view.ViewSize=[size,size];view.UseFXAA=1
    view.Background=[1,1,1];view.UseColorPaletteForBackground=0
    view.OrientationAxesVisibility=0;view.CenterAxesVisibility=0
    view.CameraParallelProjection=1;view.CameraViewUp=[0,1,0]
    points=np.concatenate([f['x'][f['active']] for f in frames]);tips=np.array([f['record']['virtual_tip_m'] for f in frames])
    lower=np.minimum(points.min(axis=0),tips.min(axis=0));upper=np.maximum(points.max(axis=0),tips.max(axis=0))
    center=.5*(lower+upper);scale=.55*float((upper-lower).max())
    if args.parallel_scale is not None:scale=args.parallel_scale
    view.CameraParallelScale=scale
    palette=[[.91,.94,.97],[.22,.60,.78],[.99,.73,.25],[.82,.10,.08]]
    checks={}
    for field,name in FIELDS.items():
        images=[]
        for index,frame in enumerate(frames):
            active=frame['active'];values=frame[field][active]
            if args.geometry=='affine':
                xy=frame['corners'][active].reshape(-1,2);values=np.repeat(values,4)
            else:xy=frame['x'][active]
            xyz=np.column_stack((xy,np.zeros(len(xy)))).astype(np.float64)
            write(out/'particles.vtp',xyz,field,values,quads=args.geometry=='affine')
            tip=np.array(frame['record']['virtual_tip_m'])
            polygon=tool_polygon(tip,case['geometry'],args.tool_length)
            write(out/'tool.vtp',np.column_stack((polygon,np.full(len(polygon),-1e-9))),polygon=True)
            camera=tip+np.array(args.follow_offset) if args.follow_tool else center
            view.CameraFocalPoint=[*camera,0];view.CameraPosition=[*camera,max(1.,10*scale)]
            source=pv.XMLPolyDataReader(FileName=[str(out/'particles.vtp')]);source.UpdatePipeline()
            fetched=servermanager.Fetch(source)
            if not np.array_equal(vtk_to_numpy(fetched.GetPoints().GetData()),xyz):fail('ParaView coordinates changed')
            if not np.array_equal(vtk_to_numpy(fetched.GetPointData().GetArray(field)),values):fail('ParaView scalar data changed')
            display=pv.Show(source,view);display.Ambient=1;display.Diffuse=0;display.Specular=0
            if args.geometry=='centres':
                display.Representation='Points';display.PointSize=4*args.supersample;display.RenderPointsAsSpheres=0
            pv.ColorBy(display,('POINTS',field));lo,hi=ranges[field];display_hi=max(hi,lo+1e-12)
            lut=pv.GetColorTransferFunction(field);lut.ColorSpace='RGB'
            lut.RGBPoints=[lo,*palette[0],lo+(display_hi-lo)/3,*palette[1],lo+2*(display_hi-lo)/3,*palette[2],display_hi,*palette[3]]
            lut.RescaleTransferFunction(lo,display_hi);lut.AutomaticRescaleRangeMode='Never';display.LookupTable=lut;display.SetScalarBarVisibility(view,False)
            tool=pv.XMLPolyDataReader(FileName=[str(out/'tool.vtp')]);td=pv.Show(tool,view)
            td.Ambient=1;td.Diffuse=0;td.DiffuseColor=[.1,.1,.1];td.AmbientColor=[.1,.1,.1];td.ColorArrayName=['POINTS','']
            pv.Render(view);path=png/(name+'_%04d.png'%index);pv.SaveScreenshot(str(path),view,ImageResolution=[size,size])
            with Image.open(path) as image:image=image.convert('RGB').resize((400,400),Image.Resampling.LANCZOS)
            image.save(path);images.append(image);pv.Delete(source);pv.Delete(tool)
        target=out/(name+'.gif')
        images[0].save(target,save_all=True,append_images=images[1:],duration=args.frame_ms,loop=0,disposal=2,optimize=False)
        images[-1].save(out/(name+'-final.png'))
        with Image.open(target) as gif:
            checks[field]={'sha256':sha(target),'encoded_frames':gif.n_frames,'distinct_frames':len({f.convert('RGB').tobytes() for f in ImageSequence.Iterator(gif)}),'size':list(gif.size)}
        print(json.dumps({'gif':str(target),**checks[field]}),flush=True)
    receipt.update(checks=checks,renderer='ParaView',renderer_sha256=sha(__file__),size=[400,400],supersample=args.supersample,
                   camera={'follow_saved_tool':args.follow_tool,'offset_m':args.follow_offset,'center_m':center.tolist(),'parallel_scale_m':scale},
                   fetched_coordinates_and_scalars_equal_saved=True)
    (out/'render.json').write_text(json.dumps(receipt,indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=61);parser.add_argument('--start',type=float);parser.add_argument('--end',type=float)
    parser.add_argument('--geometry',choices=['centres','affine'],default='centres')
    parser.add_argument('--supersample',type=int,default=3);parser.add_argument('--frame-ms',type=int,default=100)
    parser.add_argument('--ranges',type=Path,help='JSON saved-field -> [min,max], fixed for all rendered frames')
    parser.add_argument('--tool-length',type=float,default=.256e-3,help='actual public prescribed wedge length, metres')
    parser.add_argument('--follow-tool',action='store_true');parser.add_argument('--follow-offset',type=float,nargs=2,default=[12e-6,5e-6])
    parser.add_argument('--parallel-scale',type=float,help='camera half-height in metres; no geometry scaling')
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    if args.frames<2 or args.supersample<1 or args.frame_ms<1 or args.tool_length<=0:parser.error('frames>=2 and positive render settings required')
    if args.parallel_scale is not None and args.parallel_scale<=0:parser.error('parallel-scale must be positive')
    case,checkpoint,frames,ranges,hashes=load_inputs(args)
    receipt={'input':str(args.input.resolve()),**hashes,'accepted_checkpoint_frame':checkpoint['frame'],
        'actual_time_window_s':[frames[0]['record']['time_s'],frames[-1]['record']['time_s']],
        'states':[dict(**f['record'],sha256=f['source_sha256'],failed_count=int((~f['active']).sum())) for f in frames],
        'fixed_ranges':ranges,'geometry':args.geometry,'deformation_scale':1.,'tool_length_m':args.tool_length,
        'rate_definition':'actual saved history.plastic_rate (constitutive equivalent plastic strain rate); not adjacent-frame PEEQ increment',
        'failed_policy':'only saved failed particles hidden in display; input archives unchanged',
        'completion_claim':'renders only the actual accepted saved interval; does not imply full planned schedule completion'}
    if args.validate_only:
        args.output.mkdir(parents=True,exist_ok=True);(args.output/'input-validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps({'valid':True,'frames':len(frames),'actual_time_window_s':receipt['actual_time_window_s']}));return
    render(args,case,frames,ranges,receipt)

if __name__=='__main__':main()
