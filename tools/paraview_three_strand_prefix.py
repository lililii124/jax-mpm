"""Render only the authenticated saved three-strand VTK prefix, using ParaView."""
import hashlib
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
from paraview.simple import (
    _DisableFirstRenderCameraReset, CreateView, GetParaViewVersion, PVDReader,
    Render, SaveScreenshot, SaveState, Show, Threshold,
)

folder = Path(sys.argv[1]).resolve()
metadata = json.loads((folder / 'vtk_metadata.json').read_text())
assert metadata['frames'] == 13 and metadata['field_ranges']['family_id'] == [0., 2.]
assert metadata['field_ranges']['phase'] == [1., 1.]
inputs = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(folder.glob('vtk/*'))}
inputs['particles.pvd'] = hashlib.sha256((folder / 'particles.pvd').read_bytes()).hexdigest()
_DisableFirstRenderCameraReset()
reader = PVDReader(FileName=str(folder / 'particles.pvd'))
reader.UpdatePipeline()
times = list(reader.TimestepValues)
assert len(times) == 13
view = CreateView('RenderView')
view.ViewSize = [400, 400]
view.Background = [1., 1., 1.]
view.UseColorPaletteForBackground = 0
view.OrientationAxesVisibility = 0
view.CenterAxesVisibility = 0
view.CameraParallelProjection = 1
colors = [[.045, .36, .67], [.94, .36, .08], [.32, .66, .48]]
for family, color in enumerate(colors):
    selected = Threshold(registrationName='Actual strand '+str(family), Input=reader)
    selected.Scalars = ['POINTS', 'family_id']
    selected.LowerThreshold = family
    selected.UpperThreshold = family
    selected.UpdatePipeline()
    assert selected.GetDataInformation().GetNumberOfPoints() == metadata['particles'] // 3
    display = Show(selected, view)
    display.Representation = 'Point Gaussian'
    display.GaussianRadius = .62 * metadata['spacing']
    display.ShaderPreset = 'Sphere'
    display.Emissive = 0
    display.Ambient = .3
    display.Diffuse = .7
    display.Specular = .35
    display.SpecularPower = 35
    display.Opacity = 1.
    display.ColorArrayName = ['POINTS', '']
    display.DiffuseColor = color
    display.AmbientColor = color
bounds = metadata['bounds']
center = np.array([.5*(bounds[2*i]+bounds[2*i+1]) for i in range(3)])
length = max(bounds[2*i+1]-bounds[2*i] for i in range(3))
direction = np.array([.4, -1.8, .8]); direction /= np.linalg.norm(direction)
up = np.array([0., 0., 1.])
right = np.cross(up, direction); right /= np.linalg.norm(right)
upward = np.cross(direction, right)
corners = np.array(list(product(*[(bounds[2*i], bounds[2*i+1]) for i in range(3)])))
view.CameraFocalPoint = center.tolist()
view.CameraPosition = (center + 3*length*direction).tolist()
view.CameraViewUp = up.tolist()
view.CameraParallelScale = .56 * max(np.ptp(corners @ right), np.ptp(corners @ upward))
frames = folder/'frames'; frames.mkdir(exist_ok=False)
hashes = []
for index, time in enumerate(times):
    view.ViewTime = time
    reader.UpdatePipeline(time)
    Render(view)
    target = frames/f'frame_{index:05d}.png'
    SaveScreenshot(str(target), view, ImageResolution=[400,400])
    hashes.append(hashlib.sha256(target.read_bytes()).hexdigest())
    print(json.dumps({'frame':index,'time_s':time,'png_sha256':hashes[-1]}), flush=True)
SaveState(str(folder/'scene.pvsm'))
receipt = {'renderer':'ParaView','version':str(GetParaViewVersion()),'frames_rendered':13,
    'resolution':[400,400],'geometry_scale':1,'axes':False,'annotations':False,
    'background':[1.,1.,1.],'field':'family_id','family_colors':colors,
    'source_frames':list(range(13)),'times_s':times,'no_interpolation':True,
    'representation':'Point Gaussian spheres, radius 0.62 times actual particle spacing',
    'gaussian_radius_m':.62*metadata['spacing'],'camera_position':list(view.CameraPosition),
    'camera_focal_point':list(view.CameraFocalPoint),'camera_view_up':list(view.CameraViewUp),
    'camera_parallel_scale':view.CameraParallelScale,'fixed_camera':True,
    'source_vtk_sha256':inputs,'png_sha256':hashes,
    'scene_sha256':hashlib.sha256((folder/'scene.pvsm').read_bytes()).hexdigest(),
    'scene_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'source_checkpoint_sha256':'4152797ee8f85ab2b551bf721efeaabd48402775ee74bd08d5c1fea3dc8d316e',
    'visual_review_passed':False}
(folder/'render.json').write_text(json.dumps(receipt,indent=2)+'\n')
