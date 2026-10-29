"""Render actual rectangular three-point-bending AT2 fields, unamplified."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=61)
    parser.add_argument('--size', type=int, default=400)
    parser.add_argument('--height',type=int,default=400)
    parser.add_argument('--gif-name',default='at2-three-point-bending.gif')
    parser.add_argument('--allow-prefix', action='store_true')
    args = parser.parse_args()
    if args.frames < 2 or args.size < 128:
        parser.error('At least two frames and size >= 128 are required')
    if args.height!=args.size or args.size%2:
        parser.error('Use an even square canvas; two actual views share its height')
    panel_height=args.size//2
    if Path(args.gif_name).name!=args.gif_name or not args.gif_name.endswith('.gif'):
        raise ValueError('GIF name must be a plain GIF filename')
    summary = json.loads((args.input / 'summary.json').read_text())
    if not args.allow_prefix and not summary.get('completed'):
        raise ValueError('The requested mechanical load schedule has not completed')
    with (args.input / 'history.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        raise ValueError('No changing accepted trajectory to render')
    with np.load(args.input / 'reference-grid.npz', allow_pickle=False) as saved:
        xy = saved['coordinates'][:, :2].copy()
        if 'display_quads' in saved:
            quads = saved['display_quads'].astype(np.int64)
            display_xy = saved['display_coordinates'][:, :2].copy()
            display_nodes = saved['display_nodes'].astype(np.int64)
            display_weights = saved['display_weights'].copy()
        else:
            quads = np.unique(saved['nodes'].astype(np.int64), axis=0)
            display_xy = xy
            display_nodes = np.arange(len(xy))[:, None]
            display_weights = np.ones_like(display_nodes, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or quads.ndim != 2 or quads.shape[1] != 4:
        raise ValueError('Unexpected reference-grid surface layout')
    if (display_nodes.ndim != 2 or display_nodes.shape != display_weights.shape
            or len(display_nodes) != len(display_xy)
            or display_nodes.min() < 0 or display_nodes.max() >= len(xy)
            or not np.isfinite(display_weights).all()
            or np.any(display_weights < -1e-14)
            or not np.allclose(display_weights.sum(axis=1), 1., rtol=0, atol=1e-12)):
        raise ValueError('Invalid display constraint map')
    reference_display = np.einsum('ij,ijk->ik', display_weights, xy[display_nodes])
    if not np.allclose(reference_display, display_xy, rtol=0, atol=1e-12):
        raise ValueError('Display constraints do not reproduce physical coordinates')
    if quads.min() < 0 or quads.max() >= len(display_xy):
        raise ValueError('Surface connectivity does not match the coefficient grid')

    def display_fields(u, d):
        # Virtual display vertices follow the actual hanging-node constraints.
        # Their coefficients are independent of the solved damage field.
        du = np.einsum('ij,ijk->ik', display_weights, u[display_nodes])
        dd = np.sum(display_weights * d[display_nodes], axis=1)
        return np.column_stack((display_xy + du, np.zeros(len(display_xy)))), dd

    from paraview import simple as pv, servermanager
    from paraview import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
    from PIL import Image

    pv._DisableFirstRenderCameraReset()
    args.output.mkdir(parents=True, exist_ok=True)
    vtk_points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()
    packed = np.column_stack((np.full(len(quads), 4), quads)).ravel()
    cells.SetCells(len(quads), numpy_to_vtkIdTypeArray(packed, deep=True))
    mesh = vtk.vtkPolyData()
    mesh.SetPoints(vtk_points)
    mesh.SetPolys(cells)
    first_step = int(rows[0]['step'])
    with np.load(args.input / f'frame-{first_step:04d}.npz', allow_pickle=False) as first:
        xyz, display_damage = display_fields(first['displacement'], first['damage'])
        vtk_points.SetData(numpy_to_vtk(xyz, deep=True))
        field = numpy_to_vtk(display_damage, deep=True)
        field.SetName('damage')
        mesh.GetPointData().SetScalars(field)
    source = pv.TrivialProducer()
    source.GetClientSideObject().SetOutput(mesh)
    view = pv.CreateView('RenderView')
    view.ViewSize = [4 * args.size, 4 * panel_height]
    view.Background = [1., 1., 1.]
    view.UseColorPaletteForBackground = 0
    view.OrientationAxesVisibility = 0
    view.CenterAxesVisibility = 0
    view.CameraParallelProjection = 1
    center=.5*(xy.min(axis=0)+xy.max(axis=0))
    view.CameraPosition = [float(center[0]),float(center[1]),3.]
    view.CameraFocalPoint = [float(center[0]),float(center[1]),0.]
    view.CameraViewUp = [0., 1., 0.]
    overview_scale=.56*panel_height/args.size
    zoom_scale=.5*overview_scale
    view.CameraParallelScale = overview_scale
    view.UseFXAA = 1
    display = pv.Show(source, view)
    display.Representation = 'Surface'
    display.Ambient = 1.
    display.Diffuse = 0.
    pv.ColorBy(display, ('POINTS', 'damage'))
    lut = pv.GetColorTransferFunction('damage')
    lut.RGBPoints = [0., .94, .95, .97, .25, .64, .78, .89,
                     .5, .20, .46, .66, .75, .77, .32, .21, 1., .12, .02, .04]
    lut.ColorSpace = 'RGB'
    lut.RescaleTransferFunction(0., 1.)
    display.SetScalarBarVisibility(view, False)
    indices = np.unique(np.rint(np.linspace(0, len(rows) - 1,
                                          min(args.frames, len(rows)))).astype(int))
    images, proof, distinct = [], [], set()
    boundary_source=boundary_display=None
    for index in indices:
        step = int(rows[index]['step'])
        path = args.input / f'frame-{step:04d}.npz'
        with np.load(path, allow_pickle=False) as saved:
            u, d = saved['displacement'], saved['damage']
            if u.shape != xy.shape or d.shape != (len(xy),):
                raise ValueError('Accepted frame does not match the reference grid')
            if not np.isfinite(u).all() or not np.isfinite(d).all():
                raise ValueError('Nonfinite saved field')
            if d.min() < -1e-12 or d.max() > 1 + 1e-12:
                raise ValueError('Damage is outside its numerical bounds')
            # Replacing the producer also invalidates ParaView representation
            # caches, which otherwise can retain the first scalar field.
            pv.Delete(display)
            pv.Delete(source)
            if boundary_source is not None:
                pv.Delete(boundary_display);pv.Delete(boundary_source)
            xyz, display_damage = display_fields(u, d)
            vtk_points = vtk.vtkPoints()
            vtk_points.SetData(numpy_to_vtk(xyz, deep=True))
            mesh = vtk.vtkPolyData()
            mesh.SetPoints(vtk_points)
            mesh.SetPolys(cells)
            field = numpy_to_vtk(display_damage, deep=True)
            field.SetName('damage')
            mesh.GetPointData().SetScalars(field)
            mesh.Modified()
            source = pv.TrivialProducer()
            source.GetClientSideObject().SetOutput(mesh)
            source.UpdatePipeline()
            fetched = servermanager.Fetch(source)
            if not np.array_equal(vtk_to_numpy(fetched.GetPointData().GetArray('damage')), display_damage):
                raise ValueError('ParaView scalar field differs from the saved field and its display constraints')
            if not np.array_equal(vtk_to_numpy(fetched.GetPoints().GetData()), xyz):
                raise ValueError('ParaView coordinates differ from the saved frame')
            display = pv.Show(source, view)
            display.Representation = 'Surface'
            display.Ambient = 1.
            display.Diffuse = 0.
            pv.ColorBy(display, ('POINTS', 'damage'))
            display.SetScalarBarVisibility(view, False)
            lut.RescaleTransferFunction(0., 1.)
            # Actual one-sided mesh edges include the EXISTING geometric notch.
            # There is no line beyond that notch and no damage-derived path.
            edges=vtk.vtkFeatureEdges();edges.SetInputData(mesh)
            edges.BoundaryEdgesOn();edges.FeatureEdgesOff();edges.NonManifoldEdgesOff();edges.ManifoldEdgesOff();edges.Update()
            boundary=vtk.vtkPolyData();boundary.DeepCopy(edges.GetOutput())
            boundary_source=pv.TrivialProducer();boundary_source.GetClientSideObject().SetOutput(boundary)
            boundary_display=pv.Show(boundary_source,view);pv.ColorBy(boundary_display,None)
            boundary_display.Representation='Surface';boundary_display.DiffuseColor=[.30,.33,.38];boundary_display.AmbientColor=[.30,.33,.38]
            boundary_display.Ambient=1.;boundary_display.Diffuse=0.;boundary_display.LineWidth=4.;boundary_display.SetScalarBarVisibility(view,False)
            view.CameraParallelScale=overview_scale
            pv.Render(view)
            png = args.output / f'frame-{index:04d}.png'
            overview_png=args.output/f'overview-{index:04d}.png'
            zoom_png=args.output/f'zoom-{index:04d}.png'
            pv.SaveScreenshot(str(overview_png),view,ImageResolution=[4*args.size,4*panel_height])
            # Camera-only close-up of the SAME producer, displacement, damage,
            # physical time and LUT. No geometry or field is rotated/rescaled.
            view.CameraParallelScale=zoom_scale
            pv.Render(view)
            pv.SaveScreenshot(str(zoom_png),view,ImageResolution=[4*args.size,4*panel_height])
            image=Image.new('RGB',(args.size,args.size),'white')
            for panel,source_png in enumerate([overview_png,zoom_png]):
                with Image.open(source_png) as original:
                    panel_image=original.convert('RGB').resize((args.size,panel_height),Image.Resampling.LANCZOS)
                image.paste(panel_image,(0,panel*panel_height))
            image.save(png)
            pixels = hashlib.sha256(image.tobytes()).hexdigest()
            distinct.add(pixels)
            images.append(image)
            proof.append(dict(step=step, source_sha256=sha(path),
                              displacement_m=float(rows[index]['displacement_m']),
                              damage_min=float(d.min()), damage_max=float(d.max()),
                              actual_boundary_edge_points=boundary.GetNumberOfPoints(),
                              boundary_coordinates_sha256=hashlib.sha256(vtk_to_numpy(boundary.GetPoints().GetData()).tobytes()).hexdigest(),
                              fetched_fields_equal_saved=True,
                              rendered_pixels_sha256=pixels))
    if len(distinct) < 2:
        raise ValueError('Rendered frames are static; no GIF is delivered')
    gif = args.output / args.gif_name
    images[0].save(gif, save_all=True, append_images=images[1:],
                   duration=100, loop=0, optimize=False, disposal=2)
    receipt = dict(renderer='ParaView', renderer_source_sha256=sha(__file__),
                   gif_sha256=sha(gif), size=[args.size, args.height],
                   frames=len(images), distinct_rendered_frames=len(distinct),
                   deformation_scale=1., damage_range=[0., 1.],
                   display_vertices=len(display_xy), physical_coefficients=len(xy),
                   display_mapping='Saved geometry constraints; no damage-dependent or mirror averaging',
                   overlays=False, artificial_crack_lines=False,
                   actual_geometry_boundary_edges='One-sided edges extracted from saved split display-quads at the actual displacement, including only the existing notch. No propagated crack line is drawn.',
                   source_summary_sha256=sha(args.input / 'summary.json'),
                   reference_grid_sha256=sha(args.input / 'reference-grid.npz'),
                   source_summary=summary, publication_qualified=False,
                   physical_boundary_conditions=json.loads((args.input/'material-case.json').read_text()),
                   views={'upper':'Full-beam overview','lower':'Central notch/crack camera close-up','same_saved_fields_and_time_in_both':True,'overview_x_bounds_m':[float(center[0]-2*overview_scale),float(center[0]+2*overview_scale)],'overview_y_bounds_m':[float(center[1]-overview_scale),float(center[1]+overview_scale)],'zoom_x_bounds_m':[float(center[0]-2*zoom_scale),float(center[0]+2*zoom_scale)],'zoom_y_bounds_m':[float(center[1]-zoom_scale),float(center[1]+zoom_scale)],'camera_only_zoom':True,'field_rotation_or_artificial_extrusion':False},
                   note='Rendering does not establish mesh or load-step convergence.',
                   sources=proof)
    (args.output / 'render.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(gif=str(gif), frames=len(images),
                          distinct_frames=len(distinct))), flush=True)


if __name__ == '__main__':
    main()
