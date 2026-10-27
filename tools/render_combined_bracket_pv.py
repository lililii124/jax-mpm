"""Render saved bracket MPM frames with ParaView 6.2 OSMesa; no solver rerun."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence
from paraview import servermanager
from paraview.simple import (
    ColorBy, Delete, GetActiveViewOrCreate, GetColorTransferFunction, Hide,
    Render, SaveScreenshot, Show, TrivialProducer, _DisableFirstRenderCameraReset,
)
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonDataModel import vtkStructuredGrid
from vtkmodules.vtkFiltersCore import vtkContourFilter
from vtkmodules.vtkCommonCore import vtkPoints


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def make_grid_cache(reference, grid_n):
    """Build regular-grid indices and hole interpolation weights once per render."""
    h = 0.5 / grid_n
    bounds = ((.10,.90),(.22,.66),(.38,.62))
    axes = [np.arange(lo+h/2, hi, h) for lo,hi in bounds]
    nx,ny,nz = (len(axis) for axis in axes)
    dimensions = (nx+2,ny+2,nz+2)
    origin = (axes[0][0]-h,axes[1][0]-h,axes[2][0]-h)
    particle_ids = np.full((nz+2,ny+2,nx+2),-1,dtype=np.int32)
    ii=np.rint((reference[:,0]-axes[0][0])/h).astype(np.int32)+1
    jj=np.rint((reference[:,1]-axes[1][0])/h).astype(np.int32)+1
    kk=np.rint((reference[:,2]-axes[2][0])/h).astype(np.int32)+1
    assert np.all((1<=ii)&(ii<=nx)&(1<=jj)&(jj<=ny)&(1<=kk)&(kk<=nz))
    particle_ids[kk,jj,ii]=np.arange(len(reference),dtype=np.int32)
    # The exact through-hole has no MPM particles. Fill only its unused grid
    # displacement values from the nearest eight real particles; its scalar
    # occupancy remains zero, so the isosurface preserves the open hole.
    holes=np.argwhere(particle_ids[1:-1,1:-1,1:-1] < 0)+1
    query=np.column_stack((origin[0]+holes[:,2]*h,origin[1]+holes[:,1]*h,origin[2]+holes[:,0]*h))
    try:
        from scipy.spatial import cKDTree
        distance, nearest = cKDTree(reference).query(query,k=8)
    except ImportError:
        # Grid-local fallback: the nearest solid is within a small stencil of
        # the circular through-hole boundary, so never scan every particle.
        nearest=np.empty((len(holes),8),dtype=np.int32)
        distance=np.empty((len(holes),8),dtype=np.float64)
        radius=9
        for index,(k,j,i) in enumerate(holes):
            slab=particle_ids[max(1,k-radius):min(nz+1,k+radius+1),
                              max(1,j-radius):min(ny+1,j+radius+1),
                              max(1,i-radius):min(nx+1,i+radius+1)]
            ids=slab[slab>=0]
            delta=reference[ids]-query[index]
            d2=np.sum(delta*delta,axis=1)
            take=np.argpartition(d2,7)[:8]
            nearest[index]=ids[take]
            distance[index]=np.sqrt(d2[take])
    weight=1.0/np.maximum(distance,h*.1)
    weight/=weight.sum(axis=1,keepdims=True)
    kz,ky,kx=np.indices((nz+2,ny+2,nx+2))
    ck=np.clip(kz,1,nz); cy=np.clip(ky,1,ny); cx=np.clip(kx,1,nx)
    pad=(kz==0)|(kz==nz+1)|(ky==0)|(ky==ny+1)|(kx==0)|(kx==nx+1)
    pad_indices=(kz[pad],ky[pad],kx[pad],ck[pad],cy[pad],cx[pad])
    return {'h':h,'axes':axes,'nx':nx,'ny':ny,'nz':nz,'dimensions':dimensions,
            'origin':origin,'ii':ii,'jj':jj,'kk':kk,'holes':holes,
            'nearest':np.asarray(nearest,dtype=np.int32),
            'weights':np.asarray(weight,dtype=np.float64),'pad_indices':pad_indices}


def structured_surface(reference, coordinates, volume_ratio, cache):
    """Contour the particle occupancy on its actual regular MPM neighbor grid."""
    h=cache['h']; nx=cache['nx']; ny=cache['ny']; nz=cache['nz']
    dimensions=cache['dimensions']; origin=cache['origin']
    phi_grid=np.zeros((nz+2,ny+2,nx+2),dtype=np.float32)
    vector_grid=np.zeros((nz+2,ny+2,nx+2,3),dtype=np.float64)
    ii,jj,kk=cache['ii'],cache['jj'],cache['kk']
    phi_grid[kk,jj,ii]=volume_ratio
    vector_grid[kk,jj,ii]=coordinates-reference
    holes=cache['holes']; nearest=cache['nearest']; weight=cache['weights']
    hole_vectors=coordinates[nearest]-reference[nearest]
    vector_grid[holes[:,0],holes[:,1],holes[:,2]]=np.sum(hole_vectors*weight[:,:,None],axis=1)
    kz,ky,kx,ck,cy,cx=cache['pad_indices']
    vector_grid[kz,ky,kx]=vector_grid[ck,cy,cx]

    grid=vtkStructuredGrid()
    grid.SetDimensions(*dimensions)
    zcoords=np.asarray((origin[2]+np.arange(nz+2)*h),dtype=np.float64)
    ycoords=np.asarray((origin[1]+np.arange(ny+2)*h),dtype=np.float64)
    xcoords=np.asarray((origin[0]+np.arange(nx+2)*h),dtype=np.float64)
    gz,gy,gx=np.meshgrid(zcoords,ycoords,xcoords,indexing='ij')
    reference_grid=np.stack((gx,gy,gz),axis=-1).reshape((-1,3))
    deformed_grid=reference_grid+vector_grid.reshape((-1,3),order='C')
    points=vtkPoints()
    points.SetData(numpy_to_vtk(deformed_grid,deep=True))
    grid.SetPoints(points)
    scalar=numpy_to_vtk(phi_grid.ravel(order='C'),deep=True)
    scalar.SetName('volume_ratio')
    grid.GetPointData().SetScalars(scalar)
    magnitude=numpy_to_vtk(np.linalg.norm(vector_grid,axis=-1).ravel(order='C').astype(np.float32),deep=True)
    magnitude.SetName('displacement_m')
    grid.GetPointData().AddArray(magnitude)
    contour=vtkContourFilter()
    contour.SetInputData(grid)
    contour.SetValue(0,.5)
    contour.ComputeNormalsOn()
    contour.Update()
    surface=contour.GetOutput()
    assert surface.GetNumberOfPoints()>0 and surface.GetNumberOfCells()>0
    return surface


def save_gif(png_paths, output: Path, durations_ms=90):
    images = [Image.open(p).convert('RGB') for p in png_paths]
    assert all(image.size == (400, 400) for image in images)
    keyed = []
    for index, image in enumerate(images):
        pixels = image.tobytes()
        if keyed and keyed[-1]['pixels'] == pixels:
            keyed[-1]['source_frames'].append(index)
        else:
            keyed.append({'image': image, 'pixels': pixels, 'source_frames': [index]})
    assert len(keyed) >= 2
    frames = [entry['image'].convert('P', palette=Image.ADAPTIVE, colors=256) for entry in keyed]
    frames[0].save(output, save_all=True, append_images=frames[1:],
                   duration=durations_ms, loop=0, optimize=False, disposal=2)
    with Image.open(output) as saved:
        decoded = [frame.convert('RGB').tobytes() for frame in ImageSequence.Iterator(saved)]
    assert len(set(decoded)) >= 2
    return {'source_png_frames': len(png_paths), 'distinct_source_frames': len(keyed),
            'decoded_gif_frames': len(decoded), 'resolution_px': [400, 400],
            'frame_duration_ms': durations_ms, 'sha256': sha256(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True,
                        help='contains forward-summary.json and trajectory.npz')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    summary_path = args.input / 'forward-summary.json'
    trajectory_path = args.input / 'trajectory.npz'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    archive = np.load(trajectory_path, allow_pickle=False)
    reference = archive['points'].astype(np.float64, copy=False)
    phi = {
        'initial': archive['initial_volume_ratio'].astype(np.float32, copy=False),
        'optimized': archive['optimized_volume_ratio'].astype(np.float32, copy=False),
    }
    positions = {
        'initial': archive['initial_bending_positions'].astype(np.float32, copy=False),
        'optimized': archive['optimized_bending_positions'].astype(np.float32, copy=False),
    }
    count = int(summary['trajectory']['frames'])
    assert len(reference) == 71553 and count == 21
    assert all(value.shape == (count, len(reference), 3) for value in positions.values())
    assert all(np.isfinite(value).all() for value in positions.values())
    assert all(value.shape == (len(reference),) for value in phi.values())

    frame_times = np.asarray(summary['trajectory']['times_s'], dtype=float)
    displacement = {key: np.linalg.norm(value-reference[None, :, :], axis=2)
                    for key, value in positions.items()}
    color_max = max(float(value.max()) for value in displacement.values())
    assert color_max > 0 and np.isfinite(color_max)
    png_root = args.output / 'png'
    png_root.mkdir()

    _DisableFirstRenderCameraReset()
    view = GetActiveViewOrCreate('RenderView')
    view.ViewSize = [800, 800]
    view.UseColorPaletteForBackground = 0
    view.Background = [0.025, 0.040, 0.075]
    view.OrientationAxesVisibility = 0
    if hasattr(view, 'CenterAxesVisibility'):
        view.CenterAxesVisibility = 0
    view.CameraParallelProjection = 1
    view.CameraFocalPoint = [0.50, 0.44, 0.50]
    view.CameraPosition = [1.18, 1.03, 1.15]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    view.CameraParallelScale = 0.46
    if hasattr(view, 'UseFXAA'):
        view.UseFXAA = 1

    lut = GetColorTransferFunction('displacement_m')
    lut.ColorSpace = 'RGB'
    lut.RGBPoints = [
        0.0, 0.05, 0.18, 0.56,
        0.28*color_max, 0.00, 0.70, 0.76,
        0.62*color_max, 0.98, 0.77, 0.22,
        color_max, 0.91, 0.20, 0.12,
    ]
    lut.RescaleTransferFunction(0.0, color_max)
    lut.AutomaticRescaleRangeMode = 'Never'

    records = {}
    grid_n=int(summary['verification_grid'])
    grid_cache=make_grid_cache(reference,grid_n)
    print(json.dumps({'regular_grid_dimensions':grid_cache['dimensions'],
                      'particle_count':len(reference),'hole_ghost_points':len(grid_cache['holes']),
                      'cached_neighbours_per_hole':8}),flush=True)
    for design in ('initial', 'optimized'):
        design_dir = png_root / design
        design_dir.mkdir()
        hires_dir = args.output / 'png-hires' / design
        hires_dir.mkdir(parents=True)
        design_records = []
        for index in range(count):
            coords = positions[design][index].astype(np.float64, copy=False)
            surface=structured_surface(reference,coords,phi[design],grid_cache)
            # Contour the voxelized particle field with actual grid adjacency,
            # then rebuild the producer each frame to avoid ParaView caching.
            producer = TrivialProducer(registrationName=f'JAX MPM {design} frame {index}')
            producer.GetClientSideObject().SetOutput(surface)
            producer.UpdatePipeline()
            display = Show(producer, view)
            display.Representation = 'Surface'
            display.Emissive = 0
            display.Ambient = 0.30
            display.Diffuse = 0.70
            display.Specular = 0.24
            display.SpecularPower = 30
            ColorBy(display, ('POINTS', 'displacement_m'))
            display.LookupTable = lut
            display.SetScalarBarVisibility(view, False)
            fetched = servermanager.Fetch(producer)
            fetched_coords = vtk_to_numpy(fetched.GetPoints().GetData())
            expected_coords = vtk_to_numpy(surface.GetPoints().GetData())
            assert fetched_coords.shape == expected_coords.shape and np.array_equal(fetched_coords, expected_coords)
            Render(view)
            png = design_dir / f'frame-{index:03d}.png'
            hires_png = hires_dir / f'frame-{index:03d}.png'
            SaveScreenshot(str(hires_png), view, ImageResolution=[800, 800])
            with Image.open(hires_png) as high:
                resampling = getattr(Image, 'Resampling', Image).LANCZOS
                high.convert('RGB').resize((400,400),resampling).save(png)
            design_records.append({
                'index': index, 'time_s': float(frame_times[index]),
                'max_displacement_m': float(displacement[design][index].max()),
                'source_particle_count': len(reference), 'surface_point_count': int(fetched_coords.shape[0]),
                'surface_cell_count': int(fetched.GetNumberOfCells()),
                'fetched_surface_coordinates_exact': True,
                'png': str(png.relative_to(args.output)), 'png_sha256': sha256(png),
                'hires_png': str(hires_png.relative_to(args.output)), 'hires_png_sha256': sha256(hires_png),
            })
            Hide(producer, view)
            Delete(display)
            Delete(producer)
        gif = args.output / f'{design}-bending.gif'
        gif_info = save_gif([design_dir / f'frame-{i:03d}.png' for i in range(count)], gif)
        records[design] = {'frames': design_records, 'gif': gif.name, **gif_info}

    manifest = {
        'render_engine': 'ParaView 6.2 OSMesa',
        'summary_sha256': sha256(summary_path), 'trajectory_sha256': sha256(trajectory_path),
        'solver_grid': summary['verification_grid'], 'particle_count': len(reference),
        'frame_count': count, 'frame_times_s': frame_times.tolist(),
        'same_camera_and_colour_range_for_both_designs': True,
        'same_physical_trajectory_and_time_sampling': True,
        'surface_reconstruction': 'vtkContourFilter at volume_ratio=0.5 on a deformed vtkStructuredGrid using actual regular MPM particle-neighbour adjacency; through-hole grid nodes remain zero occupancy',
        'colour': {'field': 'actual displacement magnitude (m)', 'shared_range_m': [0.0, color_max]},
        'deformation_scale': 1.0, 'visualization_iso_fraction': 0.5,
        'rendered_resolution_each_px': [800, 800], 'downsampled_output_each_px': [400, 400],
        'downsampling_filter': 'Pillow LANCZOS', 'axes_titles_labels': False,
        'camera': {'parallel_projection': True, 'position': [1.18, 1.03, 1.15],
                   'focal_point': [0.50, 0.44, 0.50], 'view_up': [0.0, 0.0, 1.0],
                   'parallel_scale': 0.46},
        'phi_ranges': {key: [float(value.min()), float(value.max())] for key, value in phi.items()},
        'designs': records,
    }
    manifest_path = args.output / 'render-manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'manifest': str(manifest_path), 'initial_gif': records['initial']['gif'],
                      'optimized_gif': records['optimized']['gif'],
                      'color_range_m': [0.0,color_max], 'render_manifest_sha256':sha256(manifest_path)},
                     indent=2), flush=True)


if __name__ == '__main__':
    main()
