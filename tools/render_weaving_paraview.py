"""Render actual saved weaving particle domains with ParaView.

Run with pvpython --force-offscreen-rendering; ordinary Python supports
--validate-only. Inputs are explicit directories containing frame_*.npz and
plan.json. No solver, geometry generator, or intermediate states are run.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

COLORS = [[.045, .36, .67], [.94, .36, .08], [.32, .66, .48]]
REQUIRED = {'x', 'F', 'reference', 'reference_volume', 'fibre_id', 'failed', 'time', 'step'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_domains(reference, ids, volume, arc_segments):
    """Recover straight reference annular quadrature cells from saved samples.

    Every ring particle owns an equal-area annular sector. Its reference
    position is the saved radial quadrature point, not the sector centroid.
    Longitudinal cells are bounded halfway between saved reference stations.
    """
    offsets = np.empty((len(reference), 4 * (arc_segments + 1), 3))
    diagnostics = []
    for fibre in np.unique(ids):
        take = np.flatnonzero(ids == fibre)
        q = reference[take]
        _, _, basis = np.linalg.svd(q - q.mean(0), full_matrices=False)
        tangent = basis[0]
        if np.dot(q[-1] - q[0], tangent) < 0:
            tangent = -tangent
        axial = q @ tangent
        different = np.flatnonzero(abs(axial - axial[0]) > 1e-10)
        if not len(different):
            raise ValueError('fibre has no longitudinal stations')
        cross = int(different[0])
        if cross < 4 or len(q) % cross:
            raise ValueError('unsupported saved section ordering')
        sections = q.reshape(-1, cross, 3)
        centers = sections.mean(1)
        steps = np.diff(centers, axis=0) @ tangent
        ds = float(np.mean(steps))
        np.testing.assert_allclose(steps, ds, rtol=1e-9, atol=1e-13)
        radial = sections[0] - centers[0]
        radii = np.linalg.norm(radial, axis=1)
        ring_values = []
        for value in sorted(radii):
            if not ring_values or abs(value - ring_values[-1]) > 1e-10:
                ring_values.append(value)
        counts = [int(np.count_nonzero(abs(radii - r) < 1e-10)) for r in ring_values]
        boundaries = np.r_[0., np.cumsum(counts) / cross]
        radius = np.sqrt(float(np.mean(volume[take])) * cross / (np.pi * ds))
        templates = np.empty((cross, 4 * (arc_segments + 1), 3))
        worst_area_error = 0.
        for ring, (r, count) in enumerate(zip(ring_values, counts)):
            expected = radius * np.sqrt((boundaries[ring] + boundaries[ring + 1]) / 2)
            np.testing.assert_allclose(r, expected, rtol=1e-8, atol=1e-13)
            lower, upper = radius * np.sqrt(boundaries[ring:ring + 2])
            for index in np.flatnonzero(abs(radii - r) < 1e-10):
                e = radial[index] / r
                perpendicular = np.cross(tangent, e)
                theta = np.linspace(-np.pi / count, np.pi / count, arc_segments + 1)
                directions = np.cos(theta)[:, None] * e + np.sin(theta)[:, None] * perpendicular
                polygon = np.concatenate((upper * directions, lower * directions[::-1]))
                template = np.concatenate((polygon - .5 * ds * tangent, polygon + .5 * ds * tangent))
                templates[index] = template - radial[index]
                angle = 2 * np.pi / count
                analytic = .5 * (upper ** 2 - lower ** 2) * angle * ds
                np.testing.assert_allclose(analytic, volume[take[index]], rtol=1e-8, atol=1e-24)
                worst_area_error = max(worst_area_error, 1 - np.sin(angle / arc_segments) / (angle / arc_segments))
        offsets[take] = np.tile(templates, (len(sections), 1, 1))
        np.testing.assert_allclose(sections - centers[:, None, :], np.broadcast_to(radial, sections.shape), rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(volume[take], volume[take[0]], rtol=1e-10, atol=1e-24)
        diagnostics.append(dict(fibre_id=int(fibre), particles=len(take), sections=len(sections),
            cross_section_particles=cross, ring_counts=counts, radius_m=radius,
            axial_cell_length_m=ds, maximum_relative_arc_volume_deficit=worst_area_error))
    return offsets, diagnostics


def load(args):
    records = []
    plans = [json.loads((folder / 'plan.json').read_text()) for folder in args.input]
    plan = plans[0]
    # The endpoint is the only stage-specific plan field; bind everything else.
    stable_plan = {key: value for key, value in plan.items() if key != 'bounded_prefix_target_s'}
    for other in plans[1:]:
        other_stable = {key: value for key, value in other.items() if key != 'bounded_prefix_target_s'}
        if other_stable != stable_plan:
            different = sorted(key for key in set(stable_plan) | set(other_stable)
                               if stable_plan.get(key) != other_stable.get(key))
            raise ValueError('input windows have different stable physical/numerical plans: ' + str(different))
    for folder in args.input:
        for path in folder.glob('frame_*.npz'):
            with np.load(path, allow_pickle=False) as saved:
                if not REQUIRED <= set(saved.files):
                    raise ValueError(str(path) + ': missing ' + str(REQUIRED - set(saved.files)))
                stamp, step = float(saved['time']), int(saved['step'])
            if (args.start is None or stamp >= args.start) and (args.end is None or stamp <= args.end):
                records.append(dict(path=str(path.resolve()), time_s=stamp, step=step))
    records.sort(key=lambda row: (row['time_s'], row['step']))
    unique = {}
    times_by_step = {}
    for row in records:
        step, stamp = row['step'], row['time_s']
        if not np.isfinite(stamp):
            raise ValueError('saved physical time is not finite')
        if step in times_by_step and times_by_step[step] != stamp:
            raise ValueError('same saved step has different physical times: ' + str(step))
        times_by_step[step] = stamp
        key = (stamp, step)
        if key in unique:
            previous = unique[key]
            with np.load(previous['path'], allow_pickle=False) as first, np.load(row['path'], allow_pickle=False) as second:
                for field in REQUIRED:
                    a, b = first[field], second[field]
                    if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
                        raise ValueError('conflicting duplicate saved state ' + str(key) + ': ' + field)
        else:
            unique[key] = row
    records = list(unique.values())
    if any(right['time_s'] <= left['time_s'] or right['step'] <= left['step']
           for left, right in zip(records[:-1], records[1:])):
        raise ValueError('saved physical times and step indices must be strictly increasing')
    if len(records) < 2:
        raise ValueError('at least two actual saved states are required')
    indices = sorted(set(np.rint(np.linspace(0, len(records) - 1, min(args.frames, len(records)))).astype(int)))
    frames = []
    reference = ids = volume = offsets = domains = None
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    for index in indices:
        row = records[index]
        path = Path(row['path'])
        before = sha(path)
        with np.load(path, allow_pickle=False) as saved:
            data = {key: np.array(saved[key]) for key in REQUIRED}
        if sha(path) != before:
            raise ValueError('saved frame changed while loading: ' + str(path))
        count = len(data['x'])
        for key, shape in [('x', (count, 3)), ('reference', (count, 3)), ('F', (count, 3, 3)),
                           ('reference_volume', (count,)), ('fibre_id', (count,)), ('failed', (count,))]:
            if data[key].shape != shape or not np.isfinite(data[key]).all():
                raise ValueError('invalid saved ' + key)
        if not np.isin(data['failed'], [False, True]).all() or count != plan['particles']:
            raise ValueError('invalid saved failed flags or particle count')
        if reference is None:
            reference, ids, volume = data['reference'], data['fibre_id'], data['reference_volume']
            if not np.array_equal(np.unique(ids), np.arange(plan['fibres'])) or plan['fibres'] != 3 * plan['per_family']:
                raise ValueError('expected three contiguous fibre ID families')
            offsets, domains = reference_domains(reference, ids, volume, args.arc_segments)
        else:
            for key, expected in [('reference', reference), ('fibre_id', ids), ('reference_volume', volume)]:
                if not np.array_equal(data[key], expected):
                    raise ValueError('particle ordering/reference changed: ' + key)
        active = ~data['failed'].astype(bool)
        if not active.any() or np.any(np.linalg.det(data['F'][active]) <= 0):
            raise ValueError('no live material or nonpositive live deformation determinant')
        corners = data['x'][active, None] + np.einsum('pij,pkj->pki', data['F'][active], offsets[active])
        low = np.minimum(low, corners.min((0, 1)))
        high = np.maximum(high, corners.max((0, 1)))
        frames.append(dict(x=data['x'], F=data['F'], active=active,
            record=dict(**row, sha256=before, failed_count=int((~active).sum()))))
    return plan, frames, ids, offsets, domains, low, high


def camera(low, high, direction):
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    up = np.array([0., 0., 1.])
    right = np.cross(up, direction)
    right /= np.linalg.norm(right)
    upward = np.cross(direction, right)
    from itertools import product
    corners = np.array(list(product(*zip(low, high))))
    center = .5 * (low + high)
    scale = .56 * max(np.ptp(corners @ right), np.ptp(corners @ upward))
    return dict(position_m=(center + 3 * np.max(high - low) * direction).tolist(),
        focal_point_m=center.tolist(), view_up=up.tolist(), parallel_scale_m=float(scale))


def render(args, frames, ids, offsets, receipt):
    from PIL import Image
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
    from paraview import simple as pv, servermanager
    from paraview.simple import _DisableFirstRenderCameraReset
    _DisableFirstRenderCameraReset()
    view = pv.CreateView('RenderView')
    size = 400 * args.supersample
    view.ViewSize = [size, size]
    view.Background = [1., 1., 1.]
    view.UseColorPaletteForBackground = 0
    view.OrientationAxesVisibility = view.CenterAxesVisibility = 0
    view.CameraParallelProjection = 1
    c = receipt['camera']
    view.CameraPosition = c['position_m']; view.CameraFocalPoint = c['focal_point_m']
    view.CameraViewUp = c['view_up']; view.CameraParallelScale = c['parallel_scale_m']
    folder = args.output / 'png'; folder.mkdir(exist_ok=True)
    images, hashes = [], []
    vertices = offsets.shape[1]; half = vertices // 2
    local = [list(range(half - 1, -1, -1)), list(range(half, vertices))]
    local += [[i, (i + 1) % half, (i + 1) % half + half, i + half] for i in range(half)]
    sizes = np.array([len(face) for face in local])
    pattern = np.concatenate(local)
    for frame_index, frame in enumerate(frames):
        sources = []
        for family, color in enumerate(COLORS):
            active = frame['active'] & (ids // receipt['per_family'] == family)
            if not active.any():
                continue
            xyz = (frame['x'][active, None] + np.einsum('pij,pkj->pki', frame['F'][active], offsets[active])).reshape(-1, 3)
            count = int(active.sum())
            connectivity = (pattern[None] + np.arange(count)[:, None] * vertices).ravel().astype(np.int64)
            cell_offsets = np.r_[0, np.cumsum(np.tile(sizes, count))].astype(np.int64)
            points = vtkPoints(); points.SetData(numpy_to_vtk(xyz, deep=True))
            cells = vtkCellArray(); cells.SetData(numpy_to_vtkIdTypeArray(cell_offsets, deep=True), numpy_to_vtkIdTypeArray(connectivity, deep=True))
            poly = vtkPolyData(); poly.SetPoints(points); poly.SetPolys(cells)
            source = pv.TrivialProducer(registrationName='Actual family ' + str(family))
            source.GetClientSideObject().SetOutput(poly); source.UpdatePipeline()
            fetched = servermanager.Fetch(source)
            if not np.array_equal(vtk_to_numpy(fetched.GetPoints().GetData()), xyz):
                raise ValueError('ParaView modified actual affine domain coordinates')
            display = pv.Show(source, view); display.Representation = 'Surface'
            display.ColorArrayName = ['POINTS', '']; display.DiffuseColor = display.AmbientColor = color
            display.Ambient = .35; display.Diffuse = .65; display.Specular = .15
            display.SetScalarBarVisibility(view, False); sources.append(source)
        pv.Render(view)
        path = folder / ('frame_%05d.png' % frame_index)
        pv.SaveScreenshot(str(path), view, ImageResolution=[size, size])
        with Image.open(path) as image:
            image = image.convert('RGB').resize((400, 400), Image.Resampling.LANCZOS)
        image.save(path); images.append(image); hashes.append(sha(path))
        for source in sources:
            pv.Delete(source)
        print(json.dumps(dict(frame=frame_index, **frame['record'])), flush=True)
    images[-1].save(args.output / 'weaving-prefix.png')
    images[0].save(args.output / 'weaving-prefix.gif', save_all=True, append_images=images[1:],
        duration=[args.frame_ms] * (len(images) - 1) + [max(600, args.frame_ms)], loop=0, disposal=2, optimize=False)
    receipt.update(renderer='ParaView', png_sha256=hashes,
        gif_sha256=sha(args.output / 'weaving-prefix.gif'), fetched_coordinates_equal_affine_saved_domains=True,
        visual_review_passed=False)
    (args.output / 'render.json').write_text(json.dumps(receipt, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=13)
    parser.add_argument('--start', type=float); parser.add_argument('--end', type=float)
    parser.add_argument('--arc-segments', type=int, default=8)
    parser.add_argument('--supersample', type=int, default=3)
    parser.add_argument('--frame-ms', type=int, default=150)
    parser.add_argument('--direction', type=float, nargs=3, default=[.18, -1.8, .6])
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.frames < 2 or args.arc_segments < 2 or args.supersample < 1 or args.frame_ms < 1:
        parser.error('frames>=2, arc-segments>=2 and positive rendering settings required')
    direction = np.asarray(args.direction)
    if not np.isfinite(direction).all() or np.linalg.norm(np.cross(direction, [0, 0, 1])) == 0:
        parser.error('camera direction must be finite and not parallel to Z')
    plan, frames, ids, offsets, domains, low, high = load(args)
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = dict(schema='actual-saved-weaving-paraview-v1', renderer_script_sha256=sha(__file__),
        states=[frame['record'] for frame in frames], full_protocol_duration_s=plan['duration_s'],
        displayed_time_window_s=[frames[0]['record']['time_s'], frames[-1]['record']['time_s']],
        completion_claim='Incomplete saved prefix; no full-process or crossing/contact success claim.',
        geometry='saved x + saved F times offsets of recovered reference annular-sector particle domains',
        deformation_scale=1., geometry_scale=1., failed_policy='hide only saved failed particles',
        reference_domains=domains, arc_segments_per_particle_sector=args.arc_segments,
        camera=camera(low, high, args.direction), fixed_camera=True,
        family_mapping='fibre_id // per_family', per_family=plan['per_family'], family_colors=COLORS,
        resolution=[400, 400], supersample=args.supersample, axes=False, annotations=False,
        interpolation=False, artificial_particle_displacement=False)
    (args.output / 'input-validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if args.validate_only:
        print(json.dumps(dict(valid=True, frames=len(frames), displayed_time_window_s=receipt['displayed_time_window_s'])), flush=True)
    else:
        render(args, frames, ids, offsets, receipt)


if __name__ == '__main__':
    main()
