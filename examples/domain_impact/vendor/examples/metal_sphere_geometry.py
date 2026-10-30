"""Two deformable aluminium-like bodies for a 5000 m/s debris-impact example."""

import math

import numpy as np

from .metal_plate_geometry import sample_metal_plate


def sample_sphere(*, diameter=.002, spacing=(.0001, .00005, .0001),
                  center=(0., .00155, 0.), subdivisions=4, max_particles=1_000_000):
    """One volume centroid per occupied Cartesian cell; cut cells use subcells."""
    spacing, center = map(lambda a: np.asarray(a, dtype=np.float64), (spacing, center))
    if (spacing.shape != (3,) or center.shape != (3,) or not np.isfinite(center).all()
            or not np.isfinite(spacing).all() or np.any(spacing <= 0)
            or not np.isfinite(diameter) or diameter <= 0
            or type(subdivisions) is not int or not 2 <= subdivisions <= 8
            or type(max_particles) is not int or max_particles < 1):
        raise ValueError('Positive dimensions, a finite center, 2..8 subcells and a positive budget are required')
    ratios = diameter/spacing
    if not np.isfinite(ratios).all() or np.any(ratios > max_particles):
        raise ValueError('Sphere bounding cells exceed the particle budget before allocation')
    counts = np.rint(ratios).astype(np.int64)
    if np.any(counts < 2) or not np.allclose(counts*spacing, diameter, rtol=0, atol=1e-10*spacing):
        raise ValueError('Each sphere bounding-box length must contain an integer number of at least two cells')
    bounding_count = math.prod(map(int, counts))
    if bounding_count > max_particles:
        raise ValueError('Sphere bounding cells exceed the particle budget before allocation')
    axes = [(np.arange(n)+.5)*spacing[d]-diameter/2 for d, n in enumerate(counts)]
    cells = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    fraction = (np.arange(subdivisions)+.5)/subdivisions-.5
    offsets = np.stack(np.meshgrid(fraction, fraction, fraction, indexing='ij'), axis=-1).reshape(-1, 3)*spacing
    positions, volumes = [], []
    for start in range(0, bounding_count, 1024):
        points = cells[start:start+1024, None, :]+offsets
        inside = np.sum(points**2, axis=-1) < (diameter/2)**2
        count = inside.sum(axis=1)
        occupied = count > 0
        centroid = (points*inside[..., None]).sum(axis=1)/np.maximum(count, 1)[:, None]
        positions.append(centroid[occupied]+center)
        volumes.append(count[occupied]*(np.prod(spacing)/subdivisions**3))
    x, volume = np.concatenate(positions), np.concatenate(volumes)
    analytic_volume = np.pi*diameter**3/6
    return dict(x=x, volume=volume, metadata=dict(
        diameter_m=float(diameter), center_m=center.tolist(), spacing_m=spacing.tolist(),
        subdivisions=subdivisions, bounding_cell_count=bounding_count, particle_count=len(x),
        volume_m3=float(volume.sum()), analytic_volume_m3=float(analytic_volume),
        volume_relative_error=float(volume.sum()/analytic_volume-1),
        sampling='Occupied subcell volume and centroid; no mass renormalization or rigid-body constraint'))


def plan_case(*, speed=5000., sphere_diameter=.002, plate_size=(.02, .02),
              plate_thickness=.0005, gap=.0003, clamp_width=.001,
              dx=.0001, spacing=(.0001, .00005, .0001), density=2700.,
              end_time=1e-6, frames=60, max_particles=1_000_000):
    """NumPy-only geometry and fixed protocol; the shock solver is a separate stage."""
    plate_size, spacing = np.asarray(plate_size, dtype=float), np.asarray(spacing, dtype=float)
    positive = [speed, sphere_diameter, plate_thickness, gap, clamp_width, dx, density, end_time]
    if (plate_size.shape != (2,) or spacing.shape != (3,) or not np.isfinite(positive).all()
            or min(positive) <= 0 or not np.isfinite(plate_size).all() or np.any(plate_size <= 0)
            or not np.isfinite(spacing).all() or np.any(spacing <= 0)
            or type(frames) is not int or frames < 2
            or type(max_particles) is not int or max_particles < 1):
        raise ValueError('Finite positive geometry, speed, density, duration and integer budgets are required')
    if np.any(plate_size <= sphere_diameter+2*clamp_width) or 2*clamp_width >= plate_size.min():
        raise ValueError('The sphere must fit inside the unclamped plate span')
    if np.any(spacing > dx) or gap < 3*dx*(1-1e-12):
        raise ValueError('Sampling cannot exceed dx; initial physical gap must be at least three grid cells')
    lengths = np.array([plate_size[0], plate_thickness, plate_size[1]])
    plate_ratios, sphere_ratios = lengths/spacing, sphere_diameter/spacing
    if (not np.isfinite(plate_ratios).all() or not np.isfinite(sphere_ratios).all()
            or np.any(plate_ratios > max_particles) or np.any(sphere_ratios > max_particles)):
        raise ValueError('Combined geometry exceeds the particle budget before allocation')
    plate_counts = np.rint(plate_ratios).astype(np.int64)
    sphere_counts = np.rint(sphere_ratios).astype(np.int64)
    if (not np.allclose(plate_counts*spacing, lengths, rtol=0, atol=1e-10*spacing)
            or not np.allclose(sphere_counts*spacing, sphere_diameter, rtol=0, atol=1e-10*spacing)):
        raise ValueError('Explicit particle spacings must divide plate and sphere bounding dimensions')
    plate_count, sphere_bound = math.prod(map(int, plate_counts)), math.prod(map(int, sphere_counts))
    if plate_count+sphere_bound > max_particles:
        raise ValueError('Combined geometry exceeds the particle budget before allocating either body')
    plate = sample_metal_plate(size=plate_size, thickness=plate_thickness, spacing=spacing,
                               max_particles=max_particles-sphere_bound)
    center = (0., plate_thickness/2+gap+sphere_diameter/2, 0.)
    sphere = sample_sphere(diameter=sphere_diameter, spacing=spacing, center=center,
                           max_particles=max_particles-plate_count)
    x = np.concatenate((sphere['x'], plate.positions))
    volume = np.concatenate((sphere['volume'], plate.volume))
    ids = np.concatenate((np.zeros(len(sphere['x']), dtype=np.int32), np.ones(plate_count, dtype=np.int32)))
    velocity = np.zeros_like(x)
    velocity[ids == 0, 1] = -speed
    impulse = (density*volume[:, None]*velocity).sum(axis=0)
    kinetic = .5*np.sum(density*volume*np.sum(velocity**2, axis=1))
    plan = dict(
        case='deformable-metal-sphere-impact', units='SI', speed_m_s=float(speed),
        material='Shared illustrative Al-like shock/strength parameters for sphere and plate; no rigid sphere',
        density_kg_m3=float(density), dx_m=float(dx), particle_spacing_m=spacing.tolist(),
        sphere=sphere['metadata'], plate=dict(plate.metadata),
        boundary=dict(type='Four plate edges fixed in all three displacement components',
                      clamp_width_m=float(clamp_width), backing=False),
        initial_physical_gap_m=float(gap), initial_sphere_velocity_m_s=[0., -float(speed), 0.],
        initial_momentum_kg_m_s=impulse.tolist(), initial_kinetic_energy_J=float(kinetic),
        sphere_mass_kg=float(density*sphere['volume'].sum()), plate_mass_kg=float(density*plate.volume.sum()),
        particle_count=len(x), end_time_s=float(end_time), frames=frames,
        output_times_s=np.linspace(0, end_time, frames+1).tolist(),
        interpretation='Geometry and requested protocol only. Requires actual spatial shock/EOS/strength verification and domain checks before rendering; no fracture, phase-mixture or experimental-validation claim.')
    return dict(x=x, volume=volume, body_id=ids, v=velocity), plan
