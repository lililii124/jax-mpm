"""Volume samples for a homogeneous rectangular metal plate."""

import numpy as np

from examples.woven_geometry import WeaveSample


def solid_sphere_mass(diameter, density=7830.):
    if not np.isfinite([diameter, density]).all() or diameter <= 0 or density <= 0:
        raise ValueError("positive finite diameter and density are required")
    return float(density*np.pi*diameter**3/6)


def sample_metal_plate(*, size=(.1, .1), thickness=.002,
                       spacing=(.0005, .00025, .0005), origin=(0., 0., 0.),
                       max_particles=1_000_000):
    size, spacing, origin = (np.asarray(value, dtype=np.float64) for value in (size, spacing, origin))
    if size.shape != (2,) or spacing.shape != (3,) or origin.shape != (3,):
        raise ValueError("size, spacing and origin require 2, 3 and 3 coordinates")
    if (not all(np.isfinite(a).all() for a in (size, spacing, origin))
            or not np.isfinite(thickness) or thickness <= 0
            or np.any(size <= 0) or np.any(spacing <= 0)
            or not isinstance(max_particles, int) or isinstance(max_particles, bool)
            or max_particles < 1):
        raise ValueError("invalid plate dimensions, spacing, origin or particle budget")
    lengths = np.array((size[0], thickness, size[1]), dtype=np.float64)
    ratios = lengths/spacing
    if np.any(ratios > max_particles):
        raise ValueError("particle budget exceeded before allocation")
    counts = np.rint(ratios).astype(np.int64)
    if np.any(counts < 1) or not np.allclose(counts*spacing, lengths, rtol=0, atol=1e-10*spacing):
        raise ValueError("each explicit spacing must divide its specimen length")
    n = int(np.prod(counts, dtype=object))
    if n > max_particles:
        raise ValueError(f"particle count {n} exceeds budget {max_particles}")
    axes = [-lengths[d]/2+(np.arange(counts[d])+.5)*spacing[d] for d in range(3)]
    nx, ny, nz = map(int, counts)
    x = np.empty((n, 3), dtype=np.float64)
    x[:, 0] = np.repeat(axes[0], ny*nz)+origin[0]
    x[:, 1] = np.tile(np.repeat(axes[1], nz), nx)+origin[1]
    x[:, 2] = np.tile(axes[2], nx*ny)+origin[2]
    volume = np.full(n, float(np.prod(spacing)), dtype=np.float64)
    return WeaveSample(
        x, volume, np.zeros(n, dtype=np.uint8), np.full(n, -1, dtype=np.int32),
        np.full(n, -1, dtype=np.int8), np.broadcast_to(np.eye(3), (n, 3, 3)).copy(),
        {"geometry_type": "homogeneous-metal-plate", "geometry_units": "m",
         "size_xz_m": size.tolist(), "thickness_m": float(thickness),
         "origin_m": origin.tolist(), "particle_spacing_m": spacing.tolist(),
         "particle_spacing_equivalent_m": float(np.prod(spacing)**(1/3)),
         "sampling": "explicit anisotropic Cartesian volume lattice",
         "lattice_counts": counts.tolist(), "particle_count": n,
         "particle_budget": max_particles, "reference_volume_m3": float(volume.sum()),
         "phase_definition": "0 homogeneous metal; no yarn or matrix partition",
         "orientation_definition": "identity bookkeeping field; isotropic metal has no material axes"},
    )
