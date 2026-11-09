"""Compatible quadrature prototype for a constant-stress straight crack.

Integrates the actual quadratic MPM basis over a rectangle cut by a straight
line. Bulk and face integrals are geometry quadrature; no residual fitting.
This is an audit prototype, not a production solver operator.
"""

from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mpm_jax.core import Config, stencil


_G, _W = np.polynomial.legendre.leggauss(3)
_G = 0.5 * (_G + 1.)
_W = 0.5 * _W


def _clip(poly, value, keep_positive, tol=1e-14):
    """Clip convex polygon by value(x)>=0 (or <=0), preserving vertices."""
    if len(poly) == 0:
        return np.empty((0, 2))
    out = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        va, vb = value(a), value(b)
        ia = va >= -tol if keep_positive else va <= tol
        ib = vb >= -tol if keep_positive else vb <= tol
        if ia:
            out.append(a)
        if ia != ib:
            t = va / (va - vb)
            out.append(a + t * (b - a))
    if not out:
        return np.empty((0, 2))
    return np.asarray(out)


def _split_knots(poly, dx):
    """Split at quadratic B-spline base changes x=(k+1/2)dx, y likewise."""
    pieces = [poly]
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    for axis in range(2):
        first = ceil(lo[axis] / dx - 0.5)
        last = int(np.floor(hi[axis] / dx - 0.5))
        for k in range(first, last + 1):
            knot = (k + 0.5) * dx
            next_pieces = []
            for p in pieces:
                if p[:, axis].min() < knot - 1e-13 and p[:, axis].max() > knot + 1e-13:
                    next_pieces.extend((_clip(p, lambda x, a=axis, k0=knot: x[a]-k0, True),
                                            _clip(p, lambda x, a=axis, k0=knot: x[a]-k0, False)))
                else:
                    next_pieces.append(p)
            pieces = [p for p in next_pieces if len(p) >= 3 and abs(_area(p)) > 1e-18]
    return pieces


def _area(poly):
    if len(poly) < 3:
        return 0.
    return .5 * abs(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1)
                           - poly[:, 1] * np.roll(poly[:, 0], -1)))


def _triangle_points(a, b, c):
    """3x3 Duffy tensor Gauss rule, exact for the piecewise cubic gradients."""
    u, v = b-a, c-a
    cross = abs(u[0]*v[1]-u[1]*v[0])
    pts, weights = [], []
    for u, wu in zip(_G, _W):
        for v, wv in zip(_G, _W):
            pts.append(a + u*(b-a) + (1-u)*v*(c-a))
            weights.append(cross*(1-u)*wu*wv)
    return np.asarray(pts), np.asarray(weights)


def _integrate_basis(points, weights, config, want_grad):
    """Accumulate basis integrals onto the flattened grid using core.stencil."""
    n = (config.n_grid + 1)**2
    values = np.zeros((n, 2)) if want_grad else np.zeros((n,))
    block = 2048
    for start in range(0, len(points), block):
        p = points[start:start+block]
        q = stencil(jnp.asarray(p), config)
        if not bool(q.valid):
            raise ValueError('quadrature points reached an invalid grid stencil')
        indices = np.asarray(q.indices).reshape(-1)
        factors = np.repeat(weights[start:start+block], q.indices.shape[1])
        data = np.asarray(q.gradients if want_grad else q.weights).reshape((-1, 2) if want_grad else (-1,))
        np.add.at(values, indices, factors[:, None]*data if want_grad else factors*data)
    return values


def _bulk_integral(poly, config):
    pts, weights = [], []
    for piece in _split_knots(poly, config.dx):
        origin = piece[0]
        for i in range(1, len(piece)-1):
            p, w = _triangle_points(origin, piece[i], piece[i+1])
            pts.extend(p); weights.extend(w)
    return _integrate_basis(np.asarray(pts), np.asarray(weights), config, True)


def _line_interval(point, direction, bounds):
    lower, upper = np.asarray(bounds[0]), np.asarray(bounds[1])
    t0, t1 = -np.inf, np.inf
    for axis in range(2):
        if abs(direction[axis]) < 1e-14:
            if point[axis] < lower[axis] or point[axis] > upper[axis]:
                return None
            continue
        a, b = (lower[axis]-point[axis])/direction[axis], (upper[axis]-point[axis])/direction[axis]
        t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
    return (t0, t1) if t1 > t0 else None


def _face_integral(point, direction, config, bounds):
    interval = _line_interval(point, direction, bounds)
    if interval is None:
        raise ValueError('crack line does not cross rectangle')
    lo, hi = interval
    cuts = [lo, hi]
    for axis in range(2):
        if abs(direction[axis]) < 1e-14:
            continue
        low, high = point[axis]+lo*direction[axis], point[axis]+hi*direction[axis]
        first = ceil(min(low, high)/config.dx - 0.5)
        last = int(np.floor(max(low, high)/config.dx - 0.5))
        for k in range(first, last+1):
            knot = (k+0.5)*config.dx
            t = (knot-point[axis])/direction[axis]
            if lo+1e-13 < t < hi-1e-13:
                cuts.append(t)
    cuts = sorted(set(cuts))
    pts, weights = [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        ts = a+(b-a)*_G
        pts.extend(point[None, :]+ts[:, None]*direction)
        weights.extend((b-a)*_W)  # direction is unit length, so ds=dt
    return _integrate_basis(np.asarray(pts), np.asarray(weights), config, False)


def audit(n_grid=16, offset=0.0, angle=0.0):
    """Return discrete equilibrium and geometric invariants for one cut."""
    jax.config.update('jax_enable_x64', True)
    config = Config(dim=2, n_grid=n_grid, domain_size=1., dt=1e-5, gravity=(0., 0.), boundary_cells=0)
    direction = np.array([np.cos(angle), np.sin(angle)])
    normal = np.array([-direction[1], direction[0]])  # points to plus half-plane
    point = np.array([.5, .5]) + offset*config.dx*normal
    body_lo, body_hi = .125, .875
    base = np.array([[body_lo, body_lo], [body_hi, body_lo],
                     [body_hi, body_hi], [body_lo, body_hi]])
    plus_poly = _clip(base, lambda x: np.dot(x-point, normal), True)
    minus_poly = _clip(base, lambda x: np.dot(x-point, normal), False)
    bulk_plus, bulk_minus = _bulk_integral(plus_poly, config), _bulk_integral(minus_poly, config)
    bounds = (np.array([body_lo, body_lo]), np.array([body_hi, body_hi]))
    face = _face_integral(point, direction, config, bounds)
    intact = _bulk_integral(base, config)
    # Symmetric constant stress with nonzero shear and unequal normal components.
    sigma = np.array([[2.3, -.7], [-.7, 1.1]])
    coords = np.array([(i*config.dx, j*config.dx) for i in range(n_grid+1) for j in range(n_grid+1)])
    force_plus = -bulk_plus@sigma.T - face[:, None]*(sigma@normal)[None, :]
    force_minus = -bulk_minus@sigma.T + face[:, None]*(sigma@normal)[None, :]
    residual = force_plus+force_minus
    interior = np.all((coords >= body_lo+1.5*config.dx-1e-12)
                      & (coords <= body_hi-1.5*config.dx+1e-12), axis=1)
    # Each separated field is balanced independently at interior nodes.
    err_plus = np.linalg.norm(force_plus[interior]) / max(np.linalg.norm((bulk_plus@sigma.T)[interior]), 1e-30)
    err_minus = np.linalg.norm(force_minus[interior]) / max(np.linalg.norm((bulk_minus@sigma.T)[interior]), 1e-30)
    face_force = face @ coords
    interval = _line_interval(point, direction, bounds)
    length = interval[1]-interval[0]
    midpoint = point + .5*(interval[0]+interval[1])*direction
    face_plus = -face[:, None]*(sigma@normal)[None, :]
    face_minus = -face_plus
    interface_resultant = face_plus.sum(axis=0)+face_minus.sum(axis=0)
    interface_moment = np.sum(coords[:, 0]*face_plus[:, 1]-coords[:, 1]*face_plus[:, 0]
                              + coords[:, 0]*face_minus[:, 1]-coords[:, 1]*face_minus[:, 0])
    actual_face_moment = np.sum(coords[:, 0]*face_plus[:, 1]-coords[:, 1]*face_plus[:, 0])
    actual_face_resultant = face_plus.sum(axis=0)
    result = dict(n_grid=n_grid, offset_over_dx=offset, angle_degrees=float(np.degrees(angle)),
                  plus_relative_interior_residual=float(err_plus), minus_relative_interior_residual=float(err_minus),
                  face_partition_error=float(abs(face.sum()-length)),
                  face_first_moment_error=float(np.linalg.norm(face_force-length*midpoint)),
                  bulk_partition_error=float(np.linalg.norm(bulk_plus+bulk_minus-intact)),
                  pair_resultant_error=float(np.linalg.norm(interface_resultant)),
                  pair_moment_error=float(abs(interface_moment)),
                  face_resultant_error=float(np.linalg.norm(actual_face_resultant+length*(sigma@normal))),
                  face_moment_error=float(abs(actual_face_moment
                                               + (midpoint[0]*(sigma@normal)[1]
                                                  - midpoint[1]*(sigma@normal)[0])*length)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--grid', type=int, default=16)
    args = parser.parse_args()
    rows = [audit(args.grid, offset, angle) for angle in (0., .31, .68)
            for offset in (-.375, 0., .25)]
    payload = dict(method='knot-split polygon/triangle bulk quadrature; knot-split Gauss face quadrature',
                   stress_dependent_fitting=False, rows=rows)
    rendered = json.dumps(payload, indent=2)
    if args.output:
        with open(args.output, 'w', encoding='utf8') as stream:
            stream.write(rendered+'\n')
    print(rendered)


if __name__ == '__main__':
    main()
