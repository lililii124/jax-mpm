"""Finite straight edge-crack patch audit for endpoint visibility quadrature.

Host-side exact-geometry quadrature uses core.stencil for MPM basis values and
gradients. It compares legacy node visibility with the optional tip-support
tie map; it is a patch audit, not a dynamics or crack-tip benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mpm_jax.core import Config, stencil
from mpm_jax.polyline_crack import link_fields, shifted_nodes
from tools.audit_cut_interface import _area, _clip, _line_interval, _split_knots, _triangle_points, _G, _W


def _split_plane(poly, normal, offset):
    if len(poly) < 3:
        return []
    fn = lambda x: float(np.dot(normal, x)-offset)
    sides = (_clip(poly, fn, True), _clip(poly, fn, False))
    return [p for p in sides if len(p) >= 3 and _area(p) > 1e-18]


def _support_polygon(node, body, dx):
    x, y = node
    poly = body.copy()
    for axis, low, high in ((0, x-1.5*dx, x+1.5*dx), (1, y-1.5*dx, y+1.5*dx)):
        e = np.eye(2)[axis]
        poly = _clip(poly, lambda q, a=axis, l=low: q[a]-l, True)
        poly = _clip(poly, lambda q, a=axis, h=high: q[a]-h, False)
    if len(poly) < 3 or _area(poly) < 1e-18:
        return np.empty((0, 2))
    return poly


def _split_target_support(node, body, root, tip, dx):
    """Clip one B-spline support, then cut all parity and knot boundaries."""
    poly = _support_polygon(node, body, dx)
    if len(poly) < 3:
        return []
    pieces = _split_knots(poly, dx)
    crack = tip-root
    direction = crack/np.linalg.norm(crack)
    crack_normal = np.array([-direction[1], direction[0]])
    lines = [(crack_normal, np.dot(crack_normal, root))]
    for endpoint in (root, tip):
        ray = node-endpoint
        if np.linalg.norm(ray) > 1e-14:
            n = np.array([-ray[1], ray[0]])/np.linalg.norm(ray)
            lines.append((n, np.dot(n, endpoint)))
    for normal, offset in lines:
        pieces = [q for p in pieces for q in _split_plane(p, normal, offset)]
    return pieces


def _bulk_quadrature(nodes, root, tip, config, tied):
    body_lo, body_hi = .125, .875
    body = np.array([[body_lo, body_lo], [body_hi, body_lo],
                     [body_hi, body_hi], [body_lo, body_hi]])
    dx = config.dx
    eligible = np.all((nodes >= body_lo+1.5*dx-1e-12)
                      & (nodes <= body_hi-1.5*dx+1e-12), axis=1)
    targets, points, weights = [], [], []
    for node_id in np.flatnonzero(eligible):
        for poly in _split_target_support(nodes[node_id], body, root, tip, dx):
            origin = poly[0]
            for i in range(1, len(poly)-1):
                q, w = _triangle_points(origin, poly[i], poly[i+1])
                points.extend(q)
                weights.extend(w)
                targets.extend([node_id]*len(q))
    points, weights, targets = np.asarray(points), np.asarray(weights), np.asarray(targets)
    sigma = np.array([[2.3, -.7], [-.7, 1.1]])
    out = np.zeros((len(nodes), 2, 2))  # node, field, force component
    vis_nodes = shifted_nodes(jnp.asarray(nodes), jnp.asarray([root, tip]), jnp.asarray(2))
    for start in range(0, len(points), 2048):
        stop = min(start+2048, len(points))
        p = points[start:stop]
        q = stencil(jnp.asarray(p), config)
        if not bool(q.valid):
            raise ValueError('bulk quadrature reached an invalid stencil')
        f = link_fields(jnp.asarray(p), q, jnp.asarray([root, tip]), jnp.asarray(2),
                        vis_nodes[q.indices], tip_support_dx=config.dx if tied else None)
        index = np.asarray(q.indices)
        is_target = index == targets[start:stop, None]
        grads = np.asarray(q.gradients)
        target_grad = np.sum(grads*is_target[..., None], axis=1)
        target_field = np.sum(np.asarray(f)*is_target, axis=1).astype(int)
        target_ids = targets[start:stop]
        contribution = -np.einsum('pj,ij->pi', target_grad, sigma)
        np.add.at(out, (target_ids, target_field), weights[start:stop, None]*contribution)
    return out, eligible


def _face_quadrature(nodes, root, tip, config, tied):
    dx = config.dx
    body_lo, body_hi = .125, .875
    bounds = (np.array([body_lo, body_lo]), np.array([body_hi, body_hi]))
    vector = tip-root
    length = np.linalg.norm(vector)
    direction = vector/length
    normal = np.array([-direction[1], direction[0]])
    interval = _line_interval(root, direction, bounds)
    t0, t1 = max(0., interval[0]), min(length, interval[1])
    if t1 <= t0:
        raise ValueError('physical crack has no segment inside body')
    cuts = [t0, t1]
    for axis in range(2):
        if abs(direction[axis]) < 1e-14:
            continue
        lo, hi = root[axis]+t0*direction[axis], root[axis]+t1*direction[axis]
        first = int(np.ceil(min(lo, hi)/dx-.5))
        last = int(np.floor(max(lo, hi)/dx-.5))
        for k in range(first, last+1):
            t = ((k+.5)*dx-root[axis])/direction[axis]
            if t0+1e-13 < t < t1-1e-13:
                cuts.append(t)
    cuts = sorted(set(cuts))
    params, ds = [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        params.extend(a+(b-a)*_G)
        ds.extend((b-a)*_W)
    positions = root+np.asarray(params)[:, None]*direction
    jax_positions = jnp.asarray(positions)
    vis_nodes = shifted_nodes(jnp.asarray(nodes), jnp.asarray([root, tip]), jnp.asarray(2))
    sigma = np.array([[2.3, -.7], [-.7, 1.1]])
    traction = sigma@normal
    out = np.zeros((2, len(nodes), 2, 2))
    # The infinitesimal side probes implement the same one-sided trace used by
    # face_velocity; their displacement is far below quadrature resolution.
    scale = max(1., float(np.max(np.abs(nodes))), float(np.max(np.abs(root))),
                float(np.max(np.abs(tip))))
    eps = 256*np.finfo(float).eps*scale
    for side_index, (side, sign) in enumerate(((1., -1.), (-1., 1.))):
        support = stencil(jax_positions, config)
        if not bool(support.valid):
            raise ValueError('face quadrature reached an invalid stencil')
        fields = link_fields(jax_positions+side*eps*jnp.asarray(normal), support,
                             jnp.asarray([root, tip]), jnp.asarray(2),
                             vis_nodes[support.indices], tip_support_dx=dx if tied else None)
        ids = np.asarray(support.indices).reshape(-1)
        field_ids = np.asarray(fields).reshape(-1)
        factors = (np.asarray(support.weights)*np.asarray(ds)[:, None]).reshape(-1)
        force = sign*traction
        np.add.at(out, (side_index, ids, field_ids), factors[:, None]*force[None, :])
    return out, (t0, t1)


def audit(n_grid=16, angle=0., offset=0.):
    """Compare legacy and endpoint-tied cut/face forces for a finite edge crack."""
    jax.config.update('jax_enable_x64', True)
    config = Config(dim=2, n_grid=n_grid, domain_size=1., dt=1e-5,
                    gravity=(0., 0.), boundary_cells=0)
    dx = config.dx
    direction = np.array([np.cos(angle), np.sin(angle)])
    normal = np.array([-direction[1], direction[0]])
    tip = np.array([.55, .5])+offset*dx*normal
    root = tip-.8*direction  # safely outside the body, including inclined cases
    idx = np.arange((n_grid+1)**2)
    nodes = dx*np.stack((idx//(n_grid+1), idx%(n_grid+1)), axis=-1)
    sigma = np.array([[2.3, -.7], [-.7, 1.1]])
    rows = {}
    body_lo, body_hi = .125, .875
    eligible = np.all((nodes >= body_lo+1.5*dx-1e-12)
                      & (nodes <= body_hi-1.5*dx+1e-12), axis=1)
    intact = _intact_bulk_for_nodes(nodes, eligible, config, sigma)
    for tied in (False, True):
        bulk, eligible = _bulk_quadrature(nodes, root, tip, config, tied)
        face_sides, face_interval = _face_quadrature(nodes, root, tip, config, tied)
        face = face_sides.sum(axis=0)
        residual = bulk+face
        # Only compare nodal supports fully inside the specimen body.
        scale = max(np.linalg.norm(bulk[eligible]), 1e-30)
        face_length = face_interval[1]-face_interval[0]
        face_midpoint = root+.5*(face_interval[0]+face_interval[1])*direction
        face_resultants = face_sides.sum(axis=(1, 2))
        face_moments = np.sum(nodes[None, :, 0, None]*face_sides[:, :, :, 1]
                              - nodes[None, :, 1, None]*face_sides[:, :, :, 0], axis=(1, 2))
        expected_plus = -sigma@normal*face_length
        expected_minus = sigma@normal*face_length
        expected_plus_moment = -(face_midpoint[0]*(sigma@normal)[1]
                                 - face_midpoint[1]*(sigma@normal)[0])*face_length
        rows['tied' if tied else 'legacy'] = dict(
            relative_interior_residual=float(np.linalg.norm(residual[eligible])/scale),
            max_interior_residual=float(np.max(np.linalg.norm(residual[eligible], axis=(1, 2)))),
            physical_face_length=float(face_length),
            pair_resultant_error=float(np.linalg.norm(face.sum(axis=(0, 1)))),
            face_resultant_error=float(max(np.linalg.norm(face_resultants[0]-expected_plus),
                                           np.linalg.norm(face_resultants[1]-expected_minus))),
            face_moment_error=float(max(abs(face_moments[0]-expected_plus_moment),
                                        abs(face_moments[1]+expected_plus_moment))),
            bulk_partition_error=float(np.linalg.norm(bulk.sum(axis=1)[eligible]
                                                       - intact[eligible])))
    return dict(n_grid=n_grid, angle_degrees=float(np.degrees(angle)), offset_over_dx=offset,
                nodes_checked=int(np.count_nonzero(eligible)), legacy=rows['legacy'], tied=rows['tied'])


def _intact_bulk_for_nodes(nodes, eligible, config, sigma):
    body_lo, body_hi = .125, .875
    body = np.array([[body_lo, body_lo], [body_hi, body_lo],
                     [body_hi, body_hi], [body_lo, body_hi]])
    out = np.zeros((len(nodes), 2))
    targets, points, weights = [], [], []
    for node_id in np.flatnonzero(eligible):
        support_poly = _support_polygon(nodes[node_id], body, config.dx)
        for poly in _split_knots(support_poly, config.dx):
            origin = poly[0]
            for i in range(1, len(poly)-1):
                pts, w = _triangle_points(origin, poly[i], poly[i+1])
                points.extend(pts); weights.extend(w); targets.extend([node_id]*len(pts))
    points, weights, targets = np.asarray(points), np.asarray(weights), np.asarray(targets)
    for start in range(0, len(points), 2048):
        stop = min(start+2048, len(points))
        support = stencil(jnp.asarray(points[start:stop]), config)
        if not bool(support.valid):
            raise ValueError('intact quadrature reached an invalid stencil')
        mask = np.asarray(support.indices) == targets[start:stop, None]
        grad = np.sum(np.asarray(support.gradients)*mask[..., None], axis=1)
        force = -np.einsum('pj,ij->pi', grad, sigma)
        np.add.at(out, targets[start:stop], weights[start:stop, None]*force)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--grid', type=int, default=16)
    args = parser.parse_args()
    rows = [audit(args.grid, angle, offset) for angle, offset in ((0., 0.), (.31, .25))]
    payload = dict(method='polygon cut quadrature split by B-spline knots, crack line and endpoint rays; physical face line quadrature',
                   stress_dependent_fitting=False, rows=rows)
    rendered = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(rendered+'\n', encoding='utf8')
    print(rendered)


if __name__ == '__main__':
    main()
