"""Rigidly exact contact between advected material-particle cuboids.

The cuboids are the initial quadrature cells, with half sizes set by the
material sampling spacings.  Current domains are their affine images under F.
No background-grid support or MLS extrapolation participates in broadphase or
the contact point velocity.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np


_SIGNS = np.asarray(tuple(product((-1., 1.), repeat=3)), dtype=np.float64)
_EDGE_PAIRS = tuple((i, j) for i in range(8) for j in range(i+1, 8)
                    if np.count_nonzero(_SIGNS[i] != _SIGNS[j]) == 1)


@dataclass(frozen=True)
class DomainIntersection:
    first: int
    second: int
    point: np.ndarray
    normal: np.ndarray
    xi_first: np.ndarray
    xi_second: np.ndarray
    penetration: float


def _world_corners(center, F, half):
    local = _SIGNS * np.asarray(half)[None, :]
    return np.asarray(center)[None, :] + local @ np.asarray(F).T


def _inside(point, center, inverse_F, half, tol):
    xi = inverse_F @ (np.asarray(point)-np.asarray(center))
    return bool(np.all(np.abs(xi) <= np.asarray(half)+tol)), xi


def _edges(corners):
    return ((corners[i], corners[j]) for i, j in _EDGE_PAIRS)


def _append_unique(vertices, point, tol):
    point = np.asarray(point, dtype=np.float64)
    if not any(np.linalg.norm(point-old) <= tol for old in vertices):
        vertices.append(point)


def _edge_face_vertices(edge_owner_center, edge_owner_F, edge_owner_half,
                        face_center, face_inverse_F, face_half, corners,
                        vertices, tol):
    for p0, p1 in _edges(corners):
        a = face_inverse_F @ (p0-face_center)
        b = face_inverse_F @ (p1-face_center)
        d = b-a
        for axis in range(3):
            if abs(d[axis]) <= 1e-14:
                continue
            for side in (-1., 1.):
                t = (side*face_half[axis]-a[axis])/d[axis]
                if t < -1e-12 or t > 1+1e-12:
                    continue
                q = p0 + np.clip(t, 0., 1.)*(p1-p0)
                local = face_inverse_F @ (q-face_center)
                other = [k for k in range(3) if k != axis]
                if all(abs(local[k]) <= face_half[k]+tol for k in other):
                    # Also check that the point lies on the source edge of its
                    # owner. It is constructed parametrically, so this is true.
                    _append_unique(vertices, q, tol)


def parallelepiped_intersection(center_a, F_a, half_a,
                                center_b, F_b, half_b,
                                *, relative_tolerance=1e-12):
    """Return SAT MTV normal/depth and mean of true convex-intersection vertices.

    `normal` points from A's center toward B's center. A nonintersection
    returns None. SAT axes are the 3 face normals of each cuboid and all 9
    cross products of edge directions.
    """
    ca, cb = np.asarray(center_a, float), np.asarray(center_b, float)
    Fa, Fb = np.asarray(F_a, float), np.asarray(F_b, float)
    ha, hb = np.asarray(half_a, float), np.asarray(half_b, float)
    if (ca.shape != (3,) or cb.shape != (3,) or Fa.shape != (3, 3)
            or Fb.shape != (3, 3) or ha.shape != (3,) or hb.shape != (3,)
            or np.any(ha <= 0) or np.any(hb <= 0)):
        raise ValueError('Expected finite 3D centers, nonsingular F and positive half sizes')
    if not np.isfinite(np.r_[ca, cb, Fa.ravel(), Fb.ravel(), ha, hb]).all():
        raise ValueError('Nonfinite material-domain geometry')
    det_a, det_b = np.linalg.det(Fa), np.linalg.det(Fb)
    if det_a <= 0 or det_b <= 0:
        raise FloatingPointError('Inverted or singular material-domain deformation')
    scale = max(float(np.max(ha)), float(np.max(hb)),
                float(np.linalg.norm(cb-ca)), 1e-30)
    tol = relative_tolerance*scale
    ea, eb = Fa*ha[None, :], Fb*hb[None, :]
    axes = [np.cross(ea[:, (i+1) % 3], ea[:, (i+2) % 3]) for i in range(3)]
    axes += [np.cross(eb[:, (i+1) % 3], eb[:, (i+2) % 3]) for i in range(3)]
    axes += [np.cross(ea[:, i], eb[:, j]) for i in range(3) for j in range(3)]
    min_depth = np.inf
    min_axis = None
    delta = cb-ca
    for axis in axes:
        norm = np.linalg.norm(axis)
        if norm <= 1e-14*max(np.linalg.norm(ea), np.linalg.norm(eb), 1e-30):
            continue
        axis = axis/norm
        ra = sum(abs(float(np.dot(axis, ea[:, i]))) for i in range(3))
        rb = sum(abs(float(np.dot(axis, eb[:, i]))) for i in range(3))
        signed = float(np.dot(delta, axis))
        depth = ra+rb-abs(signed)
        if depth < -tol:
            return None
        if depth < min_depth:
            min_depth, min_axis = depth, (axis if signed >= 0 else -axis)
    if min_axis is None:
        raise FloatingPointError('No valid SAT axes for material domains')

    inv_a, inv_b = np.linalg.inv(Fa), np.linalg.inv(Fb)
    corners_a, corners_b = _world_corners(ca, Fa, ha), _world_corners(cb, Fb, hb)
    vertices = []
    for point in corners_a:
        inside, _ = _inside(point, cb, inv_b, hb, tol)
        if inside:
            _append_unique(vertices, point, tol)
    for point in corners_b:
        inside, _ = _inside(point, ca, inv_a, ha, tol)
        if inside:
            _append_unique(vertices, point, tol)
    _edge_face_vertices(ca, Fa, ha, cb, inv_b, hb, corners_a, vertices, tol)
    _edge_face_vertices(cb, Fb, hb, ca, inv_a, ha, corners_b, vertices, tol)
    if not vertices:
        # SAT admitted a touch/intersection, so an empty vertex set indicates
        # a degenerate or inconsistent polygon clip and must fail closed.
        raise FloatingPointError('SAT reports intersection but no clipped vertices were found')
    point = np.mean(vertices, axis=0)
    xi_a, xi_b = inv_a@(point-ca), inv_b@(point-cb)
    if np.any(np.abs(xi_a) > ha+tol) or np.any(np.abs(xi_b) > hb+tol):
        raise FloatingPointError('Clipped contact point lies outside a material cuboid')
    return min_axis/np.linalg.norm(min_axis), float(max(min_depth, 0.)), point, xi_a, xi_b


def parallelepiped_sat_gap(center_a, F_a, half_a, center_b, F_b, half_b):
    """Maximum separating-axis gap (positive means disjoint, <=0 intersects)."""
    ca, cb = np.asarray(center_a, float), np.asarray(center_b, float)
    Fa, Fb = np.asarray(F_a, float), np.asarray(F_b, float)
    ha, hb = np.asarray(half_a, float), np.asarray(half_b, float)
    ea, eb = Fa*ha[None, :], Fb*hb[None, :]
    axes = [np.cross(ea[:, (i+1) % 3], ea[:, (i+2) % 3]) for i in range(3)]
    axes += [np.cross(eb[:, (i+1) % 3], eb[:, (i+2) % 3]) for i in range(3)]
    axes += [np.cross(ea[:, i], eb[:, j]) for i in range(3) for j in range(3)]
    delta = cb-ca
    gaps = []
    for axis in axes:
        norm = np.linalg.norm(axis)
        if norm <= 1e-14*max(np.linalg.norm(ea), np.linalg.norm(eb), 1e-30):
            continue
        axis = axis/norm
        ra = sum(abs(float(np.dot(axis, ea[:, i]))) for i in range(3))
        rb = sum(abs(float(np.dot(axis, eb[:, i]))) for i in range(3))
        gaps.append(abs(float(np.dot(delta, axis)))-ra-rb)
    if not gaps:
        raise FloatingPointError('No valid SAT axes')
    return float(max(gaps))


def material_domain_candidates(x, F, body_id, spacing, *, max_pairs=8192):
    """Current true material-cuboid intersections via cKDTree + 15-axis SAT."""
    from scipy.spatial import cKDTree
    x, F = np.asarray(x, float), np.asarray(F, float)
    body_id = np.asarray(body_id, np.int32)
    spacing = np.asarray(spacing, float)
    if x.shape != (len(F), 3) or F.shape != (len(body_id), 3, 3):
        raise ValueError('Particle position, deformation, and body arrays disagree')
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError('Three positive material sampling spacings are required')
    half = .5*spacing
    by_body = [np.flatnonzero(body_id == b) for b in (0, 1)]
    if not len(by_body[0]) or not len(by_body[1]):
        return []
    local_corners = half[None, None, :]*_SIGNS[None, :, :]
    transformed_corners = np.einsum('pij,pkj->pki', F, local_corners)
    radii = np.max(np.linalg.norm(transformed_corners, axis=-1), axis=1)
    aabb_half = np.einsum('pij,j->pi', np.abs(F), half)
    max_radius = [float(radii[idx].max()) for idx in by_body]
    tree = cKDTree(x[by_body[1]])
    candidates = []
    for ia in by_body[0]:
        local = tree.query_ball_point(x[ia], float(radii[ia]+max_radius[1]))
        for j in local:
            ib = int(by_body[1][j])
            # The tree uses a circumscribed-radius query; this componentwise
            # AABB test cheaply rejects corner-near pairs before SAT.
            if np.any(np.abs(x[ia]-x[ib]) > aabb_half[ia]+aabb_half[ib]):
                continue
            result = parallelepiped_intersection(x[ia], F[ia], half,
                                                 x[ib], F[ib], half)
            if result is None:
                continue
            normal, penetration, point, xi_a, xi_b = result
            candidates.append(DomainIntersection(int(ia), ib, point, normal,
                                                  xi_a, xi_b, penetration))
            if len(candidates) > max_pairs:
                raise OverflowError(f'True material-domain contact count exceeded guarded capacity {max_pairs}')
    return candidates


def minimum_material_domain_gap(x, F, body_id, spacing, *, search_margin=None):
    """Minimum positive SAT gap among nearby opposite-body material cuboids."""
    from scipy.spatial import cKDTree
    x, F, body_id = np.asarray(x, float), np.asarray(F, float), np.asarray(body_id, int)
    spacing = np.asarray(spacing, float); half = spacing*.5
    by_body = [np.flatnonzero(body_id == b) for b in (0, 1)]
    if not len(by_body[0]) or not len(by_body[1]):
        raise ValueError('Both material bodies are required to measure the initial gap')
    local = half[None, None, :]*_SIGNS[None, :, :]
    radii = np.max(np.linalg.norm(np.einsum('pij,pkj->pki', F, local), axis=-1), axis=1)
    max_b = float(radii[by_body[1]].max())
    margin = float(2*np.max(spacing) if search_margin is None else search_margin)
    tree = cKDTree(x[by_body[1]])
    best = (np.inf, None, None)
    nearest_center = (np.inf, None, None)
    for ia in by_body[0]:
        nearby = tree.query_ball_point(x[ia], float(radii[ia]+max_b+margin))
        distance, nearest_j = tree.query(x[ia], k=1)
        if distance < nearest_center[0]:
            nearest_center = (float(distance), int(ia), int(by_body[1][nearest_j]))
        # Always include the nearest opposite-body center as a finite fallback.
        # The margin query remains the broad neighborhood for the exact SAT min.
        for j in set(nearby + [int(nearest_j)]):
            ib = int(by_body[1][j])
            gap = parallelepiped_sat_gap(x[ia], F[ia], half, x[ib], F[ib], half)
            if gap < best[0]:
                best = (gap, int(ia), ib)
    if best[1] is None:
        raise FloatingPointError('No opposite-body material domains in gap search envelope')
    return {'minimum_sat_gap_m': max(best[0], 0.), 'signed_sat_gap_m': best[0],
            'particle_pair': [best[1], best[2]], 'search_margin_m': margin,
            'nearest_center_distance_m': nearest_center[0],
            'nearest_center_pair': [nearest_center[1], nearest_center[2]]}


def point_velocity_coefficients(weights, gradients, xi):
    """N + grad(N) xi, evaluated at one material point inside a particle cell."""
    weights, gradients, xi = np.asarray(weights), np.asarray(gradients), np.asarray(xi)
    coeff = weights + gradients@xi
    if abs(float(coeff.sum())-1.) > 1e-10:
        raise FloatingPointError('Particle point interpolation lost partition of unity')
    return coeff


def assemble_contact_jacobian(rows, node_count, fixed_nodes):
    """Sparse J from same-point shape coefficients; fixed-node columns removed."""
    from scipy.sparse import coo_matrix
    fixed_nodes = np.asarray(fixed_nodes, bool)
    rr, cc, vv = [], [], []
    for r, row in enumerate(rows):
        for nodes, coeff, sign in ((row['nodes_a'], row['coeff_a'], 1.),
                                   (row['nodes_b'], row['coeff_b'], -1.)):
            for node, weight in zip(nodes, coeff):
                for axis in range(3):
                    if not fixed_nodes[node, axis]:
                        value = sign*weight*row['normal'][axis]
                        if value != 0.:
                            rr.append(r); cc.append(3*int(node)+axis); vv.append(value)
    return coo_matrix((vv, (rr, cc)), shape=(len(rows), 3*node_count)).tocsr()


def _independent_point_velocity(velocity, reference, particle, xi):
    idx = np.asarray(reference.indices[particle])
    weights = np.asarray(reference.weights[particle])
    gradients = np.asarray(reference.gradients[particle])
    local_v = np.asarray(velocity)[idx]
    center_v = np.sum(weights[:, None]*local_v, axis=0)
    grad_v = np.einsum('ki,kj->ij', local_v, gradients)
    return center_v + grad_v@xi


def _nnls_nonnegative(mobility, closing):
    from scipy.optimize import nnls
    eig, vec = np.linalg.eigh(.5*(mobility+mobility.T))
    maxeig = float(eig.max(initial=0.))
    threshold = max(len(closing), 1)*np.finfo(float).eps*max(maxeig, 1e-300)
    if float(eig.min(initial=0.)) < -threshold:
        raise FloatingPointError('Material-domain contact mobility is not PSD')
    keep = eig > threshold
    lam = np.zeros(len(closing))
    if np.max(closing, initial=0.) > .005:
        if not keep.any():
            raise FloatingPointError('Closing true-domain contact has zero diagonal-mass mobility')
        A = np.sqrt(eig[keep])[:, None]*vec[:, keep].T
        b = (vec[:, keep].T@closing)/np.sqrt(eig[keep])
        lam = nnls(A, b, maxiter=max(1024, 3*len(closing)))[0]
    return lam, int(keep.sum()), threshold


def material_domain_contact_jump(velocity_preimpact, reference, particle_mass,
                                 body_id, x1, F1, spacing, fixed,
                                 *, tolerance=.005, max_pairs=8192):
    """Endpoint e0 QP on intersections of deformed initial material cells."""
    v = np.asarray(velocity_preimpact, float)
    mass = np.asarray(reference.nodal_mass, float)
    fixed = np.asarray(fixed, bool)
    candidates = material_domain_candidates(np.asarray(x1), np.asarray(F1),
                                            body_id, spacing, max_pairs=max_pairs)
    rows = []
    for hit in candidates:
        ca = point_velocity_coefficients(reference.weights[hit.first],
                                         reference.gradients[hit.first], hit.xi_first)
        cb = point_velocity_coefficients(reference.weights[hit.second],
                                         reference.gradients[hit.second], hit.xi_second)
        rows.append(dict(nodes_a=np.asarray(reference.indices[hit.first]), coeff_a=ca,
            nodes_b=np.asarray(reference.indices[hit.second]), coeff_b=cb,
            normal=hit.normal, point=hit.point, penetration=hit.penetration,
            first=hit.first, second=hit.second, xi_first=hit.xi_first, xi_second=hit.xi_second))
    if not rows:
        return velocity_preimpact, np.zeros_like(v), 0., 0., dict(
            valid=True, pairs=0, candidate_pairs=0, resolved_normal_pairs=0,
            initial_closing_m_s=0., final_closing_m_s=0., complementarity_m_s=0.,
            physical_route_pre_closing_m_s=0., physical_route_post_closing_m_s=0.,
            physical_route_complementarity_m_s=0., maximum_penetration_m=0.,
            newly_active_e0=False, nodal_impulse_norm_Ns=0., jump_work_J=0., kinetic_loss_J=0.)
    J = assemble_contact_jacobian(rows, len(mass), fixed)
    inv = np.divide(1., np.repeat(mass, 3), out=np.zeros(3*len(mass)),
                    where=np.repeat(mass, 3) > 0)
    flat_v = v.reshape(-1)
    pre = np.asarray(J@flat_v).ravel()
    mobility = (J.multiply(inv[None, :])@J.T).toarray()
    lam, rank, threshold = _nnls_nonnegative(mobility, pre)
    generalized_impulse = np.asarray(J.T@lam).ravel()
    dv = -inv*generalized_impulse
    vpost = v+(dv.reshape(v.shape))
    post = np.asarray(J@vpost.ravel()).ravel()
    active_resid = np.where(lam > 0., np.abs(post), np.maximum(post, 0.))
    closure = float(np.max(np.maximum(post, 0.), initial=0.))
    native_kkt = float(np.max(active_resid, initial=0.))
    # Independent route: reconstruct point velocity as v_p + (grad_X v) xi,
    # not by multiplying the assembled contact Jacobian.
    pre_geom, post_geom, geom_kkt = [], [], []
    for row, multiplier, vp, vq in zip(rows, lam, [v]*len(rows), [vpost]*len(rows)):
        va0 = _independent_point_velocity(vp, reference, row['first'], row['xi_first'])
        vb0 = _independent_point_velocity(vp, reference, row['second'], row['xi_second'])
        va1 = _independent_point_velocity(vq, reference, row['first'], row['xi_first'])
        vb1 = _independent_point_velocity(vq, reference, row['second'], row['xi_second'])
        pre_geom.append(float(np.dot(row['normal'], va0-vb0)))
        post_geom.append(float(np.dot(row['normal'], va1-vb1)))
    pre_geom, post_geom = np.asarray(pre_geom), np.asarray(post_geom)
    geom_closure = float(np.max(np.maximum(post_geom, 0.), initial=0.))
    geom_resid = np.where(lam > 0., np.abs(post_geom), np.maximum(post_geom, 0.))
    geom_kkt = float(np.max(geom_resid, initial=0.))
    impulse = (vpost-v)*mass[:, None]
    work = float(np.sum(impulse*.5*(v+vpost)))
    before = .5*float(np.sum(mass[:, None]*v*v))
    loss = before-.5*float(np.sum(mass[:, None]*vpost*vpost))
    # Equal-and-opposite material-body impulses at each shared world point.
    body_impulse = [np.zeros(3), np.zeros(3)]
    for row, multiplier in zip(rows, lam):
        body_impulse[0] -= multiplier*row['normal']
        body_impulse[1] += multiplier*row['normal']
    # Compare the independent material-point power with the assembled sparse
    # transpose action. Bound their floating-point dot-product roundoff from
    # the actual nonzero term magnitudes and summation lengths.
    adjoint_impulse = np.asarray(J.T@lam).ravel()
    phase_power_pre = float(np.dot(pre_geom, lam))
    node_power_pre = float(np.dot(v.reshape(-1), adjoint_impulse))
    phase_power_post = float(np.dot(post_geom, lam))
    node_power_post = float(np.dot(vpost.reshape(-1), adjoint_impulse))
    eps = np.finfo(float).eps
    terms_node = max(int(np.count_nonzero(adjoint_impulse)), 1)
    terms_phase = max(int(np.count_nonzero(lam)), 1)
    gamma_node = terms_node*eps/max(1-terms_node*eps, .5)
    gamma_phase = terms_phase*eps/max(1-terms_phase*eps, .5)
    power_bound_pre = gamma_node*float(np.sum(np.abs(v.reshape(-1)*adjoint_impulse))) + gamma_phase*float(np.sum(np.abs(pre_geom*lam)))
    power_bound_post = gamma_node*float(np.sum(np.abs(vpost.reshape(-1)*adjoint_impulse))) + gamma_phase*float(np.sum(np.abs(post_geom*lam)))
    valid = (np.min(lam, initial=0.) >= 0. and closure <= tolerance
             and native_kkt <= tolerance and geom_closure <= tolerance
             and geom_kkt <= tolerance and loss >= -1e-10*max(before, 1e-30)
             and work <= 1e-10*max(loss, 1e-30)
             and abs(work+loss) <= 1e-10*max(before, 1e-30))
    report = dict(valid=bool(valid), pairs=len(rows), candidate_pairs=len(candidates),
        resolved_normal_pairs=len(rows), initial_closing_m_s=float(np.max(pre, initial=0.)),
        final_closing_m_s=closure, complementarity_m_s=native_kkt,
        physical_route_pre_closing_m_s=float(np.max(pre_geom, initial=0.)),
        physical_route_post_closing_m_s=geom_closure,
        physical_route_complementarity_m_s=geom_kkt,
        JphaseA_vs_Jnode_pre_max_abs_m_s=float(np.max(np.abs(pre_geom-pre), initial=0.)),
        JphaseA_vs_Jnode_post_max_abs_m_s=float(np.max(np.abs(post_geom-post), initial=0.)),
        dot_v_Atlambda_vs_Av_lambda_pre_abs_error=abs(node_power_pre-phase_power_pre),
        dot_v_Atlambda_vs_Av_lambda_pre_roundoff_bound=power_bound_pre,
        dot_v_Atlambda_vs_Av_lambda_post_abs_error=abs(node_power_post-phase_power_post),
        dot_v_Atlambda_vs_Av_lambda_post_roundoff_bound=power_bound_post,
        mobility_rank=rank, mobility_eigen_threshold=threshold,
        maximum_multiplier=float(lam.max(initial=0.)),
        positive_multiplier_pairs=int(np.count_nonzero(lam > 0.)),
        nodal_impulse_norm_Ns=float(np.linalg.norm(impulse)),
        body_contact_impulse_Ns=[v.tolist() for v in body_impulse],
        maximum_penetration_m=float(max(r['penetration'] for r in rows)),
        contact_points_world_m=[r['point'].tolist() for r in rows],
        contact_normals=[r['normal'].tolist() for r in rows],
        contact_local_coordinates_m=[[r['xi_first'].tolist(), r['xi_second'].tolist()] for r in rows],
        jump_work_J=work, kinetic_loss_J=loss,
        impulse_identity_J=.5*float(np.sum(mass[:, None]*dv.reshape(v.shape)**2))
            +float(np.dot(lam, post)),
        newly_active_e0=bool(np.max(pre, initial=0.) > tolerance and closure <= tolerance))
    return vpost, impulse, work, loss, report
