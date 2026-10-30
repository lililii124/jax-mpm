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



def _cross3(a,b):
    # Exact scalar arithmetic of np.cross for finite 3-vectors, without its
    # repeated generic axis normalization/moveaxis Python dispatch.
    return np.asarray((a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]))

def _conservative_face_filter(xa,Fa,xb,Fb,half):
    # Broadphase only. Stable face axes and a deliberately wider margin may
    # retain extra candidates; original15-axis SAT/clipping remains final.
    ea=Fa*half[None,None,:];eb=Fb*half[None,None,:]
    ar=ea.swapaxes(-1,-2);br=eb.swapaxes(-1,-2)
    af=np.cross(ar[:,[1,2,0]],ar[:,[2,0,1]])
    bf=np.cross(br[:,[1,2,0]],br[:,[2,0,1]])
    axes=np.concatenate((af,bf),axis=1);norm=np.linalg.norm(axes,axis=2)
    norm2a=np.sum(ea*ea,axis=(1,2));norm2b=np.sum(eb*eb,axis=(1,2))
    stable=norm>1e-4*np.concatenate((np.repeat(norm2a[:,None],3,axis=1),np.repeat(norm2b[:,None],3,axis=1)),axis=1)
    direction=axes/np.where(norm>0,norm,1.)[:,:,None]
    ra=np.sum(np.abs(np.einsum('nac,nce->nae',direction,ea)),axis=2)
    rb=np.sum(np.abs(np.einsum('nac,nce->nae',direction,eb)),axis=2)
    delta=xb-xa;depth=ra+rb-np.abs(np.einsum('nac,nc->na',direction,delta))
    margin=1e-8*(np.linalg.norm(np.abs(Fa)@half,axis=1)+np.linalg.norm(np.abs(Fb)@half,axis=1)+np.linalg.norm(delta,axis=1)+float(np.max(half)))
    separated=np.any(stable&(depth < -margin[:,None]),axis=1)
    safe=np.isfinite(Fa).all(axis=(1,2))&np.isfinite(Fb).all(axis=(1,2))&(np.linalg.det(Fa)>0)&(np.linalg.det(Fb)>0)
    return ~separated|~safe

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


# Deterministic geometric measure centroid; never weight intersection vertices equally.
_GEOMETRIC_CENTROID_COUNTS = {}

def _planar_measure_centroid(points):
    from scipy.spatial import ConvexHull, QhullError
    points=np.asarray(points,dtype=float)
    if len(points)==1:return points[0],0
    origin=points.mean(axis=0);q=points-origin;scale=max(float(np.linalg.norm(q,axis=1).max()),1e-300)
    try:
        hull=ConvexHull(q/scale)
        polygon=q[hull.vertices]/scale;nextp=np.roll(polygon,-1,axis=0)
        cross=polygon[:,0]*nextp[:,1]-nextp[:,0]*polygon[:,1];area2=float(cross.sum())
        if area2!=0.:
            center=np.sum((polygon+nextp)*cross[:,None],axis=0)/(3*area2)
            return origin+scale*center,2
    except QhullError:
        pass
    # A zero-area section is an actual line/point: use length measure, not vertex count.
    _,s,Vh=np.linalg.svd(q,full_matrices=False)
    if not len(s) or s[0]<=128*np.finfo(float).eps*scale:return points[0],0
    direction=Vh[0];t=q@direction
    return origin+.5*(float(t.min())+float(t.max()))*direction,1

def _true_intersection_normal_midplane(ca,Fa,ha,cb,Fb,hb,n,tol):
    # Reconstruct halfspace extrema independently of tolerance-merged clip vertices.
    from itertools import combinations
    midpoint=.5*(ca+cb);scale=max(float(np.linalg.norm(Fa*ha[None,:])),float(np.linalg.norm(Fb*hb[None,:])),float(np.linalg.norm(cb-ca)),1e-300)
    normals=[];bounds=[];slack=[]
    for center,F,half in [(ca,Fa,ha),(cb,Fb,hb)]:
        inverse=np.linalg.inv(F)
        for k in range(3):
            length=float(np.linalg.norm(inverse[k]));direction=inverse[k]/length
            for sign in (-1.,1.):
                normals.append(sign*direction);bounds.append((half[k]/length+sign*float(direction@(center-midpoint)))/scale);slack.append(tol/(length*scale))
    N=np.asarray(normals);c=np.asarray(bounds);eps=np.asarray(slack);triples=np.asarray(list(combinations(range(len(c)),3)));mat=N[triples];det=np.linalg.det(mat);valid=np.abs(det)>128*np.finfo(float).eps
    points=np.linalg.solve(mat[valid],c[triples[valid]][...,None])[...,0]
    allowance=eps[:,None]+128*np.finfo(float).eps*np.maximum(np.linalg.norm(points,axis=1),1.)[None,:]
    points=points[np.all(N@points.T<=c[:,None]+allowance,axis=0)]
    if not len(points):raise FloatingPointError('No true halfspace intersection extrema')
    height=points@n;level=.5*(float(height.min())+float(height.max()))
    return midpoint+scale*level*n

def _true_contact_plane_centroid(vertices,ca,Fa,ha,cb,Fb,hb,normal,tol):
    n=np.asarray(normal,float);n=n/np.linalg.norm(n);vertices=np.asarray(vertices,float)
    origin=_true_intersection_normal_midplane(ca,Fa,ha,cb,Fb,hb,n,tol)
    axis=np.zeros(3);axis[int(np.argmin(np.abs(n)))]=1.
    e1=np.cross(n,axis);e1=e1/np.linalg.norm(e1);e2=np.cross(n,e1);basis=np.column_stack((e1,e2))
    normals=[];bounds=[];slack=[]
    for center,F,half in [(ca,Fa,ha),(cb,Fb,hb)]:
        inverse=np.linalg.inv(F);ab=inverse@basis;offset=inverse@(origin-center)
        for k in range(3):
            for sign in (-1.,1.):
                row=sign*ab[k];bound=float(half[k]-sign*offset[k]);length=float(np.linalg.norm(row))
                if length<=128*np.finfo(float).eps*max(float(np.linalg.norm(inverse[k])),1e-300):
                    if bound < -tol:raise FloatingPointError('Contact plane is outside a cuboid')
                    continue
                normals.append(row/length);bounds.append(bound/length);slack.append(tol/length)
    N=np.asarray(normals);c=np.asarray(bounds);eps=np.asarray(slack);points=[]
    for i in range(len(c)):
        for k in range(i+1,len(c)):
            det=float(N[i,0]*N[k,1]-N[i,1]*N[k,0])
            if abs(det)<=128*np.finfo(float).eps:continue
            q=np.asarray(((c[i]*N[k,1]-N[i,1]*c[k])/det,(N[i,0]*c[k]-c[i]*N[k,0])/det))
            if np.all(N@q<=c+eps):
                uniqueness=max(tol*max(float(np.linalg.norm(Fa)),float(np.linalg.norm(Fb)),1.),128*np.finfo(float).eps*max(float(np.linalg.norm(q)),1e-300))
                if not any(np.linalg.norm(q-old)<=uniqueness for old in points):points.append(q)
    if not points:raise FloatingPointError('No true contact-plane section polygon')
    center,dimension=_planar_measure_centroid(points)
    return origin+basis@center,dimension

def _intersection_measure_centroid(vertices,ca,Fa,ha,cb,Fb,hb,normal,tol):
    from scipy.spatial import ConvexHull, QhullError
    vertices=np.asarray(vertices,dtype=float);origin=vertices.mean(axis=0)
    centered=vertices-origin;scale=max(float(np.linalg.norm(centered,axis=1).max(initial=0.)),1e-300)
    q=centered/scale;method=None;point=None
    if len(vertices)>=4:
        singular=np.linalg.svd(q,compute_uv=False)
        # Volume centroid conditioning loses at least half the float64 digits below sqrt(eps).
        resolution=8*tol*max(float(np.linalg.norm(Fa)),float(np.linalg.norm(Fb)),1.)
        aspect=singular[-1]/max(float(singular[0]),1e-300)
        centroid_budget=max(16*tol*max(float(np.linalg.norm(Fa)),float(np.linalg.norm(Fb)),1.),16*np.finfo(float).eps*scale)
        conditioning_floor=max(np.sqrt(np.finfo(float).eps),np.finfo(float).eps*scale/centroid_budget)
        if aspect>conditioning_floor and singular[-1]*scale>resolution:
            try:
                hull=ConvexHull(q)  # No QJ/joggle; coplanar subdivisions do not alter measure.
                triangles=q[hull.simplices]
                volume6=np.abs(np.einsum('fi,fi->f',triangles[:,0],np.cross(triangles[:,1],triangles[:,2])))
                total=float(volume6.sum())
                if total>0. and np.isfinite(total):
                    point=origin+scale*np.sum(volume6[:,None]*triangles.sum(axis=1),axis=0)/(4*total)
                    inside_a,_=_inside(point,ca,np.linalg.inv(Fa),ha,tol);inside_b,_=_inside(point,cb,np.linalg.inv(Fb),hb,tol)
                    if inside_a and inside_b:method='volume'
                    else:point=None
            except QhullError:
                pass
    if point is None:
        point,dimension=_true_contact_plane_centroid(vertices,ca,Fa,ha,cb,Fb,hb,normal,tol)
        method={2:'section_area',1:'section_length',0:'section_point'}[dimension]
    _GEOMETRIC_CENTROID_COUNTS[method]=_GEOMETRIC_CENTROID_COUNTS.get(method,0)+1
    return point


def _edge_face_vertices(edge_owner_center, edge_owner_F, edge_owner_half,
                        face_center, face_inverse_F, face_half, corners,
                        vertices, tol):
    pairs=np.asarray(_EDGE_PAIRS);p0=corners[pairs[:,0]];p1=corners[pairs[:,1]]
    a=(p0-face_center)@face_inverse_F.T;b=(p1-face_center)@face_inverse_F.T;d=b-a
    sides=np.asarray((-1.,1.));den=d[:,:,None]
    valid=np.abs(den)>1e-14
    t=np.divide(np.asarray(face_half)[None,:,None]*sides[None,None,:]-a[:,:,None],den,out=np.zeros((len(pairs),3,2)),where=valid)
    valid=np.broadcast_to(valid,t.shape)&(t>=-1e-12)&(t<=1+1e-12)
    ids=np.argwhere(valid)
    if not len(ids):return
    edge=ids[:,0];axis=ids[:,1];q=p0[edge]+np.clip(t[valid],0.,1.)[:,None]*(p1[edge]-p0[edge]);local=(q-face_center)@face_inverse_F.T
    inside=np.abs(local)<=np.asarray(face_half)+tol;inside[np.arange(len(ids)),axis]=True
    for point in q[np.all(inside,axis=1)]:_append_unique(vertices,point,tol)


def parallelepiped_intersection(center_a, F_a, half_a,
                                center_b, F_b, half_b,
                                *, relative_tolerance=1e-12):
    """Return SAT MTV normal/depth and geometric measure centroid of the true intersection.

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
    axes = [_cross3(ea[:, (i+1) % 3], ea[:, (i+2) % 3]) for i in range(3)]
    axes += [_cross3(eb[:, (i+1) % 3], eb[:, (i+2) % 3]) for i in range(3)]
    axes += [_cross3(ea[:, i], eb[:, j]) for i in range(3) for j in range(3)]
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
    point = _intersection_measure_centroid(vertices, ca, Fa, ha, cb, Fb, hb, min_axis, tol)
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
    axes = [_cross3(ea[:, (i+1) % 3], ea[:, (i+2) % 3]) for i in range(3)]
    axes += [_cross3(eb[:, (i+1) % 3], eb[:, (i+2) % 3]) for i in range(3)]
    axes += [_cross3(ea[:, i], eb[:, j]) for i in range(3) for j in range(3)]
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


def material_domain_candidates(x, F, body_id, spacing, *, max_pairs=8192,
                               component_id=None):
    """Current true material-cuboid intersections via cKDTree + 15-axis SAT.

    With no component labels, preserve the original opposite-body-only scope.
    With labels, admit cross-body pairs and same-body pairs from distinct
    material-support components, while never self-contacting one component.
    """
    from scipy.spatial import cKDTree
    x, F = np.asarray(x, float), np.asarray(F, float)
    body_id = np.asarray(body_id, np.int32)
    spacing = np.asarray(spacing, float)
    if x.shape != (len(F), 3) or F.shape != (len(body_id), 3, 3):
        raise ValueError('Particle position, deformation, and body arrays disagree')
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError('Three positive material sampling spacings are required')
    half = .5*spacing
    if component_id is not None:
        component_id = np.asarray(component_id, np.int64)
        if component_id.shape != body_id.shape or np.any(component_id < 0):
            raise ValueError('A nonnegative support-component label is required per particle')
        search_groups = np.arange(len(body_id), dtype=np.int64)
    else:
        by_body = [np.flatnonzero(body_id == b) for b in (0, 1)]
        if not len(by_body[0]) or not len(by_body[1]):
            return []
        search_groups = by_body[0]
    aabb_half = np.einsum('pij,j->pi', np.abs(F), half)
    if component_id is None:
        local_corners = half[None, None, :]*_SIGNS[None, :, :]
        transformed_corners = np.einsum('pij,pkj->pki', F, local_corners)
        radii = np.max(np.linalg.norm(transformed_corners, axis=-1), axis=1)
    else:
        # AABB circumspheres are a looser but conservative broadphase for the
        # many newly independent same-body components; SAT remains exact.
        radii = np.linalg.norm(aabb_half, axis=1)
    if component_id is None:
        target_ids = by_body[1]
        max_target_radius = float(radii[target_ids].max())
        tree = cKDTree(x[target_ids])
    else:
        target_ids = np.arange(len(body_id), dtype=np.int64)
        max_target_radius = float(radii.max(initial=0.))
        tree = cKDTree(x)
    eligible_a=[];eligible_b=[]
    neighborhoods=tree.query_ball_point(x[search_groups],radii[search_groups]+max_target_radius,return_sorted=False)
    for ia,local in zip(search_groups,neighborhoods):
        ibs=target_ids[np.asarray(local,dtype=np.int64)]
        if component_id is None:
            ibs=ibs[body_id[ibs]!=body_id[ia]]
        else:
            ibs=ibs[(ibs>ia)&(component_id[ibs]!=component_id[ia])]
        if not len(ibs):continue
        ibs=ibs[np.all(np.abs(x[ia]-x[ibs])<=aabb_half[ia]+aabb_half[ibs],axis=1)]
        if len(ibs):eligible_a.append(np.full(len(ibs),ia,dtype=np.int64));eligible_b.append(ibs)
    if not eligible_a:return []
    aa=np.concatenate(eligible_a);bb=np.concatenate(eligible_b)
    keep=_conservative_face_filter(x[aa],F[aa],x[bb],F[bb],half)
    candidates=[]
    for ia,ib in zip(aa[keep],bb[keep]):
        result=parallelepiped_intersection(x[ia],F[ia],half,x[ib],F[ib],half)
        if result is None:continue
        normal,penetration,point,xi_a,xi_b=result
        candidates.append(DomainIntersection(int(ia),int(ib),point,normal,xi_a,xi_b,penetration))
        if len(candidates)>max_pairs:
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
                                 *, tolerance=.005, max_pairs=8192,
                                 component_id=None):
    """Endpoint e0 QP on intersections of deformed initial material cells."""
    v = np.asarray(velocity_preimpact, float)
    # Transfer static stencil arrays once, avoiding thousands of tiny JAX
    # indexing dispatches in the exact host SAT/contact and independent audit.
    reference = reference._replace(weights=np.asarray(reference.weights),
        gradients=np.asarray(reference.gradients), indices=np.asarray(reference.indices))
    mass = np.asarray(reference.nodal_mass, float)
    fixed = np.asarray(fixed, bool)
    candidates = material_domain_candidates(np.asarray(x1), np.asarray(F1),
        body_id, spacing, max_pairs=max_pairs, component_id=component_id)
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
        contact_particle_pairs=[[int(r['first']), int(r['second'])] for r in rows],
        contact_points_world_m=[r['point'].tolist() for r in rows],
        contact_normals=[r['normal'].tolist() for r in rows],
        contact_multipliers_Ns=lam.tolist(),
        contact_local_coordinates_m=[[r['xi_first'].tolist(), r['xi_second'].tolist()] for r in rows],
        jump_work_J=work, kinetic_loss_J=loss,
        impulse_identity_J=.5*float(np.sum(mass[:, None]*dv.reshape(v.shape)**2))
            +float(np.dot(lam, post)),
        newly_active_e0=bool(np.max(pre, initial=0.) > tolerance and closure <= tolerance))
    return vpost, impulse, work, loss, report


_nnls_nonnegative_original = _nnls_nonnegative
LAST_QP_DIAGNOSTIC = {}
def _nnls_nonnegative(mobility, closing):
    global LAST_QP_DIAGNOSTIC
    from scipy.linalg import cho_factor, cho_solve
    H=.5*(mobility+mobility.T)
    from scipy.linalg import eigh
    eig=eigh(H, eigvals_only=True, driver='evr', check_finite=False)
    threshold=max(len(closing),1)*np.finfo(float).eps*max(float(eig.max(initial=0.)),1e-300)
    if float(eig.min(initial=0.)) < -threshold:raise FloatingPointError('Material-domain contact mobility is not PSD')
    rank=int(np.count_nonzero(eig>threshold));n=len(closing)
    if np.max(closing,initial=0.)<=.005:
        LAST_QP_DIAGNOSTIC=dict(iterations=0,fullrank=(rank==n),fallback=False)
        return np.zeros(n),rank,threshold
    if rank!=n:
        LAST_QP_DIAGNOSTIC=dict(iterations=0,fullrank=False,fallback=True)
        return _nnls_nonnegative_original(mobility,closing)
    d=np.sqrt(np.diag(H));active=closing>0;seen=set()
    for iteration in range(1,33):
        lam=np.zeros(n);ids=np.flatnonzero(active)
        if len(ids):
            block=H[np.ix_(ids,ids)]/(d[ids,None]*d[None,ids]);rhs=closing[ids]/d[ids]
            try:
                lam[ids]=cho_solve(cho_factor(block,lower=True,check_finite=False),rhs,check_finite=False)/d[ids]
            except np.linalg.LinAlgError:
                LAST_QP_DIAGNOSTIC=dict(iterations=iteration,fullrank=True,fallback=True,reason='factorization')
                return _nnls_nonnegative_original(mobility,closing)
        w=H@lam-closing
        wrong_active=active&(lam<=0);wrong_inactive=(~active)&(w < -1e-8)
        if not np.any(wrong_active|wrong_inactive):
            if np.max(np.where(active,abs(w),0),initial=0)>1e-6:
                LAST_QP_DIAGNOSTIC=dict(iterations=iteration,fullrank=True,fallback=True,reason='active_residual')
                return _nnls_nonnegative_original(mobility,closing)
            LAST_QP_DIAGNOSTIC=dict(iterations=iteration,fullrank=True,fallback=False,minimum_multiplier=float(lam.min(initial=0)),inactive_closing_m_s=float(np.max(np.where(~active,-w,0),initial=0)),active_residual_m_s=float(np.max(np.where(active,abs(w),0),initial=0)))
            return lam,rank,threshold
        key=np.packbits(active).tobytes();new=(active&~wrong_active)|wrong_inactive
        if key in seen:
            # Deterministic single principal pivot on cycle; same convexQP.
            new=active.copy();violation=np.where(wrong_active,-lam*d,np.where(wrong_inactive,-w/d,0));k=int(np.argmax(violation));new[k]=not active[k]
        seen.add(key);active=new
    LAST_QP_DIAGNOSTIC=dict(iterations=32,fullrank=True,fallback=True)
    return _nnls_nonnegative_original(mobility,closing)

# Scope BLAS parallelism to the host contactQP; force and all physical gates unchanged.
_nnls_nonnegative_serial_blas = _nnls_nonnegative
_contact_blas_handles = None
def _nnls_nonnegative(mobility, closing):
    global _contact_blas_handles
    import ctypes
    from pathlib import Path
    if _contact_blas_handles is None:
        handles=[]
        for package in ['numpy.libs','scipy.libs']:
            for lib in (Path(np.__file__).parents[1]/package).glob('*openblas*'):
                handle=ctypes.CDLL(str(lib))
                for prefix in ['openblas_','scipy_openblas_']:
                    found=False
                    for suffix in ['', '64_']:
                        try:
                            setter=getattr(handle,prefix+'set_num_threads'+suffix);getter=getattr(handle,prefix+'get_num_threads'+suffix)
                            setter.argtypes=[ctypes.c_int];setter.restype=None;getter.restype=ctypes.c_int
                            handles.append((handle,setter,getter));found=True;break
                        except AttributeError:pass
                    if found:break
        _contact_blas_handles=handles
    before=[getter() for _,_,getter in _contact_blas_handles]
    for _,setter,_ in _contact_blas_handles:setter(4)
    try:return _nnls_nonnegative_serial_blas(mobility,closing)
    finally:
        for (_,setter,_),count in zip(_contact_blas_handles,before):setter(count)
