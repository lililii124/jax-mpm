"""NumPy-only conservation/topology audit; safe on low-memory hosts."""
import numpy as np


def invariants(state, dx):
    """APIC angular momentum includes affine spin with D=dx² I/4."""
    get = lambda name: np.asarray(getattr(state, name), dtype=np.float64)
    m, x, v, C = (get(k) for k in ('mass', 'x', 'v', 'C'))
    spin = np.stack((C[:,2,1]-C[:,1,2], C[:,0,2]-C[:,2,0], C[:,1,0]-C[:,0,1]), axis=-1)
    return dict(mass_kg=float(m.sum()), linear_momentum_kg_m_s=(m[:,None]*v).sum(0).tolist(),
                angular_momentum_kg_m2_s=(m[:,None]*(np.cross(x,v)+dx**2/4*spin)).sum(0).tolist(),
                translation_J=float(.5*np.sum(m[:,None]*v*v)),
                affine_J=float(dx**2/8*np.sum(m[:,None,None]*C*C)),
                internal_J=float(np.sum(m*get('internal_energy'))))


def topology(x, reference, edges, alive, damage, physical, mass):
    """Use the existing damage>=.95 AND opening>1.10 reference-bond rule."""
    edges = np.asarray(edges); alive = np.asarray(alive, dtype=bool)
    opening = np.linalg.norm(x[edges[:,0]]-x[edges[:,1]], axis=1)
    original = np.linalg.norm(reference[edges[:,0]]-reference[edges[:,1]], axis=1)
    failed = np.maximum(damage[edges[:,0]],damage[edges[:,1]]) >= .95
    remaining = alive & ~(failed & (opening > 1.10*original))
    parent = np.arange(len(x))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for a,b in edges[remaining]:
        a,b = find(a),find(b)
        if a != b: parent[b] = a
    roots = np.array([find(i) for i in range(len(x))])
    _, labels = np.unique(roots, return_inverse=True)
    groups = []
    for label in range(int(labels.max())+1):
        take = labels == label
        bodies = np.unique(physical[take])
        if len(bodies) != 1: raise ValueError('Reference graph crosses physical bodies')
        groups.append(dict(particles=int(take.sum()), mass_kg=float(mass[take].sum()), physical_body=int(bodies[0])))
    return dict(remaining_bonds=int(remaining.sum()), newly_broken_bonds=int((alive & ~remaining).sum()),
                components=groups, component_count=len(groups),
                interpretation='Reference connectivity only; not a validated fragment or surface geometry.')


def audit_state_pair(before, after, reference, edges, alive, physical, dx):
    initial, final = invariants(before,dx), invariants(after,dx)
    impulse = np.asarray(after.boundary_impulse)-np.asarray(before.boundary_impulse)
    residual = np.array(final['linear_momentum_kg_m_s'])-initial['linear_momentum_kg_m_s']-impulse
    return dict(initial=initial, final=final,
                mass_exact=np.asarray(before.mass).tobytes()==np.asarray(after.mass).tobytes(),
                linear_momentum_minus_boundary_impulse_kg_m_s=residual.tolist(),
                angular_momentum_change_kg_m2_s=(np.array(final['angular_momentum_kg_m2_s'])-initial['angular_momentum_kg_m2_s']).tolist(),
                angular_momentum_gate='Informational: frozen carry lacks boundary torque; change alone is not a conservation residual.',
                topology=topology(np.asarray(after.x),reference,edges,alive,np.asarray(after.damage),physical,np.asarray(after.mass)))
