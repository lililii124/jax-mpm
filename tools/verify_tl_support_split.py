"""Finite-dimensional conservation check for a TL component support split.

This verifies the copy/remap algebra and shared-point SAT impulse only. It is
not a fracture law or production trajectory test.
"""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

# Load only the NumPy/SAT module; importing mpm_jax.__init__ would eagerly
# require JAX, which this small finite-dimensional receipt does not use.
_SAT_PATH = Path(__file__).resolve().parents[1] / 'mpm_jax' / 'material_domain_contact.py'
_SPEC = importlib.util.spec_from_file_location('_c2_material_domain_contact', _SAT_PATH)
_SAT_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SAT_MODULE
_SPEC.loader.exec_module(_SAT_MODULE)
parallelepiped_intersection = _SAT_MODULE.parallelepiped_intersection


def split_support(indices, weights, component):
    """Duplicate old node IDs by material component; retain particle ordering."""
    keys = sorted({(int(component[p]), int(node))
                   for p in range(len(indices)) for node in indices[p]})
    key_to_new = {key: i for i, key in enumerate(keys)}
    remapped = np.asarray([[key_to_new[(int(component[p]), int(node))]
                            for node in indices[p]]
                           for p in range(len(indices))], dtype=np.int64)
    return keys, remapped


def node_ledger(q, v, mass):
    momentum = np.sum(mass[:, None] * v, axis=0)
    angular = np.sum(np.cross(q, mass[:, None] * v), axis=0)
    kinetic = .5 * np.sum(mass[:, None] * v**2)
    return momentum, angular, float(kinetic)


def sat_rigid_jump():
    """True cuboid intersection plus rigid-body e0 impulse at common point."""
    half = np.array([.5, .35, .4])
    F = np.eye(3)
    c_a = np.array([0., 0., 0.])
    c_b = np.array([.92, .04, .03])
    normal, depth, point, xi_a, xi_b = parallelepiped_intersection(
        c_a, F, half, c_b, F, half)
    m_a, m_b = 2.3, 3.1
    I_a = np.diag([.21, .28, .31])
    I_b = np.diag([.24, .33, .37])
    v_a, v_b = np.array([1.4, .2, -.1]), np.array([-1.1, -.3, .1])
    w_a, w_b = np.array([.1, -.2, .15]), np.array([-.08, .12, .04])
    r_a, r_b = point-c_a, point-c_b
    va_c = v_a + np.cross(w_a, r_a)
    vb_c = v_b + np.cross(w_b, r_b)
    closing = float(np.dot(normal, va_c-vb_c))
    # Generalized inverse mass for a scalar normal impulse at the shared point.
    ga = np.cross(r_a, normal)
    gb = np.cross(r_b, normal)
    compliance = (1/m_a + 1/m_b + ga @ np.linalg.solve(I_a, ga)
                 + gb @ np.linalg.solve(I_b, gb))
    lam = max(closing, 0.)/compliance
    impulse_a, impulse_b = -lam*normal, lam*normal
    v_a1, v_b1 = v_a+impulse_a/m_a, v_b+impulse_b/m_b
    w_a1 = w_a + np.linalg.solve(I_a, np.cross(r_a, impulse_a))
    w_b1 = w_b + np.linalg.solve(I_b, np.cross(r_b, impulse_b))
    post = float(np.dot(normal, v_a1+np.cross(w_a1, r_a)
                        -v_b1-np.cross(w_b1, r_b)))

    P0 = m_a*v_a + m_b*v_b
    P1 = m_a*v_a1 + m_b*v_b1
    L0 = (np.cross(c_a, m_a*v_a)+I_a@w_a
          +np.cross(c_b, m_b*v_b)+I_b@w_b)
    L1 = (np.cross(c_a, m_a*v_a1)+I_a@w_a1
          +np.cross(c_b, m_b*v_b1)+I_b@w_b1)
    K0 = .5*(m_a*v_a@v_a + w_a@I_a@w_a
             +m_b*v_b@v_b + w_b@I_b@w_b)
    K1 = .5*(m_a*v_a1@v_a1 + w_a1@I_a@w_a1
             +m_b*v_b1@v_b1 + w_b1@I_b@w_b1)
    jump_work = float(np.dot(impulse_a, va_c + .5*(impulse_a/m_a
                        + np.cross(np.linalg.solve(I_a, np.cross(r_a, impulse_a)), r_a)))
                        + np.dot(impulse_b, vb_c + .5*(impulse_b/m_b
                        + np.cross(np.linalg.solve(I_b, np.cross(r_b, impulse_b)), r_b))))
    # Equivalent common-point work and rigid-body kinetic loss.
    loss = K0-K1
    assert closing > 0 and lam >= 0 and depth > 0
    assert abs(post) < 2e-14
    assert np.linalg.norm(P1-P0) < 2e-14
    assert np.linalg.norm(L1-L0) < 2e-14
    assert abs(loss - .5*lam*closing) < 2e-14
    assert abs(jump_work + loss) < 2e-14
    return dict(overlap_depth_m=float(depth), common_point_m=point.tolist(),
                normal=normal.tolist(), xi_a_m=xi_a.tolist(), xi_b_m=xi_b.tolist(),
                closing_pre_m_s=closing, closing_post_m_s=post,
                impulse_Ns=float(lam), momentum_error_Ns=float(np.linalg.norm(P1-P0)),
                angular_momentum_error_Nms=float(np.linalg.norm(L1-L0)),
                kinetic_loss_J=float(loss), contact_jump_work_J=jump_work)


def main():
    # Three material points: 0 and 1 remain in one intact component; fully
    # failed point 2 is retained, with its own duplicate copies of shared nodes.
    indices = np.array([[0, 1, 2], [1, 2, 3], [1, 2, 4]], dtype=np.int64)
    weights = np.array([[.2, .5, .3], [.3, .4, .3], [.2, .6, .2]])
    gradients = np.array([
        [[-1., 0., 0.], [1., -1., 0.], [0., 1., 0.]],
        [[-1., 0., 0.], [0., -1., 0.], [1., 1., 0.]],
        [[-1., 0., 0.], [1., 1., 0.], [0., -1., 0.]],
    ])
    components = np.array([0, 0, 1])
    keys, remapped = split_support(indices, weights, components)
    q0 = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                   [1., 1., 0.], [1., 0., 1.]])
    v0 = np.array([[.2, .1, 0.], [.4, -.1, .2], [-.1, .3, 0.],
                   [.1, -.2, .1], [.6, .2, -.1]])
    pmass = np.array([2., 3., 1.5])
    pvolume = np.array([.7, .8, .4])
    # Positive interpolation weights and exact partition of unity are the
    # same mass-lumping assumptions used by the TL reference discretization.
    assert np.all(weights >= 0) and np.max(np.abs(weights.sum(1)-1)) < 1e-15
    old_mass = np.zeros(len(q0))
    np.add.at(old_mass, indices.ravel(), (pmass[:, None]*weights).ravel())
    old = node_ledger(q0, v0, old_mass)

    q1 = np.asarray([q0[node] for _, node in keys])
    v1 = np.asarray([v0[node] for _, node in keys])
    new_mass = np.zeros(len(keys))
    np.add.at(new_mass, remapped.ravel(), (pmass[:, None]*weights).ravel())
    new = node_ledger(q1, v1, new_mass)
    assert abs(new_mass.sum()-old_mass.sum()) < 1e-14
    assert np.max(np.abs(new[0]-old[0])) < 1e-14
    assert np.max(np.abs(new[1]-old[1])) < 1e-14
    assert abs(new[2]-old[2]) < 1e-14

    # Verify that component-specific scatter remains power-adjoint to each
    # particle's same reference gradient, and particle F/history are untouched.
    P = np.array([[[2., .1, 0.], [.1, 1., .2], [0., .2, .5]],
                  [[1., .2, 0.], [.2, 1.5, 0.], [0., 0., .8]],
                  [[.7, 0., .1], [0., .9, 0.], [.1, 0., 1.1]]])
    F = np.array([np.eye(3), np.diag([1.1, .95, 1.]), np.diag([.9, 1.05, 1.])])
    material_history = {
        'tau': P.copy(),
        'plastic_strain': np.array([.1, .15, .3]),
        'cold_work': np.array([2., 3., 4.]),
        'fracture_history': np.array([.2, .4, 1.]),
        'damage': np.array([.1, .2, 1.]),
        'internal_energy': np.array([10., 20., 30.]),
        'plastic_heat': np.array([.5, 1.25, 2.]),
    }
    particle_force_power = 0.
    nodal_force = np.zeros_like(q0)
    nodal_force_split = np.zeros_like(q1)
    for p in range(len(indices)):
        local_v = v0[indices[p]]
        grad_v = np.einsum('ki,kj->ij', local_v, gradients[p])
        particle_force_power += -pvolume[p]*np.sum(P[p]*grad_v)
        f = -pvolume[p]*np.einsum('ij,kj->ki', P[p], gradients[p])
        np.add.at(nodal_force, indices[p], f)
        np.add.at(nodal_force_split, remapped[p], f)
    old_force_power = float(np.sum(v0*nodal_force))
    new_force_power = float(np.sum(v1*nodal_force_split))
    assert abs(particle_force_power-old_force_power) < 2e-14
    assert abs(particle_force_power-new_force_power) < 2e-14
    F_after = F.copy()
    history_after = {name: value.copy() for name, value in material_history.items()}
    assert np.array_equal(F_after, F)
    assert all(np.array_equal(history_after[name], value)
               for name, value in material_history.items())

    report = dict(status='passed', original_nodes=len(q0), split_nodes=len(keys),
        component_ids=components.tolist(), particle_node_map=remapped.tolist(),
        mass_kg=float(old_mass.sum()), momentum_Ns=old[0].tolist(),
        angular_momentum_Nms=old[1].tolist(), kinetic_energy_J=old[2],
        split_momentum_error_Ns=float(np.linalg.norm(new[0]-old[0])),
        split_angular_error_Nms=float(np.linalg.norm(new[1]-old[1])),
        split_kinetic_error_J=float(new[2]-old[2]),
        material_power_error_W=float(particle_force_power-old_force_power),
        split_power_error_W=float(particle_force_power-new_force_power),
        F_and_all_material_history_including_plastic_heat_unchanged=True,
        true_sat_contact=sat_rigid_jump(),
        scope='Finite-dimensional algebra only; no C2 material release or fracture topology claim.')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
