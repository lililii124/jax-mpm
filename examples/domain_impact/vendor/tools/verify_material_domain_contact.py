"""Finite material-cell SAT, clipping, rigid-motion and virtual-work checks."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

_MODULE = Path(__file__).resolve().parents[1]/'mpm_jax'/'material_domain_contact.py'
_SPEC = spec_from_file_location('material_domain_contact', _MODULE)
contact = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = contact
_SPEC.loader.exec_module(contact)


def main():
    I = np.eye(3)
    half = np.array((.5, .5, .5))
    overlap = contact.parallelepiped_intersection(
        np.zeros(3), I, half, np.array((0., .9, 0.)), I, half)
    assert overlap is not None
    normal, penetration, point, xi_a, xi_b = overlap
    assert np.allclose(normal, (0., 1., 0.), atol=1e-13)
    assert abs(penetration-.1) < 1e-12
    assert np.allclose(point, (0., .45, 0.), atol=1e-12)
    assert np.all(np.abs(xi_a) <= half+1e-12)
    assert np.all(np.abs(xi_b) <= half+1e-12)

    touching = contact.parallelepiped_intersection(
        np.zeros(3), I, half, np.array((0., 1., 0.)), I, half)
    assert touching is not None and touching[1] == 0.
    assert np.allclose(touching[2], (0., .5, 0.), atol=1e-12)
    assert contact.parallelepiped_intersection(
        np.zeros(3), I, half, np.array((0., 1.01, 0.)), I, half) is None

    theta = .37
    R = np.array(((np.cos(theta), -np.sin(theta), 0.),
                  (np.sin(theta), np.cos(theta), 0.), (0., 0., 1.)))
    oblique = contact.parallelepiped_intersection(
        np.zeros(3), I, half, np.array((.25, .38, .12)), R, half)
    assert oblique is not None
    assert np.all(np.abs(oblique[3]) <= half+1e-11)
    assert np.all(np.abs(oblique[4]) <= half+1e-11)

    # Material interpolation on each side sums to one and supplies a single
    # shared-point row with equal/opposite generalized impulse.
    weights = np.array(((.5, .5), (.4, .6)))
    gradients = np.array((((-1., 0., 0.), (1., 0., 0.)),
                          ((-.5, 0., 0.), (.5, 0., 0.))))
    xi_a = np.array((.2, 0., 0.)); xi_b = np.array((.1, 0., 0.))
    coeff_a = contact.point_velocity_coefficients(weights[0], gradients[0], xi_a)
    coeff_b = contact.point_velocity_coefficients(weights[1], gradients[1], xi_b)
    row = {'nodes_a': np.array((0, 1)), 'coeff_a': coeff_a,
           'nodes_b': np.array((2, 3)), 'coeff_b': coeff_b,
           'normal': np.array((1., 0., 0.))}
    J = contact.assemble_contact_jacobian([row], 4, np.zeros((4, 3), bool))
    velocity = np.array(((1., .2, 0.), (1., -.1, .3),
                         (0., 0., 0.), (0., 0., 0.)))
    lam = np.array((2.3,))
    generalized = np.asarray(J.T@lam).reshape(4, 3)
    power_j = float(np.dot(np.asarray(J@velocity.ravel()).ravel(), lam))
    power_adjoint = float(np.vdot(velocity, generalized))
    assert abs(power_j-power_adjoint) <= 2e-14*max(1., abs(power_j))
    assert np.allclose(generalized[:2].sum(0), lam[0]*row['normal'], atol=1e-14)
    assert np.allclose(generalized[2:].sum(0), -lam[0]*row['normal'], atol=1e-14)

    ref = SimpleNamespace(indices=np.array(((0, 1), (2, 3))),
                          weights=weights, gradients=gradients)
    va = contact._independent_point_velocity(velocity, ref, 0, xi_a)
    vb = contact._independent_point_velocity(velocity, ref, 1, xi_b)
    direct_closing = float(np.dot(row['normal'], va-vb))
    assert abs(direct_closing-float(np.asarray(J@velocity.ravel())[0])) < 1e-14
    common = np.ones_like(velocity)
    common_closing = float(np.asarray(J@common.ravel())[0])
    assert abs(common_closing) < 1e-14
    # One-row diagonal-mass e0 jump: the same shared-point transpose produces
    # zero post-closing and equal/opposite two-body momentum transfer.
    row_closing = float(np.asarray(J@velocity.ravel())[0])
    row_mobility = float((J.multiply(np.ones((1, 12)))@J.T).toarray()[0, 0])
    row_lambda = max(row_closing, 0.)/row_mobility
    vpost = velocity.reshape(-1)-np.asarray(J.T@np.array((row_lambda,))).ravel()
    assert abs(float(np.asarray(J@vpost)[0])) < 2e-14
    assert np.linalg.norm(vpost.reshape(4, 3).sum(0)-velocity.sum(0)) < 2e-14

    # A common infinitesimal rigid rotation is sampled identically at the
    # same material point on both independent body supports.
    grid_xy = np.array(((-1., -1., 0.), (1., -1., 0.),
                        (-1., 1., 0.), (1., 1., 0.)))
    gx = np.array(((-.25, 0., 0.), (.25, 0., 0.),
                   (-.25, 0., 0.), (.25, 0., 0.)))
    gy = np.array(((0., -.25, 0.), (0., -.25, 0.),
                   (0., .25, 0.), (0., .25, 0.)))
    omega = np.array((0., 0., 1.7))
    rigid_velocity = np.vstack((np.cross(omega, grid_xy),
                                np.cross(omega, grid_xy)))
    rigid_ref = SimpleNamespace(indices=np.array((tuple(range(4)), tuple(range(4, 8)))),
        weights=np.full((2, 4), .25), gradients=np.stack((gx+gy, gx+gy)))
    c = np.array((.2, -.3, 0.))
    rigid_points = [contact._independent_point_velocity(
        rigid_velocity, rigid_ref, p, c) for p in range(2)]
    assert np.allclose(rigid_points[0], np.cross(omega, c), atol=2e-14)
    assert np.allclose(rigid_points[1], rigid_points[0], atol=2e-14)
    print({'parallel_overlap_m': penetration, 'oblique_overlap_m': oblique[1],
           'contact_point': point.tolist(), 'power_error': abs(power_j-power_adjoint),
           'common_translation_closing': common_closing,
           'e0_row_lambda': row_lambda,
           'e0_post_closing_m_s': float(np.asarray(J@vpost)[0]),
           'e0_total_momentum_change': float(np.linalg.norm(vpost.reshape(4, 3).sum(0)-velocity.sum(0))),
           'rigid_rotation_point_velocity_error': float(np.linalg.norm(rigid_points[0]-np.cross(omega, c))),
           'action_reaction_error': float(np.linalg.norm(generalized[:2].sum(0)+generalized[2:].sum(0)))})


if __name__ == '__main__':
    main()
