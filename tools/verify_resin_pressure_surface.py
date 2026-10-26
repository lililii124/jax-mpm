"""Pressure-domain controls for empty and particle-owned unsupported cells."""
import json

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as j
import numpy as np

from mpm_jax.resin_apic import MACLayout
from mpm_jax.resin_mac import pressure_projection
from mpm_jax.resin_transport_extension import support_closed_domain


def verify():
    layout = MACLayout((3, 3, 3), (0., 0., 0.), 1.)
    mass = [j.ones((4, 3, 3)), j.ones((3, 4, 3)), j.ones((3, 3, 4))]
    fixed = tuple(j.zeros_like(v, dtype=bool) for v in mass)
    baseline = j.zeros(layout.shape, dtype=bool).at[1, 1, 1].set(True)
    open_cells = j.ones(layout.shape, dtype=bool)
    x = j.array([[1.5, 1.5, 1.5]])
    active = j.array([True])
    missing = (mass[0].at[1, 1, 1].set(0.), mass[1], mass[2])

    empty, _, valid_empty = support_closed_domain(
        x, ~active, missing, fixed, baseline, open_cells, layout)
    assert bool(valid_empty) and not bool(empty[1, 1, 1])
    owned, _, valid_owned = support_closed_domain(
        x, active, missing, fixed, baseline, open_cells, layout)
    assert not bool(valid_owned) and not bool(owned[1, 1, 1])

    # One supported pressure equality has six unit face masses. Its exact
    # multiplier is the divergence divided by six; all other pressures are zero.
    velocity = (j.zeros_like(mass[0]).at[2, 1, 1].set(1.),
                j.zeros_like(mass[1]), j.zeros_like(mass[2]))
    result = pressure_projection(velocity, tuple(mass), fixed, baseline,
                                 1., .1, preconditioner='mixed-lsqr',
                                 max_iterations=3200)
    assert bool(result.valid)
    expected = [np.asarray(v).copy() for v in velocity]
    for k in range(3):
        lower = [1, 1, 1]
        upper = lower.copy()
        upper[k] += 1
        expected[k][tuple(lower)] += 1./6.
        expected[k][tuple(upper)] -= 1./6.
    error = max(float(np.max(np.abs(np.asarray(v)-ref)))
                for v, ref in zip(result.velocity, expected))
    assert error < 1e-14
    rejected = pressure_projection(velocity, missing, fixed, baseline,
                                   1., .1, preconditioner='mixed-lsqr',
                                   max_iterations=3200)
    assert not bool(rejected.valid) and not bool(rejected.coverage_valid)
    return dict(empty_unsupported_surface_excluded=True,
                occupied_unsupported_cell_rejected=True,
                original_pressure_coverage_gate_retained=True,
                analytic_supported_projection_error=error,
                projection_work_identity_J=float(result.work_identity_J))


if __name__ == '__main__':
    print(json.dumps(verify()))
