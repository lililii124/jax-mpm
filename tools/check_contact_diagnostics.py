import json
from dataclasses import replace
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax.contact import coupled_step, contact_impulse, _mass_gradient
from mpm_jax.core import Config, State, particle_stencil, particle_to_grid, grid_update
from mpm_jax.materials import neo_hookean
from mpm_jax.materials.history import initial_history


def body(xlo, xhi, velocity, mass_density=1000.0):
    axis = [np.linspace(xlo[k], xhi[k], 2) for k in range(3)]
    x = np.stack(np.meshgrid(*axis, indexing="ij"), axis=-1).reshape(-1, 3)
    x = jnp.asarray(x, dtype=jnp.float64)
    n = len(x)
    v = jnp.broadcast_to(jnp.asarray(velocity, dtype=x.dtype), x.shape)
    eye = jnp.eye(3, dtype=x.dtype)
    return State(x, v, jnp.broadcast_to(eye, (n, 3, 3)), jnp.zeros((n, 3, 3), x.dtype),
                 jnp.full((n,), mass_density * 0.025**3, x.dtype),
                 jnp.full((n,), 0.025**3, x.dtype), initial_history(n, x.dtype),
                 jnp.asarray(0., x.dtype), jnp.asarray(True))


def close(a, b):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    return all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))


def contact_energy_independent(states, material, config, friction):
    supports = tuple(particle_stencil(s, config) for s in states)
    grids = tuple(particle_to_grid(s, material, config, st, None) for s, st in zip(states, supports))
    velocities = tuple(grid_update(m, p, replace(config, boundary_cells=0, boundaries=()))
                       for m, p in grids)
    ma, mb = grids[0][0], grids[1][0]
    ga, gb = tuple(_mass_gradient(s, st, ma.shape[0]) for s, st in zip(states, supports))
    normal = gb / jnp.where(mb > 0, mb, 1.)[:, None] - ga / jnp.where(ma > 0, ma, 1.)[:, None]
    center = [jnp.sum(s.mass[:, None] * s.x, axis=0) / jnp.sum(s.mass) for s in states]
    normal = jnp.where((jnp.linalg.norm(normal, axis=-1) > 1e-12)[:, None], normal,
                       center[1] - center[0])
    va, vb, impulse = contact_impulse(ma, mb, *velocities, normal, friction)
    before = .5 * jnp.sum(ma[:, None] * velocities[0]**2) + .5 * jnp.sum(mb[:, None] * velocities[1]**2)
    after = .5 * jnp.sum(ma[:, None] * va**2) + .5 * jnp.sum(mb[:, None] * vb**2)
    return float(before - after), np.asarray(jnp.sum(impulse, axis=0)), int(jnp.count_nonzero((ma > 0) & (mb > 0)))


def run(label, states, config, material, friction):
    default_state, default_impulse = coupled_step(states, (material, material), (config, config), friction)
    diagnostic_state, impulse, loss, normal, tangent = coupled_step(
        states, (material, material), (config, config), friction, return_diagnostics=True)
    same_state = close(default_state, diagnostic_state)
    same_impulse = np.array_equal(np.asarray(default_impulse), np.asarray(impulse))
    ref_loss, ref_impulse, overlap_nodes = contact_energy_independent(states, material, config, friction)
    out = {"case": label, "state_identical": same_state, "impulse_identical": same_impulse,
           "impulse": np.asarray(impulse).tolist(), "normal_impulse": float(normal),
           "tangent_impulse": float(tangent), "diagnostic_contact_loss_J": float(loss),
           "independent_grid_kinetic_loss_J": ref_loss,
           "loss_abs_diff_J": abs(float(loss) - ref_loss), "overlap_grid_nodes": overlap_nodes,
           "valid": bool(np.asarray(diagnostic_state[0].valid) & np.asarray(diagnostic_state[1].valid))}
    assert same_state and same_impulse and out["valid"]
    assert np.allclose(np.asarray(default_impulse), ref_impulse, rtol=0, atol=1e-14)
    assert abs(float(loss) - ref_loss) <= max(1e-13, abs(ref_loss) * 1e-12)
    if label == "no_contact":
        assert overlap_nodes == 0 and float(loss) == 0. and float(normal) == 0. and float(tangent) == 0.
        assert np.array_equal(np.asarray(impulse), np.zeros(3))
    else:
        assert overlap_nodes > 0 and float(normal) > 0. and float(loss) > 0.
    return out


def main():
    jax.config.update("jax_enable_x64", True)
    config = Config(n_grid=24, dt=1e-5, domain_size=1., gravity=(0., 0., 0.),
                    boundary_cells=0, material="neo-hookean", dim=3)
    material = neo_hookean(young=2e4, poisson=.2)
    contact = (body((.25,.35,.35),(.275,.375,.375),(1.,.25,0.)),
               body((.275,.35,.35),(.30,.375,.375),(-.8,-.15,0.)))
    apart = (body((.18,.35,.35),(.205,.375,.375),(1.,.25,0.)),
             body((.72,.35,.35),(.745,.375,.375),(-.8,-.15,0.)))
    result = {"source": "mpm_jax/contact.py coupled_step", "friction": .2,
              "contact": run("contact", contact, config, material, .2),
              "no_contact": run("no_contact", apart, config, material, .2)}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
