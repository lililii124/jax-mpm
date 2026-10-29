"""Geometry, checkpoints and gravity balance for the toe excavation example."""
import json
import math
import os
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax.core import State
from mpm_jax.contact import coupled_step
from mpm_jax.materials import initial_history
from examples.mat173_toe_guards import step_guards


def atomic_json(p, x):
    def clean(v):
        if isinstance(v, np.generic):
            return clean(v.item())
        if isinstance(v, dict):
            return {k: clean(value) for k, value in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(value) for value in v]
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v

    p = Path(p)
    q = p.with_name(p.name + ".tmp")
    q.write_text(json.dumps(clean(x), indent=2, allow_nan=False) + "\n")
    os.replace(q, p)


def save_checkpoint(p, states):
    a = {}
    for n, s in zip(("soil_", "base_"), states):
        for k in ("x", "v", "F", "C", "mass", "volume", "time", "valid"):
            a[n + k] = np.asarray(getattr(s, k))
        for k in s.history._fields:
            a[n + "history_" + k] = np.asarray(getattr(s.history, k))
    q = Path(p).with_suffix(".npz.tmp")
    with q.open("wb") as f:
        np.savez_compressed(f, **a)
    os.replace(q, p)


def load_checkpoint(p):
    from mpm_jax.materials.history import history_from_arrays

    with np.load(p, allow_pickle=False) as a:
        out = []
        for n in ("soil_", "base_"):
            h = history_from_arrays(a, n + "history_")
            out.append(
                State(
                    *(jnp.asarray(a[n + k]) for k in ("x", "v", "F", "C", "mass", "volume")),
                    h,
                    jnp.asarray(a[n + "time"]),
                    jnp.asarray(a[n + "valid"]),
                )
            )
    return tuple(out)


def particles(sp):
    xs = np.arange(0.2 + sp / 2, 3.0, sp)
    ys = np.arange(0.5 + sp / 2, 1.8, sp)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    top = np.where(X <= 1.0, 1.8, 1.8 - 1.3 * (X - 1.0) / 2.0)
    m = (Y <= top) & (Y >= 0.5) & (X <= 3.0)
    return np.column_stack((X[m], Y[m]))


def foundation(sp=0.025):
    xs = np.arange(0.2 + sp / 2, 3.3, sp)
    ys = np.arange(0.2 + sp / 2, 0.5, sp)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack((X.ravel(), Y.ravel()))


def soil_state(x, sp, rho, m, dtype):
    n = len(x)
    xx = jnp.asarray(x, dtype)
    return State(
        xx,
        jnp.zeros_like(xx),
        jnp.broadcast_to(jnp.eye(3, dtype=dtype), (n, 3, 3)),
        jnp.zeros((n, 2, 2), dtype),
        jnp.full(n, rho * sp**2, dtype),
        jnp.full(n, sp**2, dtype),
        initial_history(n, dtype, m, rho),
        jnp.asarray(0.0, dtype),
        jnp.asarray(True),
    )


def base_state(x, sp, rho, m, dtype):
    n = len(x)
    xx = jnp.asarray(x, dtype)
    return State(
        xx,
        jnp.zeros_like(xx),
        jnp.broadcast_to(jnp.eye(2, dtype=dtype), (n, 2, 2)),
        jnp.zeros((n, 2, 2), dtype),
        jnp.full(n, rho * sp**2, dtype),
        jnp.full(n, sp**2, dtype),
        initial_history(n, dtype, m, rho),
        jnp.asarray(0.0, dtype),
        jnp.asarray(True),
    )


@partial(jax.jit, static_argnames=("configs", "steps"))
def advance(states, materials, configs, mu, dt, ramp_time, damping, steps):
    def body(carry, _):
        pair, ci, cl, ni, ti, di, gi, dipsoil, dipbase, gsoil, gbase, gmax, gvalid = carry
        nt = pair[0].time + dt
        phase = jnp.clip(nt / ramp_time, 0.0, 1.0)
        scale = 0.5 * (1 - jnp.cos(jnp.pi * phase))
        g = jnp.asarray((0.0, -9.81), dtype=pair[0].x.dtype)
        forces = tuple(s.mass[:, None] * scale * g[None, :] for s in pair)
        updated, imp, loss, normal, tangent = coupled_step(
            pair, materials, configs, mu, forces, return_diagnostics=True
        )

        def damp(s):
            ke = 0.5 * jnp.sum(s.mass[:, None] * s.v * s.v)
            newv = s.v * damping
            dp = jnp.sum(s.mass[:, None] * (newv - s.v), axis=0)
            return s._replace(v=newv, C=s.C * damping), ke * (1 - damping * damping), dp

        damped = tuple(damp(s) for s in updated)
        out = tuple(x[0] for x in damped)
        dl = tuple(x[1] for x in damped)
        dps = tuple(x[2] for x in damped)
        gsi = pair[0].mass.sum() * scale * g * dt
        gbi = pair[1].mass.sum() * scale * g * dt
        gp = gsi + gbi
        guard, good = step_guards(pair, out)
        return (
            out,
            ci + imp,
            cl + loss,
            ni + normal,
            ti + tangent,
            di + sum(dl),
            gi + gp,
            dipsoil + dps[0],
            dipbase + dps[1],
            gsoil + gsi,
            gbase + gbi,
            jnp.maximum(gmax, guard),
            gvalid & good,
        ), None

    z = jnp.zeros((2,), dtype=states[0].x.dtype)
    ini = (
        states,
        z,
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
        z,
        z,
        z,
        z,
        z,
        jnp.full((4,), -jnp.inf),
        jnp.asarray(True),
    )
    return jax.lax.scan(body, ini, None, length=steps)[0]
