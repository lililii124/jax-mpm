"""Compare original and factored domain transfers on the saved grid-48 frame 6."""

from functools import partial
import json
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpm_jax import Config, MovingVelocityBox, neo_hookean
from mpm_jax.cohesive_crack import CohesiveLaw
from mpm_jax.cut_geometry import polygon
from mpm_jax.cut_fragments import fragment, split_fragment
from mpm_jax import domain_mpm, domain_transfer, implicit_contact, normal_cohesive


RUN = ROOT / "outputs" / "domain-cohesive-notch-grid48"
OUT = ROOT / "outputs" / "domain-stencil-separable-benchmark.json"


@partial(jax.jit, static_argnames=("config", "tile_radius", "batch_size"))
def batch_stencils_separable(states, config, tile_radius=1, batch_size=32):
    return domain_transfer._batch(
        lambda s: domain_transfer.domain_stencil_separable(s, config, tile_radius),
        states, batch_size,
    )


def ready(value):
    return jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, value)


def max_difference(a, b):
    aa, bb = np.asarray(a), np.asarray(b)
    absolute = float(np.max(np.abs(aa - bb), initial=0.0))
    scaled = absolute / (1.0 + float(np.max(np.abs(aa), initial=0.0)))
    return {"max_abs": absolute, "max_scaled": scaled}


def timed(call):
    ready(call())
    samples = []
    value = None
    for _ in range(3):
        started = time.perf_counter()
        value = ready(call())
        samples.append(time.perf_counter() - started)
    return value, samples


def build_input():
    definition = json.loads((RUN / "input.json").read_text())
    saved0 = np.load(RUN / "frame-000.npz")
    saved = np.load(RUN / "frame-006.npz")
    grid = int(definition["grid"])
    dx = float(definition["dx_m"])
    dt = float(definition["dt_s"])
    boundaries = tuple(MovingVelocityBox(
        tuple(s["lower"]), tuple(s["upper"]), tuple(s["velocity"]),
        start=float(s["start"]), end=float(s["end"]), ramp_time=float(s["ramp_time"]),
    ) for s in definition["boundary_conditions"])
    config = Config(dim=2, n_grid=grid, dt=dt, domain_size=0.06,
                    material="neo-hookean", gravity=(0.0, 0.0),
                    boundary_cells=0, boundaries=boundaries)
    h, cut, tip = dx / 2.0, 0.03, 0.022
    x_axis = 0.01 + (np.arange(round(0.04 / h)) + 0.5) * h
    y_axis = 0.02 + (np.arange(round(0.02 / h)) + 0.5) * h
    positions = jnp.asarray(np.stack(np.meshgrid(x_axis, y_axis, indexing="ij"), axis=-1).reshape(-1, 2))
    reference = polygon(h * jnp.asarray([[-0.5, -0.5], [0.5, -0.5],
                                        [0.5, 0.5], [-0.5, 0.5]]))
    parents = jax.vmap(lambda x: fragment(reference, definition["density_kg_m3"], x=x))(positions)
    children = jax.vmap(lambda s: split_fragment(s, jnp.asarray([0.0, 1.0]), cut))(parents)
    state = jax.tree.map(lambda a: a.reshape((-1,) + a.shape[2:]), children)
    fields = jnp.tile(jnp.arange(2), len(positions))
    keep = np.flatnonzero(np.asarray(state.active))
    state = jax.tree.map(lambda a: a[keep], state)
    fields = fields[keep]
    initial_error = float(np.max(np.abs(np.asarray(state.x) - saved0["x"])))
    if initial_error > 1e-13:
        raise ValueError(f"Reconstructed reference positions differ from frame 000: {initial_error}")
    state = state._replace(x=jnp.asarray(saved["x"]), F=jnp.asarray(saved["F"]),
                           v=jnp.asarray(saved["v"]), C=jnp.asarray(saved["C"]))

    cracked = implicit_contact.line_interface((0.01, cut), (tip, cut), (0.0, 1.0), config)
    intact = implicit_contact.line_interface((tip, cut), (0.05, cut), (0.0, 1.0), config)
    line = implicit_contact.Interface(
        jnp.concatenate((cracked.plus, intact.plus)), jnp.concatenate((cracked.minus, intact.minus)),
        jnp.concatenate((cracked.area, intact.area)), intact.normal, cracked.valid & intact.valid,
    )
    bonded = jnp.arange(len(line.area)) >= len(cracked.area)
    contact = normal_cohesive.initialize(line, bonded)
    contact = contact._replace(plus=jnp.asarray(saved["plus"]), minus=jnp.asarray(saved["minus"]),
                               maximum=jnp.asarray(saved["maximum"]))
    return definition, config, state, fields, contact, float(saved["time_s"]), initial_error


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    jax.config.update("jax_enable_x64", True)
    definition, config, state, fields, contact, time0, initial_error = build_input()
    material = neo_hookean(float(definition["young_Pa"]), float(definition["poisson"]))
    cohesive_law = CohesiveLaw(**definition["law"])

    original_support, original_support_times = timed(
        lambda: domain_transfer.batch_stencils(states=state, config=config,
                                               tile_radius=1, batch_size=32)
    )
    candidate_support, candidate_support_times = timed(
        lambda: batch_stencils_separable(state, config, tile_radius=1, batch_size=32)
    )
    support_differences = {
        name: max_difference(getattr(original_support, name), getattr(candidate_support, name))
        for name in ("weights", "gradients", "dpos", "moment", "inverse_moment", "volume")
    }
    support_differences["indices_equal"] = bool(jnp.array_equal(original_support.indices,
                                                                candidate_support.indices))
    support_differences["valid_equal"] = bool(jnp.array_equal(original_support.valid,
                                                              candidate_support.valid))

    def step_call(batch_size=32):
        return domain_mpm.step(
            state, material, config, jnp.asarray(time0), fields=fields, n_fields=2,
            tile_radius=1, batch_size=batch_size, contact=contact, cohesive_law=cohesive_law,
        )

    original_result, original_step_times = timed(step_call)
    original_batch = domain_mpm.batch_stencils
    jax.clear_caches()
    domain_mpm.batch_stencils = batch_stencils_separable
    try:
        candidate_result, candidate_step_times = timed(step_call)
    finally:
        domain_mpm.batch_stencils = original_batch

    original_state, original_contact, original_work, original_contact_work = original_result
    candidate_state, candidate_contact, candidate_work, candidate_contact_work = candidate_result
    state_differences = {
        name: max_difference(getattr(original_state, name), getattr(candidate_state, name))
        for name in ("x", "F", "v", "C")
    }
    state_differences.update({
        "interface_plus": max_difference(original_contact.plus, candidate_contact.plus),
        "interface_minus": max_difference(original_contact.minus, candidate_contact.minus),
        "interface_maximum": max_difference(original_contact.maximum, candidate_contact.maximum),
        "work": [max_difference(a, b) for a, b in zip(original_work, candidate_work)],
        "contact_work": [max_difference(a, b) for a, b in zip(original_contact_work, candidate_contact_work)],
        "valid_equal": bool(jnp.array_equal(original_state.valid, candidate_state.valid)
                            & jnp.array_equal(original_contact.valid, candidate_contact.valid)),
    })
    result = {
        "case": "separable integrated basis candidate; saved grid-48 frame 006 support and one coupled step",
        "source_frame_time_s": time0,
        "dt_s": float(definition["dt_s"]),
        "grid": int(definition["grid"]),
        "domains": len(state.x),
        "interface_quadrature": len(contact.area),
        "frame000_reconstructed_position_max_abs_m": initial_error,
        "support": {
            "original_mean_seconds": float(np.mean(original_support_times)),
            "candidate_mean_seconds": float(np.mean(candidate_support_times)),
            "speed_ratio_original_over_candidate": float(np.mean(original_support_times)
                                                          / np.mean(candidate_support_times)),
            "differences": support_differences,
        },
        "coupled_step": {
            "original_mean_seconds": float(np.mean(original_step_times)),
            "candidate_mean_seconds": float(np.mean(candidate_step_times)),
            "speed_ratio_original_over_candidate": float(np.mean(original_step_times)
                                                          / np.mean(candidate_step_times)),
            "original_valid": bool(jnp.all(original_state.valid) & original_contact.valid),
            "candidate_valid": bool(jnp.all(candidate_state.valid) & candidate_contact.valid),
            "differences": state_differences,
        },
        "interpretation": "Candidate-only comparison; domain_mpm default remains the original transfer. Timing is execution-only, not a physics result.",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
