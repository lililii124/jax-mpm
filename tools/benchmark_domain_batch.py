"""Compare domain midpoint batch sizes from the saved grid-48 frame 6."""

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
from mpm_jax.domain_mpm import step, kinetic_energy, elastic_energy
from mpm_jax import implicit_contact, normal_cohesive


RUN = ROOT / "outputs" / "domain-cohesive-notch-grid48"
OUT = ROOT / "outputs" / "domain-batch-benchmark.json"


def ready(value):
    return jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, value)


def compare_arrays(left, right):
    report = {}
    for name, a, b in zip(("x", "F", "v", "C", "plus", "minus", "maximum"), left, right):
        aa, bb = np.asarray(a), np.asarray(b)
        report[name] = {
            "max_abs": float(np.max(np.abs(aa - bb), initial=0.0)),
            "max_scaled": float(np.max(np.abs(aa - bb), initial=0.0) /
                                (1.0 + np.max(np.abs(aa), initial=0.0))),
        }
    return report


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    jax.config.update("jax_enable_x64", True)
    definition = json.loads((RUN / "input.json").read_text())
    saved0 = np.load(RUN / "frame-000.npz")
    saved = np.load(RUN / "frame-006.npz")
    grid = int(definition["grid"])
    domain = 0.06
    dx = float(definition["dx_m"])
    dt = float(definition["dt_s"])
    boundary_specs = definition["boundary_conditions"]
    boundaries = tuple(MovingVelocityBox(
        tuple(spec["lower"]), tuple(spec["upper"]), tuple(spec["velocity"]),
        start=float(spec["start"]), end=float(spec["end"]),
        ramp_time=float(spec["ramp_time"]),
    ) for spec in boundary_specs)
    config = Config(dim=2, n_grid=grid, dt=dt, domain_size=domain,
                    material="neo-hookean", gravity=(0.0, 0.0),
                    boundary_cells=0, boundaries=boundaries)
    material = neo_hookean(float(definition["young_Pa"]), float(definition["poisson"]))
    cohesive_law = CohesiveLaw(**definition["law"])

    h = dx / 2.0
    cut = 0.03
    tip = 0.022
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
    initial_position_error = float(np.max(np.abs(np.asarray(state.x) - saved0["x"])))
    if initial_position_error > 1e-13:
        raise ValueError(f"Reconstructed reference positions differ from frame 000: {initial_position_error}")

    cracked = implicit_contact.line_interface((0.01, cut), (tip, cut), (0.0, 1.0), config)
    intact = implicit_contact.line_interface((tip, cut), (0.05, cut), (0.0, 1.0), config)
    line = implicit_contact.Interface(
        jnp.concatenate((cracked.plus, intact.plus)),
        jnp.concatenate((cracked.minus, intact.minus)),
        jnp.concatenate((cracked.area, intact.area)),
        intact.normal,
        cracked.valid & intact.valid,
    )
    bonded = jnp.arange(len(line.area)) >= len(cracked.area)
    contact = normal_cohesive.initialize(line, bonded)
    contact = contact._replace(plus=jnp.asarray(saved["plus"]),
                               minus=jnp.asarray(saved["minus"]),
                               maximum=jnp.asarray(saved["maximum"]))
    state = state._replace(x=jnp.asarray(saved["x"]), F=jnp.asarray(saved["F"]),
                           v=jnp.asarray(saved["v"]), C=jnp.asarray(saved["C"]))
    time0 = float(saved["time_s"])

    def run(batch_size):
        def one_step():
            return step(state, material, config, jnp.asarray(time0), fields=fields,
                        n_fields=2, tile_radius=1, batch_size=batch_size,
                        contact=contact, cohesive_law=cohesive_law)

        warmed = ready(one_step())
        durations = []
        result = None
        for _ in range(3):
            started = time.perf_counter()
            result = ready(one_step())
            durations.append(time.perf_counter() - started)
        following, new_contact, work, contact_work = result
        stored, dissipated = normal_cohesive.energy(new_contact, cohesive_law)
        metrics = {
            "valid": bool(jnp.all(following.valid) & new_contact.valid),
            "mean_seconds": float(np.mean(durations)),
            "samples_seconds": durations,
            "kinetic_energy": float(kinetic_energy(following, config)),
            "elastic_energy": float(elastic_energy(following, material)),
            "interface_stored": float(stored),
            "interface_dissipated": float(dissipated),
            "work": [float(x) for x in work],
            "contact_work": [float(x) for x in contact_work],
        }
        arrays = (following.x, following.F, following.v, following.C,
                  new_contact.plus, new_contact.minus, new_contact.maximum)
        return metrics, arrays

    report32, arrays32 = run(32)
    report128, arrays128 = run(128)
    result = {
        "case": "domain midpoint batch-size timing; single step from saved grid-48 frame 006",
        "source_frame_time_s": time0,
        "dt_s": dt,
        "grid": grid,
        "domains": len(state.x),
        "interface_quadrature": len(contact.area),
        "reconstructed_frame000_position_max_abs_m": initial_position_error,
        "runs": {"batch_size_32": report32, "batch_size_128": report128},
        "speed_ratio_32_over_128": report32["mean_seconds"] / report128["mean_seconds"],
        "cross_batch_max_differences": compare_arrays(arrays32, arrays128),
        "interpretation": "Execution-only batching comparison on one frozen saved input; not a physics or grid-convergence result.",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
