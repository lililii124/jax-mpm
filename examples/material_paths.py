"""Prescribed material-point loading paths, without a spatial discretization."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax.materials import initial_history
from mpm_jax.materials.history import equivalent
from mpm_jax.materials.plastic import update_thermal_plastic
from .scenarios import PLASTIC, material_model


PATHS = ("compression-unload", "hydrostatic", "shear-reversal", "confined-shear")


def loading_path(name, steps=160):
    t = np.linspace(0.0, 1.0, steps + 1)
    strain = np.zeros((steps + 1, 3, 3))
    if name == "compression-unload":
        strain[:, 1, 1] = -0.004 * (1 - abs(2 * t - 1))
    elif name == "hydrostatic":
        strain[:] = (-0.03 * (1 - abs(2 * t - 1)))[:, None, None] * np.eye(3)
    elif name == "shear-reversal":
        strain[:] = (-0.003 * np.minimum(4 * t, 1.0))[:, None, None] * np.eye(3)
        shear = 0.006 * np.sin(2 * np.pi * np.maximum((t - 0.25) / 0.75, 0.0))
        strain[:, 0, 1] = strain[:, 1, 0] = shear
    elif name == "confined-shear":
        strain[:] = (-0.0006 * t)[:, None, None] * np.eye(3)
        strain[:, 0, 1] = strain[:, 1, 0] = 0.06 * t
    else:
        raise ValueError(name)
    return strain


def execute(model, path, output, steps=160):
    material, density, kind, _ = material_model(model)
    strain = loading_path(path, steps)
    increments = jnp.asarray(np.diff(strain, axis=0))
    volumes = jnp.asarray(np.exp(np.trace(strain[1:], axis1=1, axis2=2)))
    history = initial_history(1, jnp.float64, material, density)
    from mpm_jax.materials import (
        update_bilinear,
        update_hjc,
        update_jh2,
        update_johnson_cook,
        update_rht,
        update_steinberg,
    )

    update = {
        "bilinear": update_bilinear,
        "thermal-plastic": update_thermal_plastic,
        "johnson-cook": update_johnson_cook,
        "jh2": update_jh2,
        "hjc": update_hjc,
        "rht": update_rht,
        "steinberg-guinan": update_steinberg,
    }[kind]
    dt = 1e-6

    def advance(h, inputs):
        de, J = inputs
        h = update(h, de[None], J[None], dt, material)
        return h, h

    _, histories = jax.jit(lambda h: jax.lax.scan(advance, h, (increments, volumes)))(history)
    if not np.all(histories.valid):
        raise FloatingPointError(f"{model}/{path}: constitutive integration failed")
    output = Path(output) / model / path
    output.mkdir(parents=True, exist_ok=True)
    stress = np.asarray(histories.stress[:, 0])
    data = np.column_stack(
        (
            np.arange(1, steps + 1) * dt,
            strain[1:, 1, 1],
            strain[1:, 0, 1],
            stress[:, 1, 1],
            stress[:, 0, 1],
            np.asarray(equivalent(histories.stress))[:, 0],
            np.asarray(histories.pressure)[:, 0],
            np.asarray(histories.plastic_strain)[:, 0],
            np.asarray(histories.damage)[:, 0],
            np.asarray(histories.temperature)[:, 0],
        )
    )
    columns = (
        "time",
        "strain_yy",
        "strain_xy",
        "stress_yy",
        "stress_xy",
        "equivalent_stress",
        "pressure",
        "plastic_strain",
        "damage",
        "temperature",
    )
    np.savetxt(output / "history.csv", data, delimiter=",", header=",".join(columns), comments="")
    np.savez_compressed(output / "history.npz", strain=strain[1:], **histories._asdict())
    summary = {
        "model": model,
        "path": path,
        "steps": steps,
        "dt": dt,
        "max_equivalent_stress": float(data[:, 5].max()),
        "max_plastic_strain": float(data[:, 7].max()),
        "max_damage": float(data[:, 8].max()),
        "max_temperature": float(data[:, 9].max()),
        "all_valid": True,
        "kinematics": "prescribed symmetric logarithmic strain increments; no rigid rotation",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("all",) + PLASTIC, default="all")
    parser.add_argument("--path", choices=("all",) + PATHS, default="all")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--output", type=Path, default=Path("outputs/material_paths"))
    args = parser.parse_args()
    if args.steps < 8:
        parser.error("steps must be >= 8")
    jax.config.update("jax_enable_x64", True)
    summaries = []
    for model in PLASTIC if args.model == "all" else (args.model,):
        for path in PATHS if args.path == "all" else (args.path,):
            summaries.append(execute(model, path, args.output, args.steps))
        jax.clear_caches()
    (args.output / "index.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
