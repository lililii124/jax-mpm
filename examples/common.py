"""Shared output and time integration for the examples."""

import csv
from dataclasses import asdict
import json
import hashlib
import os
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import coupled_step, step
from mpm_jax.cli import render
from mpm_jax.initialization import diagnostics
from mpm_jax.core import State
from mpm_jax.materials import History


def _atomic_npz(path, **values):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        writer = np.savez if path.name.startswith("checkpoint_") else np.savez_compressed
        writer(stream, **values)
    temporary.replace(path)


def _atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _state_arrays(state):
    data = {k: np.asarray(v) for k, v in state._asdict().items() if k != "history"}
    data.update({f"history_{k}": np.asarray(v) for k, v in state.history._asdict().items()})
    return data


def _load_state(path):
    with np.load(path) as data:
        history = History(*(jnp.asarray(data[f"history_{k}"]) for k in History._fields))
        return State(*(history if k == "history" else jnp.asarray(data[k]) for k in State._fields))


def run(
    states,
    materials,
    configs,
    *,
    frames=30,
    substeps=20,
    output,
    label,
    forces=None,
    friction=0.0,
    gif=False,
    observer=None,
    metadata=None,
    render_output=True,
    resume=False,
):
    if frames < 1 or substeps < 1:
        raise ValueError("frames and substeps must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "frames_data").mkdir(exist_ok=True)
    dim = configs[0].dim
    dt = configs[0].dt

    @jax.jit
    def batch(states, time):
        def body(carry, i):
            states, impulse, external_impulse, external_work = carry
            load = tuple(None for _ in states) if forces is None else forces(states, time + i * dt)
            old_states = states
            if len(states) == 1:
                states = (step(states[0], materials[0], configs[0], load[0]),)
                change = jnp.zeros(dim, dtype=states[0].x.dtype)
            else:
                states, change = coupled_step(states, materials, configs, friction, load)
            for old, new, force in zip(old_states, states, load):
                if force is not None:
                    external_impulse += dt * jnp.sum(force, axis=0)
                    external_work += 0.5 * dt * jnp.sum(force * (old.v + new.v))
            return (states, impulse + change, external_impulse, external_work), None

        zero = jnp.zeros(dim, dtype=states[0].x.dtype)
        return jax.lax.scan(
            body, (states, zero, zero, jnp.asarray(0.0, dtype=zero.dtype)), jnp.arange(substeps)
        )[0]

    start = perf_counter()
    warm, *_ = batch(states, 0.0)
    warm[0].x.block_until_ready()
    compile_seconds = perf_counter() - start
    rows, positions, speeds, damages, temperatures, times, impulses = [], [], [], [], [], [], []
    solver_seconds = 0.0
    accumulated_impulse = np.zeros(dim)
    accumulated_external = np.zeros(dim)
    accumulated_work = 0.0
    initial_momentum = sum(
        np.sum(np.asarray(s.mass)[:, None] * np.asarray(s.v), axis=0) for s in states
    )
    gravity_force = sum(
        float(s.mass.sum()) * np.asarray(c.gravity) for s, c in zip(states, configs)
    )
    observations = []
    signature = hashlib.sha256(
        json.dumps(
            {
                "configs": [asdict(c) for c in configs],
                "materials": [
                    {k: np.asarray(v).tolist() for k, v in m._asdict().items()} for m in materials
                ],
                "frames": frames,
                "substeps": substeps,
                "label": label,
                "definition": metadata,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    first_frame = 0
    checkpoint_path = output / "checkpoint.json"
    if resume and checkpoint_path.exists():
        saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if saved["signature"] != signature:
            raise ValueError(
                "checkpoint belongs to a different case, resolution, material or duration"
            )
        states = tuple(_load_state(output / name) for name in saved["states"])
        rows, observations = saved["rows"], saved["observations"]
        accumulated_impulse = np.asarray(saved["contact_impulse"])
        accumulated_external = np.asarray(saved["external_impulse"])
        accumulated_work = saved["external_work"]
        solver_seconds = saved["solver_seconds"]
        first_frame = saved["frame"] + 1
        for i in range(first_frame):
            with np.load(output / "frames_data" / f"frame_{i:05d}.npz") as data:
                positions.append(data["x"])
                speeds.append(data["speed"])
                damages.append(data["damage"])
                temperatures.append(data["temperature"])
                times.append(float(data["time"]))
                impulses.append(data["contact_impulse"])
        reaction = np.array([observations[-1][f"boundary_impulse_{a}"] for a in "xyz"[:dim]])
        print(f"Resuming {label} at frame {first_frame - 1}/{frames}", flush=True)
    del warm
    for frame in range(first_frame, frames + 1):
        time = frame * substeps * dt
        if frame:
            start = perf_counter()
            states, impulse, external, work = batch(states, time - substeps * dt)
            states[0].x.block_until_ready()
            solver_seconds += perf_counter() - start
            accumulated_impulse += np.asarray(impulse)
            accumulated_external += np.asarray(external)
            accumulated_work += float(work)
        for body_id, (state, config, material) in enumerate(zip(states, configs, materials)):
            rows.append({"time": time, "body": body_id, **diagnostics(state, config, material)})
        momentum = sum(
            np.sum(np.asarray(s.mass)[:, None] * np.asarray(s.v), axis=0) for s in states
        )
        reaction = momentum - initial_momentum - time * gravity_force - accumulated_external
        observation = {
            "time": time,
            "external_work_trapezoidal": accumulated_work,
            **{f"boundary_impulse_{a}": float(reaction[d]) for d, a in enumerate("xyz"[:dim])},
        }
        if observer is not None:
            observation.update(observer(states, time))
        observations.append(observation)
        positions.append(np.concatenate([np.asarray(s.x) for s in states]))
        speeds.append(np.concatenate([np.linalg.norm(s.v, axis=1) for s in states]))
        damages.append(np.concatenate([np.asarray(s.history.damage) for s in states]))
        temperatures.append(np.concatenate([np.asarray(s.history.temperature) for s in states]))
        times.append(time)
        impulses.append(accumulated_impulse.copy())
        _atomic_npz(
            output / "frames_data" / f"frame_{frame:05d}.npz",
            x=positions[-1],
            speed=speeds[-1],
            damage=damages[-1],
            temperature=temperatures[-1],
            time=time,
            contact_impulse=impulses[-1],
        )
        state_files = []
        for i, state in enumerate(states):
            name = f"checkpoint_{frame % 2}_{i}.npz"
            _atomic_npz(output / name, **_state_arrays(state))
            state_files.append(name)
        _atomic_json(
            checkpoint_path,
            {
                "signature": signature,
                "frame": frame,
                "states": state_files,
                "rows": rows,
                "observations": observations,
                "contact_impulse": accumulated_impulse.tolist(),
                "external_impulse": accumulated_external.tolist(),
                "external_work": accumulated_work,
                "solver_seconds": solver_seconds,
            },
        )
        _atomic_json(
            output / "status.json",
            {
                "case": label,
                "pid": os.getpid(),
                "frame": frame,
                "frames": frames,
                "steps_completed": frame * substeps,
                "time": time,
                "solver_seconds": solver_seconds,
                "completed": False,
                "particles": sum(s.x.shape[0] for s in states),
            },
        )
        if frame % 5 == 0 or frame == frames:
            print(
                f"{label}: {frame}/{frames}, t={time:.6g}, Jmin={min(r['min_J'] for r in rows[-len(states):]):.4g}, Dmax={max(r['max_damage'] for r in rows[-len(states):]):.4g}",
                flush=True,
            )
    positions, speeds, damages, temperatures = map(
        np.asarray, (positions, speeds, damages, temperatures)
    )
    body = np.concatenate([np.full(s.x.shape[0], i) for i, s in enumerate(states)])
    np.savez_compressed(
        output / "trajectory.npz",
        x=positions,
        speed=speeds,
        damage=damages,
        temperature=temperatures,
        time=times,
        body=body,
        contact_impulse=impulses,
    )
    for i, state in enumerate(states):
        data = {k: v for k, v in state._asdict().items() if k != "history"}
        data.update({f"history_{k}": v for k, v in state.history._asdict().items()})
        np.savez_compressed(output / f"final_state_{i}.npz", **data)
    with (output / "diagnostics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (output / "observables.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=observations[0].keys())
        writer.writeheader()
        writer.writerows(observations)
    summary = {
        "case": label,
        "completed": True,
        "configs": [asdict(c) for c in configs],
        "materials": [
            {k: np.asarray(v).tolist() for k, v in m._asdict().items()} for m in materials
        ],
        "steps": frames * substeps,
        "simulation_time": times[-1],
        "particles": [s.x.shape[0] for s in states],
        "dtype": str(states[0].x.dtype),
        "jax_version": jax.__version__,
        "compile_and_warmup_seconds": compile_seconds,
        "devices": [str(d) for d in jax.devices()],
        "solver_seconds": solver_seconds,
        "min_J": min(r["min_J"] for r in rows),
        "max_damage": max(r["max_damage"] for r in rows),
        "max_plastic_strain": max(r["max_plastic_strain"] for r in rows),
        "max_temperature": max(r["max_temperature"] for r in rows),
        "contact_impulse_on_body_1": accumulated_impulse.tolist(),
        "boundary_impulse": reaction.tolist(),
        "external_impulse": accumulated_external.tolist(),
        "observables": {"initial": observations[0], "final": observations[-1]},
        "definition": metadata or {},
        "initial": rows[: len(states)],
        "final": rows[-len(states) :],
    }

    def finite_json(value):
        if isinstance(value, dict):
            return {k: finite_json(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite_json(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    summary = finite_json(summary)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _atomic_json(
        output / "status.json",
        {
            "case": label,
            "pid": os.getpid(),
            "frame": frames,
            "frames": frames,
            "steps_completed": frames * substeps,
            "time": times[-1],
            "solver_seconds": solver_seconds,
            "completed": True,
            "particles": sum(s.x.shape[0] for s in states),
        },
    )
    values, field = (damages, "Damage") if damages.max() > 0 else (speeds, "Speed")
    if render_output:
        render(
            positions,
            values,
            np.asarray(times),
            configs[0],
            output,
            label,
            gif,
            field_label=field,
        )
    return states, summary
