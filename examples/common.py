"""Shared output and time integration for the examples."""

import csv
from dataclasses import asdict
import json
import hashlib
import os
from pathlib import Path
import platform
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import coupled_step, step
from mpm_jax.cli import render
from mpm_jax.initialization import diagnostics
from mpm_jax.core import State
from mpm_jax.materials import History
from mpm_jax.particle_fields import (
    array_descriptor, checked_particle_fields, material_metadata,
)


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
    point_fields=None,
    stop_after_frame=None,
):
    if frames < 1 or substeps < 1:
        raise ValueError("frames and substeps must be positive")
    if stop_after_frame is not None and (type(stop_after_frame) is not int or not 0 < stop_after_frame <= frames):
        raise ValueError("stop_after_frame must be an integer between 1 and the full frame count")
    count = sum(s.x.shape[0] for s in states)
    point_fields = checked_particle_fields(point_fields, count)
    field_descriptors = {name: array_descriptor(value) for name, value in point_fields.items()}
    material_records = [material_metadata(material) for material in materials]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "frames_data").mkdir(exist_ok=True)
    dim = configs[0].dim
    dt = configs[0].dt
    spatial_materials = any(config.material == "woven-composite" for config in configs)
    material_arguments = materials if spatial_materials else ()

    @jax.jit
    def batch(states, time, material_arguments):
        constitutive = material_arguments if spatial_materials else materials

        def body(carry, i):
            states, impulse, external_impulse, external_work = carry
            load = tuple(None for _ in states) if forces is None else forces(states, time + i * dt)
            old_states = states
            if len(states) == 1:
                states = (step(states[0], constitutive[0], configs[0], load[0]),)
                change = jnp.zeros(dim, dtype=states[0].x.dtype)
            else:
                states, change = coupled_step(states, constitutive, configs, friction, load)
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
    warm, *_ = batch(states, 0.0, material_arguments)
    warm[0].x.block_until_ready()
    compile_seconds = perf_counter() - start
    del warm
    rows, times, impulses = [], [], []
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
    signature_data = {
        "configs": [asdict(c) for c in configs],
        "materials": material_records,
        "frames": frames,
        "substeps": substeps,
        "label": label,
        "definition": metadata,
    }
    if point_fields:
        signature_data["particle_fields"] = field_descriptors
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    first_frame = 0
    checkpoint_path = output / "checkpoint.json"
    saved = (
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if resume and checkpoint_path.exists()
        else None
    )
    if saved is not None and saved["signature"] != signature:
        raise ValueError("checkpoint belongs to a different case, resolution, material or duration")
    if saved is not None and stop_after_frame is not None and saved["frame"] < frames and stop_after_frame <= saved["frame"]:
        raise ValueError("A continuation endpoint must follow the saved checkpoint frame")
    restored = tuple(_load_state(output / name) for name in saved["states"]) if saved is not None else None
    if restored is not None and not _valid_checkpoint_states(restored):
        raise ValueError("An invalid numerical state cannot be resumed")
    if restored is not None and (len(restored) != len(states) or any(
        old.x.shape != reference.x.shape or not np.array_equal(old.mass, reference.mass)
        or not np.array_equal(old.volume, reference.volume) for old, reference in zip(restored, states)
    )):
        raise ValueError("Checkpoint particle shapes, mass or reference volume differ from the requested case")
    if restored is not None:
        _validate_resume_prefix(output, saved, restored, frames, substeps, dt,
                                any(config.failure_mode == "ballistic" for config in configs))
    if point_fields:
        _atomic_npz(output / "particle_fields.npz", **point_fields)
    dtype = np.result_type(*(np.asarray(s.x).dtype for s in states))

    def storage(name, shape, array_dtype=None):
        return np.lib.format.open_memmap(
            output / f"trajectory_{name}.npy", mode="w+",
            dtype=dtype if array_dtype is None else array_dtype, shape=shape
        )

    positions = storage("x", (frames + 1, count, dim))
    speeds = storage("speed", (frames + 1, count))
    damages = storage("damage", (frames + 1, count))
    temperatures = storage("temperature", (frames + 1, count))
    failures = (
        storage("failed", (frames + 1, count), np.uint8)
        if any(config.failure_mode == "ballistic" for config in configs)
        else None
    )
    if saved is not None:
        states = restored
        rows, observations = saved["rows"], saved["observations"]
        accumulated_impulse = np.asarray(saved["contact_impulse"])
        accumulated_external = np.asarray(saved["external_impulse"])
        accumulated_work = saved["external_work"]
        solver_seconds = saved["solver_seconds"]
        first_frame = saved["frame"] + 1
        for i in range(first_frame):
            with np.load(output / "frames_data" / f"frame_{i:05d}.npz") as data:
                positions[i] = data["x"]
                speeds[i] = data["speed"]
                damages[i] = data["damage"]
                temperatures[i] = data["temperature"]
                if failures is not None:
                    failures[i] = data["failed"]
                times.append(float(data["time"]))
                impulses.append(data["contact_impulse"])
        reaction = np.array([observations[-1][f"boundary_impulse_{a}"] for a in "xyz"[:dim]])
        print(f"Resuming {label} at frame {first_frame - 1}/{frames}", flush=True)
    for frame in range(first_frame, frames + 1):
        time = frame * substeps * dt
        if frame:
            start = perf_counter()
            states, impulse, external, work = batch(states, time - substeps * dt, material_arguments)
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
        positions[frame] = np.concatenate([np.asarray(s.x) for s in states])
        speeds[frame] = np.concatenate([np.linalg.norm(s.v, axis=1) for s in states])
        damages[frame] = np.concatenate([np.asarray(s.history.damage) for s in states])
        temperatures[frame] = np.concatenate([np.asarray(s.history.temperature) for s in states])
        extra_fields = {}
        if failures is not None:
            failures[frame] = np.concatenate([np.asarray(s.history.failed) for s in states])
            failures.flush()
            extra_fields["failed"] = failures[frame]
        for field in (positions, speeds, damages, temperatures):
            field.flush()
        times.append(time)
        impulses.append(accumulated_impulse.copy())
        _atomic_npz(
            output / "frames_data" / f"frame_{frame:05d}.npz",
            x=positions[frame],
            speed=speeds[frame],
            damage=damages[frame],
            temperature=temperatures[frame],
            time=time,
            contact_impulse=impulses[-1],
            **extra_fields,
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
        if stop_after_frame == frame and frame < frames:
            valid = _valid_checkpoint_states(states)
            segment = {
                "case": label, "completed": False, "segment_complete": valid,
                "phase": "checkpoint-ready" if valid else "failed",
                "signature": signature, "frame": frame, "frames": frames,
                "steps_completed": frame * substeps, "target_steps": frames * substeps,
                "time": time, "target_time": frames * substeps * dt,
                "particles": [s.x.shape[0] for s in states],
                "checkpoint": checkpoint_path.name,
                "scope": "Planned saved-frame continuation; the full physical protocol is incomplete.",
            }
            _atomic_json(output / "status.json", segment)
            _atomic_json(output / "segment.json", segment)
            _atomic_json(output / "summary.json", segment)
            for field in (positions, speeds, damages, temperatures, failures):
                if field is not None:
                    field._mmap.close()
            if not valid:
                raise FloatingPointError("Planned stop reached an invalid state; no resumable segment accepted")
            return states, segment
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
        **({"failed": failures} if failures is not None else {}),
    )
    for i, state in enumerate(states):
        data = {k: v for k, v in state._asdict().items() if k != "history"}
        data.update({f"history_{k}": v for k, v in state.history._asdict().items()})
        np.savez_compressed(output / f"final_state_{i}.npz", **data)
    with (output / "diagnostics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=dict.fromkeys(key for row in rows for key in row))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "observables.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=observations[0].keys())
        writer.writeheader()
        writer.writerows(observations)
    summary = {
        "case": label,
        "completed": True,
        "history_storage": "file-backed arrays",
        "configs": [asdict(c) for c in configs],
        "materials": material_records,
        "steps": frames * substeps,
        "simulation_time": times[-1],
        "particles": [s.x.shape[0] for s in states],
        "dtype": str(states[0].x.dtype),
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
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
    if point_fields:
        summary["particle_fields"] = {
            "file": "particle_fields.npz",
            "fields": field_descriptors,
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


def _valid_checkpoint_states(states):
    for state in states:
        shape = state.x.shape
        if len(shape) != 2 or shape[0] < 1 or shape[1] not in (2, 3):
            return False
        count, dim = shape
        if (state.v.shape != shape or state.F.shape != (count, dim, dim) or state.C.shape != (count, dim, dim)
                or state.mass.shape != (count,) or state.volume.shape != (count,)):
            return False
        if not bool(state.valid) or not bool(np.asarray(state.history.valid).all()):
            return False
        values = [value for key, value in state._asdict().items() if key != "history"]
        values.extend(state.history)
        if not all(np.isfinite(np.asarray(value)).all() for value in values):
            return False
        if not np.all(np.linalg.det(np.asarray(state.F)) > 0):
            return False
        if not np.all(np.asarray(state.mass) > 0) or not np.all(np.asarray(state.volume) > 0):
            return False
    return True


def _validate_resume_prefix(output, saved, states, frames, substeps, dt, ballistic):
    frame = saved["frame"]
    if type(frame) is not int or not 0 <= frame <= frames:
        raise ValueError("Checkpoint frame is outside the physical plan")
    expected_times = np.arange(frame + 1) * substeps * dt
    tolerance = max(dt * 1e-5, 1e-15)
    if len(saved["rows"]) != (frame + 1) * len(states) or len(saved["observations"]) != frame + 1:
        raise ValueError("Checkpoint tables do not cover the saved prefix")
    for index, row in enumerate(saved["rows"]):
        if row["body"] != index % len(states) or not np.isclose(
            row["time"], expected_times[index // len(states)], rtol=0, atol=tolerance
        ):
            raise ValueError("Checkpoint diagnostic time or body differs from the saved prefix")
    if not np.allclose([row["time"] for row in saved["observations"]], expected_times, rtol=0, atol=tolerance):
        raise ValueError("Checkpoint observation time differs from the saved prefix")
    for state in states:
        clock = np.asarray(state.time)
        epsilon = np.finfo(clock.dtype).eps if clock.dtype.kind == "f" else np.finfo(float).eps
        accumulated = frame * substeps * epsilon
        clock_error = (2 * accumulated / (1 - accumulated) * abs(expected_times[-1])
                       if accumulated < 1 else 0.)
        if accumulated >= 1 or not np.isclose(float(clock), expected_times[-1], rtol=0,
                                             atol=max(tolerance, clock_error)):
            raise ValueError("Checkpoint state time differs from the saved frame")
    count, dim = sum(len(state.x) for state in states), states[0].x.shape[1]
    shapes = {"x": (count, dim), "speed": (count,), "damage": (count,),
              "temperature": (count,), "time": (), "contact_impulse": (dim,)}
    if ballistic:
        shapes["failed"] = (count,)
    for index, expected_time in enumerate(expected_times):
        path = output / "frames_data" / f"frame_{index:05d}.npz"
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != set(shapes):
                raise ValueError("Checkpoint history fields do not match the requested run")
            for name, shape in shapes.items():
                value = data[name]
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError("Checkpoint history contains malformed or nonfinite arrays")
            if not np.isclose(float(data["time"]), expected_time, rtol=0, atol=tolerance):
                raise ValueError("Checkpoint history time differs from its saved frame")
            if not np.all((data["damage"] >= 0) & (data["damage"] <= 1)):
                raise ValueError("Checkpoint history damage is outside [0,1]")
            if ballistic and not np.all((data["failed"] == 0) | (data["failed"] == 1)):
                raise ValueError("Checkpoint history failure flags are not binary")
            if index == frame:
                for name, expected in (
                    ("x", np.concatenate([np.asarray(state.x) for state in states])),
                    ("damage", np.concatenate([np.asarray(state.history.damage) for state in states])),
                    ("temperature", np.concatenate([np.asarray(state.history.temperature) for state in states])),
                ):
                    if not np.array_equal(data[name], expected):
                        raise ValueError("Checkpoint state and last saved history frame differ")
                if ballistic and not np.array_equal(data["failed"], np.concatenate(
                    [np.asarray(state.history.failed) for state in states]
                )):
                    raise ValueError("Checkpoint state and last failure history frame differ")
