"""Large four-edge-clamped plate loading on a rectangular background grid."""

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import time

import numpy as np

from examples.woven_geometry import ARCHITECTURES
from examples.woven_large_geometry import sample_large_weave


LOADS = ("sphere-impact", "hemisphere-indentation", "pressure-bending")
ROOT = Path(__file__).resolve().parents[1]
WOVEN_MATERIAL = {
    "matrix_young": 2e9, "matrix_poisson": .3, "matrix_density": 1100.,
    "yarn_young": [40e9, 4e9, 4e9], "yarn_poisson": [.25, .25, .30],
    "yarn_shear": [1.5e9, 2e9, 2e9], "yarn_density": 1400.,
}
SOURCE_FILES = (
    "examples/woven_large_plate.py", "examples/woven_large_geometry.py",
    "examples/woven_geometry.py", "mpm_jax/packed_woven.py",
    "mpm_jax/rigid_sphere_contact.py", "mpm_jax/fibre_mpm.py",
    "mpm_jax/materials/woven.py", "mpm_jax/materials/elastic.py",
    "mpm_jax/materials/johnson_cook.py", "mpm_jax/materials/history.py",
    "mpm_jax/materials/__init__.py", "mpm_jax/particle_fields.py",
)


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def digest(value):
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def array_descriptor(value):
    array = np.ascontiguousarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest()}


def source_record():
    files = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    return {"files": files, "fingerprint": digest(files)}


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(json_value(value), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def save_npz(path, **values):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def acoustic_speed(parameters):
    E, nu, G = (np.asarray(parameters[name]) for name in ("yarn_young", "yarn_poisson", "yarn_shear"))
    S = np.diag(np.r_[1/E, 1/G[[2, 1, 0]]])
    for i, j, k in ((0, 1, 0), (0, 2, 1), (1, 2, 2)):
        S[i, j] = S[j, i] = -nu[k]/E[i]
    kelvin = np.diag([1., 1., 1., np.sqrt(2), np.sqrt(2), np.sqrt(2)])
    yarn = np.linalg.eigvalsh(kelvin@np.linalg.inv(S)@kelvin).max()/parameters["yarn_density"]
    matrix_E, matrix_nu = parameters["matrix_young"], parameters["matrix_poisson"]
    mu = matrix_E/(2*(1+matrix_nu))
    bulk = matrix_E/(3*(1-2*matrix_nu))
    matrix = max(3*bulk, 2*mu)/parameters["matrix_density"]
    return float(np.sqrt(max(matrix, yarn)))


def make_plan(architecture="plain", loading="sphere-impact", *, frames=60, probe_steps=0, dx=.0005):
    if loading not in LOADS:
        raise ValueError("unknown loading")
    if type(frames) is not int or frames < 1 or type(probe_steps) is not int or probe_steps < 0:
        raise ValueError("frames must be positive and probe_steps nonnegative integers")
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("dx must be finite and positive")
    sample = sample_large_weave(architecture)
    parameters, duration = dict(WOVEN_MATERIAL), 200e-6
    diameter, tool_mass, initial_speed = .1, .01, 20.
    label, case_path = "woven-large", [architecture, loading]
    mass = sample.volume*np.where(sample.phase == 1, parameters["yarn_density"], parameters["matrix_density"])
    tool_type = "finite-mass rigid hollow sphere; illustrative mass independent of diameter"
    speed = acoustic_speed(parameters)
    substeps = int(np.ceil(duration/(frames*.15*dx/speed)))
    full_steps = frames*substeps
    if probe_steps >= full_steps:
        raise ValueError("a probe must stop before the full protocol")
    dt = duration/full_steps
    frame_steps = list(range(0, full_steps+1, substeps))
    stop_steps = probe_steps or full_steps
    output_steps = [step for step in frame_steps if step <= stop_steps]
    if output_steps[-1] != stop_steps:
        output_steps.append(stop_steps)
    # A positive-support grid node is above the physical particle-cell surface.
    top = float(sample.positions[:, 1].max())
    fraction = top/dx-np.floor(top/dx-.5)
    weights = np.array([.5*(1.5-fraction)**2, .75-(fraction-1)**2, .5*(fraction-.5)**2])
    top_nodes = (np.floor(top/dx-.5)+np.arange(3))*dx
    occupied_top = float(top_nodes[weights > 0].max())
    center = [0., occupied_top+.0005+diameter/2, 0.]
    shape = np.ceil(np.array([sample.metadata["size_xz_m"][0], .012, sample.metadata["size_xz_m"][1]])/dx).astype(int)+7
    plan = {
        "schema": 1, "series": label, "case_path": case_path,
        "architecture": architecture,
        "loading": loading, "material_kind": "woven", "material_parameters": parameters,
        "material_scope": "assumed SI demonstration parameters, not a calibrated impact or fracture experiment",
        "geometry": sample.metadata, "particles": len(sample.positions), "mass_kg": float(mass.sum()),
        "grid": {"dx_m": float(dx), "type": "single bonded rectangular APIC field",
                 "y_bounds_m": [-.008, .004], "padding_cells": 3, "max_nodes": 8_000_000,
                 "estimated_shape": shape.tolist(), "estimated_nodes": int(np.prod(shape)),
                 "contact_half_width_m": .045,
                 "contact_capacity": 65536},
        "clamp": {"profile": "four-edge-full-clamp", "band_width_m": .005,
                  "components": [0, 1, 2], "owner_order": ["xmin", "xmax", "zmin", "zmax"],
                  "corner_assignment": "first matching owner; one impulse per node", "backing": False,
                  "support_margin_m": 1.5*dx},
        "tool": {"kind": tool_type, "diameter_m": diameter, "mass_kg": tool_mass,
                 "initial_center_m": center,
                 "initial_velocity_m_s": [0., -initial_speed if loading == "sphere-impact" else 0., 0.],
                 "initial_occupied_grid_top_m": occupied_top, "initial_nodal_gap_m": .0005,
                 "initial_physical_surface_gap_m": center[1]-diameter/2-sample.metadata["thickness_m"]/2,
                 "motion": "free finite mass" if loading == "sphere-impact" else "prescribed half-cosine displacement" if loading == "hemisphere-indentation" else "no contact tool",
                 "stroke_m": .004 if loading == "hemisphere-indentation" else 0.},
        "pressure": {"peak_Pa": .25e6 if loading == "pressure-bending" else 0.,
                     "scope": "vertical dead traction on fixed-reference top particles inside the four clamp bands"},
        "loading_duration_s": 150e-6, "duration_s": duration,
        "CFL": .15, "initial_acoustic_bound_m_s": speed, "dt_s": dt,
        "full_steps": full_steps, "frame_steps": frame_steps, "output_steps": output_steps,
        "frames": frames, "probe_steps": probe_steps, "stop_steps": stop_steps,
        "production_eligible": False,
        "scope": "coarse structural demonstration; fixed 2mm physical thickness and actual rigid-tool inertia",
        "reference_arrays": {name: array_descriptor(value) for name, value in {
            "x": sample.positions, "volume": sample.volume, "mass": mass, "phase": sample.phase,
            "yarn_id": sample.yarn_id, "family_id": sample.family_id, "orientation": sample.orientation}.items()},
        "source": source_record(),
    }
    return sample, plan


def pressure_force(sample, plan):
    force = np.zeros_like(sample.positions)
    if plan["loading"] != "pressure-bending":
        return force
    x = sample.positions
    size = np.asarray(sample.metadata["size_xz_m"])
    spacing = np.asarray(sample.metadata["particle_spacing_m"])
    band = plan["clamp"]["band_width_m"]
    selected = ((x[:, 1] == x[:, 1].max()) & (np.abs(x[:, 0]) < size[0]/2-band)
                & (np.abs(x[:, 2]) < size[1]/2-band))
    force[selected, 1] = -plan["pressure"]["peak_Pa"]*spacing[0]*spacing[2]
    return force


def run_case(sample, plan, output, *, resume=False):
    """Integrate the declared full protocol or an explicitly separate prefix probe."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from mpm_jax.materials.woven import woven_elastic, woven_energy
    from mpm_jax.packed_woven import (PackedWovenConfig, WovenState, make_plate_grid,
                                     make_state, step_with_diagnostics, loading_fraction)
    from mpm_jax.rigid_sphere_contact import SphereState
    from mpm_jax.particle_fields import material_metadata

    if source_record() != plan["source"]:
        raise ValueError("source differs from the declared plan")
    folder = Path(output).joinpath(*plan["case_path"])
    folder.mkdir(parents=True, exist_ok=True)
    signature = digest(plan)
    if (folder/"plan.json").exists():
        if not resume:
            raise FileExistsError("output exists; use an exact-plan resume or a new output root")
        if json.loads((folder/"plan.json").read_text()) != plan:
            raise ValueError("resume requires the identical physical plan, reference arrays and source")
    elif resume and (folder/"checkpoint.npz").exists():
        raise ValueError("checkpoint has no immutable plan")
    else:
        save_json(folder/"plan.json", plan)
    kind = plan["material_kind"]
    if kind != "woven":
        raise ValueError("this driver is restricted to the declared elastic woven material")
    material = woven_elastic(orientation=sample.orientation, phase=sample.phase, **plan["material_parameters"])
    mass = sample.volume*np.where(sample.phase == 1, WOVEN_MATERIAL["yarn_density"], WOVEN_MATERIAL["matrix_density"])
    settings = plan["grid"]
    grid = make_plate_grid(sample.positions, size=sample.metadata["size_xz_m"], dx=settings["dx_m"],
        band=plan["clamp"]["band_width_m"], y_bounds=settings["y_bounds_m"],
        padding_cells=settings["padding_cells"], max_nodes=settings["max_nodes"],
        contact_half_width=settings["contact_half_width_m"])
    config = PackedWovenConfig(plan["dt_s"], plan["loading"], material_model=kind,
        loading_duration=plan["loading_duration_s"], stroke=plan["tool"]["stroke_m"],
        initial_center=tuple(plan["tool"]["initial_center_m"]),
        contact_half_width=settings["contact_half_width_m"], contact_capacity=settings["contact_capacity"])
    state = make_state(sample.positions, sample.volume, mass, material, kind=kind)
    sphere = SphereState(jnp.asarray(plan["tool"]["initial_center_m"]),
        jnp.asarray(plan["tool"]["initial_velocity_m_s"]), jnp.asarray(plan["tool"]["mass_kg"]),
        jnp.asarray(plan["tool"]["diameter_m"]/2))
    force_base = jnp.asarray(pressure_force(sample, plan))
    # accepted, valid, residual, velocity violation, iterations, contact nodes,
    # overflows, invalid windows, external work, peak yarn strain, min J.
    aggregate = jnp.asarray([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1.])
    rows, diagnostics, times = [], [], []
    start_frame, solver_seconds, compile_seconds = -1, 0., 0.
    fields = {"phase": sample.phase, "yarn_id": sample.yarn_id, "family_id": sample.family_id}
    field_record = {"file": "particle_fields.npz", "fields": {key: array_descriptor(v) for key, v in fields.items()}}
    layout_record = {"lower": np.asarray(grid.layout.lower[0]).tolist(),
        "shape": np.asarray(grid.layout.shape[0]).tolist(), "origin": np.asarray(grid.layout.origin).tolist(),
        "dx_m": grid.layout.dx, "nodes": grid.layout.node_count,
        "clamp_owner": array_descriptor(np.asarray(grid.clamp_owner)),
        "contact_indices": array_descriptor(np.asarray(grid.contact_indices))}
    if resume and (folder/"checkpoint.npz").exists():
        with np.load(folder/"checkpoint.npz", allow_pickle=False) as data:
            if str(data["signature"]) != signature or str(data["jax_version"]) != jax.__version__:
                raise ValueError("checkpoint source/plan or JAX version differs")
            aggregate = jnp.asarray(data["aggregate"])
            if float(aggregate[1]) != 1. or not bool(data["state_valid"]):
                raise ValueError("a rejected step is diagnostic evidence, not an automatically resumable state")
            values = {name: jnp.asarray(data["state_"+name]) for name in WovenState._fields if name != "history"}
            for name, value in state._asdict().items():
                if name != "history":
                    saved_value = data["state_"+name]
                    if saved_value.shape != value.shape or saved_value.dtype != np.asarray(value).dtype or not np.isfinite(saved_value).all():
                        raise ValueError(f"checkpoint state {name} has invalid shape, dtype or values")
            if not np.array_equal(data["state_mass"], mass) or not np.array_equal(data["state_volume"], sample.volume):
                raise ValueError("checkpoint mass or reference volume differs from the immutable reference")
            if not np.all(data["state_fibre_id"] == 0) or not np.all(np.linalg.det(data["state_F"]) > 0):
                raise ValueError("checkpoint field identity or deformation is invalid")
            state = WovenState(**values, history=None)
            sphere = SphereState(*(jnp.asarray(data["sphere_"+name]) for name in SphereState._fields))
            start_frame, solver_seconds = int(data["frame"]), float(data["solver_seconds"])
            tables = json.loads(str(data["tables"]))
            rows, diagnostics, times = tables["rows"], tables["diagnostics"], tables["times"]
            if (not 0 <= start_frame < len(plan["output_steps"])
                    or any(len(table) != start_frame+1 for table in (rows, diagnostics, times))
                    or int(aggregate[0]) != plan["output_steps"][start_frame]
                    or not np.isclose(float(state.time), int(aggregate[0])*config.dt, rtol=1e-12, atol=0)
                    or times[-1] != float(state.time)):
                raise ValueError("checkpoint and saved time history are not the same complete prefix")
            if float(sphere.mass) != plan["tool"]["mass_kg"] or float(sphere.radius) != plan["tool"]["diameter_m"]/2:
                raise ValueError("checkpoint rigid-tool inertia or radius differs from the declared plan")
        if json.loads((folder/"layout.json").read_text()) != layout_record:
            raise ValueError("reconstructed grid or clamp differs from saved layout")
    else:
        save_npz(folder/"particle_fields.npz", **fields)
        save_npz(folder/"reference.npz", x=sample.positions, volume=sample.volume, mass=mass,
                 orientation=sample.orientation, **fields)
        save_npz(folder/"layout_arrays.npz", clamp_owner=np.asarray(grid.clamp_owner),
                 contact_indices=np.asarray(grid.contact_indices))
        save_json(folder/"layout.json", layout_record)
    n, saved = len(sample.positions), len(plan["output_steps"])
    arrays = {}
    for name, shape, dtype in (("x", (saved, n, 3), np.float64), ("speed", (saved, n), np.float32),
                               ("damage", (saved, n), np.float32), ("temperature", (saved, n), np.float32)):
        path = folder/f"trajectory_{name}.npy"
        array = np.lib.format.open_memmap(path, mode="r+" if start_frame >= 0 else "w+", dtype=dtype, shape=shape)
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise ValueError("trajectory shape/dtype differs from the complete output plan")
        arrays[name] = array
    if start_frame >= 0 and not np.array_equal(arrays["x"][start_frame], np.asarray(state.x)):
        raise ValueError("checkpoint coordinates differ from saved trajectory frame")

    @jax.jit
    def advance(current, ball, totals, parameters, packed_grid, force, steps):
        def body(_, carry):
            before, previous_ball, values = carry
            def attempt(_):
                candidate, next_ball, detail = step_with_diagnostics(before, parameters, packed_grid, config, previous_ball, force)
                updated = values.at[2].set(jnp.maximum(values[2], detail.contact_impulse_residual))
                updated = updated.at[3].set(jnp.maximum(values[3], detail.contact_velocity_violation))
                updated = updated.at[4].set(jnp.maximum(values[4], detail.contact_iterations))
                updated = updated.at[5].set(jnp.maximum(values[5], detail.contact_nodes))
                updated = updated.at[6].add(detail.contact_overflow).at[7].add(~detail.contact_window_valid)
                def accept(_):
                    result = updated.at[0].add(1.)
                    applied = force*loading_fraction(candidate.time, config.loading_duration) if plan["loading"] == "pressure-bending" else force
                    result = result.at[8].add(jnp.sum(applied*(candidate.x-before.x)))
                    if kind == "woven":
                        green = .5*(jnp.swapaxes(candidate.F, -1, -2)@candidate.F-jnp.eye(3))
                        local = jnp.swapaxes(parameters.orientation, -1, -2)@green@parameters.orientation
                        strain = jnp.max(jnp.where(parameters.phase == 1, jnp.linalg.norm(local, axis=(-2, -1)), 0.))
                        result = result.at[9].set(jnp.maximum(result[9], strain))
                    result = result.at[10].set(jnp.minimum(result[10], jnp.min(jnp.linalg.det(candidate.F))))
                    return candidate, next_ball, result
                return jax.lax.cond(detail.valid, accept,
                    lambda _: (before, previous_ball, updated.at[1].set(0.)), None)
            return jax.lax.cond(values[1] > 0, attempt, lambda _: carry, None)
        return jax.lax.fori_loop(0, steps, body, (current, ball, totals))

    def state_arrays():
        result = {"state_"+name: np.asarray(value) for name, value in state._asdict().items() if name != "history"}
        if state.history is not None:
            result.update({"history_"+name: np.asarray(value) for name, value in state.history._asdict().items()})
        result.update({"sphere_"+name: np.asarray(value) for name, value in sphere._asdict().items()})
        return result

    initial_ball_momentum = plan["tool"]["mass_kg"]*np.asarray(plan["tool"]["initial_velocity_m_s"])
    reference_edges = ((np.abs(sample.positions[:, 0]) >= sample.metadata["size_xz_m"][0]/2-.005)
                       | (np.abs(sample.positions[:, 2]) >= sample.metadata["size_xz_m"][1]/2-.005))
    center_roi = np.linalg.norm(sample.positions[:, (0, 2)], axis=1) <= .005

    def record(frame):
        x, v, F, C = (np.asarray(getattr(state, name)) for name in ("x", "v", "F", "C"))
        J = np.linalg.det(F)
        instant = float(state.time)
        arrays["x"][frame], arrays["speed"][frame] = x, np.linalg.norm(v, axis=1)
        arrays["damage"][frame] = 0. if state.history is None else np.asarray(state.history.damage)
        arrays["temperature"][frame] = 293.15 if state.history is None else np.asarray(state.history.temperature)
        for array in arrays.values():
            array.flush()
        translational = .5*float(np.sum(mass*np.sum(v*v, axis=1)))
        affine = grid.layout.dx**2/8*float(np.sum(mass*np.einsum("pij,pij->p", C, C)))
        ball_v, ball_x = np.asarray(sphere.velocity), np.asarray(sphere.center)
        ball_ke = .5*float(sphere.mass)*float(ball_v@ball_v) if plan["loading"] == "sphere-impact" else None
        energy = float(jnp.sum(state.volume*woven_energy(state.F, material))) if kind == "woven" else None
        momentum = np.sum(mass[:, None]*v, axis=0)
        boundary = np.asarray(state.boundary_impulse)
        contact = np.asarray(state.contact_impulse)
        external = np.asarray(state.external_impulse)
        plate_residual = momentum-boundary-contact-external
        total_residual = momentum+float(sphere.mass)*ball_v-initial_ball_momentum-boundary-external if plan["loading"] == "sphere-impact" else None
        interval = instant-times[-1] if times else 0.
        previous = np.array([[rows[-1][f"edge_{g}_impulse_{axis}_Ns"] for axis in "xyz"] for g in range(4)]) if rows else np.zeros((4, 3))
        applied_force = (np.asarray(state.group_impulses)-previous)/interval if interval else np.zeros((4, 3))
        row = {"time": instant, "accepted_steps": int(aggregate[0]),
            "center_mean_displacement_y_m": float(np.mean(x[center_roi, 1]-sample.positions[center_roi, 1])),
            "minimum_displacement_y_m": float(np.min(x[:, 1]-sample.positions[:, 1])),
            "maximum_displacement_m": float(np.max(np.linalg.norm(x-sample.positions, axis=1))),
            "clamp_position_error_m": float(np.max(np.linalg.norm(x[reference_edges]-sample.positions[reference_edges], axis=1))),
            "clamp_speed_error_m_s": float(np.max(np.linalg.norm(v[reference_edges], axis=1))),
            "kinetic_translational_J": translational, "kinetic_apic_affine_J": affine,
            "kinetic_energy_J": translational+affine, "stored_elastic_energy_J": energy,
            "ball_kinetic_energy_J": ball_ke, "actual_fixed_clamp_work_J": 0.,
            "prescribed_tool_work_J": float(state.tool_work), "particle_external_work_J": float(aggregate[8]),
            "contact_kinetic_loss_J": float(state.contact_loss), "grid_projection_work_J": float(state.grid_projection_work),
            "peak_step_yarn_green_strain_norm": float(aggregate[9]) if kind == "woven" else None,
            "max_plastic_strain": float(np.max(np.asarray(state.history.plastic_strain))) if state.history is not None else 0.,
            "max_damage": float(np.max(arrays["damage"][frame])),
            "plate_momentum_residual_Ns": float(np.linalg.norm(plate_residual)),
            "system_momentum_residual_Ns": float(np.linalg.norm(total_residual)) if total_residual is not None else None,
            "max_contact_impulse_residual_Ns": float(aggregate[2]),
            "max_contact_velocity_violation_m_s": float(aggregate[3]),
            "max_contact_iterations": int(aggregate[4]), "max_contact_nodes": int(aggregate[5]),
            "contact_overflow_events": int(aggregate[6]), "contact_invalid_window_events": int(aggregate[7])}
        for axis, name in enumerate("xyz"):
            row[f"ball_center_{name}_m"], row[f"ball_velocity_{name}_m_s"] = float(ball_x[axis]), float(ball_v[axis])
            row[f"plate_momentum_{name}_Ns"] = float(momentum[axis])
            row[f"boundary_impulse_{name}_Ns"], row[f"contact_impulse_{name}_Ns"], row[f"external_impulse_{name}_Ns"] = float(boundary[axis]), float(contact[axis]), float(external[axis])
            for group in range(4):
                row[f"edge_{group}_impulse_{name}_Ns"] = float(state.group_impulses[group, axis])
                row[f"edge_{group}_interval_force_{name}_N"] = float(applied_force[group, axis])
        rows.append(row)
        diagnostics.append({"time": instant, "mass": float(mass.sum()), "reference_volume": float(sample.volume.sum()),
            "current_volume": float(np.sum(sample.volume*J)), "min_J": float(J.min()), "max_J": float(J.max()),
            "minimum_step_J": float(aggregate[10]), "state_valid": bool(state.valid),
            "coupling_valid": bool(aggregate[1]), "plate_momentum_residual_Ns": row["plate_momentum_residual_Ns"],
            "system_momentum_residual_Ns": row["system_momentum_residual_Ns"]})
        times.append(instant)
        for filename, table in (("observables.csv", rows), ("diagnostics.csv", diagnostics)):
            with (folder/filename).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=table[0], lineterminator="\n")
                writer.writeheader()
                writer.writerows(table)
        save_npz(folder/"trajectory.npz", time=np.asarray(times), body=np.zeros(n, dtype=np.int32))
        save_npz(folder/"checkpoint.npz", **state_arrays(), aggregate=np.asarray(aggregate), frame=np.asarray(frame),
                 signature=np.array(signature), solver_seconds=np.asarray(solver_seconds), jax_version=np.array(jax.__version__),
                 tables=np.array(json.dumps(json_value({"rows": rows, "diagnostics": diagnostics, "times": times}))))
        save_json(folder/"status.json", {"completed": False, "phase": "running", "frame": frame,
            "frames": saved-1, "steps": int(aggregate[0]), "target_steps": plan["stop_steps"],
            "full_steps": plan["full_steps"], "time": instant, "signature": signature,
            "coupling_valid": bool(aggregate[1])})

    try:
        if start_frame < 0:
            record(0)
            start_frame = 0
        started = time.perf_counter()
        compiled = advance.lower(state, sphere, aggregate, material, grid, force_base, jnp.asarray(0, dtype=jnp.int32)).compile()
        compile_seconds = time.perf_counter()-started
        for frame in range(start_frame+1, saved):
            steps = plan["output_steps"][frame]-int(aggregate[0])
            if steps <= 0:
                raise ValueError("saved step index is not a strict prefix of the plan")
            started = time.perf_counter()
            state, sphere, aggregate = compiled(state, sphere, aggregate, material, grid, force_base, jnp.asarray(steps, dtype=jnp.int32))
            jax.block_until_ready(state.x)
            solver_seconds += time.perf_counter()-started
            record(frame)
            if not bool(aggregate[1]):
                break
        if source_record() != plan["source"]:
            raise ValueError("numerical source changed during execution")
        finished = bool(aggregate[1]) and int(aggregate[0]) == plan["stop_steps"]
        completed = finished and plan["probe_steps"] == 0
        phase = "completed" if completed else "probe-complete" if finished else "failed"
        final = {key.removeprefix("state_"): value for key, value in state_arrays().items()}
        save_npz(folder/"final_state_0.npz", **final)
        definition = {**plan, "particle_spacing_xyz_m": sample.metadata["particle_spacing_m"],
                      "particle_spacing_m": min(sample.metadata["particle_spacing_m"]),
                      "display_geometry_scale": 1.}
        if plan["loading"] == "sphere-impact":
            definition["rigid_spheres"] = [{"radius": plan["tool"]["diameter_m"]/2,
                "times": times, "centers": [[row[f"ball_center_{axis}_m"] for axis in "xyz"] for row in rows],
                "motion_source": "actual integrated finite-mass sphere states; no imposed impact trajectory"}]
        elif plan["loading"] == "hemisphere-indentation":
            definition["rigid_hemisphere"] = {"radius": plan["tool"]["diameter_m"]/2,
                "center": plan["tool"]["initial_center_m"], "displacement": [0., -.004, 0.],
                "loading_duration": plan["loading_duration_s"], "motion_kind": "half-cosine-displacement-hold"}
        summary = {"case": "/".join([plan["series"], *plan["case_path"]]), "completed": completed,
            "probe_complete": bool(finished and plan["probe_steps"]), "production_eligible": False,
            "phase": phase, "particles": [n], "steps": int(aggregate[0]), "full_steps": plan["full_steps"],
            "simulation_time": float(state.time), "target_simulation_time": plan["duration_s"],
            "solver_seconds": solver_seconds, "compile_seconds": compile_seconds,
            "jax_version": jax.__version__, "numpy_version": np.__version__, "python_version": platform.python_version(),
            "dtype": "float64", "history_storage": "file-backed arrays",
            "configs": [{"dim": 3, "dt": config.dt, "domain_size": sample.metadata["size_xz_m"][0],
                         "n_grid": int(round(sample.metadata["size_xz_m"][0]/grid.layout.dx)),
                         "grid_type": "packed-rectangular", "failure_mode": "retain", "gravity": [0., 0., 0.]}],
            "materials": [material_metadata(material)], "definition": definition,
            "particle_fields": field_record, "layout": layout_record, "source": plan["source"],
            "initial": diagnostics[0], "final": diagnostics[-1],
            "observables": {"initial": rows[0], "final": rows[-1]},
            "output_scope": "damage is identically zero and temperature fixed because the elastic woven model has neither damage nor heat evolution",
            "energy_scope": "APIC translational and affine kinetic energy and woven stored energy; separate contact loss, fixed-grid projection loss and actual discrete particle external work.",
            "angular_momentum_scope": "radial frictionless sphere has no spin degree of freedom; global fixture angular-impulse closure is not reported"}
        save_json(folder/"summary.json", summary)
        save_json(folder/"status.json", {"completed": completed, "probe_complete": summary["probe_complete"],
            "phase": phase, "frame": len(rows)-1, "frames": saved-1, "steps": int(aggregate[0]),
            "target_steps": plan["stop_steps"], "full_steps": plan["full_steps"], "time": float(state.time),
            "signature": signature, "coupling_valid": bool(aggregate[1])})
        if not finished:
            raise FloatingPointError("plate/contact step rejected; last accepted state and failure diagnostics preserved")
        return summary
    except BaseException as error:
        save_json(folder/"exception.json", {"type": type(error).__name__, "message": str(error),
            "last_saved_frame": len(rows)-1, "completed": False, "source": plan["source"]})
        raise
    finally:
        for array in arrays.values():
            array.flush()
            array._mmap.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="plain")
    parser.add_argument("--case", choices=LOADS, default="sphere-impact")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--probe-steps", type=int, default=0)
    parser.add_argument("--dx", type=float, default=.0005)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/woven-large"))
    args = parser.parse_args(argv)
    sample, plan = make_plan(args.architecture, args.case, frames=args.frames,
                            probe_steps=args.probe_steps, dx=args.dx)
    if args.dry_run:
        print(json.dumps(json_value(plan), indent=2, allow_nan=False))
    else:
        summary = run_case(sample, plan, args.output, resume=args.resume)
        print(json.dumps({"completed": summary["completed"], "probe_complete": summary["probe_complete"],
                          "steps": summary["steps"], "particles": summary["particles"]}))


if __name__ == "__main__":
    main()
