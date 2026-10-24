"""A common displacement cycle for local Hertz, JKR, MD and cohesive BEM models."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax.local_contact_models import (
    MODELS, initial_local_contact_history, local_contact_model, update_local_contact,
)


DEFAULTS = dict(effective_modulus=1e6, effective_radius=.01,
                work_of_adhesion=4e-2/(3*np.pi), interaction_range=2e-6/np.pi,
                outer_radius=4e-4, rings=32)


def displacement_cycle(*, samples=121, initial=-4e-6, peak=2e-6, final=-4e-6):
    """Two linear displacement legs, sampled at the same states for every law."""
    if (type(samples) is not int or samples < 5 or samples % 2 != 1
            or not np.isfinite([initial, peak, final]).all() or peak <= max(initial, final)):
        raise ValueError("Odd samples >=5 and a finite peak above both endpoints required")
    parameter = np.linspace(0., 1., samples)
    indentation = np.where(parameter <= .5, initial+2*parameter*(peak-initial),
                            peak+2*(parameter-.5)*(final-peak))
    return parameter, indentation


def evaluate_cycle(model, indentation, *, parameters=None, max_sweeps=1200, tolerance=1e-9):
    """Run actual attachment/ring histories with lax.scan; no MPM body is solved."""
    settings = dict(DEFAULTS) if parameters is None else dict(parameters)
    law = local_contact_model(model, **settings)
    delta = np.asarray(indentation, dtype=float)
    if delta.ndim != 1 or not len(delta) or not np.all(np.isfinite(delta)):
        raise ValueError("One finite indentation history is required")
    if any(np.ndim(getattr(law, name)) != 0 for name in ("effective_modulus", "effective_radius", "work_of_adhesion")):
        raise ValueError("The reference cycle is one scalar contact")
    initial = initial_local_contact_history(law)

    @jax.jit
    def execute(values):
        return jax.lax.scan(lambda h, d: update_local_contact(
            h, d, law, max_sweeps=max_sweeps, tolerance=tolerance,
        ), initial, values)

    started = time.perf_counter()
    history, response = execute(jnp.asarray(delta))
    response.normal_force.block_until_ready()
    elapsed = time.perf_counter()-started
    return law, history, response, elapsed


def records(model, parameter, delta, response):
    values = {name: np.asarray(getattr(response, name)) for name in
              ("normal_force", "radius", "area", "potential", "active", "valid")}
    c = np.asarray(getattr(response, "adhesive_radius", response.radius))
    adhesive_area = np.asarray(getattr(response, "adhesive_area", np.zeros_like(delta)))
    phase = np.asarray(getattr(response, "phase", np.where(values["active"],
                       np.where(values["area"] > 0, 2, 1), 0)))
    if model == "self-consistent":
        equilibrium = np.asarray(response.details.equilibrium_residual)
        elastic = np.asarray(response.details.elastic_energy)
        cohesive = np.asarray(response.details.cohesive_energy)
        iterations = np.asarray(response.details.sweeps)
        domain_clear = np.asarray(response.details.domain_clear)
    else:
        equilibrium = np.zeros(len(delta))
        elastic = cohesive = None
        iterations = np.zeros(len(delta), dtype=int)
        domain_clear = None
    interval_work = .5*(values["normal_force"][1:]+values["normal_force"][:-1])*np.diff(delta)
    work = np.concatenate(([0.], np.cumsum(interval_work)))
    previous_active = np.concatenate(([False], values["active"][:-1]))
    result = []
    for i in range(len(delta)):
        result.append(dict(
            sample=i, protocol_fraction=float(parameter[i]), indentation_m=float(delta[i]),
            force_N=float(values["normal_force"][i]), contact_radius_m=float(values["radius"][i]),
            cohesive_radius_m=float(c[i]), contact_area_m2=float(values["area"][i]),
            cohesive_annulus_area_m2=float(adhesive_area[i]), potential_J=float(values["potential"][i]),
            sampled_trapezoidal_work_J=float(work[i]), active=bool(values["active"][i]),
            phase=int(phase[i]), attached_event=bool(values["active"][i] and not previous_active[i]),
            detached_event=bool(not values["active"][i] and previous_active[i]), valid=bool(values["valid"][i]),
            displacement_equation_residual_m=float(np.asarray(getattr(response, "displacement_residual", np.zeros(len(delta))))[i]),
            cohesive_energy_condition_residual=float(np.asarray(getattr(response, "energy_condition_residual", np.zeros(len(delta))))[i]),
            equilibrium_pressure_residual_normalized=float(equilibrium[i]) if model == "self-consistent" else None,
            elastic_energy_J=float(elastic[i]) if elastic is not None else None,
            cohesive_energy_J=float(cohesive[i]) if cohesive is not None else None,
            sweeps=int(iterations[i]) if model == "self-consistent" else None,
            domain_clear=bool(domain_clear[i]) if domain_clear is not None else None,
            exact_open_solution=bool(np.asarray(response.open_exact)[i]) if model == "self-consistent" else None,
        ))
    return result


def source_record():
    root = Path(__file__).resolve().parents[1]
    names = ["examples/contact_pull_off.py", "mpm_jax/local_contact_models.py",
             "mpm_jax/fibre_contact.py", "mpm_jax/maugis_dugdale.py", "mpm_jax/cohesive_bem.py"]
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}


def run_reference(model, *, output, samples=121, initial=-4e-6, peak=2e-6, final=-4e-6,
                  parameters=None, max_sweeps=1200, tolerance=1e-9):
    settings = dict(DEFAULTS) if parameters is None else dict(parameters)
    parameter, delta = displacement_cycle(samples=samples, initial=initial, peak=peak, final=final)
    law, history, response, elapsed = evaluate_cycle(model, delta, parameters=settings,
                                                  max_sweeps=max_sweeps, tolerance=tolerance)
    table = records(model, parameter, delta, response)
    valid = np.asarray(response.valid)
    completed = bool(valid.all() and np.asarray(history.valid).all())
    folder = Path(output)/model
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/"observables.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=table[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    saved = {"protocol_fraction": parameter, "indentation_m": delta}
    saved.update({"history_final_"+name: np.asarray(value) for name, value in history._asdict().items()})
    saved.update({"response_"+name: np.asarray(value) for name, value in response._asdict().items()
                  if name != "details"})
    if hasattr(response, "details"):
        saved.update({"details_"+name: np.asarray(value) for name, value in response.details._asdict().items()})
        saved.update({"model_"+name: np.asarray(value) for name, value in law.base._asdict().items()})
    np.savez_compressed(folder/"history.npz", **saved)
    summary = dict(schema=1, model=model, completed=completed, samples=samples,
                   coupled_mpm=False, production_eligible=False,
                   scope="local quasistatic circular contact reference; no deformable MPM body",
                   parameters=settings, protocol=dict(initial_m=initial, peak_m=peak, final_m=final,
                                                       fraction_is_not_dynamic_time=True),
                   solver=dict(max_sweeps=max_sweeps, tolerance=tolerance,
                               seconds_including_JIT_and_execution=elapsed),
                   first_invalid_sample=int(np.flatnonzero(~valid)[0]) if not completed else None,
                   all_equilibria_valid=completed, final_active=bool(table[-1]["active"]),
                   force_min_N=min(row["force_N"] for row in table),
                   force_max_N=max(row["force_N"] for row in table),
                   work_scope="saved-state trapezoidal force-displacement quadrature; snap energy is not friction loss",
                   runtime=dict(jax=jax.__version__, numpy=np.__version__, x64=bool(jax.config.jax_enable_x64)),
                   source_sha256=source_record(),
                   files={name: hashlib.sha256((folder/name).read_bytes()).hexdigest()
                          for name in ("observables.csv", "history.npz")})
    (folder/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODELS, "all"), default="all")
    parser.add_argument("--output", type=Path, default=Path("outputs/contact-reference"))
    parser.add_argument("--samples", type=int, default=121)
    parser.add_argument("--initial", type=float, default=-4e-6)
    parser.add_argument("--peak", type=float, default=2e-6)
    parser.add_argument("--final", type=float, default=-4e-6)
    for flag, key in (("modulus", "effective_modulus"), ("radius", "effective_radius"),
                      ("work", "work_of_adhesion"), ("range", "interaction_range"),
                      ("outer-radius", "outer_radius")):
        parser.add_argument("--"+flag, dest=key, type=float, default=DEFAULTS[key])
    parser.add_argument("--rings", type=int, default=32)
    parser.add_argument("--max-sweeps", type=int, default=1200)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    settings = {name: getattr(args, name) for name in DEFAULTS}
    summaries = []
    for name in MODELS if args.model == "all" else (args.model,):
        summary = run_reference(name, output=args.output, samples=args.samples,
                                initial=args.initial, peak=args.peak, final=args.final,
                                parameters=settings, max_sweeps=args.max_sweeps, tolerance=args.tolerance)
        summaries.append(summary)
        print(f"{name}: {summary['samples']} states, valid={summary['completed']}, "
              f"final_active={summary['final_active']}", flush=True)
    if not all(value["completed"] for value in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
