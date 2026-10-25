"""Native MAT002 stress, hyperelastic identities and one3D release prefix."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from examples.common import _state_arrays, _valid_checkpoint_states
from examples.elastic_models import setup
from mpm_jax import advance
from mpm_jax.materials.elastic import (
    LinearMaterial, kirchhoff_stress, stvenant_kirchhoff_energy,
)


def check(reference, output):
    jax.config.update('jax_enable_x64', True)
    output.mkdir(parents=True, exist_ok=True)
    native = json.loads(reference.read_text())
    F = jnp.asarray([row['F'] for row in native['cases']], dtype=jnp.float64)
    expected = np.asarray([row['tau'] for row in native['cases']])
    mat = LinearMaterial(jnp.asarray(native['stiffness_public_voigt']), jnp.asarray(native['orientation']))
    tau = np.asarray(jax.jit(lambda f: kirchhoff_stress(f, mat, 'orthotropic-svk'))(F))
    P = jax.vmap(jax.grad(lambda f: stvenant_kirchhoff_energy(f, mat)))(F)
    ad_tau = np.asarray(P @ jnp.swapaxes(F, -1, -2))
    angle = .713
    R = jnp.asarray([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
    rotated_tau = np.asarray(kirchhoff_stress(R @ F, mat, 'orthotropic-svk'))
    rotated_expected = np.asarray(R @ jnp.asarray(tau) @ R.T)
    energy = np.asarray(stvenant_kirchhoff_energy(F, mat))
    rotated_energy = np.asarray(stvenant_kirchhoff_energy(R @ F, mat))
    relative = lambda a, b: float(np.max(np.abs(a-b))/max(1., float(np.max(np.abs(b)))))
    errors = dict(native_stress=relative(tau, expected), energy_AD_stress=relative(ad_tau, tau),
                  stress_objectivity=relative(rotated_tau, rotated_expected), energy_objectivity=relative(rotated_energy, energy))
    assert len(F) == 64 and min(np.linalg.eigvalsh(np.asarray(mat.stiffness))) > 0
    assert all(error < 2e-12 for error in errors.values()), errors
    np.savez_compressed(output/'constitutive-arrays.npz', F=F, native_tau=expected, jax_tau=tau, AD_tau=ad_tau,
                        rotated_tau=rotated_tau, rotated_expected_tau=rotated_expected, energy=energy, rotated_energy=rotated_energy)
    state, material, config = setup('orthotropic-svk', grid=128)
    assert _valid_checkpoint_states((state,))
    np.savez_compressed(output/'initial-state.npz', **_state_arrays(state))
    t = perf_counter()
    executable = advance.lower(state, material, config, 20).compile()
    compile_seconds = perf_counter()-t
    t = perf_counter(); final = executable(state, material); final.x.block_until_ready()
    solve_seconds = perf_counter()-t
    assert _valid_checkpoint_states((final,))
    assert np.array_equal(state.mass, final.mass) and np.array_equal(state.volume, final.volume)
    summaries = []
    for item in (state, final):
        mass = np.asarray(item.mass); velocity = np.asarray(item.v)
        summaries.append(dict(time_s=float(item.time), mass_kg=float(mass.sum()),
            momentum_kg_m_per_s=np.sum(mass[:, None]*velocity, axis=0).tolist(),
            minimum_J=float(np.linalg.det(np.asarray(item.F)).min()),
            kinetic_energy_J=float(.5*np.sum(mass[:, None]*velocity**2)),
            strain_energy_J=float(np.dot(np.asarray(item.volume), np.asarray(stvenant_kirchhoff_energy(item.F, material))))))
    momentum_error = float(np.linalg.norm(np.subtract(summaries[1]['momentum_kg_m_per_s'], summaries[0]['momentum_kg_m_per_s'])))
    momentum_scale = max(1., float(np.sum(np.asarray(final.mass)*np.linalg.norm(np.asarray(final.v), axis=1))))
    assert momentum_error <= 1e-10*momentum_scale
    np.savez_compressed(output/'final-state.npz', **_state_arrays(final))
    initial_energy = summaries[0]['kinetic_energy_J']+summaries[0]['strain_energy_J']
    final_energy = summaries[1]['kinetic_energy_J']+summaries[1]['strain_energy_J']
    report = dict(status='passed', native_reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                  native_source_sha256=native['source_sha256'], native_cases=64, tolerance=2e-12, normalized_errors=errors,
                  native_orientation_scope='Identity material axes only; superposed spatial rotations are covered, rotated material-axis adapter is not native-certified',
                  release=dict(grid=128, particles=len(state.x), steps=20, force='none', initial=summaries[0], final=summaries[1],
                               momentum_error=momentum_error, momentum_scale=momentum_scale,
                               energy_change_J=final_energy-initial_energy, relative_energy_change=(final_energy-initial_energy)/initial_energy,
                               compile_seconds=compile_seconds, solve_seconds=solve_seconds, finite_positive_J_mass_history_passed=True),
                  runtime=dict(jax=jax.__version__, numpy=np.__version__, devices=list(map(str, jax.devices()))),
                  scope='MAT002 finite-strain solid constitutive comparison and one free prestrain-release prefix; no temporal/spatial convergence or large-compression stability claim')
    (output/'review.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check(args.reference, args.output)
