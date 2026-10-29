"""Compare the operator-chain evaluator with the original full response graph."""
from types import SimpleNamespace
import hashlib
import json
import numpy as np
import unittest
import tempfile
from pathlib import Path
import jax
import jax.numpy as jnp

from mpm_jax.reduced_tl import newmark, response
from mpm_jax.reduced_case_pullback import FixedMassCasePullback, configure_sqp_evaluator

jax.config.update('jax_enable_x64', True)


def model(tied):
    settings = SimpleNamespace(stage='fixed-mass', horizon=.01, pulse_duration=.004,
        peak_pressure=2., displacement_limit=.01, stress_limit=1.)
    def operators(q):
        return (jnp.diag(jnp.exp(q)+1.),
            jnp.array([[5.+jnp.exp(q[0]), .2], [.2, 7.+jnp.exp(q[1])]]),
            jnp.asarray(1.8468))
    loads = np.array([[1., .2], [1., .2], [1., .2]]) if tied else np.array([[1., .2], [.3, 1.], [.7, .5]])
    projected = dict(rank=2, operators=operators, loads=jnp.asarray(loads),
        back=jnp.array([[1., .3], [.2, 1.]]),
        skin_stress=jnp.arange(24, dtype=float).reshape(2, 6, 2)/20.)
    def metrics(q):
        mass, stiffness, total_mass = operators(q)
        cases = []
        for load in projected['loads']:
            _, displacement, _, energy = newmark(mass, stiffness, load, settings, .001)
            result = response(projected, displacement, settings, 4, .005)
            error = jnp.max(jnp.abs(energy[:, 0]+energy[:, 1]-energy[:, 2]))
            relative = error/jnp.maximum(jnp.max(jnp.abs(energy)), 1e-30)
            cases.append(jnp.concatenate((result, jnp.asarray([error, relative]))))
        cases = jnp.stack(cases)
        maximum = jnp.max(cases[:, 1])
        record = dict(cases=cases, mass_kg=total_mass, response=maximum,
            actual_peak_back_m=jnp.max(cases[:, 3]), actual_peak_skin_Pa=jnp.max(cases[:, 4]),
            displacement_bound=maximum, stress_bound=jnp.max(cases[:, 2]),
            constraints=jnp.array([maximum-1, jnp.max(cases[:, 2])-1, total_mass/1.8468-1]))
        return maximum, record
    base = dict(plan=dict(uniform_mass_kg=1.8468),
        support=SimpleNamespace(positions=np.zeros((2, 3)), volume=np.ones(2)),
        geometry=dict(design=np.ones(2, dtype=bool), skin=np.zeros(2, dtype=bool)),
        properties=lambda q: (jnp.full(2, .18), jnp.full(2, 486.), jnp.full(2, 1e8), None))
    return dict(settings=settings, projected=projected, q0=np.zeros(2), dt=.001,
        time_chunk=4, ks_relative_error=.005, base=base, metrics=metrics,
        signature=lambda: ('original', dict(original=True)))


def check_full_graph_equivalence_and_dynamic_controls(tied):
    configured = model(tied)
    backend = FixedMassCasePullback(configured)
    def original(q):
        mass, stiffness, _ = configured['projected']['operators'](q)
        records = []
        for load in configured['projected']['loads']:
            _, displacement, _, _ = newmark(mass, stiffness, load,
                configured['settings'], configured['dt'])
            records.append(response(configured['projected'], displacement,
                configured['settings'], 4, .005))
        records = jnp.stack(records)
        return jnp.concatenate((jnp.max(records[:, 1])[None], 1-records[:, 2]))
    baseline_sizes = None
    for q in [np.array([.1, -.2]), np.array([-.3, .4]), np.array([.1, -.2])]:
        actual = backend(q)
        np.testing.assert_allclose(actual['values'], original(q), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(actual['jacobian'], jax.jacrev(original)(q), rtol=1e-10, atol=1e-13)
        if tied:
            np.testing.assert_array_equal(actual['diagnostics']['active_displacement_case_weights'], np.ones(3)/3)
        if baseline_sizes is None:
            baseline_sizes = backend.cache_sizes()
        assert backend.cache_sizes() == baseline_sizes
        assert backend(q) is actual


class CasePullbackTests(unittest.TestCase):
    def test_distinct_cases(self):
        check_full_graph_equivalence_and_dynamic_controls(False)

    def test_exact_maximum_ties(self):
        check_full_graph_equivalence_and_dynamic_controls(True)

    def test_binding_and_rejections(self):
        check_binding_and_rejections(self)

    def test_sqp_checkpoint_uses_opt_in_signature(self):
        from examples.blast_sandwich_reduced import optimize_sqp
        configured = model(False)
        with tempfile.TemporaryDirectory() as directory:
            result = optimize_sqp(configured, directory, iterations=1, evaluator='case-pullback')
            self.assertTrue(result['valid'] and result['feasible'])
            self.assertIn('sqp_evaluator', result['binding'])
            with np.load(Path(directory)/'sqp.npz', allow_pickle=False) as saved:
                self.assertEqual(str(saved['signature']), result['signature'])
            with self.assertRaises(ValueError):
                optimize_sqp(configured, directory, iterations=1, resume=True)


def check_binding_and_rejections(test):
    configured = model(False)
    configure_sqp_evaluator(configured, 'case-pullback')
    signature, binding = configured['signature']()
    assert signature == hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    assert binding['sqp_evaluator']['all_time_space_stress_samples_retained']
    configure_sqp_evaluator(configured)
    assert configured['signature']()[0] == 'original'
    backend = FixedMassCasePullback(configured)
    for q in [np.zeros(2, dtype=np.float32), np.array([np.nan, 0.]), np.zeros(3)]:
        with test.assertRaises(ValueError):
            backend(q)
    configured['settings'].stage = 'minimum-mass'
    with test.assertRaises(ValueError):
        FixedMassCasePullback(configured)
    with test.assertRaises(ValueError):
        configure_sqp_evaluator(configured, 'case-pullback')


if __name__ == '__main__':
    unittest.main()
