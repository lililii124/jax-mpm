"""Check fixed physical fixtures and the unchanged grid-64 numerical baseline."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from examples.brittle_engineering import setup
from mpm_jax import step
from mpm_jax.contact_laws import boundary_reaction

jax.config.update("jax_enable_x64", True)


def check(baseline_file, output):
    spec = importlib.util.spec_from_file_location("examples._fixture_baseline", baseline_file)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    records = []
    definitions = []
    for grid in (32, 64, 96):
        case = setup("concrete-bending-mat111", grid=grid, ppc=1, frames=4)
        cfg = case.configs[0]
        left, right, _, _, loading = cfg.boundaries
        physical = [asdict(item) for item in (left, right, loading)]
        definitions.append(physical)
        nodes = np.arange(grid + 1) * cfg.dx
        counts = []
        for time in (0., .0006, .0012, .0015):
            displacement, _ = loading.motion(time)
            low = loading.lower[1] + float(displacement[1])
            high = loading.upper[1] + float(displacement[1])
            count = int(np.sum((nodes >= low) & (nodes <= high)))
            assert count > 0
            counts.append(count)
        records.append(dict(grid=grid, fixtures=physical, selected_y_nodes=counts))
    assert definitions[0] == definitions[1] == definitions[2]
    try:
        setup("concrete-bending-mat111", grid=16, ppc=1, frames=4)
    except ValueError as error:
        assert "fixtures require grid >= 32" in str(error)
    else:
        raise AssertionError("Underresolved fixtures were accepted")

    before = baseline.setup("concrete-bending-mat111", grid=64, ppc=3, frames=60)
    after = setup("concrete-bending-mat111", grid=64, ppc=3, frames=60)
    assert [asdict(cfg) for cfg in before.configs] == [asdict(cfg) for cfg in after.configs]
    assert before.duration == after.duration
    for old, new in zip(jax.tree.leaves((before.states, before.materials)),
                        jax.tree.leaves((after.states, after.materials))):
        np.testing.assert_array_equal(old, new)

    def advance(case):
        def body(_, state):
            return step(state, case.materials[0], case.configs[0])
        kernel = jax.jit(lambda state: jax.lax.fori_loop(0, 12, body, state))
        state = kernel(case.states[0])
        jax.block_until_ready(state)
        return state
    old_state, new_state = advance(before), advance(after)
    for old, new in zip(jax.tree.leaves(old_state), jax.tree.leaves(new_state)):
        np.testing.assert_array_equal(old, new)
    assert bool(new_state.valid)
    assert np.isfinite(np.asarray(new_state.x)).all()
    assert np.min(np.linalg.det(np.asarray(new_state.F))) > 0

    case = setup("concrete-bending-mat111", grid=32, ppc=1, frames=4)
    state = case.states[0]._replace(time=jnp.asarray(.2 * case.duration))
    material, cfg = case.materials[0], case.configs[0]
    reactions = jnp.stack([boundary_reaction(state, material, cfg, i) for i in range(len(cfg.boundaries))])
    following = step(state, material, cfg)
    dp = jnp.sum(state.mass[:, None] * (following.v - state.v), axis=0)
    np.testing.assert_allclose(dp, -cfg.dt * reactions.sum(0), rtol=2e-12, atol=1e-12)
    assert bool(following.valid) and float(reactions[-1, 1]) > 0
    report = dict(status="passed", geometry=records, grid64_particles=len(new_state.x),
        grid64_physical_steps_per_version=12, grid64_all_state_fields_bitwise_equal=True,
        material_parameters_and_configs_unchanged_at_grid64=True,
        grid32_prescribed_boundary_momentum_residual=float(jnp.linalg.norm(dp + cfg.dt * reactions.sum(0))),
        baseline_file_sha256=hashlib.sha256(Path(baseline_file).read_bytes()).hexdigest(),
        implementation_sha256=hashlib.sha256(Path("examples/brittle_engineering.py").read_bytes()).hexdigest(),
        scope="Fixture geometry and baseline identity, not full dynamic or mesh convergence")
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "geometry"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    check(args.baseline_file, args.output)
