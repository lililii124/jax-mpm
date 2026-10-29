"""Material return, plane strain and checkpoint regressions for toe excavation."""
from itertools import permutations
import json
import numpy as np
import pytest
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from mpm_jax.materials.history import initial_history
from mpm_jax.materials.mat173_softening import mat173_softening, softening_properties
from mpm_jax.materials.mat173_tensile_cutoff import (
    with_tensile_cutoff,
    update_mat173_softening_tensile_cutoff,
    separated_principal_increments,
)


def test_separated_material_point_returns():
    G = 5.86e6
    nu = 0.28
    lam = 2 * G * nu / (1 - 2 * nu)
    Ce = lam * np.ones((3, 3)) + 2 * G * np.eye(3)
    p = mat173_softening(
        shear_modulus=G,
        poisson=nu,
        friction_peak=np.deg2rad(38),
        friction_residual=np.deg2rad(20),
        cohesion_peak=1000.0,
        cohesion_residual=0.0,
        dilation_peak=np.deg2rad(2),
        dilation_residual=0.0,
        density=1800.0,
        characteristic_length=0.0125,
        fracture_slip=0.0005,
    )
    combined = with_tensile_cutoff(p, 0.0)
    separate = with_tensile_cutoff(p, 0.0, separate_shear_softening=True)
    labels = []
    Ds = []
    trials = []

    def add(label, D, stress, dep, Q=np.eye(3)):
        labels.append(label)
        Ds.append(D)
        trials.append(Q @ np.diag(np.array(stress) + Ce @ np.array(dep)) @ Q.T)

    add("pure_T", 0, [0, 0, 0], [0.001, 0, 0])
    add("zero_flow", 0, [0, 0, 0], [0, 0, 0])
    add("saturated_ambiguous", 1, [0, 0, 0], [0.001, 0, 0])
    phi = np.deg2rad(38)
    psi = np.deg2rad(2)
    hard = 2000 * np.cos(phi)
    low = ((1 + np.sin(phi)) * (-1000) - hard) / (1 - np.sin(phi))
    B = np.array([1 + np.sin(psi), 0, -(1 - np.sin(psi))])
    add("single_MC", 0, [-1000, -3000, low], 0.0001 * B)
    for D in [0.0, 0.8]:
        phi = np.deg2rad(38 - 18 * D)
        q = 2000 * (1 - D) * np.cos(phi) / (1 - np.sin(phi))
        stress = np.array([0, 0, -q])
        flow = np.array([0.0002, 0.0004, -0.0003])
        for k, perm in enumerate(permutations(range(3))):
            add(f"corner_D{D}_perm{k}", D, stress[list(perm)], flow[list(perm)])
        if D == 0:
            axis = np.array([1.0, 2.0, 3.0])
            axis /= np.linalg.norm(axis)
            angle = 0.417
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            Q = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
            add("corner_arbitrary_rotation", D, stress, flow, Q)
    old = initial_history(len(labels), jnp.float64, density=1800.0)._replace(
        stress=jnp.asarray(np.array(trials)),
        hardening=jnp.asarray(Ds) * 0.04,
        damage=jnp.asarray(Ds),
    )
    strain = jnp.zeros_like(old.stress)
    J = jnp.ones(len(labels))
    update = jax.jit(update_mat173_softening_tensile_cutoff)
    a = update(old, strain, J, 1.0, combined)
    b = update(old, strain, J, 1.0, separate)
    physical = [
        name
        for name in old._fields
        if name
        not in [
            "hardening",
            "damage",
            "postpeak",
            "tensile_plastic_strain",
            "partition_ambiguous_steps",
        ]
    ]
    physical_equal = {
        name: bool(np.array_equal(np.asarray(getattr(a, name)), np.asarray(getattr(b, name))))
        for name in physical
    }
    assert all(physical_equal.values()), physical_equal
    assert np.asarray(b.valid).all()
    out = {
        label: {
            "combined_kappa_increment": float(a.hardening[k] - old.hardening[k]),
            "separated_kappa_increment": float(b.hardening[k] - old.hardening[k]),
            "tensile_diagnostic_increment": float(b.tensile_plastic_strain[k]),
            "damage": float(b.damage[k]),
            "ambiguous_counter": int(b.partition_ambiguous_steps[k]),
            "valid": bool(b.valid[k]),
        }
        for k, label in enumerate(labels)
    }
    assert (
        abs(out["pure_T"]["separated_kappa_increment"]) < 1e-14
        and out["pure_T"]["combined_kappa_increment"] > 0
        and out["pure_T"]["damage"] == 0
    )
    assert abs(out["pure_T"]["tensile_diagnostic_increment"] - 0.001) < 1e-14
    assert (
        abs(
            out["single_MC"]["separated_kappa_increment"]
            - out["single_MC"]["combined_kappa_increment"]
        )
        < 1e-14
    )
    assert (
        out["saturated_ambiguous"]["separated_kappa_increment"] == 0
        and out["saturated_ambiguous"]["damage"] == 1
        and out["saturated_ambiguous"]["ambiguous_counter"] == 1
    )
    for D in [0.0, 0.8]:
        values = [out[f"corner_D{D}_perm{k}"]["separated_kappa_increment"] for k in range(6)]
        assert np.ptp(values) < 1e-14
    assert (
        abs(
            out["corner_arbitrary_rotation"]["separated_kappa_increment"]
            - out["corner_D0.0_perm0"]["separated_kappa_increment"]
        )
        < 1e-14
    )
    rotindex = labels.index("corner_arbitrary_rotation")
    refindex = labels.index("corner_D0.0_perm0")
    assert (
        np.max(abs(np.asarray(b.stress[rotindex]) - Q @ np.asarray(b.stress[refindex]) @ Q.T))
        < 1e-7
    )
    # Explicit diagnostic: unresolved allzero near-apex before saturation fails.
    diagnostic = separated_principal_increments(
        jnp.zeros((1, 3)),
        jnp.asarray([[0.001, 0, 0.0]]),
        jnp.asarray([0.0]),
        jnp.asarray([1e-15]),
        jnp.asarray([1e-6]),
        jnp.asarray([G]),
        jnp.asarray([False]),
    )
    assert not bool(diagnostic[2][0])
    phi_after, c_after, psi_after, _ = softening_properties(b.hardening, p)
    assert float(phi_after[0]) == float(p.friction_peak) and float(c_after[0]) == 1000


def test_plane_strain_particle_step_preserves_out_of_plane_constraints():
    from mpm_jax import Config, block, step
    from mpm_jax.materials import mat173

    base = mat173(
        shear_modulus=5.86e6,
        poisson=0.28,
        friction_angle=np.deg2rad(38),
        cohesion=1000.0,
        dilation_angle=np.deg2rad(2),
        density=1800.0,
    )
    material = with_tensile_cutoff(base)
    config = Config(
        n_grid=16,
        dt=3e-5,
        dim=2,
        gravity=(0.0, -9.81),
        boundary_cells=0,
        material="mat173-tensile-cutoff",
    )
    state = block(
        config,
        (0.4, 0.4),
        (0.5, 0.5),
        particles_per_cell=1,
        density=1800.0,
        material=material,
        dtype=jnp.float64,
    )
    state = state._replace(v=state.x @ jnp.asarray([[0.01, 0.02], [0.0, -0.01]]))
    result = step(state, material, config)
    assert result.x.shape[1] == 2 and result.C.shape[1:] == (2, 2)
    assert result.F.shape[1:] == (3, 3)
    np.testing.assert_array_equal(result.F[:, 2, 2], 1.0)
    np.testing.assert_array_equal(result.F[:, :2, 2], 0.0)
    np.testing.assert_array_equal(result.F[:, 2, :2], 0.0)
    assert bool(result.valid) and np.asarray(result.history.valid).all()
    assert np.max(np.abs(np.asarray(result.F[:, :2, :2]) - np.eye(2))) > 0
    with pytest.raises(ValueError, match="dim=3"):
        Config(material="mat173", dim=2)


def test_old_checkpoint_defaults_only_new_diagnostics(tmp_path):
    from examples.common import _state_arrays, _load_state
    from mpm_jax import Config, block
    from mpm_jax.materials.history import history_from_arrays
    from examples.mat173_toe_helpers import load_checkpoint, save_checkpoint

    state = block(
        Config(n_grid=16), (0.4, 0.4), (0.5, 0.5), particles_per_cell=1, dtype=jnp.float64
    )
    fields = {
        "return_tolerance",
        "return_faces",
        "tensile_plastic_strain",
        "partition_ambiguous_steps",
    }
    arrays = _state_arrays(state)
    for name in fields:
        arrays.pop("history_" + name)
    path = tmp_path / "old.npz"
    np.savez(path, **arrays)
    restored = _load_state(path)
    for name in restored.history._fields:
        np.testing.assert_array_equal(getattr(restored.history, name), getattr(state.history, name))
    assert restored.history.return_faces.shape == (len(state.x), 9)
    assert restored.history.partition_ambiguous_steps.dtype == jnp.int32
    arrays.pop("history_plastic_strain")
    with pytest.raises(KeyError, match="plastic_strain"):
        history_from_arrays(arrays)
    checkpoint = tmp_path / "pair.npz"
    save_checkpoint(checkpoint, (state, state))
    with np.load(checkpoint) as archive:
        old = {
            k: archive[k]
            for k in archive.files
            if not any(k.endswith("history_" + name) for name in fields)
        }
    np.savez(checkpoint, **old)
    pair = load_checkpoint(checkpoint)
    np.testing.assert_array_equal(pair[0].history.return_faces, state.history.return_faces)


def test_toe_cli_fixed_defaults_and_short_limits(capsys, tmp_path):
    from examples.mat173_toe_excavation import main

    output = tmp_path / "dry-run"
    main(["--dry-run", "--output", str(output)])
    plan = json.loads(capsys.readouterr().out)
    assert plan["grid"] == 160 and plan["dt_s"] == 3e-5
    assert plan["max_balance_steps"] == 9000 and plan["postcut_steps"] == 3000
    assert plan["guards"]["total_Hencky_postcut"] == 0.2
    assert not output.exists()
    main(["--dry-run", "--max-balance-steps", "100", "--postcut-steps", "0"])
    short = json.loads(capsys.readouterr().out)
    assert short["max_balance_steps"] == 100 and short["postcut_steps"] == 0
    with pytest.raises(SystemExit):
        main(["--max-balance-steps", "99"])
