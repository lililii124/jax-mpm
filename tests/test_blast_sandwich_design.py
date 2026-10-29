from pathlib import Path
import importlib.util
import json

import numpy as np
import pytest

from examples.blast_sandwich_design import Settings, describe, geometry


def test_layer_quadrature_mass_and_three_pressure_resultants():
    settings = Settings()
    geo = geometry(settings)
    volume = geo['volume']
    np.testing.assert_allclose(volume.sum(), .3*.3*.024, rtol=2e-15)
    np.testing.assert_allclose(volume[geo['skin']].sum(), .3*.3*.004, rtol=2e-15)
    density = np.where(geo['skin'], 2700., 2700*.18)
    np.testing.assert_allclose(volume@density, 1.8468, rtol=1e-12)
    np.testing.assert_allclose(geo['footprints']@geo['face_area'], [.09]*3, rtol=1e-12)
    assert not np.any(geo['design'] & geo['skin'])
    assert describe(settings)['skin_cells'] == 2


def test_thin_layer_plan_uses_rectangular_counts_and_fixed_metal_cfl():
    plan = describe(Settings(xy_step=.005, controls=(24,24,3)))
    assert plan['particles'] == 691200 and plan['nodes'] == 107163
    assert plan['design_variables'] == 1728
    assert plan['dt_s'] < 3e-8 and plan['steps'] > 60000
    finer = describe(Settings(xy_step=.005, z_step=.0005))
    assert finer['skin_cells'] == 4 and finer['particles'] == 2*plan['particles']
    assert finer['dt_s'] < plan['dt_s']


def micro_model():
    pytest.importorskip('jax')
    from examples.blast_sandwich_design import build_model
    return build_model(Settings(length=.012,width=.012,skin=.002,core=.004,edge=.002,
        xy_step=.004,z_step=.001,controls=(2,2,2),filter_radius=.002,
        pulse_duration=5e-8,horizon=1e-7,displacement_limit=1e-10,block_steps=2))


def transient_model():
    pytest.importorskip('jax')
    from examples.blast_sandwich_design import build_model
    return build_model(Settings(length=.012,width=.012,skin=.002,core=.004,edge=.002,
        xy_step=.004,z_step=.001,controls=(2,2,2),filter_radius=.002,
        pulse_duration=2e-6,horizon=8e-6,displacement_limit=1e-6,block_steps=16))


def test_anisotropic_partition_heterogeneous_inertia_and_property_mapping():
    pytest.importorskip('jax')
    import jax.numpy as jnp
    from mpm_jax.compact_tl import reference_grid
    model = micro_model()
    support = model['support']
    affine = np.einsum('pki,pkj->pij', support.nodes[support.indices],support.gradients)
    np.testing.assert_allclose(affine,np.broadcast_to(np.eye(3),affine.shape),atol=2e-14)
    q=np.linspace(-.8,.8,len(model['q0']))
    phi,density,young,material=model['properties'](jnp.asarray(q))
    baseline=model['properties'](jnp.asarray(model['q0']))
    ref=reference_grid(support,density)
    np.testing.assert_array_equal(ref.volume,support.volume)
    np.testing.assert_allclose(ref.particle_mass.sum(),baseline[1]@ref.volume,rtol=2e-15)
    np.testing.assert_allclose(ref.nodal_mass.sum(),ref.particle_mass.sum(),rtol=2e-15)
    assert material.mu.shape==(len(phi),1,1)
    np.testing.assert_array_equal(np.asarray(young)[model['geometry']['skin']],70e9)
    np.testing.assert_allclose(np.asarray(young)[~model['geometry']['skin']],
        .3*70e9*np.asarray(phi)[~model['geometry']['skin']]**2,rtol=2e-15)
    np.testing.assert_allclose(np.asarray(model['load_shape']).sum(axis=1)[:,2],[-.012**2]*3,rtol=2e-15)


def test_heterogeneous_internal_force_is_the_reference_energy_gradient():
    jax=pytest.importorskip('jax')
    import jax.numpy as jnp
    from types import SimpleNamespace
    from mpm_jax.compact_tl import reference_grid,elastic_energy
    from mpm_jax.total_lagrangian import deformation,forces
    model=micro_model()
    _,density,_,material=model['properties'](jnp.asarray(np.linspace(-1,1,len(model['q0']))))
    ref=reference_grid(model['support'],density)
    matrix=jnp.asarray([[.002,.0003,0.],[.0001,-.001,0.],[0.,.0002,.001]])
    u=ref.nodes@matrix.T
    gradient=jax.grad(lambda u:elastic_energy(deformation(u,ref),ref,material))(u)
    force=forces(deformation(u,ref),None,ref,material,SimpleNamespace(material='elastic'),jnp.zeros_like(ref.positions))
    np.testing.assert_allclose(gradient,-force,rtol=2e-11,atol=2e-10)
    np.testing.assert_allclose(force.sum(axis=0),0,atol=2e-10)


@pytest.mark.skipif(importlib.util.find_spec('jax') is None,reason='JAX runtime required')
def test_five_step_transient_directional_derivative_and_complete_resume(tmp_path):
    jax=pytest.importorskip('jax')
    from examples.blast_sandwich_design import directional_check,forward_case
    model=micro_model()
    assert model['plan']['steps']==5
    q=model['q0']
    check=directional_check(transient_model(),q,step=1e-3)
    (tmp_path/'directional-check.json').write_text(json.dumps(check,indent=2))
    assert check['passed'],check
    full=forward_case(model,q,1,tmp_path/'full',save_every=5)
    partial=forward_case(model,q,1,tmp_path/'split',stop_step=2,save_every=2)
    assert not partial['completed']
    final=forward_case(model,q,1,tmp_path/'split',resume=True,save_every=5)
    assert full['completed'] and final['completed']
    with np.load(tmp_path/'full/checkpoint.npz') as a,np.load(tmp_path/'split/checkpoint.npz') as b:
        for key in a.files:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
        rows=a['rows']
        assert rows.shape==(6,19) and rows[-1,7]>0
        assert np.max(rows[:,15])<1e-12
    assert (tmp_path/'full/history.csv').read_bytes()==(tmp_path/'split/history.csv').read_bytes()
    jax.clear_caches()


@pytest.mark.skipif(importlib.util.find_spec('jax') is None,reason='JAX runtime required')
def test_micro_optimizer_checkpoint_and_saved_particle_export(tmp_path):
    jax=pytest.importorskip('jax')
    from examples.blast_sandwich_design import optimize,forward_case,export_vtk
    model=transient_model()
    first=optimize(model,tmp_path/'optimizer',iterations=1,gradient_mode='reverse')
    resumed=optimize(model,tmp_path/'optimizer',iterations=0,gradient_mode='reverse',resume=True)
    np.testing.assert_array_equal(first['controls'],resumed['controls'])
    assert resumed['iterations_completed']==1 and not resumed['convergence_claim']
    forward_case(model,np.asarray(resumed['controls']),0,tmp_path/'forward',stop_step=1,save_every=1)
    exported=export_vtk(tmp_path/'forward')
    assert exported['frames']==2 and Path(exported['pvd']).is_file()
    jax.clear_caches()
