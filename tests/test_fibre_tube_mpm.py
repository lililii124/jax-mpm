from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mpm_jax.fibre_mpm import (
    FibreConfig, advance, build_layout, build_tube_layout, grid_coordinates,
    grid_to_particle, make_state, particle_to_grid, reference_grips, stencil,
    step_with_diagnostics, validate_state,
)
from mpm_jax.materials.isotropic_brittle import isotropic_brittle

jax.config.update('jax_enable_x64', True)


def material():
    return isotropic_brittle(young=1000., poisson=.25, density=1000., tensile_strength=100.,
                            fracture_energy=10., characteristic_length=.01)


def two_fibres():
    t = np.linspace(.04,.24,17)
    centers = np.column_stack((t,.05+.003*np.sin(20*t),.025+.6*t))
    offsets = np.array([[0,-.002,-.002],[0,-.002,.002],[0,.002,-.002],[0,.002,.002]])
    one = (centers[:,None,:]+offsets).reshape(-1,3)
    return np.concatenate((one,one)), np.repeat(np.arange(2,dtype=np.int32),len(one))


def layouts(x, ids, **kwargs):
    return tuple(builder(x,ids,dx=.01,**kwargs) for builder in (build_layout,build_tube_layout))


def compare_trees(left,right,atol=2e-13):
    for a,b in zip(jax.tree.leaves(left),jax.tree.leaves(right)):
        np.testing.assert_allclose(a,b,rtol=2e-12,atol=atol)


def test_sparse_scatter_gather_and_coordinates_equal_dense_affine_field():
    x,ids=two_fibres()
    dense,tube=layouts(x,ids)
    matrix=np.array([[.03,.02,0],[0,-.01,.01],[.005,0,.02]])
    bias=np.array([[.2,.03,-.1],[-.3,.02,.1]])
    state=make_state(x,np.full(len(x),1e-6),ids,material(),dense,v=x@matrix.T+bias[ids],C=np.broadcast_to(matrix,(len(x),3,3)))
    dense_mass,dense_momentum=particle_to_grid(state,dense,FibreConfig(1e-4))
    tube_mass,tube_momentum=particle_to_grid(state,tube,FibreConfig(1e-4))
    np.testing.assert_allclose(tube_mass,dense_mass[tube.dense_indices],rtol=2e-14,atol=1e-18)
    np.testing.assert_allclose(tube_momentum,dense_momentum[tube.dense_indices],rtol=2e-14,atol=1e-18)
    np.testing.assert_array_equal(dense_mass[np.asarray(tube.node_map)<0],0.)
    coordinates,fibres=grid_coordinates(tube,jnp.arange(tube.node_count))
    dense_coordinates,dense_fibres=grid_coordinates(dense,tube.dense_indices)
    np.testing.assert_array_equal(coordinates,dense_coordinates)
    np.testing.assert_array_equal(fibres,dense_fibres)
    velocity=coordinates@matrix.T+bias[np.asarray(fibres)]
    v,C=grid_to_particle(velocity,stencil(state.x,state.fibre_id,tube),tube)
    np.testing.assert_allclose(v,state.v,atol=2e-15)
    np.testing.assert_allclose(C,state.C,atol=2e-13)


@pytest.mark.parametrize('motion',['translation','rotation','affine'])
def test_sparse_and_dense_steps_preserve_same_state_mass_and_external_momentum(motion):
    x,ids=two_fibres()
    dense,tube=layouts(x,ids)
    matrix=np.zeros((3,3))
    if motion=='rotation':
        matrix=np.array([[0,-.2,.1],[.2,0,-.1],[-.1,.1,0]])
    elif motion=='affine':
        matrix=np.array([[.03,.02,0],[0,-.01,.01],[.005,0,.02]])
    rotation=np.array([[np.cos(.3),-np.sin(.3),0],[np.sin(.3),np.cos(.3),0],[0,0,1.]])
    initial=make_state(x,np.full(len(x),1e-6),ids,material(),dense,
        v=x@matrix.T+np.where(ids[:,None]==0,np.array([.1,.05,0]),np.array([-.1,-.02,.01])),
        C=np.broadcast_to(matrix,(len(x),3,3)),F=np.broadcast_to(rotation,(len(x),3,3)))
    config=FibreConfig(1e-4,gravity=(.1,-.2,.05))
    force=initial.mass[:,None]*jnp.asarray(np.column_stack((np.sin(x[:,0]),np.cos(x[:,2]),x[:,1])))
    a,b=initial,initial
    for _ in range(3):
        a,da=step_with_diagnostics(a,material(),dense,config,force)
        b,db=step_with_diagnostics(b,material(),tube,config,force)
        compare_trees(a,b)
        compare_trees(da,db)
        assert validate_state(a,dense) and validate_state(b,tube)
    np.testing.assert_allclose(np.sum(b.mass[:,None]*(b.v-initial.v),axis=0),b.external_impulse,atol=2e-15)
    np.testing.assert_array_equal(b.mass,initial.mass)
    np.testing.assert_allclose(db.grid_mass,initial.mass.sum(),atol=2e-15)


def test_reference_grip_impulses_and_work_do_not_depend_on_empty_grid_storage():
    x,ids=two_fibres()
    dense,tube=layouts(x,ids)
    state=make_state(x,np.full(len(x),1e-6),ids,material(),dense)
    boundary=reference_grips((x[:,0]<.07,x[:,0]>.21),
        (lambda t:jnp.array([-.05,0.,0.]),lambda t:jnp.array([.05,0.,0.])))
    a,da=step_with_diagnostics(state,material(),dense,FibreConfig(1e-4),boundary=boundary)
    b,db=step_with_diagnostics(state,material(),tube,FibreConfig(1e-4),boundary=boundary)
    compare_trees(a,b)
    compare_trees(da,db)
    assert bool(b.valid)
    np.testing.assert_allclose(np.sum(state.mass[:,None]*(b.v-state.v),axis=0),db.group_impulses.sum(0),atol=1e-16)


def test_future_stencil_outside_tube_is_rejected_even_inside_dense_aabb():
    x,ids=np.array([[.105,.105,.105]]),np.array([0],dtype=np.int32)
    dense,tube=layouts(x,ids,reference_bounds=[[0.,.3,0.,.3,0.,.3]])
    state=make_state(x,np.array([1e-6]),ids,material(),tube,v=[[100.,0.,0.]])
    dense_result=advance(state,material(),dense,FibreConfig(.001),1)
    sparse_result=advance(state,material(),tube,FibreConfig(.001),3)
    assert bool(dense_result.valid) and not bool(sparse_result.valid)
    assert float(sparse_result.time)==.001
    np.testing.assert_allclose(sparse_result.x,dense_result.x,atol=1e-15)
    assert bool(stencil(sparse_result.x,sparse_result.fibre_id,dense).valid)
    support=stencil(sparse_result.x,sparse_result.fibre_id,tube)
    assert not bool(support.valid) and bool(jnp.any(support.indices==tube.node_count))
    with pytest.raises(FloatingPointError,match='outside its active tube'):
        validate_state(sparse_result,tube)


def test_missing_active_map_entry_never_wraps_to_another_fibre():
    x,ids=two_fibres()
    _,tube=layouts(x,ids)
    state=make_state(x,np.full(len(x),1e-6),ids,material(),tube)
    support=stencil(state.x,state.fibre_id,tube)
    active=int(support.indices[0,np.argmax(np.asarray(support.weights[0]))])
    dense=int(tube.dense_indices[active])
    broken=replace(tube,node_map=tube.node_map.at[dense].set(-1))
    updated=stencil(state.x,state.fibre_id,broken)
    assert not bool(updated.valid)
    assert bool(jnp.any(updated.indices[0]==tube.node_count))
    assert bool(jnp.all(updated.indices[ids==1]<tube.node_count))
    with pytest.raises(FloatingPointError,match='outside its active tube'):
        validate_state(state,broken)
