import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mpm_jax import Config, block, neo_hookean
from mpm_jax.cohesive_crack import CohesiveLaw, elastic_energy, kinetic_energy
from mpm_jax.extrinsic_crack import (discrete_surface_gradient, initialize, insert_tip,
                                    segment_positions, step, surface_gradient, surface_response, transfers)
from mpm_jax.cohesive_crack import face_velocity, scatter_face_force

jax.config.update('jax_enable_x64', True)


def positions(opening):
    return jnp.asarray([[[0.,opening/2],[.001,opening/2],[0.,-opening/2],[.001,-opening/2]]])


def test_inserted_surface_retains_traction_and_spends_prescribed_energy():
    law = CohesiveLaw(120e3,2.,1e11)
    birth, bonded = jnp.asarray([[0.,law.strength]]),jnp.asarray([True])
    maximum = jnp.full((1,2),law.strength/law.stiffness)
    q = positions(0.)
    response = surface_response(q,maximum,birth,bonded,law)
    gradient = surface_gradient(q,maximum,birth,bonded,law)
    np.testing.assert_allclose(response.stored,0.,atol=1e-14)
    np.testing.assert_allclose(response.dissipated,0.,atol=1e-14)
    np.testing.assert_allclose(gradient[0,:2].sum(axis=0),[0.,law.strength],rtol=1e-13)
    work = 0.
    for opening in np.linspace(0.,2*law.energy/law.strength,65)[1:]:
        following = positions(opening)
        force,after,energy,error = discrete_surface_gradient(q,following,maximum,birth,bonded,law)
        work += float(jnp.sum(force*(following-q)))
        np.testing.assert_allclose(error,0.,atol=1e-13)
        q,maximum = following,after.maximum
    np.testing.assert_allclose(work,law.energy,rtol=1e-12)
    np.testing.assert_allclose(after.dissipated,law.energy,rtol=1e-12)
    np.testing.assert_allclose(after.stored,0.,atol=1e-13)


def test_rotating_surface_is_objective_and_discrete_work_has_no_resultant_or_couple():
    law = CohesiveLaw(120e3,2.,1e11)
    birth,bonded = jnp.asarray([[0.,120e3]]),jnp.asarray([True])
    q = positions(8e-6)
    old = surface_response(q,jnp.zeros((1,2)),birth,bonded,law)
    theta = .6
    R = jnp.asarray([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    rotated = q @ R.T+jnp.asarray([.1,-.2])
    after = surface_response(rotated,old.maximum,birth,bonded,law)
    np.testing.assert_allclose(after.stored,old.stored,rtol=1e-10)
    np.testing.assert_allclose(surface_gradient(rotated,old.maximum,birth,bonded,law),
                               surface_gradient(q,old.maximum,birth,bonded,law) @ R.T,rtol=1e-9,atol=1e-6)
    rigid_gradient,_,_,rigid_error = discrete_surface_gradient(q,rotated,old.maximum,birth,bonded,law)
    np.testing.assert_allclose(rigid_error,0.,atol=1e-8)
    np.testing.assert_allclose(rigid_gradient.sum(axis=1),0.,atol=1e-7)
    assert float(jnp.abs(rigid_gradient).max()) < 2*float(jnp.abs(surface_gradient(q,old.maximum,birth,bonded,law)).max())
    new = positions(15e-6) @ R.T+jnp.asarray([.1,-.2])
    gradient,response,energy,error = discrete_surface_gradient(q,new,old.maximum,birth,bonded,law)
    np.testing.assert_allclose(error,0.,atol=1e-10)
    np.testing.assert_allclose(gradient.sum(axis=1),0.,atol=1e-7)
    mid = .5*(new+q)
    torque = jnp.sum(mid[:,:,0]*gradient[:,:,1]-mid[:,:,1]*gradient[:,:,0])
    np.testing.assert_allclose(torque,0.,atol=1e-8)


@pytest.mark.parametrize('consistent',[False,True])
def test_coupled_midpoint_step_closes_bulk_surface_and_transfer_ledger(consistent):
    config = Config(dim=2,n_grid=16,dt=1e-5,gravity=(0.,0.),boundary_cells=0)
    material = neo_hookean(1e4,.2)
    state = block(config,(.25,.375),(.75,.625),dtype=jnp.float64)
    state = state._replace(F=state.F.at[:,1,1].set(1.01))
    path = initialize([[.25,.5],[.5,.5],[.75,.5]],capacity=6)
    law = CohesiveLaw(100.,1.,1e4)
    path = path._replace(birth_traction=path.birth_traction.at[:2,1].set(100.),
                        maximum=path.maximum.at[:2].set(.01),bonded=path.bonded.at[:2].set(True))
    old = surface_response(segment_positions(path.plus,path.minus),path.maximum,path.birth_traction,path.bonded,law)
    energy0 = kinetic_energy(state,config)+elastic_energy(state,material,config)+jnp.sum(path.area*(old.stored+old.dissipated))
    following,newpath,work = step(state,path,material,config,law,consistent_gradients=consistent)
    assert bool(following.valid & newpath.valid)
    after = surface_response(segment_positions(newpath.plus,newpath.minus),newpath.maximum,newpath.birth_traction,newpath.bonded,law)
    energy1 = kinetic_energy(following,config)+elastic_energy(following,material,config)+jnp.sum(newpath.area*(after.stored+after.dissipated))
    np.testing.assert_allclose(energy1-energy0+work.transfer_loss+work.boundary_projection_loss,work.boundary+work.gravity,atol=1e-10)
    np.testing.assert_allclose(jnp.sum(following.mass[:,None]*following.v,axis=0),0.,atol=1e-10)


def test_stress_selected_insertion_has_no_instantaneous_energy_jump():
    config = Config(dim=2,n_grid=32,dt=1e-5,gravity=(0.,0.),boundary_cells=0)
    material = neo_hookean(1e6,.2)
    state = block(config,(.3,.3),(.7,.7),dtype=jnp.float64)
    normal = jnp.asarray([-.5,np.sqrt(.75)])
    state = state._replace(F=state.F+.02*jnp.outer(normal,normal))
    law = CohesiveLaw(1000.,1.,1e8)
    path = initialize([[.2,.5],[.5,.5]],capacity=8)
    @jax.jit
    def insert(state,path):
        following,report = insert_tip(state,path,material,config,law,.01,.1,.1)
        following = jax.lax.cond(path.valid,lambda _:following,lambda _:path,None)
        return following,report
    following,report = insert(state,path)
    assert bool(following.valid & report.inserted) and int(following.count)==3
    np.testing.assert_allclose(report.direction,[np.sqrt(.75),.5],atol=1e-13)
    np.testing.assert_allclose(report.energy_jump,0.,atol=1e-12)
    assert bool(following.bonded[1]) and float(jnp.linalg.norm(following.birth_traction[1]))>law.strength


@pytest.mark.parametrize('mass_weighted',[False,True])
def test_face_transfer_reproduces_affine_velocity_with_missing_nodes(mass_weighted):
    config = Config(dim=2,n_grid=16,dt=1e-5,gravity=(0.,0.),boundary_cells=0)
    size = 17**2
    index = jnp.arange(size)
    nodes = config.dx*jnp.stack((index//17,index%17),axis=-1)
    mass = jnp.tile(jnp.where(nodes[:,0]>=.5,1.,0.),2)
    points = jnp.asarray([[.49,.49],[.51,.51]])
    normals = jnp.asarray([[0.,1.],[0.,1.]])
    far = jnp.asarray([[.1,.1],[.2,.1]])
    transfer = transfers(points,normals,1.,mass,config,far,jnp.asarray(2),nodes,mass_weighted=mass_weighted)
    A = jnp.asarray([[.3,-.4],[.2,.5]])
    b = jnp.asarray([.8,-.6])
    velocity = jnp.tile(nodes@A.T+b,(2,1))
    assert bool(transfer.valid)
    np.testing.assert_allclose(face_velocity(velocity,transfer),points@A.T+b,atol=1e-13)
    force = jnp.asarray([[3.,-2.],[-1.,4.]])
    nodal = scatter_face_force(force,transfer,velocity.shape)
    np.testing.assert_allclose(nodal.sum(axis=0),force.sum(axis=0),atol=1e-13)
    repeated = jnp.tile(nodes,(2,1))
    np.testing.assert_allclose(jnp.sum(repeated[:,0]*nodal[:,1]-repeated[:,1]*nodal[:,0]),
                               jnp.sum(points[:,0]*force[:,1]-points[:,1]*force[:,0]),atol=1e-13)


def test_grid_aligned_cohesive_patch_balances_uniform_bulk_stress():
    from tools.audit_extrinsic_insertion import audit
    result=audit(48,.5,0.,consistent=True)
    assert result['intact_max_force_N_per_m']<1e-10
    assert result['relative_residual']<1e-10
