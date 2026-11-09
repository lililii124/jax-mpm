import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import Config,block,neo_hookean
from mpm_jax.apic_projection import velocity_projection,force_adjoint
from mpm_jax.core import particle_stencil
from mpm_jax.cohesive_crack import CohesiveLaw,face_velocity,scatter_face_force,elastic_energy,kinetic_energy
from mpm_jax.extrinsic_crack import initialize,transfers,step,segment_positions,surface_response
from mpm_jax.polyline_crack import link_fields

jax.config.update('jax_enable_x64',True)


def test_split_apic_trace_preserves_affine_fields_and_force_adjoint():
    config=Config(dim=2,n_grid=16,dt=1e-5,gravity=(0.,0.),boundary_cells=0)
    state=block(config,(.25,.25),(.75,.75),dtype=jnp.float64)
    support=particle_stencil(state,config);size=17**2;idx=jnp.arange(size)
    nodes=config.dx*jnp.stack((idx//17,idx%17),axis=1)
    path=jnp.array([[.1,.5],[.9,.5]])
    fields=link_fields(state.x,support,path,jnp.asarray(2),nodes[support.indices])
    support=support._replace(indices=support.indices+fields*size)
    mass=jnp.zeros(2*size).at[support.indices].add(support.weights*state.mass[:,None])
    points=jnp.array([[.35,.5],[.65,.5]])
    transfer=transfers(points,jnp.array([[0.,1.],[0.,1.]]),1.,mass,config,path,jnp.asarray(2),nodes,mass_weighted=False)
    assert bool(transfer.valid)
    project=lambda v:velocity_projection(v,support,state.mass,mass,config)
    A=jnp.array([[.3,-.4],[.2,.5]]);b=jnp.array([.8,-.6])
    coordinates=jnp.tile(nodes,(2,1));affine=coordinates@A.T+b
    np.testing.assert_allclose(face_velocity(project(affine),transfer),points@A.T+b,atol=1e-13)
    rng=np.random.default_rng(291)
    v=jnp.asarray(rng.normal(size=(2*size,2)))
    face_force=jnp.array([[3.,-2.],[-1.,4.]])
    raw=scatter_face_force(face_force,transfer,v.shape)
    force=force_adjoint(raw,support,state.mass,mass,config)
    np.testing.assert_allclose(force,jax.linear_transpose(project,v)(raw)[0],atol=1e-13)
    np.testing.assert_allclose(jnp.sum(force*v),jnp.sum(face_force*face_velocity(project(v),transfer)),atol=1e-13)
    np.testing.assert_allclose(force.sum(axis=0),face_force.sum(axis=0),atol=1e-13)
    np.testing.assert_allclose(jnp.sum(coordinates[:,0]*force[:,1]-coordinates[:,1]*force[:,0]),
                               jnp.sum(points[:,0]*face_force[:,1]-points[:,1]*face_force[:,0]),atol=1e-13)
    assert float(jnp.sum(mass[:,None]*project(v)**2))<=float(jnp.sum(mass[:,None]*v**2))


def test_compressive_recontact_with_mediated_faces_closes_energy_ledger():
    config=Config(dim=2,n_grid=16,dt=1e-5,material='neo-hookean',gravity=(0.,0.),boundary_cells=0)
    material=neo_hookean(1e6,.2);law=CohesiveLaw(1000.,1.,1e6)
    state=block(config,(.25,.25),(.75,.75),dtype=jnp.float64)
    path=initialize([[.25,.5],[.5,.5],[.75,.5]],capacity=6)
    path=path._replace(plus=path.plus.at[:3,1].add(-1e-4),minus=path.minus.at[:3,1].add(1e-4))
    def energy(s,p):
        r=surface_response(segment_positions(p.plus,p.minus),p.maximum,p.birth_traction,p.bonded,law)
        return kinetic_energy(s,config)+elastic_energy(s,material,config)+jnp.sum(p.area*(r.stored+r.dissipated))
    following,newpath,work,audit=step(state,path,material,config,law,consistent_gradients=True,
                                      particle_mediated_faces=True,audit_transfer=True)
    assert bool(following.valid & newpath.valid)
    np.testing.assert_allclose(energy(following,newpath)-energy(state,path)+work.transfer_loss+work.boundary_projection_loss,
                               work.boundary+work.gravity,atol=1e-10)
    np.testing.assert_allclose(audit.particle_to_grid_loss+audit.grid_to_particle_loss,work.transfer_loss,atol=1e-14)
    np.testing.assert_allclose(jnp.sum(following.mass[:,None]*following.v,axis=0),0.,atol=1e-12)
