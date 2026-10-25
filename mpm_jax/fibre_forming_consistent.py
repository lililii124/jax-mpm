"""Isolated boundary-aware force/kinematic work pairing candidate.

Initial Dirichlet projection precedes the force kick. Configuration advection
uses the average of this admissible initial grid velocity and final velocity.
The signed ledger separates finite material work error from grid work identities.
This is an explicit candidate, not a claim of unconditional energy stability.
"""
import jax
import jax.numpy as jnp
from .fibre_mpm import FibreState,stencil,particle_to_grid,grid_to_particle,grid_coordinates
from .materials.isotropic_brittle import update_brittle,brittle_energy
from .fibre_forming_boundary import prescribe
from .fibre_forming_contact import project,mass_gradient,PAIRS


def make_step(material,layout,config,masks,duration):
    nodes=layout.node_count//3
    def particle_energy(s,v,C):
        return .5*jnp.sum(s.mass*jnp.sum(v*v,axis=1))+layout.dx**2/8*jnp.sum(s.mass*jnp.sum(C*C,axis=(1,2)))
    def grid_energy(m,v):return .5*jnp.sum(m*jnp.sum(v*v,axis=1))
    @jax.jit
    def step(s):
        support=stencil(s.x,s.fibre_id,layout)
        mass,p0=particle_to_grid(s._replace(tau=jnp.zeros_like(s.tau)),layout,config,support)
        _,pforce=particle_to_grid(s,layout,config,support)
        divide=jnp.where(mass>0,mass,1)[:,None]
        pre=p0/divide
        begin,clamped,valid0=prescribe(pre,mass,s,support,layout,masks,config.dt,duration)
        target=begin
        force_impulse=pforce-p0
        force=begin+force_impulse/divide
        admissible,clamped1,valid1=prescribe(force,mass,s,support,layout,masks,config.dt,duration)
        gradient=mass_gradient(s,support,layout).reshape(3,nodes,3)
        centres=jnp.stack([jnp.sum(jnp.where((s.fibre_id==i)[:,None],s.mass[:,None]*s.x,0),axis=0)/jnp.sum(jnp.where(s.fibre_id==i,s.mass,0)) for i in range(3)])
        contact=project(mass.reshape(3,nodes),admissible.reshape(3,nodes,3),gradient,centres,clamped.reshape(3,nodes))
        post=contact.velocity.reshape(-1,3)
        midpoint=(begin+post)/2
        initial_fixture=mass[:,None]*(begin-pre)
        force_fixture=mass[:,None]*(admissible-force)
        contact_fixture=contact.fixture_impulse.reshape(-1,3)
        fixture=initial_fixture+force_fixture+contact_fixture
        work=jnp.sum(fixture*target)
        lossbc=.5*jnp.sum(mass*jnp.sum((begin-pre)**2,axis=1))
        contact_field=jnp.zeros((3,nodes,3),dtype=s.x.dtype)
        for k,(a,b) in enumerate(PAIRS):contact_field=contact_field.at[a].add(-contact.pair_impulse[k]).at[b].add(contact.pair_impulse[k])
        contactloss=-jnp.sum(contact_field.reshape(-1,3)*midpoint)
        v,C=grid_to_particle(post,support,layout)
        advect,Cadv=grid_to_particle(midpoint,support,layout)
        F=(jnp.eye(3)+config.dt*Cadv)@s.F;x=s.x+config.dt*advect
        history,tau=update_brittle(s.history,F,material)
        valid=s.valid&support.valid&stencil(x,s.fibre_id,layout).valid&valid0&valid1&contact.valid&jnp.all(history.valid)&jnp.all(history.damage==0)&jnp.all(jnp.isfinite(F))&jnp.all(jnp.isfinite(x))&jnp.all(jnp.linalg.det(F)>0)
        impulse=jnp.sum(fixture,axis=0)
        new=FibreState(x,v,F,C,s.mass,s.volume,s.fibre_id,history,tau,s.time+config.dt,valid,s.boundary_impulse+impulse,s.external_impulse,s.boundary_work+work)
        K0=particle_energy(s,s.v,s.C);K1=particle_energy(s,v,C)
        U0=jnp.sum(s.volume*brittle_energy(s.F,s.history,material));U1=jnp.sum(s.volume*brittle_energy(F,history,material))
        S=config.dt*jnp.sum(s.volume*jnp.sum(s.tau*Cadv,axis=(1,2)))
        forcework=jnp.sum(force_impulse*midpoint)
        p2gloss=K0-grid_energy(mass,pre);g2ploss=grid_energy(mass,post)-K1
        remainder=work-lossbc-contactloss-(K1-K0)-(U1-U0)
        grididentity=grid_energy(mass,post)-grid_energy(mass,pre)-work+lossbc+contactloss+S
        coordinates,_=grid_coordinates(layout,jnp.arange(layout.node_count))
        fixture_torque=jnp.sum(jnp.cross(coordinates,fixture),axis=0)
        def angular(position,velocity,affine):
            spin=jnp.stack((affine[:,2,1]-affine[:,1,2],affine[:,0,2]-affine[:,2,0],affine[:,1,0]-affine[:,0,1]),axis=1)
            return jnp.sum(s.mass[:,None]*(jnp.cross(position,velocity)+layout.dx**2/4*spin),axis=0)
        Lold=angular(s.x,s.v,s.C);Lnew=angular(x,v,C)
        angular_error=jnp.linalg.norm(Lnew-Lold-fixture_torque)
        transport=jnp.linalg.norm(jnp.sum(s.mass[:,None]*jnp.cross(x-s.x,v),axis=0))
        momentum_error=jnp.linalg.norm(jnp.sum(s.mass[:,None]*(v-s.v),axis=0)-impulse)
        metrics=jnp.array([work,lossbc,contactloss,K1-K0,U1-U0,S,p2gloss,g2ploss,remainder,grididentity,forcework+S,(U1-U0)-S,contact.closing_residual,contact.sweeps,jnp.min(jnp.linalg.det(F)),momentum_error,angular_error,transport,jnp.linalg.norm(fixture_torque),jnp.linalg.norm(Lnew)])
        return new,metrics
    return step
