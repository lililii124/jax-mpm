"""Independent Newtonian resin APIC/MAC step; forming fibres remain fixed.

Source-only candidate. No cure, moving-fibre feedback, surface tension, gravity,
or artificial compressibility. Actual source births/outflow belong to runner.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from .resin_apic import particle_to_faces,faces_to_particles,cell_mass,kinetic_particle,kinetic_grid,support
from .resin_mac import viscosity_force,pressure_projection


class FlowState(NamedTuple):
    x: object
    v: object
    C: object
    mass: object
    active: object
    time: object
    valid: object


def make_step(layout,fixed,prescribed,open_cells,*,density=1100.,viscosity=.05,dt=.00004,fill_threshold=.25):
    """Freeze an actual voxelized mold/fibre boundary, not final fluid geometry.

    Pressure cells have APIC cell mass > threshold*rho*h^3, a documented
    particle-density free-surface discretization. Empty cells have p=0. No mass
    or volume is added by this mask, and face masses are never floored.
    """
    h=layout.dx
    if not (0<fill_threshold<1 and dt>0 and viscosity>0):raise ValueError('Invalid fluid parameters')
    viscous_number=12*viscosity*dt/(density*h*h)
    if viscous_number>.5:raise ValueError('Conservative explicit symmetric-viscosity prefix bound exceeded')
    @jax.jit
    def step(s):
        mass,momentum,supports=particle_to_faces(s.x,s.v,s.C,s.mass,s.active,layout)
        u0=tuple(p/jnp.where(m>0,m,1.) for m,p in zip(mass,momentum))
        u=tuple(jnp.where(f,b,a) for a,b,f in zip(u0,prescribed,fixed))
        K0=kinetic_particle(s.mass,s.v,s.C,s.active,h);Kg0=kinetic_grid(mass,u0)
        fixture_initial=tuple(m*(b-a) for m,a,b in zip(mass,u0,u))
        Wbc=sum(jnp.sum(I*b) for I,b in zip(fixture_initial,prescribed))
        Lbc=.5*sum(jnp.sum(m*(b-a)**2) for m,a,b in zip(mass,u0,u))
        force,power=viscosity_force(u,mass,fixed,h,density,viscosity)
        inverse=tuple(jnp.where((m>0)&~f,1/jnp.maximum(m,1e-300),0.) for m,f in zip(mass,fixed))
        viscous=tuple(a+dt*i*F for a,i,F in zip(u,inverse,force))
        Wvisc=-dt*sum(jnp.sum(jnp.where(f,F*b,0.)) for f,F,b in zip(fixed,force,prescribed))
        Lvisc=dt*power-.5*dt*dt*sum(jnp.sum(i*F*F) for i,F in zip(inverse,force))
        cm,cell_valid=cell_mass(s.x,s.mass,s.active,layout)
        fluid=open_cells&(cm>fill_threshold*density*h**3)
        projection=pressure_projection(viscous,mass,fixed,fluid,h,dt)
        v,C=faces_to_particles(projection.velocity,supports,layout)
        v=jnp.where(s.active[:,None],v,0.);C=jnp.where(s.active[:,None,None],C,0.)
        x=jnp.where(s.active[:,None],s.x+dt*v,s.x)
        K1=kinetic_particle(s.mass,v,C,s.active,h)
        Kg1=kinetic_grid(mass,projection.velocity)
        W=Wbc+Wvisc+projection.fixture_work_J
        losses=K0-Kg0+Lbc+Lvisc+projection.projection_loss_J+Kg1-K1
        residual=W-(K1-K0)-losses
        after_support=jnp.all(jnp.stack([support(x,s.active,layout,c).valid for c in range(3)]))
        valid=s.valid&after_support&cell_valid&jnp.all(jnp.stack([a.valid for a in supports]))&projection.valid&jnp.all(jnp.isfinite(x))&jnp.all(jnp.isfinite(C))&(Lvisc>=-1e-18)
        metrics=dict(kinetic_before_J=K0,kinetic_after_J=K1,fixture_work_J=W,p2g_loss_J=K0-Kg0,boundary_loss_J=Lbc,viscous_loss_J=Lvisc,pressure_loss_J=projection.projection_loss_J,g2p_loss_J=Kg1-K1,energy_identity_J=residual,pressure_identity_J=projection.work_identity_J,maximum_divergence_per_s=projection.maximum_divergence_per_s,pressure_relative_residual=projection.relative_residual,pressure_coverage_valid=projection.coverage_valid,pressure_cells=jnp.sum(fluid),pressure_volume_m3=h**3*jnp.sum(fluid),particle_volume_m3=jnp.sum(jnp.where(s.active,s.mass/density,0.)),maximum_speed_m_per_s=jnp.max(jnp.linalg.norm(v,axis=1)),valid=valid)
        return FlowState(x,v,C,s.mass,s.active,s.time+dt,valid),metrics
    return step
