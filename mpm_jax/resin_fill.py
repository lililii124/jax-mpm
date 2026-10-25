"""Independent Newtonian resin APIC/MAC step; forming fibres remain fixed.

Source-only candidate. No cure, moving-fibre feedback, surface tension, gravity,
or artificial compressibility. Actual source births/outflow belong to runner.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from .resin_apic import particle_to_faces,faces_to_particles,cell_mass,kinetic_particle,kinetic_grid,support
from .resin_mac import viscosity_force,pressure_projection
from .resin_particle_boundary import advect
from .resin_transport_rt0 import advect as advect_rt0
from .resin_transport_extension import Extension,support_closed_domain


class FlowState(NamedTuple):
    x: object
    v: object
    C: object
    mass: object
    active: object
    time: object
    valid: object


def make_step(layout,fixed,prescribed,open_cells,*,density=1100.,viscosity=.05,dt=.00004,fill_threshold=.25,solid=None,transport='point-toi',capture_pressure=False):
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
        domain_added=jnp.asarray(0);domain_valid=jnp.asarray(True)
        if transport=='rt0':fluid,domain_added,domain_valid=support_closed_domain(s.x,s.active,mass,fixed,fluid,open_cells,layout)
        projection=pressure_projection(viscous,mass,fixed,fluid,h,dt,preconditioner='mixed-lsqr' if transport=='rt0' else None,max_iterations=1600 if transport=='rt0' else 400)
        v,C=faces_to_particles(projection.velocity,supports,layout)
        v=jnp.where(s.active[:,None],v,0.);C=jnp.where(s.active[:,None,None],C,0.)
        Ktransfer=kinetic_particle(s.mass,v,C,s.active,h)
        wall_loss=jnp.asarray(0.);wall_impulse=jnp.zeros(3);wall_events=jnp.asarray(0);wall_valid=jnp.asarray(True)
        transport_data=None
        if transport=='rt0':
            if solid is None:raise ValueError('RT0 requires the actual fixed voxel boundary')
            extension=Extension(projection.velocity,fluid,jnp.asarray(0.),projection.maximum_divergence_per_s,jnp.asarray(0.),jnp.asarray(0.),domain_valid)
            transport_data=advect_rt0(s.x,extension.velocity,s.active,solid,extension.cells,layout,dt)
            x=transport_data.x;wall_valid=transport_data.valid&extension.valid&jnp.all(~s.active|(transport_data.projected_time>=dt*(1-1e-12)))
        elif solid is None:
            x=jnp.where(s.active[:,None],s.x+dt*v,s.x)
        else:
            collision=advect(s.x,v,s.mass,s.active,solid,layout,dt)
            x,v=collision.x,collision.v;wall_loss=collision.kinetic_loss;wall_impulse=collision.impulse;wall_events=collision.events;wall_valid=collision.valid
        K1=kinetic_particle(s.mass,v,C,s.active,h)
        Kg1=kinetic_grid(mass,projection.velocity)
        W=Wbc+Wvisc+projection.fixture_work_J
        losses=K0-Kg0+Lbc+Lvisc+projection.projection_loss_J+Kg1-Ktransfer+wall_loss
        residual=W-(K1-K0)-losses
        after_support=jnp.all(jnp.stack([support(x,s.active,layout,c).valid for c in range(3)]))
        valid=s.valid&domain_valid&wall_valid&after_support&cell_valid&jnp.all(jnp.stack([a.valid for a in supports]))&projection.valid&jnp.all(jnp.isfinite(x))&jnp.all(jnp.isfinite(C))&(Lvisc>=-1e-18)
        fixture_vector=wall_impulse+jnp.stack([jnp.sum(I)-dt*jnp.sum(jnp.where(f,F,0.))+jnp.sum(P) for I,f,F,P in zip(fixture_initial,fixed,force,projection.fixture_impulse)])
        particle_momentum0=jnp.sum(jnp.where(s.active,s.mass,0.)[:,None]*s.v,axis=0)
        particle_momentum1=jnp.sum(jnp.where(s.active,s.mass,0.)[:,None]*v,axis=0)
        momentum_error=jnp.linalg.norm(particle_momentum1-particle_momentum0-fixture_vector)
        metrics=dict(particle_wall_loss_J=wall_loss,particle_wall_events=wall_events,particle_wall_valid=wall_valid,momentum_step_residual_Ns=momentum_error,fixture_impulse_x_Ns=fixture_vector[0],fixture_impulse_y_Ns=fixture_vector[1],fixture_impulse_z_Ns=fixture_vector[2],kinetic_before_J=K0,kinetic_after_J=K1,fixture_work_J=W,p2g_loss_J=K0-Kg0,boundary_loss_J=Lbc,viscous_loss_J=Lvisc,pressure_loss_J=projection.projection_loss_J,g2p_loss_J=Kg1-Ktransfer,energy_identity_J=residual,pressure_identity_J=projection.work_identity_J,maximum_divergence_per_s=projection.maximum_divergence_per_s,pressure_relative_residual=projection.relative_residual,pressure_iterations=projection.iterations,pressure_stationarity_relative=projection.stationarity_relative,pressure_stationarity_absolute=projection.stationarity_absolute,pressure_reconstruction_relative=projection.reconstruction_relative,pressure_iteration_cap=1600 if transport=='rt0' else 400,pressure_coverage_valid=projection.coverage_valid,pressure_cells=jnp.sum(fluid),pressure_volume_m3=h**3*jnp.sum(fluid),particle_volume_m3=jnp.sum(jnp.where(s.active,s.mass/density,0.)),maximum_speed_m_per_s=jnp.max(jnp.linalg.norm(v,axis=1)),valid=valid)
        new=FlowState(x,v,C,s.mass,s.active,s.time+dt,valid)
        if transport=='rt0':
            effective=jnp.where(s.active,s.mass,0.)
            def angular(position,velocity,affine):
                spin=jnp.stack([affine[:,2,1]-affine[:,1,2],affine[:,0,2]-affine[:,2,0],affine[:,1,0]-affine[:,0,1]],axis=1)*h*h/4
                return jnp.sum(effective[:,None]*(jnp.cross(position,velocity)+spin),axis=0)
            torque=jnp.zeros(3)
            for c,(I,f,F,P) in enumerate(zip(fixture_initial,fixed,force,projection.fixture_impulse)):
                impulse=I-dt*jnp.where(f,F,0.)+P
                for r in range(3):
                    if r==c:continue
                    axis=(jnp.arange(impulse.shape[r])+.5)*h+layout.origin[r]
                    axes=tuple(k for k in range(3) if k!=r);moment=jnp.sum(jnp.sum(impulse,axis=axes)*axis)
                    cross=jnp.cross(jax.nn.one_hot(r,3),jax.nn.one_hot(c,3));torque=torque+moment*cross
            transport_torque=jnp.sum(effective[:,None]*jnp.cross(x-s.x,v),axis=0)
            actual=angular(x,v,C)-angular(s.x,s.v,s.C)-torque
            metrics.update(physical_pressure_domain_added_cells=domain_added,physical_pressure_domain_valid=domain_valid,transport_extension_relative_residual=extension.relative_residual,transport_extension_max_divergence_per_s=extension.maximum_divergence,transport_extension_protected_face_change=extension.protected_change,transport_extension_compatibility_residual=extension.compatibility_residual,transport_extension_valid=extension.valid,transport_angular_remainder_norm=jnp.linalg.norm(transport_torque),transport_angular_identity_norm=jnp.linalg.norm(actual-transport_torque),fixture_torque_norm=jnp.linalg.norm(torque),transport_torque_x=transport_torque[0],transport_torque_y=transport_torque[1],transport_torque_z=transport_torque[2],fixture_torque_x=torque[0],fixture_torque_y=torque[1],fixture_torque_z=torque[2],transport_logJ_mean=jnp.sum(effective*transport_data.logJ)/jnp.maximum(jnp.sum(effective),1e-300),transport_logJ_max=jnp.where(jnp.any(s.active),jnp.max(jnp.where(s.active,transport_data.logJ,-jnp.inf)),0.),transport_logJ_min=jnp.where(jnp.any(s.active),jnp.min(jnp.where(s.active,transport_data.logJ,jnp.inf)),0.),transport_maximum_roundoff_distance_m=jnp.max(transport_data.roundoff_distance),transport_maximum_path_over_h=jnp.max(transport_data.path_bound)/h,transport_maximum_crossings=jnp.max(transport_data.crossings),transport_unprojected_particle_fraction=jnp.sum(s.active&(transport_data.projected_time<dt*(1-1e-12)))/jnp.maximum(jnp.sum(s.active),1))
            return (new,metrics,transport_data.logJ,(viscous,mass,fixed,fluid,projection.impulse_potential_base,projection.impulse_potential_correction)) if capture_pressure else (new,metrics,transport_data.logJ)
        return new,metrics
    return step
