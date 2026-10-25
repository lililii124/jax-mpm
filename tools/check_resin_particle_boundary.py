"""Directed face/edge/corner checks of actual voxel-wall flight events."""
from pathlib import Path
import json
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from mpm_jax.resin_apic import MACLayout
from mpm_jax.resin_particle_boundary import advect


def main():
    layout=MACLayout((6,6,6),(0.,0.,0.),1.);dt=.14
    index=np.indices(layout.shape);solid=jnp.asarray(np.any(index>=3,axis=0))
    x=jnp.asarray([[2.9,.5,.5],[2.9,2.9,.5],[2.9,2.9,2.9],[1.,1.,1.]])
    v=jnp.asarray([[1.,.2,-.1],[1.,1.,.2],[1.,1.,1.],[.2,.3,.4]])
    mass=jnp.full(4,2.);active=jnp.ones(4,dtype=bool)
    solve=jax.jit(lambda x,v,wall:advect(x,v,mass,active,wall,layout,dt));result=solve(x,v,solid)
    expected_v=jnp.asarray([[0.,.2,-.1],[0.,0.,.2],[0.,0.,0.],[.2,.3,.4]])
    expected_x=jnp.asarray([[3.,.528,.486],[3.,3.,.528],[3.,3.,3.],[1.028,1.042,1.056]])
    assert bool(result.valid) and int(result.events)==6 and jnp.max(jnp.abs(result.v-expected_v))==0
    position_error=float(jnp.max(jnp.abs(result.x-expected_x)));assert position_error<2e-15
    P0=jnp.sum(mass[:,None]*v,axis=0);P1=jnp.sum(mass[:,None]*result.v,axis=0)
    K0=.5*jnp.sum(mass*jnp.sum(v*v,axis=1));K1=.5*jnp.sum(mass*jnp.sum(result.v*result.v,axis=1))
    momentum_error=float(jnp.linalg.norm(P1-P0-result.impulse));energy_error=float(K0-K1-result.kinetic_loss)
    angular_error=float(jnp.linalg.norm(jnp.sum(jnp.cross(result.x,mass[:,None]*result.v)-jnp.cross(x,mass[:,None]*v),axis=0)-result.torque_impulse))
    assert momentum_error<1e-14 and abs(energy_error)<1e-14 and angular_error<1e-14 and float(result.kinetic_loss)==6.
    free=solve(jnp.full((4,3),2.9),jnp.ones((4,3)),jnp.zeros(layout.shape,dtype=bool));assert bool(free.valid) and int(free.events)==0 and float(jnp.max(jnp.abs(free.x-3.04)))<2e-15
    invalid=solve(x,10*v,solid);assert not bool(invalid.valid)
    physical_layout=MACLayout((6,6,6),(-.00075,-.00075,-.00075),.00025)
    physical_x=jnp.asarray([[-1e-6,-.0004,-.0004]])
    physical_v=jnp.asarray([[.05,0.,0.]])
    physical_wall=jnp.asarray(np.indices((6,6,6))[0]>=3)
    zero=advect(physical_x,physical_v,jnp.ones(1),jnp.ones(1,dtype=bool),physical_wall,physical_layout,.00004)
    assert bool(zero.valid) and int(zero.events)==1 and float(zero.x[0,0])<0 and abs(float(zero.x[0,0]))<1e-18
    record=dict(passed=True,physical_steps=0,collision_events=int(result.events),position_error_grid_units=position_error,momentum_error=momentum_error,energy_error=energy_error,angular_error=angular_error,loss_J=float(result.kinetic_loss),wall_work_J=0.,tangential_velocities_unchanged=True,normal_restitution=0,free_three_face_crossing_passed=True,excess_CFL_rejected=True,physical_zero_face_position_m=float(zero.x[0,0]),scope='Face/edge/corner algebra; no physical inlet trajectory replay')
    Path('resin-wall-review.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))

if __name__=='__main__':main()
