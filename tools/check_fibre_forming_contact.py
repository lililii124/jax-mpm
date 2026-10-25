"""Small targeted algebra checks for the new contact/rotating-collar kernels."""
import json
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from mpm_jax.fibre_forming_contact import project
from mpm_jax.fibre_forming_boundary import angle


def main():
    centres=np.array([[0.,0.,0.],[0.,1.,0.],[0.,.5,1.]])
    mass=np.array([[2.],[3.],[5.]])
    v=np.array([[[0.,1.,1.]],[[0.,-1.,1.]],[[0.,0.,-1.]]])
    gradient=np.zeros_like(v);clamp=np.zeros((3,1),bool)
    solve=jax.jit(project)
    result=solve(jnp.array(mass),jnp.array(v),jnp.array(gradient),jnp.array(centres),jnp.array(clamp))
    out=np.asarray(result.velocity);pair=np.asarray(result.pair_impulse)
    assert bool(result.valid)
    balance=np.sum(mass[:,:,None]*(out-v),axis=(0,1))
    assert np.linalg.norm(balance)<1e-12
    loss=.5*np.sum(mass[:,:,None]*(v*v-out*out))
    assert loss>=-1e-12
    # Torque is conserved for coincident nodal impulses, independent of normal.
    node=np.array([.3,-.2,.1]);torque=np.cross(node,balance)
    assert np.linalg.norm(torque)<1e-12
    # Common rigid velocity/rotation of the whole input preserves the answer.
    a=.71;Q=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    translation=np.array([.6,-.1,.3])
    transformed=solve(jnp.array(mass),jnp.array(v@Q.T+translation),jnp.array(gradient@Q.T),jnp.array(centres@Q.T),jnp.array(clamp))
    assert np.max(np.abs(np.asarray(transformed.velocity)-(out@Q.T+translation)))<1e-12
    # A constrained strand exchanges its counter-impulse with the fixture.
    clamp[0]=True
    fixed=solve(jnp.array(mass),jnp.array(v),jnp.array(gradient),jnp.array(centres),jnp.array(clamp))
    assert bool(fixed.valid) and np.array_equal(np.asarray(fixed.velocity)[0],v[0])
    delta=np.sum(mass[:,:,None]*(np.asarray(fixed.velocity)-v),axis=(0,1))
    assert np.linalg.norm(delta-np.sum(np.asarray(fixed.fixture_impulse),axis=(0,1)))<1e-12
    # No tensile adhesion: already separating configuration is untouched.
    separating=solve(jnp.array(mass),jnp.array(-v),jnp.array(gradient),jnp.array(centres),jnp.zeros((3,1),bool))
    assert np.max(np.abs(np.asarray(separating.velocity)+v))<1e-12
    # Finite-angle secant BC integrates x and F to an exact orthogonal Q.
    dt=3.872966692486445e-6;t=.007
    delta=float(angle(t+dt)-angle(t));c,s=np.cos(delta),np.sin(delta)
    R=np.array([[1,0,0],[0,c,-s],[0,s,c]])
    C=(R-np.eye(3))/dt
    assert abs(np.linalg.det(np.eye(3)+dt*C)-1)<1e-14
    assert abs(float(angle(.02))-2*np.pi)<1e-14
    print(json.dumps(dict(passed=True,momentum_error=float(np.linalg.norm(balance)),contact_loss=float(loss),max_closing=float(result.closing_residual),sweeps=int(result.sweeps),pair_impulses=pair.tolist())))

if __name__=='__main__':main()
