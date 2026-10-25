"""Small targeted algebra checks before any physical resin filling pilot."""
import json
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from mpm_jax.resin_mac import divergence,transpose_divergence,symmetric_gradient,viscosity_force,pressure_projection
from mpm_jax.resin_apic import MACLayout,particle_to_faces,faces_to_particles


def main():
    rng=np.random.default_rng(41);shape=(6,5,4);h=.00025
    sizes=[tuple(n+(k==c) for k,n in enumerate(shape)) for c in range(3)]
    velocity=tuple(jnp.asarray(rng.normal(size=s)*.02) for s in sizes);p=jnp.asarray(rng.normal(size=shape))
    transpose=transpose_divergence(p,h);lhs=float(jnp.sum(p*divergence(velocity,h)));rhs=float(sum(jnp.sum(a*b) for a,b in zip(velocity,transpose)));adjoint=abs(lhs-rhs)
    assert adjoint<1e-20
    omega=np.array([.7,-.3,.2]);rigid=[]
    for c,size in enumerate(sizes):
        axes=[(np.arange(n)+(0 if k==c else .5))*h for k,n in enumerate(size)]
        x=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1);rigid.append(jnp.asarray(np.cross(omega,x)[...,c]))
    rigid_strain=max(float(jnp.max(jnp.abs(v))) for v in symmetric_gradient(tuple(rigid),h));assert rigid_strain<1e-14
    mass=tuple(jnp.full(s,1100*h**3) for s in sizes);fixed=[]
    for c,s in enumerate(sizes):
        f=np.zeros(s,dtype=bool);index=[slice(None)]*3;index[c]=0;f[tuple(index)]=True;index[c]=-1;f[tuple(index)]=True;fixed.append(jnp.asarray(f))
    velocity=tuple(jnp.where(f,0.,v) for v,f in zip(velocity,fixed))
    force,power=viscosity_force(velocity,mass,tuple(fixed),h,1100.,.05)
    viscwork=float(sum(jnp.sum(v*f) for v,f in zip(velocity,force))+power);assert abs(viscwork)<1e-18 and float(power)>=0
    result=jax.jit(lambda v:pressure_projection(v,mass,tuple(fixed),jnp.ones(shape,dtype=bool),h,.00008))(velocity)
    assert bool(result.valid) and float(result.maximum_divergence_per_s)<1e-4 and abs(float(result.work_identity_J))<1e-17
    layout=MACLayout((8,8,8),(0.,0.,0.),h);axis=np.arange(2,6)*h;points=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3);A=np.array([[.2,-.3,.1],[.1,.4,-.2],[-.4,.1,.2]]);b=np.array([.01,-.02,.03]);v=points@A.T+b;C=np.broadcast_to(A,(len(v),3,3));active=jnp.ones(len(v),dtype=bool);pm=jnp.full(len(v),1e-8)
    masses,momenta,supports=particle_to_faces(jnp.asarray(points),jnp.asarray(v),jnp.asarray(C),pm,active,layout)
    faces=tuple(P/jnp.where(M>0,M,1) for M,P in zip(masses,momenta));after,affine=faces_to_particles(faces,supports,layout)
    error_v=float(jnp.max(jnp.abs(after-v)));error_C=float(jnp.max(jnp.abs(affine-C)))
    assert error_v<1e-15 and error_C<1e-12 and all(bool(s.valid) for s in supports)
    mass_error=max(abs(float(M.sum()-pm.sum())) for M in masses);assert mass_error<1e-20
    record=dict(passed=True,physical_fill_steps=0,divergence_adjoint_error=adjoint,rigid_rotation_strain_error=rigid_strain,viscous_work_adjoint_error_W=viscwork,pressure_relative_residual=float(result.relative_residual),pressure_maximum_divergence_per_s=float(result.maximum_divergence_per_s),pressure_work_identity_J=float(result.work_identity_J),affine_velocity_error=error_v,affine_gradient_error=error_C,mass_transfer_error_kg=mass_error,scope='Operator algebra only; no obstacle/free-surface/inlet filling, CFL trajectory or resin physical qualification')
    Path('resin-operator-review.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))

if __name__=='__main__':main()
