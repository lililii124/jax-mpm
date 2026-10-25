import os,sys,json,hashlib
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4')
import numpy as np, jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
sys.path.insert(0,'/source')
from run_metal_tl_fresh import make_reference
from examples.metal_sphere_impact_fracture import plan_case
from mpm_jax.metal_tl_midpoint import MetalTLState,step,actual_q1_contact_jump
from mpm_jax.materials.linear_metal import linear_metal,eos_state
from mpm_jax.fragment_multifield_v6_mpm import elastic_energy

g,p=plan_case(speed=5000.,sphere_diameter=.002,plate_size=(.02,.02),plate_thickness=.0005,gap=.00075,clamp_width=.001,dx=.00025,spacing=(.00025,.000125,.00025),end_time=1e-6,frames=60)
ref,body,fixed,v0,rr=make_reference(g,.00025); material=linear_metal(**p['material'])
with np.load('/prior/accepted.npz',allow_pickle=False) as a:
 fields=tuple(jnp.asarray(a[k]) for k in ('F','tau','ep','cold','fracture','damage','internal'))
 state=MetalTLState(jnp.asarray(a['displacement']),jnp.asarray(a['velocity']),fields,jnp.asarray(a['time_s']),jnp.asarray(a['initial_P']),jnp.asarray(a['initial_L']),jnp.asarray(a['initial_K']),jnp.asarray(a['contact_work']),jnp.asarray(a['viscous_loss']))
 stepnum=int(np.load('/prior/accepted.npz',allow_pickle=False)['step'])+1
F,tau,ep,cold,fracture,damage,internal=fields
W=jnp.einsum('pki,pkj->pij',state.velocity[ref.indices],ref.gradients)
thermo_energy=internal-cold-fracture-elastic_energy(tau,material,damage)
thermo=eos_state(jnp.linalg.det(F),thermo_energy,material)
L=W@jnp.linalg.inv(F); C=jnp.sqrt(jnp.sum(L*L,axis=(-2,-1)))
vpart=jnp.linalg.norm(jnp.sum(ref.weights[...,None]*state.velocity[ref.indices],axis=1),axis=-1)
compression=jnp.maximum(-jnp.trace(L,axis1=-2,axis2=-1),0.)
visc_signal=material.viscosity_linear*thermo.sound_speed+2*material.viscosity_quadratic*.00025*compression
signal=thermo.longitudinal_speed+vpart+1.5*np.sqrt(3)*.00025*C+visc_signal
dt=.1*.00025/float(jnp.max(signal))
fixed_particles=jnp.asarray((body==1)&((np.abs(g['x'][:,0])>=.009)|(np.abs(g['x'][:,2])>=.009)))
def contact(vpre,x1,F1):
 return actual_q1_contact_jump(vpre,ref,ref.particle_mass,jnp.asarray(body),x1,F1,jnp.eye(3),.00025,fixed,fixed_particles=fixed_particles)
out=step(state,ref,material,dt,jnp.asarray(fixed),contact,cell_length=.00025)
report=dict(step=stepnum,prior_step=stepnum-1,prior_time_s=float(state.time),dt_s=dt,step_valid=bool(np.asarray(out.valid)),force_residual=float(np.asarray(out.force_residual)),contact=out.contact_report,input_P=np.asarray(state.initial_momentum).tolist(),output_P=np.asarray(out.geometry['momentum']).tolist(),input_L=np.asarray(state.initial_angular_momentum).tolist(),output_L=np.asarray(out.geometry['angular_momentum']).tolist(),input_K=float(state.initial_kinetic),output_K=float(out.geometry['kinetic']),source_sha256={k:hashlib.sha256(open('/source/'+k,'rb').read()).hexdigest() for k in ('mpm_jax/metal_tl_midpoint.py','run_metal_tl_fresh.py')},checkpoint_sha256=hashlib.sha256(open('/prior/accepted.npz','rb').read()).hexdigest(),state='read-only-one-step-diagnostic; no checkpoint written')
print(json.dumps(report,indent=2))

