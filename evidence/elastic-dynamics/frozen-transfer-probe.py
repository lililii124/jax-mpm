import hashlib,json,os,sys,time
from pathlib import Path
root=Path('/work');parent=Path('/parent');contract=json.loads((root/'contract.json').read_text())
assert sorted(os.sched_getaffinity(0))==[0,1,2,3]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(parent/'output/accepted.npz')==contract['checkpoint_sha256']
for name,digest in contract['source_sha256'].items():assert sha(parent/'source'/name)==digest
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',JAX_PLATFORMS='cpu',JAX_ENABLE_X64='true',XLA_PYTHON_CLIENT_PREALLOCATE='false')
sys.path.insert(0,str(parent/'source'))
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from mpm_jax.core import Config,State,stencil,particle_to_grid,grid_to_particle
from tools.check_elastic_force_refinement import parameters,field
from tools.check_elastic_dynamic_mms import momenta,PERIOD
cfg=Config(n_grid=64,dim=3,dt=1e-6,gravity=(0,0,0),boundary_cells=0,material='transverse-isotropic')
material,_,_,amplitude=parameters('transverse-isotropic')
with np.load(parent/'output/accepted.npz') as data:
 state=State(**{key:jnp.asarray(data[key]) for key in State._fields if key!='history'},history=None)
assert state.valid and len(state.x)==884736
def kinetic(s):return .5*jnp.sum(s.mass[:,None]*s.v**2)+.5*(cfg.dx**2/4)*jnp.sum(s.mass[:,None,None]*s.C**2)
@jax.jit
def probe(s,material):
 support=stencil(s.x,cfg)
 # Only the stress evaluation sees I; no physical state or clock is advanced.
 stress_free=s._replace(F=jnp.broadcast_to(jnp.eye(3),s.F.shape))
 mass,momentum=particle_to_grid(stress_free,material,cfg,support)
 velocity=momentum/jnp.where(mass>0,mass,1.)[:,None]
 v,C=grid_to_particle(velocity,support,cfg);following=s._replace(v=v,C=C)
 before=kinetic(s);grid=.5*jnp.sum(mass[:,None]*velocity**2);after=kinetic(following)
 P0,L0=momenta(s,cfg.dx);P1,L1=momenta(following,cfg.dx)
 scale=jnp.maximum(jnp.sum(s.mass)*jnp.max(jnp.linalg.norm(s.v,axis=1)),1e-30)
 return dict(K_before=before,K_grid=grid,K_after=after,p2g_loss=before-grid,g2p_loss=grid-after,
     relative_loss=(before-after)/jnp.maximum(before,1e-30),momentum_relative=jnp.linalg.norm(P1-P0)/scale,
     angular_momentum_relative=jnp.linalg.norm(L1-L0)/scale,velocity_change=jnp.linalg.norm(v-s.v)/jnp.maximum(jnp.linalg.norm(s.v),1e-30),support_valid=support.valid)
records=[]
def record(name,s):
 begin=time.monotonic();result=probe(s,material);jax.block_until_ready(result)
 result={key:bool(value) if key=='support_valid' else float(value) for key,value in result.items()}
 result.update(case=name,elapsed_s=time.monotonic()-begin)
 assert result['support_valid'] and result['momentum_relative']<1e-10 and result['angular_momentum_relative']<1e-10
 assert result['p2g_loss']>=-1e-12*result['K_before'] and result['g2p_loss']>=-1e-12*result['K_before']
 records.append(result);print(json.dumps(result),flush=True)
record('actual_final_checkpoint',state)
axes=.125+(np.arange(96)+.5)/128
X=np.stack(np.meshgrid(axes,axes,axes,indexing='ij'),axis=-1).reshape(-1,3)
x,F,_=field(X,amplitude);omega=2*np.pi/PERIOD
g=F[:,0,:]-np.eye(3)[0];direction=np.array([1.,.6,-.4]);velocity=omega*(x-X)
C=omega*direction[None,:,None]*g[:,None,:]@np.linalg.inv(F)
exact=state._replace(x=jnp.asarray(x),v=jnp.asarray(velocity),C=jnp.asarray(C),F=jnp.asarray(F))
record('exact_quarter_cycle_mode',exact)
A=jnp.array([[.12,-.08,.03],[.04,-.06,.05],[.02,.01,.09]])
affine=state._replace(v=jnp.array([.03,-.01,.02])+(state.x-.5)@A.T,C=jnp.broadcast_to(A,state.C.shape))
record('affine_control',affine)
assert abs(records[-1]['relative_loss'])<1e-12 and records[-1]['velocity_change']<1e-12
out=dict(passed=True,records=records,parent=contract,physical_time_steps_executed=0,
 scope='Frozen zero-stress APIC remap evaluation, actual final and analytic mode plus affine control; no reconstructed cumulative trajectory energy budget or proof of sole error cause')
output=root/'output';output.mkdir();(output/'review.json').write_text(json.dumps(out,indent=2)+'\n')
