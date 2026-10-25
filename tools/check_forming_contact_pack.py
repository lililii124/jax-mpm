"""Compare full/packed contact operators at one actual checkpoint, no step."""
import json,time
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from mpm_jax.fibre_mpm import FibreState,FibreConfig,build_layout,stencil,particle_to_grid
from mpm_jax.materials.isotropic_brittle import BrittleHistory,isotropic_brittle
from mpm_jax.fibre_forming_boundary import prescribe
from mpm_jax.fibre_forming_contact import mass_gradient
from mpm_jax.fibre_forming_active_set import project as full
from mpm_jax.fibre_forming_contact_packed import project as packed
from examples.three_strand_forming import geometry
p=json.loads(Path('/parent/plan.json').read_text());g=geometry()
with np.load('/parent/checkpoint.npz') as cp:
    state=FibreState(**{k[6:]:jnp.asarray(cp[k]) for k in cp.files if k.startswith('state_')},history=BrittleHistory(*(jnp.asarray(cp['history_'+k]) for k in BrittleHistory._fields)));step=int(cp['step'])
material=isotropic_brittle(**p['material']);config=FibreConfig(p['dt']);layout=build_layout(g['x'],g['ids'],dx=p['dx'],reference_bounds=np.broadcast_to(np.asarray(p['bounds']),(3,3,2)),padding_cells=3,max_nodes=1600000);nodes=layout.node_count//3;masks=tuple(jnp.asarray(m) for m in g['masks'])
@jax.jit
def inputs(s):
    support=stencil(s.x,s.fibre_id,layout);mass,p0=particle_to_grid(s._replace(tau=jnp.zeros_like(s.tau)),layout,config,support);_,pf=particle_to_grid(s,layout,config,support);divide=jnp.where(mass>0,mass,1)[:,None]
    begin,clamped,_=prescribe(p0/divide,mass,s,support,layout,masks,config.dt,p['duration']);force=begin+(pf-p0)/divide;v,_,_=prescribe(force,mass,s,support,layout,masks,config.dt,p['duration'])
    gradient=mass_gradient(s,support,layout).reshape(3,nodes,3)
    centres=jnp.stack([jnp.sum(jnp.where((s.fibre_id==i)[:,None],s.mass[:,None]*s.x,0),axis=0)/jnp.sum(jnp.where(s.fibre_id==i,s.mass,0)) for i in range(3)])
    return mass.reshape(3,nodes),v.reshape(3,nodes,3),gradient,centres,clamped.reshape(3,nodes)
args=inputs(state);jax.block_until_ready(args)
# Compilation is separated; exactly one execution of each same-state operator.
f=jax.jit(full).lower(*args).compile();h=jax.jit(packed).lower(*args).compile()
begun=time.monotonic();a=f(*args);jax.block_until_ready(a);full_seconds=time.monotonic()-begun
begun=time.monotonic();b=h(*args);jax.block_until_ready(b);packed_seconds=time.monotonic()-begun
errors={name:float(np.max(np.abs(np.asarray(getattr(a,name))-np.asarray(getattr(b,name))))) for name in ['velocity','pair_impulse','fixture_impulse']}
assert bool(a.valid) and bool(b.valid) and bool(b.shared_capacity_valid)
assert errors['velocity']<1e-12 and errors['pair_impulse']<1e-20 and errors['fixture_impulse']<1e-20
assert float(a.closing_residual)<=1e-8 and float(b.closing_residual)<=1e-8
review=dict(passed=True,checkpoint_step=step,full_grid_nodes=nodes,shared_nodes=int(b.shared_nodes),shared_capacity=16384,shared_capacity_valid=bool(b.shared_capacity_valid),max_errors=errors,full_residual=float(a.closing_residual),packed_residual=float(b.closing_residual),full_sweeps=int(a.sweeps),packed_sweeps=int(b.sweeps),full_seconds=full_seconds,packed_seconds=packed_seconds,scope='One actual-state operator comparison, no particle/time/material advance; all positive shared masses included, zero padding and drop scatter.')
Path('contact-pack-review.json').write_text(json.dumps(review,indent=2));print(json.dumps(review,indent=2),flush=True)
