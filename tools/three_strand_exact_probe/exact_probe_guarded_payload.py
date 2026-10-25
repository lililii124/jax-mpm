import os,sys,json,time
from pathlib import Path
os.environ.setdefault("JAX_PLATFORMS","cpu")
sys.path.insert(0,os.environ["MPM_REPO_ROOT"])
import numpy as np
import jax
jax.config.update("jax_enable_x64",True)
import jax.numpy as jnp
from mpm_jax.fibre_mpm import FibreState
from mpm_jax.materials.isotropic_brittle import BrittleHistory
from mpm_jax.fibre_contact import FibreContactHistory,fibre_contact_parameters
from mpm_jax.fibre_coupling import build_contact_data,FibreCouplingHistory
from mpm_jax.contact_patches import patch_neighbours
from mpm_jax.rope_hertz_contact import evaluate_rope_contact
root=Path(os.environ["MPM_REPO_ROOT"])
p=Path(os.environ["MPM_ORIGINAL_DATA"])
cp=np.load(p/"checkpoint.npz");plan=json.loads((p/"plan.json").read_text());g=plan["geometry"]
x0=np.load(p/"trajectory_x.npy",mmap_mode="r")[0]
candidates=[]
from examples.three_strand_tension import centreline
for a,b in ((0,1),(0,2),(1,2)):
 candidates.append({"fibre_ids":[a,b],"reference_centreline_points_m":[centreline(np.array([g["length_m"]/2]),i,g["orbit_radius_m"],g["pitch_m"])[0].tolist() for i in (a,b)]})
geom={"x":x0,"mass":cp["state_mass"],"fibre_id":cp["state_fibre_id"],"metadata":g,"contact_candidates":candidates}
data=build_contact_data(geom,search_window=8,coverage_skin=plan["contact"]["coverage_skin_m"],max_coverage_pairs=600000)
nei=patch_neighbours(np.asarray(data.coverage_pairs),np.asarray(data.segment_fibre))
state=FibreState(**{k:jnp.asarray(cp["state_"+k]) for k in FibreState._fields if k!="history"},history=BrittleHistory(*(jnp.asarray(cp["history_"+k]) for k in BrittleHistory._fields)))
h=FibreCouplingHistory(FibreContactHistory(jnp.asarray(cp["contact_law_active"]),jnp.asarray(cp["contact_law_valid"])),*(jnp.asarray(cp["contact_history_"+k]) for k in FibreCouplingHistory._fields if k!="law"))
law=fibre_contact_parameters(effective_modulus=2e6/(2*(1-.25**2)),effective_radius=.001,work_of_adhesion=0.,model="hertz")

import psutil
from mpm_jax.fibre_mpm import build_layout, FibreConfig, reference_grips, step_with_diagnostics,stencil
from mpm_jax.materials.isotropic_brittle import isotropic_brittle
from examples.single_fibre_tension import motion
# Resource policy is enforced before import and continuously by guarded_replay.py.
layout=build_layout(x0,geom["fibre_id"],dx=plan["grid"]["dx_m"],padding_cells=6,reference_bounds=plan["grid"]["bounds_m"],max_nodes=1800000)
material=isotropic_brittle(**plan["material"]);config=FibreConfig(plan["dt"])
axial=np.tile(np.repeat((np.arange(g["sections_per_strand"])+.5)*g["length_m"]/g["sections_per_strand"],g["cross_section_particles"]),3)
masks=tuple((geom["fibre_id"]==f)&(axial<g["grip_length_m"] if side==0 else axial>g["length_m"]-g["grip_length_m"]) for f in range(3) for side in (0,1))
def speed(t):return (motion(t,plan["protocol"],jnp)-motion(t-config.dt,plan["protocol"],jnp))/config.dt
velocities=tuple((lambda t,side=side:jnp.array([(-1 if side==0 else 1)*speed(t)/2,0.,0.])) for f in range(3) for side in (0,1))
boundary=reference_grips(masks,velocities)
@jax.jit
def exact_step(s,h):
 force,updated,m,extent,unsupported=evaluate_rope_contact(s,h,data,law,nei,capacity=4096,max_patches=32,max_iterations=256)
 new,d=step_with_diagnostics(s,material,layout,config,force,boundary=boundary)
 valid_before_contact=new.valid
 new=new._replace(valid=new.valid&m.valid)
 return new,updated,m,extent,unsupported,valid_before_contact
out=[]
for k in range(1,33):
 previous_state=state
 state,h,m,e,u,mpmvalid=exact_step(state,h)
 state.x.block_until_ready()
 row={"attempted_step":3500+k,"mpm_valid_before_contact":bool(mpmvalid),"contact_valid":bool(m.valid),"combined_valid":bool(state.valid),"extent_m":float(e),"unsupported_local_extent":bool(u),"Jmin":float(np.linalg.det(np.asarray(state.F)).min()),"history_valid":bool(np.asarray(state.history.valid).all()),"damage_max":float(np.asarray(state.history.damage).max()),"process_uss_bytes":psutil.Process().memory_full_info().uss}
 for name in m._fields:
  a=np.asarray(getattr(m,name))
  if a.ndim==0: row[name]=a.item()
  elif a.dtype.kind=="b":row[name+"_count"]=int(a.sum())
 if bool(u):
  from patch_identity import identify
  row['patch_identity']=identify(previous_state,data,nei,m)
  if not np.isclose(row['patch_identity']['maximum_extent_m'],float(e),rtol=1e-12,atol=1e-15):
   raise RuntimeError('Read-only patch extent does not reproduce original predicate')
 out.append(row);print(json.dumps(row),flush=True)
 if not bool(state.valid):break
Path(os.environ["MPM_PROBE_OUTPUT"]).write_text(json.dumps({"scope":"Isolated exact dt/kernel/BC/contact-carry diagnostic from original last-valid f12 checkpoint; no trajectory/checkpoint written; failed evidence remains never-resumable","samples":out},indent=2))
