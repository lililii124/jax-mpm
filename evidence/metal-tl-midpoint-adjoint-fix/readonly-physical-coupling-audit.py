import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS','4'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1'); os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('XLA_FLAGS','--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4')
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
sys.path.insert(0,'/work/source')
from examples.metal_sphere_impact_fracture import plan_case
from mpm_jax.total_lagrangian import ReferenceGrid, reference_grid, sample_nodal
from mpm_jax.core import Config
from mpm_jax.fragment_v6_matrixfree_projection import quadratic_reconstruction, polynomial_modes, physical_contact, physical_adjoint, physical_sample
from mpm_jax.metal_tl_midpoint import tl_velocity_map
from mpm_jax.metal_tl_midpoint import tl_velocity_map_transpose

def make_reference(g,dx):
 refs=[]; offsets=[]; no=0
 for b in (0,1):
  take=g['body_id']==b; x=np.asarray(g['x'][take],dtype=np.float64)
  lower=np.floor(x.min(0)/dx)*dx-2*dx; upper=np.ceil(x.max(0)/dx)*dx+2*dx
  n=int(max(8,np.ceil((upper-lower)/dx).max()))
  cfg=Config(n_grid=n,dt=1e-12,domain_size=n*dx,dim=3,boundary_cells=0,material='elastic',failure_mode='retain')
  r=reference_grid(jnp.asarray(x-lower),jnp.asarray(g['volume'][take]),2700.,cfg)
  refs.append(dict(positions=x,indices=np.asarray(r.indices)+no,weights=np.asarray(r.weights),gradients=np.asarray(r.gradients),volume=np.asarray(r.volume),particle_mass=np.asarray(r.particle_mass),nodal_mass=np.asarray(r.nodal_mass),nodes=np.asarray(r.nodes)+lower,body=np.full(len(x),b,np.int32)))
  offsets.append((no,no+len(r.nodes)));no+=len(r.nodes)
 pos=np.concatenate([r['positions'] for r in refs]); idx=np.concatenate([r['indices'] for r in refs]); w=np.concatenate([r['weights'] for r in refs]); grad=np.concatenate([r['gradients'] for r in refs]); vol=np.concatenate([r['volume'] for r in refs]); pm=np.concatenate([r['particle_mass'] for r in refs]); nm=np.concatenate([r['nodal_mass'] for r in refs]); nodes=np.concatenate([r['nodes'] for r in refs]); body=np.concatenate([r['body'] for r in refs])
 ref=ReferenceGrid(*[jnp.asarray(a) for a in (pos,idx,w,grad,vol,pm,nm,nodes)],jnp.asarray(True))
 fixed=np.zeros((len(nodes),3),bool); ps=offsets[1][0]; pn=nodes[ps:]; fixed[ps:]=((np.abs(pn[:,0])>=.009)|(np.abs(pn[:,2])>=.009))[:,None]
 node_body=np.concatenate([np.full(len(r['nodes']),i,np.int32) for i,r in enumerate(refs)])
 return ref,body,fixed,node_body

geom,plan=plan_case(speed=5000.,sphere_diameter=.002,plate_size=(.02,.02),plate_thickness=.0005,gap=.00075,clamp_width=.001,dx=.00025,spacing=(.00025,.000125,.00025),end_time=1e-6,frames=60)
ref,body,fixed,node_body=make_reference(geom,.00025)
cp=np.load('/work/result/accepted.npz',allow_pickle=False)
u=jnp.asarray(cp['displacement']); v=np.asarray(cp['velocity']); F=np.asarray(cp['F']); tau=np.asarray(cp['tau']); internal=np.asarray(cp['internal'])
m=np.asarray(ref.nodal_mass); nodes=np.asarray(ref.nodes); body=np.asarray(body); node_body=np.asarray(node_body)
metrics={}
for b in (0,1):
 sl=node_body==b; mb=m[sl]; vb=v[sl]; xb=nodes[sl]+np.asarray(u)[sl]
 metrics['sphere' if b==0 else 'plate']={'mass_kg':float(mb.sum()),'P_kg_m_s':(mb[:,None]*vb).sum(0).tolist(),'K_J':float(.5*np.sum(mb[:,None]*vb**2)),'moving_nodes':int(np.count_nonzero(mb))}
plate=node_body==1; fixed_node=np.asarray(fixed).any(axis=1); platep=np.asarray(geom['body_id'])==1
metrics['plate'].update({'internal_energy_J':float(np.sum(np.asarray(ref.particle_mass)[platep]*internal[platep])),'fixed_fraction_mass':float(m[plate&fixed_node].sum()/m[plate].sum()),'displacement_rms_m':float(np.sqrt(np.average(np.sum(np.asarray(u)[plate]**2,axis=1),weights=m[plate]))),'displacement_max_m':float(np.linalg.norm(np.asarray(u)[plate],axis=1).max()),'F_minus_I_fro_rms':float(np.sqrt(np.mean(np.sum((F[platep]-np.eye(3))**2,axis=(1,2))))),'F_minus_I_fro_max':float(np.linalg.norm(F[platep]-np.eye(3),axis=(1,2)).max()),'tau_fro_max_Pa':float(np.linalg.norm(tau[platep],axis=(1,2)).max()),'tau_fro_rms_Pa':float(np.sqrt(np.mean(np.sum(tau[platep]**2,axis=(1,2)))))})
x1=np.asarray(geom['x'])+np.asarray(sample_nodal(u,ref)); spherep=np.asarray(geom['body_id'])==0
X=np.asarray(geom['x']); center=np.array((0.,.001,0.)); nose_i=np.flatnonzero(spherep)[np.argmin(np.linalg.norm(X[spherep]-center,axis=1))]; nose_surface=x1[nose_i]+F[nose_i]@(center-X[nose_i])
ytop=X[platep,1].max(); top_ids=np.flatnonzero(platep&(X[:,1]>=ytop-1e-12)); topi=top_ids[np.argmin(np.linalg.norm(X[top_ids]-np.array((0.,ytop,0.)),axis=1))]; face_ref=np.array((0.,ytop+.0000625,0.)); plate_surface=x1[topi]+F[topi]@(face_ref-X[topi])
metrics['geometry']={'sphere_material_nose_xyz_m':center.tolist(),'nose_nearest_particle':int(nose_i),'nose_surface_deformed_xyz_m':nose_surface.tolist(),'plate_material_top_xyz_m':face_ref.tolist(),'plate_nearest_particle':int(topi),'plate_surface_deformed_xyz_m':plate_surface.tolist(),'nose_minus_plate_gap_m':float(nose_surface[1]-plate_surface[1]),'nose_particle_F':F[nose_i].tolist(),'plate_surface_particle_F':F[topi].tolist(),'plate_surface_particle_tau_Pa':tau[topi].tolist()}
E0=jnp.eye(3); recon=quadratic_reconstruction(jnp.asarray(x1),ref.particle_mass,jnp.asarray(body),jnp.asarray(F),E0,jnp.asarray(.00025),jnp.zeros((len(body),),bool),capacity=65536)
modes,polys,mreport=polynomial_modes(recon,jnp.asarray(.00025)); contact=physical_contact(recon,modes,polys,jnp.asarray(.00025),pair_capacity=512,particle_capacity=4096)
count=int(np.asarray(contact['pair_count'])); B=np.zeros((count,len(m),3));
for r in range(count):
 one=jnp.zeros(contact['first'].shape,dtype=jnp.float64).at[r].set(1.); phase=physical_adjoint(recon,contact,one); B[r]=np.asarray(tl_velocity_map_transpose(phase,ref,ref.particle_mass,E0,len(m)))
flat=B.reshape(count,-1); closing=flat@v.ravel(); dens=np.repeat(m,3); inv=np.divide(1.,dens,out=np.zeros_like(dens),where=dens>0); mobility=np.sum(flat*flat*inv[None,:],axis=1); eff=np.divide(1.,mobility,out=np.full_like(mobility,np.inf),where=mobility>0); row=int(np.argmin(eff)); jr=B[row]; jn=np.linalg.norm(jr,axis=1); nz=np.flatnonzero(jn>1e-3*jn.max()); metrics['contact_row']={'candidate_count':count,'row_min_effective_mass':row,'closing_at_final_state_m_s':float(closing[row]),'J_node_l2':float(np.linalg.norm(jr)),'J_node_max_component':float(np.max(np.abs(jr))),'effective_mass_kg':float(eff[row]),'participating_nodes_above_1e-3_max':len(nz),'participating_nodal_mass_min_kg':float(m[nz].min()) if len(nz) else None,'participating_nodal_mass_max_kg':float(m[nz].max()) if len(nz) else None,'effective_mass_min_median_max_kg':np.quantile(eff,[0,.5,1]).tolist()}
actual=np.asarray(physical_sample(recon,contact,tl_velocity_map(jnp.asarray(v),ref,ref.particle_mass,E0).reshape(-1,3)))[:count]; rigid_s=np.zeros_like(v); rigid_p=np.zeros_like(v); rigid_both=np.zeros_like(v); rigid_s[node_body==0,1]=1.; rigid_p[node_body==1,1]=1.; rigid_both[:,1]=1.; ns=np.asarray(contact['normals'])[:count,1]; cs=np.asarray(physical_sample(recon,contact,tl_velocity_map(jnp.asarray(rigid_s),ref,ref.particle_mass,E0).reshape(-1,3)))[:count]; cp_=np.asarray(physical_sample(recon,contact,tl_velocity_map(jnp.asarray(rigid_p),ref,ref.particle_mass,E0).reshape(-1,3)))[:count]; cb=np.asarray(physical_sample(recon,contact,tl_velocity_map(jnp.asarray(rigid_both),ref,ref.particle_mass,E0).reshape(-1,3)))[:count]; metrics['rigid_copy']={'phaseA_vs_Jnode_velocity_max_abs_m_s':float(np.max(np.abs(actual-closing))),'common_translation_max_abs_m_s':float(np.max(np.abs(cb))),'sphere_only_translation_vs_normal_max_abs':float(np.max(np.abs(cs-ns))),'plate_only_translation_vs_minus_normal_max_abs':float(np.max(np.abs(cp_+ns))),'both_body_translation_max_abs_m_s':float(np.max(np.abs(cb)))}
print(json.dumps(metrics,indent=2))









