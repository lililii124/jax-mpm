"""Conservative deformed quadrature-cell exclusion from actual Hertz disks."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.optimize import brentq
from examples.triaxial_fibre_geometry import make_geometry,_arc_samples,_integral,centreline,_frame,_disk_sampling
plan=json.loads(Path('input/plan.json').read_text());kw=json.loads(Path('input/context.json').read_text())['kwargs'];keys=('dimensions','diameter','pitch','crimp','clearance','particle_spacing','packing','density','grip_length','max_particles');g=make_geometry(theta_deg=-kw['theta'],**{k:kw[k] for k in keys});meta=g['metadata'];cross=meta['cross_section_particles'];fibres=meta['fibres'];report=json.loads(Path('/points/supported-guard-and-footprint.json').read_text());R=kw['diameter']/2;omega=np.pi*np.sqrt(3)/kw['pitch'];A=kw['crimp'];K=A*omega**2;Kprime=A*omega**3*(1+3*(A*omega)**2);denominator=1-K*R;assert denominator>0
Ls=1+R*K+R**2*Kprime/denominator;Lr=1/denominator;Lq=R/denominator
with np.load('/failed/checkpoint_0008109.npz',allow_pickle=False) as z:x=z['state_x'];F=z['state_F'];failed=z['history_failed'];ids=z['state_fibre_id']
rad,quantile,counts=_disk_sampling(R,kw['particle_spacing']);radial_boundaries=R*np.sqrt(np.r_[0,np.cumsum(counts)/sum(counts)]);ring_of=np.repeat(np.arange(len(counts)),counts);ring_start=np.r_[0,np.cumsum(counts)[:-1]];first=np.r_[0,np.cumsum([f['particle_count'] for f in fibres])[:-1]];cache={};bounds={};detail=[]

def cell_radius(particle):
 if particle in bounds:return bounds[particle]
 fid=int(ids[particle]);f=fibres[fid];local=particle-int(first[fid]);section=local//cross;ordinal=local%cross;ring=int(ring_of[ordinal]);j=ordinal-int(ring_start[ring]);count=counts[ring]
 if fid not in cache:cache[fid]=_arc_samples(*f['parameter_range_m'],kw['pitch'],kw['crimp'],kw['particle_spacing'])
 parameter,length=cache[fid];assert len(parameter)==f['longitudinal_sections'];t0=parameter[section];ds=length/len(parameter);sgrid=np.linspace(-ds/2,ds/2,33);t=[]
 for delta in sgrid:
  if delta==0:t.append(t0)
  else:t.append(brentq(lambda value:float(_integral(t0,value,omega,A))-delta,*sorted([t0,t0+delta]),xtol=1e-16))
 t=np.asarray(t);path,tangent,curvature=centreline(f['family_id'],f['periodic_line_index'],t,pitch=kw['pitch'],crimp=kw['crimp'],theta_deg=-kw['theta']);direction,normal=_frame(f['family_id'],-kw['theta']);normal3=np.array([normal[0],0.,normal[1]]);bending=np.cross(normal3,tangent);r=np.linspace(radial_boundaries[ring],radial_boundaries[ring+1],33);q=np.linspace(2*np.pi*j/count,2*np.pi*(j+1)/count,65);product=curvature[:,None,None]*r[None,:,None];angle=np.broadcast_to(q,(33,33,65)).copy()
 for _ in range(12):angle-=(angle-product*np.sin(angle)-q)/(1-product*np.cos(angle))
 assert np.max(abs(angle-product*np.sin(angle)-q))<1e-12
 points=path[:,None,None,:]+np.asarray(meta['origin_m'])+r[None,:,None,None]*(np.cos(angle)[...,None]*bending[:,None,None,:]+np.sin(angle)[...,None]*normal3)
 sample_radius=float(np.linalg.norm(points-g['x'][particle],axis=-1).max());remainder=Ls*(ds/32)/2+Lr*((r[-1]-r[0])/32)/2+Lq*((q[-1]-q[0])/64)/2;stretch=float(np.linalg.svd(F[particle],compute_uv=False).max());radius=stretch*(sample_radius+remainder);bounds[particle]=radius;detail.append(dict(particle=particle,fibre=fid,section=section,ring=ring,reference_sample_radius_m=sample_radius,reference_Lipschitz_remainder_m=remainder,F_max_singular=stretch,deformed_cell_enclosing_radius_m=radius));return radius

patches=[]
for patch in report['patches']:
 centers=np.asarray(patch['contact_points']);normal=(centers[1]-centers[0])/np.linalg.norm(centers[1]-centers[0]);surfaces=centers+np.array([[1.],[-1.]])*R*normal;rows=[]
 for side,fid in enumerate([15,60]):
  candidates=np.flatnonzero(failed&(ids==fid));minimum=np.inf;worst=None;tested=0
  for particle in candidates:
   f=fibres[fid];ds=f['reference_arclength_m']/f['longitudinal_sections'];coarse=float(np.linalg.svd(F[particle],compute_uv=False).max())*(2*R+ds/2);distance=float(np.linalg.norm(x[particle]-surfaces[side]));lower=distance-patch['contact_radius_m']-coarse
   if lower<=0:lower=distance-patch['contact_radius_m']-cell_radius(int(particle));tested+=1
   if lower<minimum:minimum=lower;worst=int(particle)
  rows.append(dict(fibre=fid,failed_MP_count=len(candidates),refined_cells=tested,minimum_cell_to_Hertz_disk_lower_bound_m=None if not len(candidates) else minimum,worst_particle=worst,healthy=bool(not len(candidates) or minimum>0)))
 patches.append(dict(slot=patch['slot'],representative_segments=patch['representative_segments'],actual_surface_points_m=surfaces.tolist(),contact_radius_m=patch['contact_radius_m'],checks=rows))
healthy=all(row['healthy'] for p in patches for row in p['checks']);result=dict(mechanical_steps_executed=0,actual_contact_points_corrected=True,important='FibrePatchMetrics.closest_points are centreline points; surface points reconstructed using actual radius/normal. Original preliminary point-only audit is not a surface certificate.',scope='Current deformed local-affine quadrature-cell enclosure using original equal-area angular-Jacobian sectors and uniform-arclength cells; not a continuum convergence proof. Dense reference sampling plus global analytic derivative bound encloses the entire cell, then actualF stretches the enclosure.',sampling_grid=[33,33,65],global_reference_derivative_bounds=dict(Ls=Ls,Lr=Lr,Lq=Lq,K=K,Kprime=Kprime),patches=patches,cell_bounds=detail,all_model_footprints_healthy=healthy,positive_support_guard_still_required=True)
Path('output').mkdir();Path('output/material-cell-footprint-proof.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='cell_bounds'},indent=2))
assert healthy,'Cannot admit positive-support continuation: actual footprint cell exclusion failed'
