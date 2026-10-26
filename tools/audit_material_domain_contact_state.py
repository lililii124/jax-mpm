"""Read-only physical audit of an accepted and rejected C2 material-domain CP."""
import argparse, json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--result',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    from examples.metal_sphere_impact_fracture import plan_case
    from run_metal_tl_fresh import make_reference
    from mpm_jax.material_domain_contact import (material_domain_candidates,
        point_velocity_coefficients, assemble_contact_jacobian, _independent_point_velocity)
    from mpm_jax.materials.linear_metal import linear_metal
    geometry,plan=plan_case(speed=5000.,sphere_diameter=.002,plate_size=(.02,.02),
        plate_thickness=.0005,gap=.00075,clamp_width=.001,dx=.00025,
        spacing=(.00025,.000125,.00025),end_time=1e-6,frames=60)
    ref,body,fixed,v0,record=make_reference(geometry,.00025)
    mass=np.asarray(ref.particle_mass); nodal=np.asarray(ref.nodal_mass)
    spacing=np.asarray(plan['particle_spacing_m']); ids=np.asarray(geometry['body_id'])
    result={'accepted':{},'rejected_attempt':{}}
    states=[('accepted','accepted.npz')]
    if (a.result/'rejected.npz').is_file(): states.append(('rejected_attempt','rejected.npz'))
    for label,file in states:
        with np.load(a.result/file,allow_pickle=False) as z: state={k:z[k].copy() for k in z.files}
        u,v,F,tau,internal=(state['displacement'],state['velocity'],state['F'],state['tau'],state['internal'])
        x=np.asarray(ref.positions)+np.sum(np.asarray(ref.weights)[...,None]*u[np.asarray(ref.indices)],axis=1)
        signs=np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]],float)
        local=signs*(spacing*.5)[None,:]
        corners=x[:,None,:]+np.einsum('pij,kj->pki',F,local)
        sphere=ids==0; plate=ids==1
        nose_flat=np.argmin(corners[sphere,:,1]); pi,ci=np.unravel_index(nose_flat,corners[sphere,:,1].shape)
        nose=corners[np.flatnonzero(sphere)[pi],ci]
        pc=corners[plate].reshape(-1,3)
        near=np.linalg.norm(pc[:,[0,2]]-nose[[0,2]],axis=1)<=.0025
        surface=float(np.max(pc[near,1]))
        ranges={}; start=0
        for b in (0,1):
            n=record['bodies'][b]['nodes']; sl=slice(start,start+n); mb=nodal[sl]; vb=v[sl]; ib=np.asarray(v0)[sl]
            ranges[str(b)]={'initial_P':(mb[:,None]*ib).sum(0).tolist(),'final_P':(mb[:,None]*vb).sum(0).tolist(),
                'delta_P':(mb[:,None]*(vb-ib)).sum(0).tolist(),'initial_K':float(.5*np.sum(mb[:,None]*ib**2)),
                'final_K':float(.5*np.sum(mb[:,None]*vb**2)),'max_speed_m_s':float(np.max(np.linalg.norm(vb,axis=1)))}
            if b==1: plate_nodes=sl
            start+=n
        sigma=.5*(tau+tau.swapaxes(-1,-2))/np.maximum(np.linalg.det(F)[:,None,None],1e-30)
        dev=sigma-np.trace(sigma,axis1=-2,axis2=-1)[:,None,None]*np.eye(3)/3
        vm=np.sqrt(np.maximum(1.5*np.sum(dev*dev,axis=(-2,-1)),0))
        candidates=material_domain_candidates(x,F,body,spacing)
        rows=[]
        for hit in candidates:
            ca=point_velocity_coefficients(ref.weights[hit.first],ref.gradients[hit.first],hit.xi_first)
            cb=point_velocity_coefficients(ref.weights[hit.second],ref.gradients[hit.second],hit.xi_second)
            rows.append(dict(nodes_a=np.asarray(ref.indices[hit.first]),coeff_a=ca,nodes_b=np.asarray(ref.indices[hit.second]),
                coeff_b=cb,normal=hit.normal,point=hit.point,penetration=hit.penetration,
                first=hit.first,second=hit.second,xi_first=hit.xi_first,xi_second=hit.xi_second))
        if rows:
            J=assemble_contact_jacobian(rows,len(nodal),fixed); im=np.divide(1.,np.repeat(nodal,3),out=np.zeros(3*len(nodal)),where=np.repeat(nodal,3)>0)
            sq=np.asarray(J.multiply(J).sum(axis=1)).ravel(); mobility=np.asarray(J.multiply(im[None,:])@J.T.diagonal()).ravel() if False else None
            mobility=np.asarray((J.multiply(J)).dot(im)).ravel()
            normal_v=np.asarray(J@v.ravel()).ravel()
            for r,row in enumerate(rows):
                cols=J.getrow(r).indices; participating=np.unique(cols//3); nm=nodal[participating]; nm=nm[nm>0]
                va=_independent_point_velocity(v,ref,row['first'],row['xi_first']); vb=_independent_point_velocity(v,ref,row['second'],row['xi_second'])
                route=float(np.dot(row['normal'],va-vb))
                row['effective_contact_mass_kg']=float(1/mobility[r]) if mobility[r]>0 else None
                row['J_euclidean_norm']=float(np.sqrt(sq[r])); row['participating_node_count']=int(len(participating))
                row['participating_mass_min_kg']=float(nm.min()) if nm.size else None
                row['participating_mass_max_kg']=float(nm.max()) if nm.size else None
                row['postclosing_assembled_m_s']=float(normal_v[r]); row['postclosing_independent_m_s']=route
                row['pair']=[row.pop('first'),row.pop('second')]; row['point_m']=row.pop('point').tolist()
                row['normal']=row['normal'].tolist(); row['penetration_m']=row.pop('penetration')
                row['xi_m']=[row.pop('xi_first').tolist(),row.pop('xi_second').tolist()]
        else: mobility=np.zeros(0)
        result[label]=dict(step=int(state['step']) if 'step' in state else None,time_s=float(state['time_s']),
            energy_error_J=float(state['energy_error_J']) if 'energy_error_J' in state else None,
            total_velocity_min=float(v.min()),total_velocity_max=float(v.max()),
            bodies=ranges,plate_max_nodal_displacement_m=float(np.max(np.linalg.norm(u[plate_nodes],axis=1))),
            plate_max_particle_von_mises_Pa=float(vm[plate].max()),plate_min_detF=float(np.linalg.det(F[plate]).min()),
            sphere_min_detF=float(np.linalg.det(F[sphere]).min()),plate_internal_energy_J=float(np.sum(mass[plate]*internal[plate])),
            sphere_internal_energy_J=float(np.sum(mass[sphere]*internal[sphere])),
            sphere_nose_m=nose.tolist(),nearby_deformed_plate_surface_y_m=surface,
            nose_surface_gap_m=float(nose[1]-surface),candidate_count=len(candidates),
            max_intersection_depth_m=float(max((h.penetration for h in candidates),default=0.)),
            representative_rows=sorted(rows,key=lambda r:r['effective_contact_mass_kg'] or np.inf)[:12])
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False,
        default=lambda value: value.tolist() if isinstance(value,np.ndarray) else value.item())+'\n')
    print(json.dumps({k:{q:v for q,v in x.items() if q not in ('representative_rows','bodies')} for k,x in result.items()},indent=2))

if __name__=='__main__': main()
