"""Independent fragment velocity fields; reference bonds only determine field splits."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np


def reference_edges(x,physical,radius):
    from scipy.spatial import cKDTree
    pairs=cKDTree(x).query_pairs(radius,output_type='ndarray')
    return pairs[physical[pairs[:,0]]==physical[pairs[:,1]]]


def split_fields(x,reference,edges,alive,damage,physical,max_fields=8192):
    length=np.linalg.norm(x[edges[:,0]]-x[edges[:,1]],axis=1)
    original=np.linalg.norm(reference[edges[:,0]]-reference[edges[:,1]],axis=1)
    # Break only after constitutive failure AND actual opening. Broken bonds never heal.
    alive=alive&~((np.maximum(damage[edges[:,0]],damage[edges[:,1]])>=.95)&(length>1.10*original))
    parent=np.arange(len(x))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for a,b in edges[alive]:
        a,b=find(a),find(b)
        if a!=b:parent[b]=a
    roots=np.array([find(i) for i in range(len(x))]);_,labels=np.unique(roots,return_inverse=True)
    count=int(labels.max()+1)
    if count>max_fields:raise RuntimeError('Fragment field budget exceeded; never merge detached matter')
    bodies=np.array([physical[np.flatnonzero(labels==i)[0]] for i in range(count)])
    return labels.astype(np.int32),alive,bodies


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--coarse',action='store_true');p.add_argument('--resolution',choices=('medium','10k','25k'),default='medium');p.add_argument('--frames',type=int,default=8);args=p.parse_args()
    import jax;jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from examples.metal_sphere_impact_fracture import plan_case,save_json,save_npz
    from mpm_jax import fragment_multifield_compact_mpm as solver
    from mpm_jax.materials.linear_metal import linear_metal
    from tools.audit_metal_fragments import components
    dx=.0005 if args.coarse else dict(medium=.00025,**{'10k':1/6000.,'25k':.000125})[args.resolution]
    plate_width=.0065 if args.resolution=='25k' else .006
    geometry,plan=plan_case(sphere_diameter=.002,plate_size=(plate_width,plate_width),plate_thickness=.0005,gap=3*dx,clamp_width=.001,dx=dx,spacing=(dx,dx/2,dx),frames=args.frames,end_time=2e-6)
    if args.output.exists():raise FileExistsError('Fresh output required')
    args.output.mkdir(parents=True)
    physical=geometry['body_id'];reference=geometry['x'];edges=reference_edges(reference,physical,1.05*dx);alive=np.ones(len(edges),bool)
    labels,alive,bodies=split_fields(reference,reference,edges,alive,np.zeros(len(reference)),physical)
    material=linear_metal();config=solver.ImpactConfig();state=None;rows=[];begun=time.monotonic();boundary=np.zeros(3)
    save_json(args.output/'assumptions.json',dict(plan=plan,damage_onset=.10,damage_failure=.30,bond_break_damage=.95,bond_opening_ratio=1.10,max_fields=8192,contact='Eight damped Jacobi frictionless impulse sweeps; split boundary projection records reaction; no friction heat',boundary='Plate edge strips fixed xyz; no backing; each detached group retains original fixed reference edge membership',intact_J_domain=[.4,2.],fully_failed_tensile_J_domain='finite J>=1, traction-free; no intact upper bound',free_energy='Compression unchanged; tensile and shear elastic free energies scaled by1-D; damage release only fracture_energy; plastic heat scaled by pre-step surviving fraction',scope='Exploratory hypotheses, resolution dependent damage, no phase model; no promised cloud'))
    for frame,target in enumerate(plan['output_times_s']):
        x=reference if state is None else np.asarray(state.x)
        if state is not None:labels,alive,bodies=split_fields(x,reference,edges,alive,np.asarray(state.damage),physical)
        duration=target-(0. if state is None else float(state.time))
        velocity=geometry['v'] if state is None else np.asarray(state.v)
        predicted=x+duration*velocity
        lower=np.full((len(bodies),3),np.inf);upper=np.full((len(bodies),3),-np.inf)
        np.minimum.at(lower,labels,np.minimum(x,predicted));np.maximum.at(upper,labels,np.maximum(x,predicted))
        bounds=np.stack((lower-2*dx,upper+2*dx),axis=-1)
        grid=solver.build_grid(x,labels,dx=dx,padding_cells=3,reference_bounds=bounds,max_nodes=2_000_000,field_physical_body=bodies,contact_sweeps=(x,predicted))
        # Clamp only original plate edge reference particles' supporting nodes.
        fixed=(physical==1)&((np.abs(reference[:,0])>=plate_width/2-.001)|(np.abs(reference[:,2])>=plate_width/2-.001))
        support=solver.stencil(jnp.asarray(x),jnp.asarray(labels),grid.layout)
        mask=np.zeros((grid.layout.node_count,3),bool);ids=np.asarray(support.indices)[fixed].ravel();ids=ids[(ids>=0)&(ids<grid.layout.node_count)];mask[ids]=True
        grid=replace(grid,clamped=jnp.asarray(mask))
        if state is None:state=solver.make_state(x,geometry['volume'],labels,material,grid,v=geometry['v'])
        else:state=state._replace(body_id=jnp.asarray(labels))
        if frame:
            state,steps=solver.advance_to(state,material,grid,config,jnp.asarray(target),max_steps=20000)
            state.x.block_until_ready()
            if not bool(state.valid):
                save_npz(args.output/'failed_state.npz',**{k:np.asarray(v) for k,v in state._asdict().items()});raise RuntimeError('Rejected interval retained')
        solver.validate_state(state,material,grid)
        audit=components(np.asarray(state.x),np.asarray(state.mass),np.asarray(state.v),physical,separation_radius=1.05*dx)
        thermo=solver.thermodynamics(state,material);ledger=np.asarray(solver.energy_ledger(state,grid))
        row=dict(frame=frame,time=float(state.time),fields=int(len(bodies)),nodes=grid.layout.node_count,broken_bonds=int((~alive).sum()),spatial_components=len(audit['components']),largest_mass_kg=audit['components'][0]['mass_kg'],mass_kg=float(state.mass.sum()),momentum_Ns=np.sum(np.asarray(state.mass)[:,None]*np.asarray(state.v),axis=0).tolist(),physical_energy_J=float(ledger[:3].sum()),boundary_reservoir_J=float(ledger[3]),numerical_reservoir_J=float(ledger[4]),accounted_energy_J=float(ledger.sum()),boundary_impulse_Ns=np.asarray(state.boundary_impulse).tolist(),momentum_residual_Ns=float(np.linalg.norm(np.sum(np.asarray(state.mass)[:,None]*(np.asarray(state.v)-geometry['v']),axis=0)-np.asarray(state.boundary_impulse))),separated_mass_kg=float(audit['total_mass_kg']-audit['components'][0]['mass_kg']),maximum_damage=float(state.damage.max()),maximum_temperature_K=float(thermo.temperature.max()),elapsed_s=time.monotonic()-begun)
        rows.append(row);save_json(args.output/'metrics.json',rows);save_json(args.output/f'groups-{frame:03}.json',audit)
        save_npz(args.output/'checkpoint.npz',**{k:np.asarray(v) for k,v in state._asdict().items()},physical_body=physical,reference=reference,edges=edges,alive=alive)
        print(json.dumps(row),flush=True)


if __name__=='__main__':main()
