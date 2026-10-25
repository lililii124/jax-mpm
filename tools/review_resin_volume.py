"""Read-only real-checkpoint volume audit; no physical time step or replay."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion


def main():
    p=argparse.ArgumentParser();p.add_argument('flow',type=Path);p.add_argument('forming',type=Path);p.add_argument('--output',type=Path,default=Path('resin-volume-review.json'));a=p.parse_args();plan=json.loads((a.flow/'output/plan.json').read_text())
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from mpm_jax.resin_apic import MACLayout,cell_mass
    from examples.three_strand_resin_geometry import obstacle_from_checkpoint,sha256
    geometry=obstacle_from_checkpoint(a.forming/'checkpoint.npz',a.forming/'plan.json');assert geometry['receipt']==plan['forming']
    with np.load(a.flow/'output/checkpoint.npz',allow_pickle=False) as d:x=d['x'].copy();m=d['mass'].copy();active=d['active'].copy();step=int(d['step']);t=float(d['time'])
    with np.load(a.flow/'output/checkpoint.npz',allow_pickle=False) as d:logJ=d['transport_logJ'][active].copy() if 'transport_logJ' in d else None
    layout=MACLayout(tuple(geometry['shape']),tuple(geometry['origin']),geometry['dx']);density=plan['density_kg_m3'];h=layout.dx
    cm,valid=jax.jit(lambda x,m,active:cell_mass(x,m,active,layout))(jnp.asarray(x),jnp.asarray(m),jnp.asarray(active));cm=np.asarray(cm);assert bool(valid) and np.isfinite(cm).all()
    assert abs(cm.sum()-m[active].sum())<1e-12*m[active].sum();phi=cm/(density*h**3);fluid=geometry['open_cells']&(phi>.25)
    bulk=binary_erosion(fluid,structure=np.ones((3,3,3),dtype=bool),iterations=2)
    def stats(value):return dict(count=int(value.size),mean=float(value.mean()),quantiles=np.quantile(value,[0,.01,.5,.99,1]).tolist()) if value.size else dict(count=0)
    out=dict(step=step,time_s=t,checkpoint_sha256=sha256(a.flow/'output/checkpoint.npz'),physical_particle_volume_m3=float(m[active].sum()/density),total_basis_mass_volume_m3=float(cm.sum()/density),open_cavity_volume_m3=float(geometry['open_cells'].sum()*h**3),pressure_mask_volume_m3=float(fluid.sum()*h**3),pressure_mask_over_particle_volume=float(fluid.sum()*h**3/(m[active].sum()/density)),basis_mass_fraction_outside_open_geometry=float(cm[~geometry['open_cells']].sum()/cm.sum()),pressure_cell_density_over_rho=stats(phi[fluid]),bulk_density_over_rho=stats(phi[bulk]),bulk_definition='two full26-neighbour erosions of pressure-fluid cells; exclude immediate free-surface and fixed-obstacle bands',assessment_time='end particle checkpoint; stored last-step grid metrics refer to pre-advection positions',physical_steps_executed=0,scope='Actual checkpoint JAX basis-mass reconstruction only; no physical time replay and no convergence certificate')
    out.update(reference_volume_over_cavity=float(m[active].sum()/density/out['open_cavity_volume_m3']),occupied_mask_over_cavity=float(fluid.sum()/geometry['open_cells'].sum()),transport_logJ_reference=plan.get('transport_reference_checkpoint_sha256'))
    if logJ is not None:out.update(transport_logJ=stats(logJ),transport_J=stats(np.exp(logJ)),integrated_incremental_transport_volume_m3=float(np.sum(m[active]/density*np.exp(logJ))))
    owned=np.zeros(tuple(geometry['shape']));indices=np.floor((x[active]-geometry['origin'])/h).astype(int);np.add.at(owned,tuple(indices.T),m[active]);out['cell_owned_bulk_density_over_rho']=stats(owned[bulk]/(density*h**3))
    axial=[];centres=geometry['origin'][0]+(np.arange(geometry['shape'][0])+.5)*h;left,right=geometry['receipt']['cylinder_limits_m']
    for lo,hi in zip(np.linspace(left,right,65)[:-1],np.linspace(left,right,65)[1:]):
        mask=(centres>=lo)&(centres<hi);den=int(geometry['open_cells'][mask].sum());axial.append(dict(x_interval_m=[float(lo),float(hi)],void_cells=den,occupied_cells=int(fluid[mask].sum()),occupied_fraction=float(fluid[mask].sum()/den) if den else None))
    out['axial_occupancy_bins']=axial;out['particle_centres_in_solid']=int(geometry['solid'][tuple(indices.T)].sum());out['fill_scope']='Voxel occupied mask and reference material volume reported separately; neither numerical completion nor kernel density alone certifies complete filling.'
    a.output.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
