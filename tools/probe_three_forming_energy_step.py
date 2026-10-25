"""One checkpoint's signed stage ledger; no continuation/checkpoint mutation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def probe(folder):
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from mpm_jax.fibre_mpm import FibreConfig,FibreState,build_layout,stencil,particle_to_grid,grid_to_particle
    from mpm_jax.materials.isotropic_brittle import BrittleHistory,isotropic_brittle,brittle_energy,update_brittle
    from mpm_jax.fibre_forming_boundary import prescribe
    from mpm_jax.fibre_forming_contact import project,mass_gradient
    from examples.three_strand_forming import geometry
    folder=Path(folder);p=json.loads((folder/'plan.json').read_text());g=geometry();cpfile=folder/'checkpoint.npz'
    with np.load(cpfile,allow_pickle=False) as cp:
        leaves={k[6:]:jnp.asarray(cp[k]) for k in cp.files if k.startswith('state_')}
        leaves['history']=BrittleHistory(*(jnp.asarray(cp['history_'+k]) for k in BrittleHistory._fields))
        state=FibreState(**leaves);step=int(cp['step'])
    material=isotropic_brittle(**p['material']);config=FibreConfig(p['dt'])
    layout=build_layout(g['x'],g['ids'],dx=p['dx'],reference_bounds=np.broadcast_to(np.asarray(p['bounds']),(3,3,2)),padding_cells=3,max_nodes=1600000)
    masks=tuple(jnp.asarray(m) for m in g['masks']);nodes=layout.node_count//3
    def kinetic(v,C):return .5*jnp.sum(state.mass*jnp.sum(v*v,axis=1))+p['dx']**2/8*jnp.sum(state.mass*jnp.sum(C*C,axis=(1,2)))
    def grid_energy(m,v):return .5*jnp.sum(m*jnp.sum(v*v,axis=1))
    @jax.jit
    def ledger(s):
        support=stencil(s.x,s.fibre_id,layout)
        mass,momentum0=particle_to_grid(s._replace(tau=jnp.zeros_like(s.tau)),layout,config,support)
        _,momentum=particle_to_grid(s,layout,config,support)
        divide=jnp.where(mass>0,mass,1)[:,None]
        pre=momentum0/divide;force=momentum/divide
        prescribed,clamped,bcvalid=prescribe(force,mass,s,support,layout,masks,config.dt,p['duration'])
        grad=mass_gradient(s,support,layout).reshape(3,nodes,3)
        centres=jnp.stack([jnp.sum(jnp.where((s.fibre_id==i)[:,None],s.mass[:,None]*s.x,0),axis=0)/jnp.sum(jnp.where(s.fibre_id==i,s.mass,0)) for i in range(3)])
        contact=project(mass.reshape(3,nodes),prescribed.reshape(3,nodes,3),grad,centres,clamped.reshape(3,nodes))
        post=contact.velocity.reshape(-1,3)
        fixture=mass[:,None]*(prescribed-force)+contact.fixture_impulse.reshape(-1,3)
        work=jnp.sum(fixture*prescribed)
        lossbc=.5*jnp.sum(mass*jnp.sum((prescribed-force)**2,axis=1))
        losscontact=work-lossbc-(grid_energy(mass,post)-grid_energy(mass,force))
        v,C=grid_to_particle(post,support,layout);F=(jnp.eye(3)+config.dt*C)@s.F
        history,tau=update_brittle(s.history,F,material)
        E0=kinetic(s.v,s.C);E1=grid_energy(mass,pre);E2=grid_energy(mass,force);E3=grid_energy(mass,post);E4=kinetic(v,C)
        U0=jnp.sum(s.volume*brittle_energy(s.F,s.history,material));U1=jnp.sum(s.volume*brittle_energy(F,history,material))
        S=config.dt*jnp.sum(s.volume*jnp.sum(s.tau*C,axis=(1,2)))
        force_work_post=jnp.sum((momentum-momentum0)*post)
        contact_metric=.5*jnp.sum(mass*jnp.sum((post-prescribed)**2,axis=1))
        return jnp.array([E0,E1,E2,E3,E4,U0,U1,work,lossbc,losscontact,contact.closing_residual,S,force_work_post,contact_metric]),bcvalid&contact.valid&jnp.all(history.valid)&jnp.all(history.damage==0)
    values,valid=ledger(state);values=np.asarray(values)
    names=['particle_K_old','grid_K_after_p2g','grid_K_after_force','grid_K_after_fixture_contact','particle_K_new','U_old','U_new','fixture_work','fixture_projection_loss','contact_projection_loss','closing_residual','old_stress_post_velocity_work','grid_force_work_post','contact_mass_metric']
    result={k:float(v) for k,v in zip(names,values)}
    E0,E1,E2,E3,E4,U0,U1,W,Lb,Lc,_,S,Fwork,Lmetric=values
    result.update(material_endpoint_remainder=float(U1-U0-S),force_vs_admissible_velocity=float(E2-E1+S),stress_virtual_work_residual=float(Fwork+S),contact_metric_residual=float(Lc-Lmetric),P2G_loss=float(E0-E1),G2P_loss=float(E3-E4),force_K_increment=float(E2-E1),elastic_increment=float(U1-U0),force_plus_elastic=float(E2-E1+U1-U0),energy_remainder=float(W-Lb-Lc-(E4-E0)-(U1-U0)),identity_residual=float(W-Lb-Lc-(E4-E0)-(U1-U0)-((E0-E1)+(E3-E4)-(E2-E1+U1-U0))))
    return dict(checkpoint_sha256=hashlib.sha256(cpfile.read_bytes()).hexdigest(),checkpoint_step=step,dt=config.dt,valid=bool(valid),one_step_ledger=result,units='All ledger entries J except closing_residual in m/s; tau is Kirchhoff Pa and volume is reference m^3.',scope='One hypothetical next step from frozen checkpoint. No accepted state advanced, no full-window convergence claim.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();r=probe(a.folder);a.output.write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
