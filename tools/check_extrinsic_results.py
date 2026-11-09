"""Reconstruct stored-field energies for stress-triggered cohesive insertion."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def check(directory):
    info = json.loads((directory/'summary.json').read_text())
    rows = list(csv.DictReader((directory/'history.csv').open()))
    assert info['all_states_valid'] and len(rows)==info['frames'] and all(r['valid']=='True' for r in rows)
    columns = {k:np.asarray([float(r[k]) for r in rows]) for k in rows[0] if k!='valid'}
    assert all(np.isfinite(v).all() for v in columns.values())
    E,nu,rho = info['young_Pa'],info['poisson'],info['density_kg_m3']
    mu,lam = E/(2*(1+nu)),E*nu/((1+nu)*(1-2*nu))
    K,Gc = info['law']['stiffness'],info['law']['energy']
    maximum, previous_bonded = None,None
    first_failure = None
    minimum_surface = np.inf
    peak_contact_pressure = 0.
    peak_compression_energy = 0.
    for i,row in enumerate(rows):
        with np.load(directory/f'frame-{i:03d}.npz',allow_pickle=False) as data:
            a = {k:data[k] for k in data.files}
        assert all(np.isfinite(v).all() for v in a.values())
        if 'ledger' in a:
            assert a['ledger'].shape==(9,)
            np.testing.assert_allclose(a['ledger'][[2,3,8]],
                                      [float(row[k]) for k in ('transfer_loss_J_per_m',
                                       'boundary_projection_loss_J_per_m','insertion_energy_J_per_m')],atol=1e-12)
            np.testing.assert_allclose(a['ledger'][0]+a['ledger'][1],float(row['external_J_per_m']),atol=1e-12)
            assert int(a['control_accepted'])==int(row['step'])
            assert int(a['control_rejected'])==int(row['rejected_attempts'])
            assert int(a['control_stop_code'])==0
        plus,minus = a['plus'],a['minus']
        assert len(plus)==int(row['vertices'])
        tangent = .5*np.diff(plus+minus,axis=0)
        tangent /= np.linalg.norm(tangent,axis=-1,keepdims=True)
        normal = np.column_stack((-tangent[:,1],tangent[:,0]))
        fraction = np.array([.5-.5/np.sqrt(3),.5+.5/np.sqrt(3)])
        jump = (1-fraction)[None,:,None]*(plus[:-1]-minus[:-1])[:,None,:]+fraction[None,:,None]*(plus[1:]-minus[1:])[:,None,:]
        opening = np.stack((np.sum(jump*tangent[:,None,:],axis=-1),np.sum(jump*normal[:,None,:],axis=-1)),axis=-1)
        bonded,area,birth = a['bonded'],a['area'],a['birth_traction']
        strength = np.where(bonded,np.linalg.norm(birth,axis=-1),1.)[:,None]
        r0,width = strength/K,2*Gc/strength
        radius = np.linalg.norm(opening+birth[:,None,:]/K,axis=-1)
        history = np.maximum(a['maximum'],r0)
        fraction_failed = np.clip((history-r0)/width,0,1)
        remaining = r0/history*(1-fraction_failed)
        stored = np.where(bonded[:,None],.5*K*remaining*radius**2-.5*strength*r0*(1-fraction_failed),0.)
        compression = .5*K*np.minimum(opening[:,:,1],0.)**2
        stored += compression
        peak_contact_pressure=max(peak_contact_pressure,float(-K*np.minimum(opening[:,:,1],0.).min()))
        peak_compression_energy=max(peak_compression_energy,float(np.sum(area*compression.mean(axis=1))))
        dissipation = Gc*np.where(bonded[:,None],fraction_failed,0.)
        surface = np.sum(area*stored.mean(axis=1))
        fracture = np.sum(area*dissipation.mean(axis=1))
        minimum_surface = min(minimum_surface,float(surface))
        np.testing.assert_allclose(a['softening'],np.where(bonded[:,None],fraction_failed,0.),atol=2e-10)
        complete = (fraction_failed>=1-1e-8).all(axis=1)
        connected = np.cumprod(~bonded|complete)
        failed_length = np.sum(area*bonded*connected)
        if first_failure is None and failed_length>0:
            first_failure = float(row['time_s'])
        if maximum is not None:
            old_count = len(maximum)
            assert np.all(a['maximum'][:old_count][previous_bonded]>=maximum[previous_bonded]-1e-12)
        maximum,previous_bonded = a['maximum'],bonded
        F,v,C = a['F'],a['v'],a['C']
        J = np.linalg.det(F)
        assert J.min()>0
        extent = np.asarray(info.get('upper_m',[.05,.04]))-np.asarray(info.get('lower_m',[.01,.02]))
        volume = np.prod(extent)/len(F)
        bulk = volume*np.sum(.5*mu*(np.sum(F*F,axis=(1,2))-2)-mu*np.log(J)+.5*lam*np.log(J)**2)
        kinetic = .5*rho*volume*np.sum(np.sum(v*v,axis=1)+.25*info['dx_m']**2*np.sum(C*C,axis=(1,2)))
        stress = (mu*(F@F.swapaxes(-1,-2)-np.eye(2))+lam*np.log(J)[:,None,None]*np.eye(2))/J[:,None,None]
        np.testing.assert_allclose(np.linalg.eigvalsh(stress)[:,-1],a['principal_stress'],atol=1e-4,rtol=1e-9)
        np.testing.assert_allclose([kinetic,bulk,surface,fracture,failed_length],
                                   [float(row[k]) for k in ('kinetic_J_per_m','elastic_J_per_m','surface_J_per_m','fracture_J_per_m','connected_failed_m')],atol=2e-8,rtol=2e-9)
        assert fracture <= Gc*float(row['potential_extension_m'])+1e-9
    energy = sum(columns[k] for k in ('kinetic_J_per_m','elastic_J_per_m','surface_J_per_m','fracture_J_per_m',
                                     'transfer_loss_J_per_m','boundary_projection_loss_J_per_m'))
    error = columns['external_J_per_m']+columns['insertion_energy_J_per_m']-energy
    scale = max(float(np.abs(columns['external_J_per_m']).max()),1.)
    assert np.abs(error).max()<1e-7*scale
    assert np.abs(columns['insertion_energy_J_per_m']).max()<1e-9
    assert columns['max_iteration_residual'].max()<=1e-8
    assert np.diff(columns['fracture_J_per_m']).min()>-1e-9
    assert columns['transfer_loss_J_per_m'].min()>-1e-9
    assert columns['boundary_projection_loss_J_per_m'].min()>-1e-9
    result = dict(grid=info['grid'],fracture_energy_J_m2=Gc,frames=len(rows),
                  nominal_dt_s=info['dt_s'],face_mass_fraction=info.get('face_mass_fraction',0.),
                  accepted_steps=int(columns['step'][-1]),
                  peak_sampled_penalty_pressure_Pa=peak_contact_pressure,
                  peak_sampled_compression_energy_J_per_m=peak_compression_energy,
                  final_potential_extension_m=float(columns['potential_extension_m'][-1]),
                  final_connected_failed_m=float(columns['connected_failed_m'][-1]),
                  max_detached_failed_m=float(np.abs(columns['detached_failed_m']).max()),
                  first_saved_failure_s=first_failure,final_fracture_J_per_m=float(columns['fracture_J_per_m'][-1]),
                  final_numerical_fraction=float(columns['numerical_fraction'][-1]),
                  max_unexplained_J_per_m=float(np.abs(error).max()),
                  max_insertion_energy_J_per_m=float(np.abs(columns['insertion_energy_J_per_m']).max()),
                  minimum_surface_J_per_m=minimum_surface,min_face_gap_m=float(columns['min_face_gap_m'].min()),
                  max_sampled_strain=float(columns['max_strain'].max()),
                  extension_limit_reached=bool(columns['potential_extension_m'][-1]>=info['max_extension_m']-1e-10),
                  mesh_convergence_established=False,nodal_traction_balance_at_insertion_verified=False,
                  history_sha256=hashlib.sha256((directory/'history.csv').read_bytes().replace(b'\r\n',b'\n')).hexdigest())
    return result,columns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result,history = {},{}
    for directory in args.runs:
        result[directory.name],history[directory.name] = check(directory)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for name,columns in history.items():
        info=result[name]
        label=f"N={info['grid']}, dt={info['nominal_dt_s']*1e9:.3g} ns, cutoff={info['face_mass_fraction']:g}"
        t=columns['time_s']*1e6
        axes[0,0].plot(t,columns['potential_extension_m']*1e3,label=label)
        axes[0,1].plot(t,columns['connected_failed_m']*1e3,label=label)
        axes[1,0].plot(t,columns['fracture_J_per_m'],label=label)
        axes[1,1].plot(t,columns['numerical_fraction']*100,label=label)
    for ax,title in zip(axes.flat,['Inserted cohesive extension (mm)','Connected fully failed length (mm)','Fracture dissipation (J/m)','Numerical loss / work (%)']):
        ax.set(xlabel='Time (microseconds)',ylabel=title);ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.savefig(args.output/'extrinsic-history.png',dpi=140)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
