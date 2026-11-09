"""Reconstruct final domain energies and compare transient audit histories."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def check(run):
    info=json.loads((run/'summary.json').read_text())
    rows=list(csv.DictReader((run/'history.csv').open()))
    assert info['completed'] and all(row['valid']=='True' for row in rows)
    values={key:np.asarray([float(row[key]) for row in rows]) for key in rows[0] if key!='valid'}
    assert all(np.isfinite(value).all() for value in values.values())
    np.testing.assert_allclose(values['time_s'][-1],info['duration_s'],rtol=1e-12)
    assert int(values['steps'][-1])==info['steps']
    assert values['max_iteration_residual'].max()<=1e-8
    with np.load(run/'final.npz',allow_pickle=False) as data:
        state={key:data[key] for key in data.files}
    assert all(np.isfinite(value).all() for value in state.values())
    F,C,v=state['F'],state['C'],state['v'];J=np.linalg.det(F)
    assert (J>0).all()
    h=info['dx_m']/2;area=h*h;mass=info['density_kg_m3']*area
    moment=.25*info['dx_m']**2*np.eye(2)+h*h/12*(F@F.swapaxes(-1,-2))
    kinetic=.5*mass*np.sum(np.sum(v*v,axis=1)+np.trace(C@moment@C.swapaxes(-1,-2),axis1=-2,axis2=-1))
    E,nu=info['young_Pa'],info['poisson'];mu=E/(2*(1+nu));lam=E*nu/((1+nu)*(1-2*nu))
    logJ=np.log(J)
    elastic=area*np.sum(.5*mu*(np.sum(F*F,axis=(1,2))-2)-mu*logJ+.5*lam*logJ**2)
    np.testing.assert_allclose([kinetic,elastic],[values['kinetic_J_per_m'][-1],values['elastic_J_per_m'][-1]],atol=1e-12,rtol=1e-10)
    ledger=state['ledger']
    balance=kinetic+elastic+ledger[3]+ledger[4]-ledger[5]-ledger[:3].sum()
    np.testing.assert_allclose(balance,values['balance_error_J_per_m'][-1],atol=1e-12)
    assert np.max(np.abs(values['balance_error_J_per_m']))<1e-9
    assert values['transfer_loss_J_per_m'].min()>-1e-10
    assert values['absolute_geometry_energy_J_per_m'][-1]>=abs(ledger[5])-1e-14
    report=dict(grid=info['grid'],dt_s=info['dt_s'],domains=info['domains'],steps=info['steps'],
                final_numerical_loss_fraction=float(values['numerical_loss_fraction'][-1]),
                final_absolute_geometry_fraction=float(values['absolute_geometry_fraction'][-1]),
                max_balance_error_J_per_m=float(np.abs(values['balance_error_J_per_m']).max()),
                final_snapshot_sha256=hashlib.sha256((run/'final.npz').read_bytes()).hexdigest(),
                history_sha256=hashlib.sha256((run/'history.csv').read_bytes().replace(b'\r\n',b'\n')).hexdigest())
    return report,values


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();reports={};histories={}
    for run in args.runs:
        reports[run.name],histories[run.name]=check(run)
    comparisons=[]
    for a,b in zip(args.runs[:-1],args.runs[1:]):
        av,bv=histories[a.name],histories[b.name]
        np.testing.assert_allclose(av['time_s'],bv['time_s'],atol=1e-15)
        keys=('kinetic_J_per_m','elastic_J_per_m','external_J_per_m','mean_right_displacement_m')
        errors={key:float(np.linalg.norm(av[key]-bv[key])/np.linalg.norm(bv[key])) for key in keys}
        comparisons.append(dict(coarser=a.name,reference=b.name,history_relative_L2=errors))
    args.output.mkdir(parents=True,exist_ok=True)
    result=dict(runs=reports,adjacent_comparisons=comparisons,
                validation_scope='Final-field reconstruction and supplied work histories; not full per-step replay',
                mesh_convergence_established=False,fracture_validation=False)
    (args.output/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
    for name,values in histories.items():
        info=reports[name];label=f"N={info['grid']}, dt={info['dt_s']*1e6:.3g} us"
        t=values['time_s']*1e3
        axes[0].plot(t,values['elastic_J_per_m'],label=label)
        axes[1].plot(t,100*values['numerical_loss_fraction'],label=label)
        axes[2].plot(t,100*values['absolute_geometry_fraction'],label=label)
    for ax,title in zip(axes,('Elastic energy (J/m)','Transfer + boundary loss (%)','Absolute changing-moment term (%)')):
        ax.set(xlabel='Time (ms)',ylabel=title);ax.grid(alpha=.25)
    axes[0].legend(fontsize=7)
    fig.savefig(args.output/'domain-dynamics.png',dpi=150)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
