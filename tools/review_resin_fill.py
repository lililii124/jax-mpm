"""Read-only full-array acceptance of an isolated actual resin fill segment."""
import argparse,hashlib,json,zipfile,re
from pathlib import Path
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def audit_lineage(task,root):
    """Check the exact accepted CP ancestry, including accepted failed-job prefixes."""
    nodes=[];rows={};current=task;seen=set();expected=None
    while True:
        assert current.parent.resolve()==root.resolve() and re.fullmatch('three-resin-[a-z0-9-]+',current.name)
        assert current.name not in seen;seen.add(current.name);folder=current/'output';cp=folder/'checkpoint.npz';cp_sha=digest(cp)
        if expected is not None:assert cp_sha==expected
        with zipfile.ZipFile(cp) as z:assert z.testzip() is None
        with np.load(cp,allow_pickle=False) as d:
            step=int(d['step']);assert bool(d['valid'])
            for name in d.files:
                a=d[name]
                if a.dtype.kind in 'fci':assert np.isfinite(a).all()
        source=json.loads((current/'source.json').read_text())
        for name,sha in source['files'].items():assert digest(current/name)==sha
        journal=json.loads((folder/'observables.json').read_text());accepted=[r for r in journal if r['step']<=step]
        assert accepted and accepted[-1]['step']==step
        for row in accepted:
            assert row['step'] not in rows and row['valid'] and row['pressure_coverage_valid'] and row['solid_cell_intrusions']==row['escaped_particles']==0
            assert row['pressure_relative_residual']<=1e-7 and row['pressure_stationarity_relative']<=1e-10
            rows[row['step']]=row
        nodes.append(dict(task=current.name,checkpoint_step=step,checkpoint_sha256=cp_sha,source=source['source'],accepted_rows=len(accepted),partial_failed_job_prefix=not (folder/'review.json').exists()))
        parent=source.get('parent_task')
        if not parent:break
        expected=source['parent_checkpoint_sha256'];current=root/parent
    assert sorted(rows)==list(range(1,max(rows)+1));first=current/'output/initial-checkpoint.npz'
    with np.load(first,allow_pickle=False) as d:
        assert int(d['step'])==0 and not d['active'].any() and not d['emitted'].any() and not np.any(d['transport_logJ'])
    return dict(passed=True,initial_checkpoint_sha256=digest(first),steps_verified=len(rows),nodes=nodes,maximum_pressure_relative_residual=max(r['pressure_relative_residual'] for r in rows.values()),maximum_pressure_stationarity=max(r['pressure_stationarity_relative'] for r in rows.values()),maximum_step_energy_identity_J=max(abs(r['energy_identity_J']) for r in rows.values()),maximum_pressure_work_identity_J=max(abs(r['pressure_identity_J']) for r in rows.values()),maximum_momentum_residual_Ns=max(r['momentum_step_residual_Ns'] for r in rows.values()),maximum_absolute_cumulative_logJ=max(max(abs(r['transport_logJ_reference_active_min']),abs(r['transport_logJ_reference_active_max'])) for r in rows.values()),scope='No physical replay; exact CP parent hashes/source files/CRC/finite accepted ancestry and contiguous step rows. Stored material logJ is carried from injection birth0; legacy plan label correction is explicit in the bridge.')


def main():
    p=argparse.ArgumentParser();p.add_argument('task',type=Path);p.add_argument('--output',type=Path,default=Path('resin-state-review.json'));p.add_argument('--lineage-root',type=Path);a=p.parse_args();folder=a.task/'output';source=json.loads((a.task/'source.json').read_text());plan=json.loads((folder/'plan.json').read_text());review=json.loads((folder/'review.json').read_text());rows=json.loads((folder/'observables.json').read_text())
    for name,sha in source['files'].items():assert digest(a.task/name)==sha
    arrays=0;files=0
    for path in folder.glob('*.npz'):
        with zipfile.ZipFile(path) as z:assert z.testzip() is None
        with np.load(path,allow_pickle=False) as d:
            for name in d.files:
                value=d[name];arrays+=1
                if value.dtype.kind in 'fci':assert np.isfinite(value).all()
        files+=1
    with np.load(folder/'checkpoint.npz',allow_pickle=False) as d:
        active=d['active'];mass=d['mass'];assert bool(d['valid']);assert np.all(active<=d['emitted']);m=float(mass[active].sum());source_mass=float(d['source_mass_kg']);outmass=float(d['outflow_mass_kg']);K=float(.5*np.sum(mass[active]*(np.sum(d['v'][active]**2,axis=1)+plan['forming']['grid_dx_m']**2/4*np.sum(d['C'][active]**2,axis=(1,2)))))
        assert abs(m-source_mass+outmass)<1e-12*source_mass
        assert int(d['step'])==rows[-1]['step'] and abs(float(d['time'])-rows[-1]['time_s'])<1e-15
        assert abs(K-rows[-1]['active_kinetic_J'])<1e-12*max(K,1e-30)
        cp_sha=digest(folder/'checkpoint.npz');last_step=int(d['step'])
    assert all(r['valid'] and r['pressure_coverage_valid'] and r['solid_cell_intrusions']==0 and r['escaped_particles']==0 for r in rows)
    result=dict(passed=True,source=source['source'],source_files_verified=len(source['files']),npz_files_crc_and_finite=files,arrays_finite_checked=arrays,last_step=last_step,time_s=rows[-1]['time_s'],active_particles=rows[-1]['active_particles'],active_mass_kg=m,source_mass_kg=source_mass,outflow_mass_kg=outmass,mass_identity_kg=m-source_mass+outmass,active_kinetic_J=K,maximum_pressure_relative_residual=max(r['pressure_relative_residual'] for r in rows),maximum_divergence_per_s=max(r['maximum_divergence_per_s'] for r in rows),maximum_step_energy_identity_J=max(abs(r['energy_identity_J']) for r in rows),global_energy_identity_J=rows[-1]['global_energy_identity_J'],pressure_cell_volume_m3=rows[-1]['pressure_volume_m3'],actual_particle_volume_m3=rows[-1]['particle_volume_m3'],pressure_volume_over_particle_volume=rows[-1]['pressure_volume_m3']/rows[-1]['particle_volume_m3'],checkpoint_sha256=cp_sha,forming_checkpoint_sha256=plan['forming']['checkpoint_sha256'],boundary_clearance_m=plan['forming']['conservative_radial_clearance_m'],scope='Actual short inlet flow only; no completed filling, no smooth surface/time/space convergence, pressure mask volume differs from physical particle volume',production_eligible=False)
    if a.lineage_root is not None:result['lineage']=audit_lineage(a.task,a.lineage_root)
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
