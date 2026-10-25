"""One saved fixed geometry: original/new QP and one allowed closing probe."""
import argparse,hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
from tools.metal_affine_dynamic_numpy import DynamicFrame,kinetic,moments,unpack,whiten


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--old-contact',type=Path,required=True);p.add_argument('--admission',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    parent=a.raw/'step-17/accepted.npz';saved=a.raw/'step-18/initial-projection.npz'
    assert sha(parent)=='73aaeb1b942742d269baf2ac90311c40ea756c24054c802298bc26773b8d422b'
    with np.load(parent,allow_pickle=False) as z:state=dict(z)
    with np.load(saved,allow_pickle=False) as z:y=z['white'];before=z['white_before']
    with np.load(a.raw/'step-18/rejected.npz',allow_pickle=False) as z:rejected=z['rejected_white']
    energy_tolerance=json.loads(a.admission.read_text())['energy_roundoff_J']
    spec=importlib.util.spec_from_file_location('original_contact_reference',a.old_contact);original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    started=time.monotonic();frame=DynamicFrame(state,state['candidate_fixed_particles'],y,lambda phase:print(phase,flush=True),direct_projection=True)
    geometry_seconds=time.monotonic()-started;contact=frame.restricted;full=contact.full_contact
    initial_closing=contact.sample(y)
    old_y,old_report,old_lambda=original.CompatibleContact.project(contact,y,return_impulse=True)
    old_report.update(original_sample_closing_m_s=float(np.max(full.sample(old_y),initial=0)),restricted_sample_difference_m_s=float(np.max(abs(contact.sample(old_y)-full.sample(old_y)),initial=0)))
    def check(label,incoming):
        following,row,reaction,physical=contact.apply(incoming)
        delta=following-incoming;loss=.5*float(np.sum(delta*delta));work_identity=kinetic(following)-kinetic(incoming)+loss
        p0,L0=moments(state,incoming);p1,L1=moments(state,following);reaction_p,reaction_L=moments(state,reaction)
        v0,_=unpack(state,incoming);v1,_=unpack(state,following)
        momentum_scale=float(np.sum(state['mass']*(np.linalg.norm(v0,axis=1)+np.linalg.norm(v1,axis=1))))+np.linalg.norm(reaction_p)
        angular_scale=float(np.sum(state['mass']*np.linalg.norm(state['x'],axis=1)*(np.linalg.norm(v0,axis=1)+np.linalg.norm(v1,axis=1))))+np.linalg.norm(reaction_L)
        dp=p1-p0-reaction_p;dL=L1-L0-reaction_L
        np.savez_compressed(a.output/(label+'.npz'),white_before=incoming,white=following,physical=physical,reaction=reaction,multiplier=contact.last_multiplier,mobility=contact.mobility,constraint_scale=contact._balanced_system()[0],closing_before=contact.sample(incoming),closing_after=contact.sample(following),full_closing_after=full.sample(following))
        checks=dict(native_KKT=row['valid'],original_full_closing=row['original_sample_closing_m_s']<=.005,original_restricted_difference=row['restricted_sample_difference_m_s']<=.005,
                    original_work_tolerance=abs(work_identity)<=energy_tolerance and abs(row['instantaneous_work_identity_J'])<=energy_tolerance,
                    original_momentum_relative=np.linalg.norm(dp)<=1e-10*max(momentum_scale,1e-30),original_angular_relative=np.linalg.norm(dL)<=1e-10*max(angular_scale,1e-30))
        checks={name:bool(value) for name,value in checks.items()}
        return dict(contact=row,checks=checks,all_original_gates_passed=bool(all(checks.values())),work_identity_J=work_identity,loss_J=loss,momentum_relative=float(np.linalg.norm(dp)/max(momentum_scale,1e-30)),angular_relative=float(np.linalg.norm(dL)/max(angular_scale,1e-30)),physical_torque=moments(state,physical)[1].tolist(),positive_multipliers=int(np.sum(contact.last_multiplier>0)))
    actual=check('saved-initial-contact',y)
    index=int(np.argmax(np.diag(contact.mobility)));direction=contact.adjoint(np.eye(len(contact.pairs))[index]);amplitude=(.1-initial_closing[index])/contact.mobility[index,index]
    probe_y=y+amplitude*direction
    probe=check('allowed-closing-probe',probe_y)
    assert probe['contact']['initial_closing_m_s']>.005 and probe['positive_multipliers']>0 and not probe['contact']['feasible_fast_path']
    report=dict(schema='metal-fixed-contact-balance-v1',parent_checkpoint_sha256=sha(parent),saved_initial_projection_sha256=sha(saved),original_contact_source_sha256=sha(a.old_contact),physical_steps_executed=0,material_updates_executed=0,geometries_reconstructed=1,closing_probes=1,geometry_seconds=geometry_seconds,
                reconstructed_white_before_error=float(np.max(abs(whiten(state)-before))),initial_projection_white_change_norm=float(np.linalg.norm(y-before)),initial_native_closing_m_s=float(np.max(initial_closing,initial=0)),
                old=old_report,old_computed_failure_matches_saved=bool(not old_report['valid'] or old_report['original_sample_closing_m_s']>.005 or old_report['restricted_sample_difference_m_s']>.005),old_vs_saved_rejected_white_max_error=float(np.max(abs(old_y-rejected))),
                new_actual=actual,allowed_closing_probe=probe,original_energy_tolerance_J=energy_tolerance,
                passed=bool(actual['all_original_gates_passed'] and probe['all_original_gates_passed']),scope='Fixed geometry contact projection only; original rejected trajectory not advanced or automatically resumed')
    (a.output/'review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True);frame.release()
    assert report['passed']


if __name__=='__main__':main()
