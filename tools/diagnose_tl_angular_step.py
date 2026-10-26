"""Read-only stage torque audit of one C2 TL midpoint candidate from a CP."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vec(value):
    return np.asarray(value, dtype=np.float64)


def skew_stats(tensor):
    sympair = tensor - np.swapaxes(tensor, -1, -2)
    norm = np.linalg.norm(sympair, axis=(-2, -1))
    scale = np.maximum(np.linalg.norm(tensor, axis=(-2, -1)), 1e-300)
    return dict(max_abs=float(np.max(norm)), max_relative=float(np.max(norm/scale)),
                rms_abs=float(np.sqrt(np.mean(norm*norm))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if sha(args.checkpoint) != args.expected_sha256:
        raise ValueError('CP SHA does not match the approved diagnostic parent')

    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from examples.metal_sphere_impact_fracture import plan_case
    from mpm_jax.materials.linear_metal import linear_metal, eos_state
    from mpm_jax.materials.shock_metal import artificial_viscosity
    from mpm_jax.metal_tl_midpoint import initial_state, step, nodal_moments
    from mpm_jax.material_domain_contact import material_domain_contact_jump
    from mpm_jax.total_lagrangian import deformation, sample_nodal
    from run_metal_tl_fresh import make_reference

    geometry, plan = plan_case(speed=5000., sphere_diameter=.002,
        plate_size=(.02, .02), plate_thickness=.0005, gap=.00075,
        clamp_width=.001, dx=.00025, spacing=(.00025, .000125, .00025),
        end_time=1e-6, frames=60)
    reference, body_id, fixed, v0, record = make_reference(geometry, .00025)
    particle_mass = np.asarray(reference.particle_mass)
    material = linear_metal(**plan['material'])
    with np.load(args.checkpoint, allow_pickle=False) as cp:
        step0, time0 = int(cp['step']), float(cp['time_s'])
        fields = tuple(jnp.asarray(cp[n]) for n in
            ('F', 'tau', 'ep', 'cold', 'fracture', 'damage', 'internal'))
        state = initial_state(reference, fields, jnp.asarray(cp['velocity']))
        state = state._replace(displacement=jnp.asarray(cp['displacement']),
            time=jnp.asarray(time0), initial_momentum=jnp.asarray(cp['initial_P']),
            initial_angular_momentum=jnp.asarray(cp['initial_L']),
            initial_kinetic=jnp.asarray(cp['initial_K']),
            viscous_loss=jnp.asarray(cp['viscous_loss']),
            contact_work=jnp.asarray(cp['contact_work']))
        viscous_before, contact_work_before = float(cp['viscous_loss']), float(cp['contact_work'])

    F0, tau0, ep0, cold0, fracture0, damage0, internal0 = state.material
    J0 = jnp.linalg.det(F0)
    elastic0 = jnp.sum(tau0*tau0, axis=(-2, -1))/(4*material.shear_modulus*
        material.density*jnp.maximum(1-damage0, 1e-12))
    thermo0 = eos_state(J0, internal0-cold0-fracture0-elastic0, material)
    W0 = jnp.einsum('pki,pkj->pij', state.velocity[reference.indices], reference.gradients)
    L0 = W0 @ jnp.linalg.inv(F0)
    C0 = jnp.sqrt(jnp.sum(L0*L0, axis=(-2, -1)))
    vpart0 = jnp.linalg.norm(jnp.sum(reference.weights[..., None]*
        state.velocity[reference.indices], axis=1), axis=-1)
    compression0 = jnp.maximum(-jnp.trace(L0, axis1=-2, axis2=-1), 0.)
    visc_signal0 = (material.viscosity_linear*thermo0.sound_speed +
        2*material.viscosity_quadratic*.00025*compression0)
    signal0 = thermo0.longitudinal_speed + vpart0 + 1.5*np.sqrt(3)*.00025*C0 + visc_signal0
    dt = .1*.00025/float(jnp.max(signal0))

    def contact(vpre, x1, F1):
        return material_domain_contact_jump(vpre, reference, particle_mass,
            body_id, x1, F1, np.asarray(plan['particle_spacing_m']), fixed)

    candidate = step(state, reference, material, dt, jnp.asarray(fixed), contact,
                     cell_length=.00025)
    if not bool(np.asarray(candidate.valid)):
        raise FloatingPointError('One-step solver candidate failed its own midpoint/material/contact admission')

    u1 = vec(candidate.state.displacement)
    vpost = vec(candidate.state.velocity)
    mnode = vec(reference.nodal_mass)
    ic = vec(candidate.contact_impulse)
    vpre = vpost - np.divide(ic, mnode[:, None], out=np.zeros_like(ic),
                              where=mnode[:, None] > 0)
    q0 = vec(reference.nodes) + vec(state.displacement)
    q1 = vec(reference.nodes) + u1
    qmid = .5*(q0+q1)
    L_initial = vec(nodal_moments(state.displacement, state.velocity, reference)[1])
    L_pre = vec(nodal_moments(candidate.state.displacement, jnp.asarray(vpre), reference)[1])
    L_post = vec(candidate.geometry['angular_momentum'])
    deltaL_force = L_pre-L_initial
    deltaL_contact = L_post-L_pre
    Pmat = vec(candidate.material_result.piola)
    Fmaterial = vec(candidate.material_result.state[0])
    Fkin = vec(candidate.geometry['F'])
    Fmid = .5*(vec(F0)+Fmaterial)
    Wmid = vec(candidate.geometry['midpoint_velocity'])[np.asarray(reference.indices)]
    Wmid = np.einsum('pki,pkj->pij', Wmid, np.asarray(reference.gradients))
    J1 = np.linalg.det(Fkin)
    thermo = eos_state(jnp.asarray(J0), internal0-cold0-fracture0-elastic0, material)
    div1 = np.trace(Wmid @ np.linalg.inv(Fkin), axis1=-2, axis2=-1)
    density1 = material.density/np.maximum(J1, 1e-30)
    q1p = np.asarray(artificial_viscosity(density1, np.asarray(thermo.sound_speed),
        div1, .00025, material))
    Pvisc_current = -(J1*q1p)[:, None, None]*np.linalg.inv(Fkin).transpose(0, 2, 1)
    Jmid = np.linalg.det(Fmid)
    Pvisc_mid_same_q = -(Jmid*q1p)[:, None, None]*np.linalg.inv(Fmid).transpose(0, 2, 1)
    divmid = np.trace(Wmid @ np.linalg.inv(Fmid), axis1=-2, axis2=-1)
    densitymid = material.density/np.maximum(Jmid, 1e-30)
    qmidp = np.asarray(artificial_viscosity(densitymid, np.asarray(thermo.sound_speed),
        divmid, .00025, material))
    Pvisc_mid_qmid = -(Jmid*qmidp)[:, None, None]*np.linalg.inv(Fmid).transpose(0, 2, 1)

    def nodal_force(P):
        particle_force = -np.asarray(reference.volume)[:, None, None]*np.einsum(
            'pij,pkj->pki', P, np.asarray(reference.gradients))
        return np.zeros_like(q0).at[np.asarray(reference.indices)].add(particle_force) if hasattr(q0, 'at') else scatter(particle_force)

    def scatter(particle_force):
        result = np.zeros_like(q0)
        np.add.at(result, np.asarray(reference.indices).reshape(-1),
                  particle_force.reshape(-1, 3))
        return result

    def force_torque(P):
        return dt*np.sum(np.cross(qmid, nodal_force(P)), axis=0)

    I_fixed = vec(candidate.fixed_reaction)
    tau_fixed = np.sum(np.cross(qmid, I_fixed), axis=0)
    tau_mat = force_torque(Pmat)
    tau_visc_current = force_torque(Pvisc_current)
    tau_visc_mid = force_torque(Pvisc_mid_same_q)
    tau_force_current = tau_mat+tau_visc_current+tau_fixed
    tau_force_mid = tau_mat+tau_visc_mid+tau_fixed

    report = candidate.contact_report
    points = np.asarray(report.get('contact_points_world_m', []), float)
    normals = np.asarray(report.get('contact_normals', []), float)
    multipliers = np.asarray(report.get('contact_multipliers_Ns', []), float)
    expected_body_torque = [np.zeros(3), np.zeros(3)]
    if len(points):
        for point, normal, lam in zip(points, normals, multipliers):
            expected_body_torque[0] -= lam*np.cross(point, normal)
            expected_body_torque[1] += lam*np.cross(point, normal)
    node_counts = [item['nodes'] for item in record['bodies']]
    split = node_counts[0]
    node_torque = [np.sum(np.cross(q1[:split], ic[:split]), axis=0),
                   np.sum(np.cross(q1[split:], ic[split:]), axis=0)]
    xi = np.asarray(report.get('contact_local_coordinates_m', []), float)
    pairs = np.asarray(report.get('contact_particle_pairs', []), int)
    xparticles = vec(candidate.geometry['position'])
    recon = []
    for k, (a, b) in enumerate(pairs):
        ca = xparticles[a]+Fkin[a]@xi[k, 0]
        cb = xparticles[b]+Fkin[b]@xi[k, 1]
        recon.append((ca, cb, points[k]))
    cp_reconstruction_error = max((max(np.linalg.norm(a-c), np.linalg.norm(b-c))
                                  for a,b,c in recon), default=0.)
    Fpred = vec(F0)+dt*Wmid
    Fscale = max(float(np.max(np.abs(Fmaterial))), 1.)
    Fkin_error = float(np.max(np.abs(Fmaterial-Fkin)))
    Fupdate_error = float(np.max(np.abs(Fmaterial-Fpred)))
    Fbound = 64*np.finfo(float).eps*Fscale
    lscale = max(float(np.sum(particle_mass))*.00025*5000., 1e-30)
    result = dict(schema='c2-tl-one-step-angular-stage-audit-v1',
        checkpoint_sha256=args.expected_sha256, step_before=step0, time_before_s=time0,
        diagnostic_dt_s=dt, candidate_step=step0+1, candidate_time_s=time0+dt,
        candidate_valid=bool(np.asarray(candidate.valid)),
        angular_guard_limit_relative=1e-10, angular_scale_kg_m2_s=lscale,
        deltaL_force_Nms=deltaL_force.tolist(), deltaL_force_relative=float(np.linalg.norm(deltaL_force)/lscale),
        dt_sum_qmid_cross_f_material_Nms=tau_mat.tolist(),
        dt_sum_qmid_cross_f_viscous_current_Nms=tau_visc_current.tolist(),
        dt_sum_qmid_cross_f_viscous_mid_same_q_Nms=tau_visc_mid.tolist(),
        fixed_reaction_impulse_torque_qmid_Nms=tau_fixed.tolist(),
        force_balance_angular_residual_current_Nms=(deltaL_force-tau_force_current).tolist(),
        force_balance_angular_residual_mid_viscosity_Nms=(deltaL_force-tau_force_mid).tolist(),
        material_P_FmidT_skew=skew_stats(Pmat@Fmid.transpose(0,2,1)),
        viscous_current_P_FmidT_skew=skew_stats(Pvisc_current@Fmid.transpose(0,2,1)),
        viscous_mid_P_FmidT_skew_same_q=skew_stats(Pvisc_mid_same_q@Fmid.transpose(0,2,1)),
        viscous_mid_P_FmidT_skew_qmid=skew_stats(Pvisc_mid_qmid@Fmid.transpose(0,2,1)),
        viscous_pressure_current_q_minmax_Pa=[float(q1p.min()),float(q1p.max())],
        viscous_pressure_mid_q_minmax_Pa=[float(qmidp.min()),float(qmidp.max())],
        F_material_vs_geometry_max_abs=Fkin_error,F_material_vs_F0_dtW_max_abs=Fupdate_error,
        F_64eps_bound=Fbound,F_material_geometry_within_64eps=bool(Fkin_error<=Fbound),
        F_material_increment_within_64eps=bool(Fupdate_error<=Fbound),
        contact_deltaL_q1_Nms=deltaL_contact.tolist(),
        contact_impulse_torque_q1_Nms=np.sum(np.cross(q1,ic),axis=0).tolist(),
        contact_impulse_torque_by_body_q1_Nms=[v.tolist() for v in node_torque],
        contact_common_point_torque_by_body_Nms=[v.tolist() for v in expected_body_torque],
        contact_deltaL_vs_q1_impulse_torque_Nms=(deltaL_contact-np.sum(np.cross(q1,ic),axis=0)).tolist(),
        contact_node_vs_commonpoint_torque_by_body_Nms=[(node_torque[k]-expected_body_torque[k]).tolist() for k in (0,1)],
        contact_common_point_reconstruction_max_abs_m=float(cp_reconstruction_error),
        contact_pairs=int(report.get('pairs',0)),
        contact_route_preclosing_m_s=float(report.get('physical_route_pre_closing_m_s',0.)),
        contact_route_postclosing_m_s=float(report.get('physical_route_post_closing_m_s',0.)),
        contact_pair_point_count=len(points),
        contact_points_world_m=points.tolist(),contact_normals=normals.tolist(),
        contact_multipliers_Ns=multipliers.tolist(),contact_local_coordinates_m=xi.tolist(),
        checkpoint_cumulative_viscous_loss_J=viscous_before,
        checkpoint_cumulative_contact_work_J=contact_work_before,
        interpretation='One unaccepted diagnostic replay from the exact accepted checkpoint. It writes no checkpoint and makes no ledger update.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in
        ('contact_points_world_m','contact_normals','contact_multipliers_Ns','contact_local_coordinates_m')},indent=2))


if __name__ == '__main__':
    main()
