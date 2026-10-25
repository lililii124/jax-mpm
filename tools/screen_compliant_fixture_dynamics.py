"""Low-order mass/beam/contact screen; never substitute for actual MPM frames."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.scan_fibre_pair_fixture import geometry_stiffness, jkr_branch, state
from tools.review_jkr_compliant_pilot import projected_pixels


def drive(t, times, values):
    displacement = speed = 0.
    for left, right, a, b in zip(times[:-1], times[1:], values[:-1], values[1:]):
        q = np.clip((t-left)/(right-left), 0., 1.)
        displacement += .5*(b-a)*(1-np.cos(np.pi*q))
        if left < t < right:
            speed += .5*(b-a)*np.pi*np.sin(np.pi*q)/(right-left)
    return displacement, speed


def solve(model, *, damping=.015, approach=.006, press=-.00015, step=1e-5,
          pull=.010, hold=.025, fixture=.7056029424546741):
    mass, kfixture, gap, radius = .00010112, fixture, 1e-5, .001
    young, W = 4e5, .08
    beam = geometry_stiffness(young)
    jkr, critical = jkr_branch(young, W, beam)
    start_pull = approach+.004
    start_hold = start_pull+pull
    end_hold = start_hold+hold
    times = [0., approach, start_pull, start_hold, end_hold, end_hold+.005, end_hold+.035]
    values = [0., press, press, .00045, .00045, .0012, .0012]
    duration = times[-1]
    count = int(np.ceil(duration/step))
    dt = duration/count
    displacement = velocity = fixture_work = contact_work = dissipation = 0.
    active = False
    maximum_radius = 0.
    records = []
    def contact(y, attached):
        if model == 'hertz':
            return state('hertz', young, W, beam, gap+y)
        return jkr(gap+y, attached)
    def rhs(t, y, v, attached):
        target, speed = drive(t, times, values)
        c = contact(y, attached)
        fixture = kfixture*(target-y)+damping*(speed-v)
        return v, (fixture+c['force_N'])/mass
    for i in range(count):
        t = i*dt
        old_y, old_v = displacement, velocity
        old_contact = contact(old_y, active)
        a = rhs(t, old_y, old_v, active)
        b = rhs(t+dt/2, old_y+dt*a[0]/2, old_v+dt*a[1]/2, active)
        c = rhs(t+dt/2, old_y+dt*b[0]/2, old_v+dt*b[1]/2, active)
        d = rhs(t+dt, old_y+dt*c[0], old_v+dt*c[1], active)
        displacement += dt*(a[0]+2*b[0]+2*c[0]+d[0])/6
        velocity += dt*(a[1]+2*b[1]+2*c[1]+d[1])/6
        new_contact = contact(displacement, active)
        active = new_contact['active']
        maximum_radius = max(maximum_radius, new_contact['radius_m'])
        target, speed = drive(t+dt/2, times, values)
        midpoint_y, midpoint_v = (old_y+displacement)/2, (old_v+velocity)/2
        fixture_force = kfixture*(target-midpoint_y)+damping*(speed-midpoint_v)
        fixture_work += fixture_force*speed*dt
        dissipation += damping*(speed-midpoint_v)**2*dt
        contact_work += .5*(old_contact['force_N']+new_contact['force_N'])*(displacement-old_y)
        if i % max(1, count//800) == 0 or i == count-1:
            target, _ = drive(t+dt, times, values)
            spring = .5*kfixture*(target-displacement)**2
            kinetic = .5*mass*velocity**2
            records.append(dict(time_s=t+dt, collar_displacement_m=displacement,
                velocity_m_per_s=velocity, gap_m=-new_contact['indentation_m'],
                contact_force_N=new_contact['force_N'], active=active,
                radius_over_R=new_contact['radius_m']/radius,
                actuator_work_J=fixture_work, damper_dissipation_J=dissipation,
                body_contact_work_J=contact_work, spring_energy_J=spring, kinetic_energy_J=kinetic,
                work_identity_residual_J=fixture_work+contact_work-dissipation-spring-kinetic))
    hold_records = [r for r in records if times[3]+.01 <= r['time_s'] <= times[4]]
    work_scale = max(abs(r['actuator_work_J'])+abs(r['body_contact_work_J']) for r in records)
    return dict(model=model, damping_Ns_per_m=damping, approach_s=approach, press_m=press,
        fixture_stiffness_N_per_m=kfixture,
        pull_s=pull, hold_s=hold, protocol_times_s=times,
        dt_s=dt, maximum_radius_over_R=maximum_radius/radius,
        hold_gap_median_m=float(np.median([r['gap_m'] for r in hold_records])),
        hold_gap_last_m=hold_records[-1]['gap_m'], hold_active_samples=sum(r['active'] for r in hold_records),
        hold_samples=len(hold_records), final_gap_m=records[-1]['gap_m'], final_active=records[-1]['active'],
        maximum_work_identity_relative_residual=max(abs(r['work_identity_residual_J']) for r in records)/work_scale,
        points=records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--slower-pull', action='store_true')
    parser.add_argument('--soft-fixture', action='store_true')
    args = parser.parse_args()
    candidates = []
    camera = dict(direction=[1.4,.2,1.8], resolution=400,
        focus_bounds_m=[-.003,.003,-.0018,.0042,-.003,.003])
    settings = ((.0015,.006,-.00015,.010),(.015,.006,-.00015,.010),
                (.015,.010,-.00015,.010),(.015,.010,-.00008,.010))
    if args.slower_pull:
        settings = ((.003,.006,-.00015,.020),(.006,.006,-.00015,.030),
                    (.010,.006,-.00015,.040),(.015,.006,-.00015,.060))
    settings = [(*row, .7056029424546741) for row in settings]
    if args.soft_fixture:
        settings = ((.003,.006,-.00015,.020,.35),(.003,.006,-.00015,.015,.25),
                    (.002,.006,-.00015,.010,.35),(.004,.006,-.00015,.020,.35))
    for damping, approach, press, pull, fixture in settings:
        hertz = solve('hertz', damping=damping, approach=approach, press=press, pull=pull, fixture=fixture)
        jkr = solve('jkr', damping=damping, approach=approach, press=press, pull=pull, fixture=fixture)
        separation = abs(hertz['hold_gap_median_m']-jkr['hold_gap_median_m'])
        candidates.append(dict(damping_Ns_per_m=damping, approach_s=approach, press_m=press, pull_s=pull,
            fixture_stiffness_N_per_m=fixture,
            reduced_hold_gap_difference_m=separation,
            projected_hold_difference_pixels=projected_pixels(separation, camera),
            hertz={k:v for k,v in hertz.items() if k != 'points'},
            jkr={k:v for k,v in jkr.items() if k != 'points'}))
    result = dict(scope='Reduced upper rigid mass plus quasistatic relative beam/contact; no distributed inertia or MPM/real pixels.',
        upper_mass_kg=.00010112, candidates=candidates,
        interpretation=['Contact and beam effective response is quasistatic; actual beam modes and contact impulses are unresolved.',
            'Work identity includes measured reduced contact work; it is not full contact-potential energy conservation.',
            'Reject radius_over_R>=.2. MPM must independently confirm contact, geometry and energy.'],
        production_eligible=False, public_visual_review_passed=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
