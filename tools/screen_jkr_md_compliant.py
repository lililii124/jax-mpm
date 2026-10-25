"""NumPy/SciPy static spring/beam screen; no JAX, MPM, or rendered pixels.

MD at mu1--2 and JKR at mu5 represent explicitly different cohesive ranges,
not a same-interface law-only comparison. Static fold differences smaller than
grid/geometry errors cannot admit a visibly different dynamic movie.
"""
import argparse
import ast
import json
import math
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import brentq, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.scan_fibre_pair_fixture import jkr_branch
from tools.review_jkr_compliant_pilot import projected_pixels


def md_algebra():
    # Pure branch equations are evaluated independently with NumPy; importing
    # the production module would initialize JAX on the shared Windows host.
    tree = ast.parse((ROOT/'mpm_jax/maugis_dugdale.py').read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in ('_coefficients', 'maugis_branch')]
    scope = {'jnp':np, 'np':np}
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<MD-algebra-NumPy>', 'exec'), scope)
    return scope['maugis_branch'], scope['_coefficients']


def screen():
    branch, coefficients = md_algebra()
    young, nu, W, R, stiffness = 4e5, .25, .08, .001, .5
    E = young/(2*(1-nu**2))
    K = 4*E/3
    L = (math.pi*W*R*R/K)**(1/3)
    delta_scale = L*L/R
    camera = dict(direction=[1.4,.2,1.8], resolution=400,
                  focus_bounds_m=[-.003,.003,-.0018,.0042,-.003,.003])
    solve_jkr, jkr_critical_drive = jkr_branch(young, W, stiffness)
    hold_drive = .00046
    jkr = solve_jkr(hold_drive, True)
    records = []
    for mu in (1., 1.5, 2., 5.):
        h = (R*W*W/(E*E*mu**3))**(1/3)
        lam = 2*(W/h)*R/(K*L)
        q = np.geomspace(1e-5, 100., 10000)
        A, C, D, P, derivative, _ = branch(q, lam)
        intrinsic = derivative > 0
        qmin = float(q[np.flatnonzero(intrinsic)[0]])
        def physical(value):
            a, c, d, p, *_ = branch(float(value), lam)
            return float(a*L), float(c*L), float(d*delta_scale), float(math.pi*W*R*p)
        drive = lambda value: -physical(value)[2]-physical(value)[3]/stiffness
        fold = minimize_scalar(lambda logq:-drive(math.exp(logq)),
            bounds=(math.log(qmin), math.log(100.)), method='bounded', options={'xatol':1e-13})
        critical_q = math.exp(fold.x)
        critical_drive = drive(critical_q)
        held_q = brentq(lambda value:drive(value)-hold_drive, critical_q, 100., xtol=1e-13)
        a, c, delta, force = physical(held_q)
        alpha, beta, *_ = coefficients(held_q, lam)
        B = branch(held_q, lam)[1]/np.hypot(1, held_q)
        energy_condition = float(alpha*B*B+beta*B-1)
        separation = abs(delta-jkr['indentation_m'])
        records.append(dict(model='md', Tabor_mu=mu, interaction_range_m=h,
            maugis_lambda=lam, traction_Pa=W/h,
            hold_contact_radius_over_R=a/R, hold_cohesive_radius_over_R=c/R,
            hold_force_N=force, hold_indentation_m=delta,
            energy_condition_residual=energy_condition,
            critical_actuator_m=critical_drive-.00001,
            critical_drive_difference_from_JKR_m=critical_drive-jkr_critical_drive,
            static_hold_centre_separation_difference_from_JKR_m=separation,
            static_hold_projection_difference_from_JKR_pixels=projected_pixels(separation,camera),
            fold_window_projection_pixels=projected_pixels(abs(critical_drive-jkr_critical_drive),camera),
            same_cohesive_range_as_JKR_screen=mu == 5.,
            applicable_local_intimate_contact=a/R < .2,
            visible_static_hold_gate=projected_pixels(separation,camera) >= 10.))
    return dict(scope='Static compliant-fixture MD equations evaluated with NumPy; no measured MPM or rendered pixels.',
        common_geometry=dict(length_m=.032, radius_m=R, particles=202240),
        common_mechanics=dict(young_Pa=young, poisson=nu, effective_modulus_Pa=E,
            series_stiffness_N_per_m=stiffness, fixture_stiffness_N_per_m=.7056029424546741,
            damping_Ns_per_m=.015, hold_drive_m=hold_drive), camera=camera,
        JKR=dict(work_J_per_m2=W, asymptotic_Tabor_screen=5.,
            critical_actuator_m=jkr_critical_drive-.00001,
            hold_radius_over_R=jkr['radius_m']/R, hold_force_N=jkr['force_N']),
        MD=records, production_eligible=False, public_visual_review_passed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = screen()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
