"""Low-memory NumPy manufactured elastic beam and contact audit qualification.

This verifies quadratic reference-quadrature algebra independently of JAX.
It is not a qualification of the production JIT/contact/damage kernel.
"""
import itertools
import json
from pathlib import Path
import numpy as np


def qualify():
    h = .1
    axes = [np.arange(h/4, end, h/2) for end in (.6,.2,.2)]
    x = np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3)
    origin = np.full(3,-2*h)
    shape = np.array([11,7,7])
    coords = np.stack(np.meshgrid(*[origin[i]+h*np.arange(shape[i]) for i in range(3)],indexing='ij'),-1).reshape(-1,3)
    q = (x-origin)/h
    base = np.floor(q-.5).astype(int)
    f = q-base
    offsets = np.array(list(itertools.product(range(3),repeat=3)))
    w = np.stack([.5*(1.5-f)**2,.75-(f-1)**2,.5*(f-.5)**2],axis=1)
    dw = np.stack([f-1.5,-2*(f-1),f-.5],axis=1)/h
    selected = w[:,offsets,np.arange(3)]
    derivative = dw[:,offsets,np.arange(3)]
    weights = np.prod(selected,axis=-1)
    grad = np.stack([derivative[...,a]*selected[...,(a+1)%3]*selected[...,(a+2)%3] for a in range(3)],axis=-1)
    ijk = base[:,None,:]+offsets
    nodes = np.ravel_multi_index(tuple(ijk[...,i] for i in range(3)),shape)
    volume = (h/2)**3
    count = len(coords)
    mass = np.zeros(count)
    np.add.at(mass,nodes,volume*weights)
    strain = lambda u: .5*(np.einsum('pki,pkj->pij',u[nodes],grad)+np.einsum('pkj,pki->pij',u[nodes],grad))
    lam, mu = 1000*.2/(1.2*.6),1000/(2*1.2)
    def stress(e):
        return lam*np.trace(e,axis1=-2,axis2=-1)[:,None,None]*np.eye(3)+2*mu*e
    def potential(u):
        e = strain(u)
        return float(.5*volume*np.sum(e*stress(e)))
    def force(u):
        local = -volume*np.einsum('pij,pkj->pki',stress(strain(u)),grad)
        result = np.zeros_like(u)
        np.add.at(result,nodes,local)
        return result
    # Euler-Bernoulli displacement mode: transverse w~x^2 and axial -y*w'.
    mode = np.zeros_like(coords)
    mode[:,0] = -2*(coords[:,1]-.1)*coords[:,0]
    mode[:,1] = coords[:,0]**2
    rng = np.random.default_rng(516)
    du = rng.normal(size=mode.shape)*1e-5
    weak_residual = float(np.sum(force(mode)*du)+volume*np.sum(stress(strain(mode))*strain(du)))
    assert abs(weak_residual) < 1e-12
    stiffness = 2*potential(mode)
    modal_mass = float(np.sum(mass[:,None]*mode**2))
    omega = np.sqrt(stiffness/modal_mass)
    period = 2*np.pi/omega
    errors = []
    for n in (80,160,320):
        dt = .75*period/n
        a, v = .001, 0.
        initial = .5*stiffness*a*a
        max_error = 0.
        mechanical_defect = 0.
        kick_drift = 0.
        for _ in range(n):
            fn = -stiffness*a
            vn = v+dt*fn/modal_mass
            an = a+dt*vn
            mechanical_defect += .5*stiffness*(an*an-a*a)+dt*fn*vn
            kick_drift += dt*fn*(.5*(v+vn)-vn)
            energy = .5*modal_mass*vn*vn+.5*stiffness*an*an
            max_error = max(max_error,abs(energy-initial)/initial)
            a,v = an,vn
        closure = energy-initial-mechanical_defect-kick_drift
        assert abs(closure) < 1e-12
        errors.append(dict(steps=n,dt=dt,maximum_relative_energy_error=max_error,
                           mechanical_defect=mechanical_defect,kick_drift_difference=kick_drift,closure=closure))
    assert errors[1]['maximum_relative_energy_error'] < .6*errors[0]['maximum_relative_energy_error']
    assert errors[2]['maximum_relative_energy_error'] < .6*errors[1]['maximum_relative_energy_error']
    # Moving rigid punch: impulse work separates actuator work from impact.
    m, free, contact = 2., -.1, -.2
    impulse = m*(contact-free)
    actuator = impulse*contact
    impact = .5*m*(contact-free)**2
    projection_closure = .5*m*(contact*contact-free*free)-actuator+impact
    assert abs(projection_closure) < 1e-14
    return dict(status='numpy_manufactured_algebra_passed_not_jax_kernel_qualified',
        quadrature_particles=len(x),weak_force_drift_identity_error=weak_residual,
        no_damage_elastic_beam_mode_dt_refinement=errors,
        moving_punch_projection_closure=projection_closure,
        claim='First-order symplectic Euler energy accuracy improves with dt refinement; not exact energy conservation')


if __name__ == '__main__':
    result = qualify()
    output = Path('.local/at2-bending-persistent-grid-manufactured.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2))
