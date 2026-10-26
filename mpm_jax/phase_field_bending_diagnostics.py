"""Read-only AT2 export checks and per-step operator accounting."""
import numpy as np


def compare_restart_frame(parent, resumed):
    """Only elastic density may differ within a strain-scaled roundoff budget."""
    assert set(parent.files)==set(resumed.files)
    exact=[]
    for key in parent.files:
        if key!='elastic_energy_density':
            assert np.array_equal(parent[key],resumed[key]), f'Actual restart field changed: {key}'
            exact.append(key)
    strain=parent['strain'];d=parent['damage']
    lam=1000*.2/(1.2*.6);mu=1000/(2*1.2)
    values=np.linalg.eigvalsh(.5*(strain+strain.swapaxes(-1,-2)))
    trace=values.sum(axis=-1)
    positive=.5*lam*np.maximum(trace,0)**2+mu*np.sum(np.maximum(values,0)**2,axis=-1)
    negative=.5*lam*np.minimum(trace,0)**2+mu*np.sum(np.minimum(values,0)**2,axis=-1)
    reference=((1-1e-8)*(1-d)**2+1e-8)*positive+negative
    budget=np.maximum(128*np.finfo(float).eps*(lam+2*mu)*np.sum(strain*strain,axis=(-2,-1)),np.finfo(float).tiny)
    a,b=parent['elastic_energy_density'],resumed['elastic_energy_density']
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert (abs(a-b)<=budget).all(), 'Elastic export exceeds roundoff budget'
    assert (abs(a-reference)<=budget).all() and (abs(b-reference)<=budget).all()
    return dict(exact_fields=exact,elastic_density_bitwise_equal=bool(np.array_equal(a,b)),
        elastic_density_maximum_absolute_difference=float(abs(a-b).max()),
        elastic_density_maximum_roundoff_budget=float(budget.max()),
        rule='Only elastic_energy_density:128*eps*(lambda+2mu)*||strain||_F^2; all other fields exact')


def step_telemetry(step, time, dt, dx, wavespeed, mass, q, old_u, old_v, new_u, new_v,
                   reaction, audit, old_punch, kkt, strain_max):
    mass=np.asarray(mass);q=np.asarray(q);old_u=np.asarray(old_u);old_v=np.asarray(old_v)
    new_u=np.asarray(new_u);new_v=np.asarray(new_v);reaction=np.asarray(reaction);audit=np.asarray(audit)
    active=mass>0;denominator=np.where(active,mass,1.)
    correction=dt*reaction/denominator[:,None]
    free_delta=new_v-old_v-correction
    free_kick=-.5*np.sum(mass[:,None]*free_delta**2)
    reaction_cross=-np.sum(mass[:,None]*free_delta*correction)
    projection=.5*np.sum(mass[:,None]*correction**2)
    actual_delta=.5*np.sum(mass[:,None]*(new_v-old_v)**2)
    weak_projection_identity=audit[4]-audit[1]+actual_delta
    left=active&np.isclose(q[:,1],0,atol=1e-10)&(abs(q[:,0]-.2)<=.06)
    right=active&np.isclose(q[:,1],0,atol=1e-10)&(abs(q[:,0]-1.4)<=.06)
    punch=active&np.isclose(q[:,1],.4,atol=1e-10)&(abs(q[:,0]-.8)<=.06)
    anchor=left&(abs(q[:,2]-.08)<=dx/2+1e-10)
    # The window starts after the ramp: punch speed is exactly -.12.
    persistent=((left|right)&(abs(old_u[:,1])<=1e-12)&(abs(old_v[:,1])<=1e-10))
    persistent|=punch&(abs(old_punch-old_u[:,1])<=1e-12)&(abs(old_v[:,1]+.12)<=1e-10)
    normal_loss=.5*mass*correction[:,1]**2
    local_work=dt*reaction*new_v
    scale=max(abs(float(audit[4])),abs(float(audit[1])),float(actual_delta),1e-20)
    vmax=float(np.linalg.norm(new_v,axis=1).max())
    row=dict(step=step,time=time,dt=dt,nominal_elastic_cfl=dt*wavespeed/dx,
        velocity_cfl=dt*vmax/dx,elastic_plus_velocity_cfl=dt*(wavespeed+vmax)/dx,
        actuator_work=float(audit[0]),absolute_step_actuator_work=abs(float(audit[0])),
        absolute_nodal_actuator_work=float(abs(local_work).sum()),
        nodal_work_reconstruction_error=float(local_work.sum()-audit[0]),
        projection_loss=float(audit[1]),projection_from_reaction=float(projection),
        frozen_strain_defect=float(audit[3]),kick_drift_difference=float(audit[4]),
        damage_split_change=float(audit[5]),algebraic_closure=float(audit[6]),
        reconstructed_free_kick_term=float(free_kick),reconstructed_reaction_cross_term=float(reaction_cross),
        kick_drift_decomposition_error=float(audit[4]-free_kick-reaction_cross),
        actual_velocity_increment_norm=float(actual_delta),
        kick_projection_identity_error=float(weak_projection_identity),
        kick_projection_identity_relative_error=float(abs(weak_projection_identity)/scale),
        kinematically_persistent_normal_projection=float(normal_loss[persistent].sum()),
        other_normal_projection=float(normal_loss[(left|right|punch)&~persistent].sum()),
        anchor_projection=float((.5*mass[anchor,None]*correction[anchor][:,[0,2]]**2).sum()),
        maximum_kkt=kkt,maximum_strain=strain_max)
    assert all(np.isfinite(value) for value in row.values())
    assert row['nominal_elastic_cfl']<=.040000000001
    assert row['kick_projection_identity_relative_error']<=1e-8
    assert abs(row['nodal_work_reconstruction_error'])<=1e-10*max(
        row['absolute_nodal_actuator_work'],abs(row['actuator_work']),1e-20)
    return row
