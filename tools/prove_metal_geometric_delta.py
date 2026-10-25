"""Executable dual-carry analytic, virtual-work, crossing and contact checks."""
import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from types import SimpleNamespace
import numpy as np


def qualify(output):
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as j
    from mpm_jax import fragment_multifield_v6_mpm as original
    from mpm_jax import fragment_v6_geometric_delta_candidate as new
    from mpm_jax.fragment_v6_delta_transfer import transfer_delta
    from mpm_jax.fragment_v6_geometric_delta_diagnostic import advance
    from mpm_jax.materials.linear_metal import linear_metal
    from tools.metal_frozen_transfer import support,remap
    material=linear_metal();eps=np.finfo(float).eps
    results={}
    # Full compiled material/contact step in an analytically stress-free shear.
    # L0^2 !=0, det(I+tL0)=1: this catches the old exponential-F limit without
    # suppressing nonzero pressure or passing a fake zero-force full step.
    rng=np.random.default_rng(9);x=rng.uniform(-.0003,.0003,(24,3))
    L0=np.array([[0.,1e7,0.],[0.,0.,1e7],[0.,0.,0.]])
    v=x@L0.T;T=1e-8;predicted=x+T*v
    bounds=np.stack((np.minimum(x.min(0),predicted.min(0))-.002,np.maximum(x.max(0),predicted.max(0))+.002),axis=-1)[None]
    grid=original.build_grid(x,np.zeros(len(x),np.int32),dx=.001,reference_bounds=bounds,field_physical_body=np.array([0]))
    config=original.ImpactConfig(strength=False)
    state=original.make_state(x,np.full(len(x),1e-12),np.zeros(len(x),np.int32),material,grid,v=v,C=np.broadcast_to(L0,(len(x),3,3)).copy())
    affine=[]
    for steps in (8,16,32):
        carry=new.initialize(state);history=[]
        for k in range(steps):
            carry,d,e=new.step(carry,material,grid,config,j.asarray(T/steps))
            gates=dict(original=bool(carry.state.valid),**new.additional_gates(carry,d,e,grid,1.414599609375e-8))
            assert all(gates.values()),gates
            history.append(dict(virtual_J=float(e.virtual_work_identity_J),angular=float(j.linalg.norm(e.angular_momentum_residual))))
        exact_F=np.eye(3)+T*L0;exact_L=L0@np.linalg.inv(exact_F)
        errors=dict(x=float(np.max(abs(np.asarray(carry.state.x)-predicted))),F=float(np.max(abs(np.asarray(carry.state.F)-exact_F))),
            L_relative=float(np.max(abs(np.asarray(carry.L_physical)-exact_L)))/np.max(abs(exact_L)),
            kinetic_J=float(np.asarray(original.energy_ledger(carry.state,grid)[:2].sum()-original.energy_ledger(state,grid)[:2].sum())))
        assert errors['x']<64*eps*.001 and errors['F']<64*eps and errors['L_relative']<64*eps
        assert np.array_equal(np.asarray(carry.state.v),v) and np.array_equal(np.asarray(carry.state.C),np.asarray(state.C))
        assert abs(errors['kinetic_J'])<64*eps*float(original.energy_ledger(state,grid)[:2].sum())
        assert np.all(np.asarray(original.thermodynamics(carry.state,material).temperature)==300.)
        new.save_checkpoint(output/f'affine-{steps}.npz',carry)
        loaded=new.load_checkpoint(output/f'affine-{steps}.npz')
        assert all(np.asarray(a).tobytes()==np.asarray(b).tobytes() for a,b in zip(carry.state,loaded.state))
        assert np.asarray(carry.L_physical).tobytes()==np.asarray(loaded.L_physical).tobytes()
        affine.append(dict(steps=steps,errors=errors,max_virtual_work_J=max(abs(row['virtual_J']) for row in history),max_angular_residual=max(row['angular'] for row in history)))
    results['actual_full_step_force_free_affine']=affine
    # Prescribed rigid rotation is an affine acceleration benchmark. Its secant
    # particle map is exactly orthogonal, so geometric midpoint strain is zero.
    from mpm_jax.fragment_v6_geometric_kinematics import geometric_kinematics
    from mpm_jax.materials.linear_metal import deviatoric_return
    angle=.2;Q=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
    dt_rotation=1e-8
    rotating=geometric_kinematics(j.eye(3),j.zeros((3,3)),j.asarray((Q-np.eye(3))/dt_rotation),dt_rotation)
    tau0=j.diag(j.array([2e6,-1e6,-1e6]))
    returned=deviatoric_return(tau0,j.asarray(0.),rotating.strain,rotating.rotation,j.asarray(1.),dt_rotation,j.asarray(300.),material)
    rotation_error=float(np.max(abs(np.asarray(rotating.rotation)-Q)))
    strain_error=float(np.max(abs(np.asarray(rotating.strain))))
    stress_error=float(np.max(abs(np.asarray(returned.kirchhoff_stress)-Q@np.asarray(tau0)@Q.T)))
    assert bool(rotating.valid) and bool(returned.valid)
    assert rotation_error<64*eps and strain_error<64*eps
    assert stress_error<256*eps*material.A+1e-6
    assert float(returned.specific_plastic_heat)==0. and float(returned.specific_cold_work)==0.
    results['objective_rotation_material']=dict(rotation_error=rotation_error,strain_error=strain_error,stress_error_Pa=stress_error,plastic_heat_J_per_kg=0.)
    # The old-coordinate grid work and geometric-midpoint physical work differ
    # by a time-quadrature term, not heat. Its smooth affine benchmark has a
    # first-order vanishing mismatch and second-order material quadrature error.
    L=np.diag([1e7,-5e6,2e6]);stress=np.diag([2.,-.5,.3]);Tq=1e-8
    exact_work=float(np.sum(np.diag(stress)*np.log1p(Tq*np.diag(L))))
    work_rows=[]
    for count in (8,16,32):
        h=Tq/count;grid_work=physical_work=0.
        for k in range(count):
            gradient=L@np.linalg.inv(np.eye(3)+k*h*L)
            midpoint=gradient@np.linalg.inv(np.eye(3)+.5*h*gradient)
            grid_work+=h*float(np.sum(stress*gradient));physical_work+=h*float(np.sum(stress*midpoint))
        work_rows.append(dict(steps=count,grid_physical_mismatch=grid_work-physical_work,physical_work_error=physical_work-exact_work))
    assert all(abs(work_rows[i+1]['grid_physical_mismatch'])<.6*abs(work_rows[i]['grid_physical_mismatch']) for i in range(2))
    assert all(abs(work_rows[i+1]['physical_work_error'])<.3*abs(work_rows[i]['physical_work_error']) for i in range(2))
    results['affine_work_time_quadrature']=work_rows
    # Arbitrary virtual nodal velocities, arbitrary symmetric stresses and
    # non-affine particle data: the force is the exact adjoint of the MLS gradient.
    x=rng.uniform(-.3,.3,(40,3));mass=rng.uniform(.5,1.5,len(x));body=np.repeat(np.arange(2),20);dx=.25
    frozen=support(x,body,mass,dx);w,r,ids,wm,gm=frozen
    stencil=SimpleNamespace(weights=j.asarray(w),dpos=j.asarray(r),indices=j.asarray(ids))
    stub=SimpleNamespace(layout=SimpleNamespace(dx=dx,node_count=len(gm)))
    velocities=rng.normal(size=(len(gm),3));virtual=rng.normal(size=velocities.shape);tau=rng.normal(size=(len(x),3,3));tau=.5*(tau+tau.swapaxes(-1,-2));volume=rng.uniform(.2,.5,len(x));dt=.01
    gradient=4/dx**2*w[...,None]*r
    local=-volume[:,None,None]*np.einsum('pij,pkj->pki',tau,gradient)
    force=np.stack([np.bincount(ids.ravel(),weights=local[:,:,a].ravel(),minlength=len(gm)) for a in range(3)],axis=-1)
    virtual_gradient=4/dx**2*np.einsum('pk,pki,pkj->pij',w,virtual[ids],r)
    virtual_identity=float(np.sum(force*virtual)+np.sum(volume[:,None,None]*tau*virtual_gradient))
    scale=float(np.sum(abs(force*virtual))+np.sum(abs(volume[:,None,None]*tau*virtual_gradient)))
    assert abs(virtual_identity)<64*eps*scale
    acceleration=np.array([.4,-.2,.7]);B=np.array([[.1,.3,0.],[-.2,.05,0.],[0.,0.,-.1]])
    positions=x[:,None,:]+r
    # Reconstruct global nodes independently, including distinct body fields.
    node_positions=np.zeros((len(gm),3));node_positions[ids.ravel()]=positions.reshape(-1,3)
    increment=dt*(node_positions@B.T+acceleration)
    dv,dC,_=original.transfer_to_particles(j.asarray(increment),stencil,stub)
    dv_error=float(np.max(abs(np.asarray(dv)-dt*(x@B.T+acceleration))))
    dC_error=float(np.max(abs(np.asarray(dC)-dt*B)))
    assert dv_error<64*eps and dC_error<64*eps
    results['virtual_work_and_affine_acceleration']=dict(virtual_identity=virtual_identity,virtual_scale=scale,velocity_increment_error=dv_error,gradient_increment_error=dC_error)
    # Zero force with generic unresolved affine moments, over two grid cells.
    crossing=[]
    base=rng.uniform(-.3,.3,(40,3));vel=rng.normal(size=(40,3));C=rng.normal(size=(40,3,3));duration=2*dx/np.max(np.linalg.norm(vel,axis=-1))
    for phase in (0.,.25,.5,.75):
        xx=base+phase*dx;vv=vel.copy();cc=C.copy()
        initial=.5*np.sum(mass[:,None]*vv*vv)+dx**2/8*np.sum(mass[:,None,None]*cc*cc)
        maximum_identity=0.
        for k in range(64):
            current=support(xx,body,mass,dx)
            vv,cc,diag=remap(vv,cc,mass,dx,current,'DELTA')
            maximum_identity=max(maximum_identity,abs(diag['balance_residual_J']))
            xx=xx+duration/64*vv
        assert np.array_equal(vv,vel) and np.array_equal(cc,C)
        final=.5*np.sum(mass[:,None]*vv*vv)+dx**2/8*np.sum(mass[:,None,None]*cc*cc)
        assert final==initial
        assert np.max(abs(xx-(base+phase*dx+duration*vel)))<128*eps
        crossing.append(dict(grid_phase=phase,grid_cells_crossed=2,remaps=64,kinetic_change_J=float(final-initial),maximum_ledger_residual_J=maximum_identity))
    results['general_zero_force_two_cell_crossing']=crossing
    actual_crossing=[]
    x0=rng.uniform(-.0003,.0003,(24,3));speed=np.array([5000.,0.,0.]);dx_physical=.001;duration=2*dx_physical/speed[0]
    for phase in (0.,.25,.5,.75):
        x=x0+phase*dx_physical;velocity=np.broadcast_to(speed,x.shape).copy();future=x+duration*velocity
        bounds=np.stack((np.minimum(x.min(0),future.min(0))-2*dx_physical,np.maximum(x.max(0),future.max(0))+2*dx_physical),axis=-1)[None]
        grid=original.build_grid(x,np.zeros(len(x),np.int32),dx=dx_physical,reference_bounds=bounds,field_physical_body=np.array([0]),contact_sweeps=(x,future))
        state=original.make_state(x,np.full(len(x),1e-12),np.zeros(len(x),np.int32),material,grid,v=velocity)
        carry=new.initialize(state);config=original.ImpactConfig();kinetic0=float(original.energy_ledger(state,grid)[:2].sum())
        carry,trace,failure=advance(carry,material,grid,config,duration,energy_roundoff_J=1.414599609375e-8,max_steps=512)
        assert failure is None,failure
        x_error=float(np.max(abs(np.asarray(carry.state.x)-future)))
        assert x_error<128*eps*dx_physical
        assert np.array_equal(np.asarray(carry.state.v),velocity)
        assert np.array_equal(np.asarray(carry.state.C),np.zeros((len(x),3,3)))
        assert np.array_equal(np.asarray(carry.L_physical),np.zeros((len(x),3,3)))
        kinetic_change=float(original.energy_ledger(carry.state,grid)[:2].sum())-kinetic0
        assert kinetic_change==0.
        assert np.all(np.asarray(original.thermodynamics(carry.state,material).temperature)==300.)
        actual_crossing.append(dict(phase=phase,steps=len(trace),distance_grid_cells=2,x_error_m=x_error,kinetic_change_J=kinetic_change,temperature_K=300.,all_original_and_extra_gates=True,own_CFL=True))
    results['actual_full_step_rigid_two_cell_crossing']=actual_crossing
    # The exact same contact projector, now followed by the new full step.
    contact=[]
    for name,masses,velocities,mask,expected in (
        ('two_free',[2.,5.],[[3.,.2,0.],[-1.,.2,0.]],[[False]*3]*2,[[1/7,.2,0.]]*2),
        ('fixed_wall',[2.,5.],[[3.,0.,0.],[0.,0.,0.]],[[False]*3,[True]*3],[[0.,0.,0.]]*2)):
        masses=j.asarray(masses);velocities=j.asarray(velocities);pairs=j.array([[0,1]]);colors,active=original.color_contact_pairs(np.asarray(pairs),np.zeros(2,np.int32))
        projected,loss,impulses,valid,closure,error,sweeps,_=original.paired_contact(velocities,masses,j.array([[1.,0.,0.]]),pairs,j.asarray(mask),1024,.005,j.asarray(colors),j.asarray(active))
        assert bool(valid) and np.max(abs(np.asarray(projected)-expected))<1e-12
        assert np.max(abs(np.asarray(impulses).sum(0)))<1e-12
        before=float(.5*j.sum(masses*j.sum(velocities**2,axis=1)));after=float(.5*j.sum(masses*j.sum(projected**2,axis=1)))
        analytic_loss=80/7 if name=='two_free' else 9.
        assert abs(before-after-analytic_loss)<64*eps*before
        contact.append(dict(case=name,kinetic_loss_J=before-after,analytic_loss_J=analytic_loss,closure=float(closure),complementarity=float(error)))
    results['analytic_contact']=contact
    x=np.array([[0.,0.,0.],[.0002,0.,0.]])
    body=np.array([0,1],np.int32);grid=original.build_grid(x,body,dx=.001,field_physical_body=np.array([0,1]),contact_sweeps=(x,x))
    grid=replace(grid,clamped=j.broadcast_to((grid.node_body==1)[:,None],(grid.layout.node_count,3)))
    config=original.ImpactConfig(contact_iterations=1024,contact_tolerance=.005)
    state=original.make_state(x,np.array([1e-9,1e-9]),body,material,grid,v=np.array([[3.,0.,0.],[0.,0.,0.]]))
    following,d,e=new.step(new.initialize(state),material,grid,config,j.asarray(1e-10))
    gates=dict(original=bool(following.state.valid),**new.additional_gates(following,d,e,grid,1.414599609375e-8))
    assert all(gates.values()),gates
    original.validate_state(following.state,material,grid)
    temperatures=np.asarray(original.thermodynamics(following.state,material).temperature)
    assert np.max(abs(temperatures-300))<1e-9
    new.save_checkpoint(output/'integrated-contact.npz',following)
    results['actual_full_step_contact']=dict(gates=gates,temperature_K=temperatures.tolist(),grid_physical_work_mismatch_J=float(e.grid_physical_work_mismatch_J),material_work_remainder_J=float(e.material_work_remainder_J),virtual_work_identity_J=float(e.virtual_work_identity_J),angular_residual=np.asarray(e.angular_momentum_residual).tolist())
    # Nonzero stresses in the executable step: do not certify virtual work only
    # on the zero-force example. All original energy/material gates remain live.
    x=rng.uniform(-.0003,.0003,(24,3));body=np.zeros(len(x),np.int32)
    grid=original.build_grid(x,body,dx=.001,field_physical_body=np.array([0]))
    H=np.diag([10.,-5.,-5.]);v=x@H.T
    state=original.make_state(x,np.full(len(x),1e-12),body,material,grid,v=v,C=np.broadcast_to(H,(len(x),3,3)).copy(),tau=np.broadcast_to(np.diag([2e6,-1e6,-1e6]),(len(x),3,3)).copy())
    carry,d,e=new.step(new.initialize(state),material,grid,config,j.asarray(1e-10))
    gates=dict(original=bool(carry.state.valid),**new.additional_gates(carry,d,e,grid,1.414599609375e-8))
    assert all(gates.values()),gates
    assert abs(float(d.stress_work))>0.
    new.save_checkpoint(output/'nonzero-stress.npz',carry)
    results['actual_nonzero_stress_work']=dict(gates=gates,stress_work_J=float(d.stress_work),virtual_identity_J=float(e.virtual_work_identity_J),
        grid_physical_mismatch_J=float(e.grid_physical_work_mismatch_J),material_remainder_J=float(e.material_work_remainder_J),
        original_constitutive_defect_J=float(d.constitutive_work_defect),decomposition_identity_J=float(e.work_decomposition_identity_J))
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    result=dict(passed=False,impact_started=False)
    try:
        result['checks']=qualify(args.output);result['passed']=True
    except BaseException as error:
        result['failure']=repr(error);raise
    finally:
        names=('mpm_jax/fragment_v6_geometric_delta_candidate.py','mpm_jax/fragment_v6_geometric_delta_diagnostic.py','mpm_jax/fragment_v6_geometric_kinematics.py','tools/prove_metal_geometric_delta.py','mpm_jax/fragment_v6_delta_transfer.py','mpm_jax/fragment_multifield_v6_mpm.py')
        result['source_sha256']={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in names}
        result['runtime']=dict(python=platform.python_version(),**{name:importlib.metadata.version(name) for name in ('jax','jaxlib','numpy')})
        (args.output/'proof.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
