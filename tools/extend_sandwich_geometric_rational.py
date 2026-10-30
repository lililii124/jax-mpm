"""Add q94934 static and Neo-Hookean second-order directions to a frozen span.

Construction reads physical operators and source status receipts only. It never
loads MPM displacement snapshots. A model freeze is written only while every
predeclared 71000--83000 source history is still strictly below step 71000.
"""
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
import hashlib, json, os, sys, time
import argparse

for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_key,'4')

ROOT=Path('/extension')
FROZEN=ROOT/'output'
OUT=Path('/work/output')
MPM_ROOT=Path('/')
CASES=('uniform','central','eccentric')
FRESH=list(range(71000,83001,1000))
Q='94934d1d5792131b22756f3e2634d25708de07ade1a0ff37610881f8bb1b6ec4'
PARENT_FREEZE_SHA='f3a30fffa434aa386e2e351fa0ed107a6291cd7e79f0bb7a8672f88aed2173b6'
PARENT_RANK=184
GATE=.05

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')

def source_heads(physical_signature):
    rows=[]
    for case in CASES:
        p=MPM_ROOT/('mpm-'+case)/'status.json'
        status=json.loads(p.read_text(encoding='utf-8'));binding=status['binding']
        control_sha=hashlib.sha256(__import__('numpy').asarray(binding['controls'],dtype='float64').tobytes()).hexdigest()
        if not status.get('valid') or binding.get('case')!=case or binding.get('source_signature')!=physical_signature or control_sha!=Q:
            raise ValueError('MPM source binding mismatch: '+case)
        step=int(status['step'])
        if step>=FRESH[0]:raise RuntimeError('Fresh window consumed before freeze: '+case+' '+str(step))
        rows.append(dict(case=case,step=step,time_s=float(status['time_s']),status_sha256=sha(p)))
    return rows

def run(extension_root=ROOT,frozen_dir=None,output_dir=OUT,mpm_root=MPM_ROOT):
    global ROOT,FROZEN,OUT,MPM_ROOT
    ROOT=Path(extension_root);FROZEN=Path(frozen_dir) if frozen_dir is not None else ROOT/'output'
    OUT=Path(output_dir);MPM_ROOT=Path(mpm_root)
    sys.path.insert(0,str(ROOT))
    import numpy as np
    from scipy import sparse
    from scipy.linalg import cho_factor,cho_solve
    from scipy.sparse.linalg import gmres,spilu,LinearOperator
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from examples.blast_sandwich_design import Settings,source_signature
    from tools.build_sandwich_rational import physics_support,material_fields_digest
    from tools.assess_sandwich_rational import integrate
    from mpm_jax.total_lagrangian import ReferenceGrid,deformation,forces
    from mpm_jax.compact_tl import point_material

    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=False)
    freeze_path=FROZEN/'model-freeze.json';freeze=json.loads(freeze_path.read_text())
    if sha(freeze_path)!=PARENT_FREEZE_SHA or freeze['basis']['rank']!=PARENT_RANK:
        raise ValueError('Unexpected parent model freeze')
    for name,digest in freeze['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Pinned source mismatch: '+name)
    candidate_source_sha=sha(Path(__file__))
    if freeze['selected_controls_sha256']!=Q or freeze['fresh_validation']['steps']!=FRESH:
        raise ValueError('Parent control or allocation mismatch')
    settings,geo,support,namespace=physics_support()
    signature,sources=source_signature(Settings(xy_step=.01,z_step=.001,controls=(24,24,3),stage='fixed-mass'))
    if signature!=freeze['full_mpm_source_signature']:raise ValueError('Physical source signature mismatch')
    preheads=source_heads(signature)
    predecl_input=json.loads(Path('/work/candidate-predeclaration.json').read_text(encoding='utf-8'))
    source_manifest=json.loads(Path('/work/source-sha256.json').read_text(encoding='utf-8'))
    if source_manifest.get('worker.py')!=candidate_source_sha:
        raise ValueError('Worker source manifest mismatch')
    if source_manifest.get('run-config.json')!=sha('/work/run-config.json'):
        raise ValueError('Runtime path configuration hash mismatch')
    if predecl_input.get('candidate_source_sha256')!=candidate_source_sha:
        raise ValueError('Predeclaration did not pin the candidate source')
    declaration=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Predeclared second model using the same unread independent MPM window as q94934 vertex184.',
        parent_freeze_sha256=PARENT_FREEZE_SHA,parent_rank=PARENT_RANK,controls_sha256=Q,
        fresh_steps=FRESH,response_gate=GATE,omitted_energy_gate=GATE,
        construction_source_sha256=sha(Path(__file__)),
        predeclaration_input_sha256=sha('/work/candidate-predeclaration.json'),
        prior_model=dict(name='q94934-vertex184',freeze_sha256=PARENT_FREEZE_SHA),
        candidate_model=dict(name='q94934-vertex184-static-neohookean-second-order',
            method='retained rank-184 span + 3 q94934 unit-load static solutions + 6 Neo-Hookean internal-force second derivatives'))
    save(OUT/'predeclaration.json',dict(declaration=declaration,input=predecl_input,
        source_heads_before_construction=preheads))
    print(json.dumps(dict(stage='predeclared',source_heads=preheads)),flush=True)

    with np.load(FROZEN/'frozen-basis.npz',allow_pickle=False) as z:
        old=z['basis'].copy();nodes=z['nodes'].copy()
    rank=old.shape[-1]
    if rank!=PARENT_RANK or hashlib.sha256(old.tobytes()).hexdigest()!=freeze['basis']['numeric_basis_sha256']:
        raise ValueError('Parent basis does not match receipt')
    with np.load(FROZEN/'material.npz',allow_pickle=False) as z:
        density=z['density'].copy();young=z['young'].copy();skin=z['skin'].astype(bool).copy()
    if sha(FROZEN/'material.npz')!=freeze['inputs']['selected_material']['sha256']:
        raise ValueError('Selected material hash mismatch')
    if material_fields_digest(FROZEN/'material.npz')!=freeze['material_fields_sha256']['selected']:
        raise ValueError('Selected material fields mismatch')
    if sha(FROZEN/'tangent.npz')!=freeze['inputs']['selected_tangent']['sha256']:
        raise ValueError('Selected tangent hash mismatch')
    with np.load(FROZEN/'controls.npz',allow_pickle=False) as z:q=z['q'].copy()
    if hashlib.sha256(q.tobytes()).hexdigest()!=Q:raise ValueError('q94934 controls mismatch')
    np.testing.assert_array_equal(nodes,support.nodes)
    Kfull=sparse.load_npz(FROZEN/'tangent.npz').tocsr()
    _,_,fixed=namespace['clamped_basis'](nodes,Settings(xy_step=.01,z_step=.001,controls=(24,24,3),stage='fixed-mass'),2)
    ids=np.flatnonzero(np.repeat(~fixed,3));free=np.repeat(~fixed,3)
    K=Kfull[ids][:,ids].tocsr();nfree=len(ids)
    if K.shape!=(nfree,nfree):raise ValueError('Unexpected free-DOF tangent shape')
    # Initial design q0 is zero in fixed-mass mode: uniform target core density.
    rho0=np.where(geo['skin'],2700.,2700.*settings.target_phi)
    mass0=np.zeros(len(nodes));np.add.at(mass0,support.indices.ravel(),
        (support.weights*(support.volume*rho0)[:,None]).ravel())
    if abs(mass0.sum()/1.8468-1)>1e-10:raise ValueError('M0 mass differs from frozen target')
    mf0=np.repeat(mass0,3)[ids];C=old.reshape(-1,rank)[ids]
    gram=C.T@(mf0[:,None]*C)
    if np.max(np.abs(gram-np.eye(rank)))>1e-9:raise ValueError('Parent span lost M0 normalization')

    top_ids,top_weights=namespace['surface_weights'](support,geo['top'])
    loads=np.zeros((len(nodes),3,3))
    for case in range(3):
        np.add.at(loads[:,2,case],top_ids.ravel(),
            (-geo['face_area'][:,None]*geo['footprints'][case,:,None]*top_weights).ravel())
    load_free=loads.reshape(-1,3)[ids]
    # Reuse the extension recipe's unchanged GMRES and ILU tolerances, with the
    # retained span as a coarse correction for all nine static solves.
    scale=1/np.sqrt(K.diagonal());scaled=sparse.diags(scale)@K@sparse.diags(scale)
    ilu=spilu(scaled.tocsc(),drop_tol=1e-3,fill_factor=4,permc_spec='MMD_AT_PLUS_A')
    coarseK=C.T@(K@C);coarse=cho_factor(.5*(coarseK+coarseK.T))
    def precondition(r):return scale*ilu.solve(scale*r)+C@cho_solve(coarse,C.T@r,check_finite=False)
    Mpre=LinearOperator(K.shape,matvec=precondition,dtype=np.float64)
    def solve_static(rhs,label):
        counter=[0]
        value,info=gmres(K,rhs,M=Mpre,rtol=1e-8,atol=0.,restart=50,maxiter=40,
            callback=lambda _:counter.__setitem__(0,counter[0]+1),callback_type='pr_norm')
        residual=float(np.linalg.norm(K@value-rhs)/max(np.linalg.norm(rhs),np.finfo(float).tiny))
        record=dict(label=label,iterations=counter[0],info=int(info),relative_residual=residual)
        if info!=0 or residual>=1e-8:raise RuntimeError('Static solve failed unchanged 1e-8 gate: '+label)
        return value,record
    static=[];static_receipts=[]
    for k,case in enumerate(CASES):
        d,receipt=solve_static(load_free[:,k],'unit-load-'+case)
        static_receipts.append(receipt);static.append(d)
        print(json.dumps(dict(stage='static-solve',**receipt)),flush=True)
    static_free=np.column_stack(static)
    static_full=np.zeros((len(nodes)*3,3));static_full[ids]=static_free
    # Independent amplitude normalization changes neither the span of the
    # six Hessian loads nor the physical mode shapes; it makes the FD probe
    # well-scaled and is pinned in the construction receipt.
    directions=[];direction_scales=[]
    for k in range(3):
        d=static_full[:,k].reshape(len(nodes),3)
        grad=np.einsum('pki,pkj->pij',d[support.indices],support.gradients)
        scale_d=1/max(float(np.max(np.abs(grad))),np.finfo(float).tiny)
        directions.append(static_full[:,k]*scale_d);direction_scales.append(scale_d)

    particle_mass=density*support.volume
    nodal_mass=np.zeros(len(nodes));np.add.at(nodal_mass,support.indices.ravel(),
        (support.weights*particle_mass[:,None]).ravel())
    ref=ReferenceGrid(jnp.asarray(support.positions),jnp.asarray(support.indices),
        jnp.asarray(support.weights),jnp.asarray(support.gradients),jnp.asarray(support.volume),
        jnp.asarray(particle_mass),jnp.asarray(nodal_mass),jnp.asarray(nodes),jnp.asarray(True))
    nu=np.where(skin,.33,.25);material=point_material(jnp.asarray(young),jnp.asarray(nu))
    config=SimpleNamespace(dim=3,material='neo-hookean')
    zero=jnp.zeros(len(nodes)*3,dtype=jnp.float64);body=jnp.zeros(3,dtype=jnp.float64)
    @jax.jit
    def internal_force(flat):
        F=deformation(flat.reshape(len(nodes),3),ref)
        return forces(F,None,ref,material,config,body).reshape(-1)
    @jax.jit
    def second(d1,d2):
        return jax.jvp(lambda u:jax.jvp(internal_force,(u,),(d1,))[1],
                       (zero,),(d2,))[1]
    fzero=np.asarray(internal_force(zero));f0_norm=float(np.linalg.norm(fzero[ids]))
    if f0_norm>1e-10:raise ValueError('Neo-Hookean internal force is nonzero at F=I')
    linear_checks=[]
    d_jax=[jnp.asarray(d) for d in directions]
    for k,d in enumerate(directions):
        tangent=np.asarray(jax.jvp(internal_force,(zero,),(d_jax[k],))[1])
        kd=np.asarray(Kfull@d)
        rel=float(np.linalg.norm(tangent[ids]+kd[ids])/max(np.linalg.norm(kd[ids]),np.finfo(float).tiny))
        if rel>1e-8:raise ValueError('Internal-force JVP sign/tangent check failed')
        linear_checks.append(dict(case=CASES[k],relative_error_vs_negative_Kd=rel,
            internal_force_sign='Df(0)d = -K d'))

    # One self-term and one mixed-term finite-difference check, both at small
    # strain, against the exact JAX JVP. No repeated system-wide test sweep.
    h=1e-3
    fd_checks=[]
    d0,d1=d_jax[0],d_jax[1]
    h_self=np.asarray(second(d0,d0));fd_self=(np.asarray(internal_force(h*d0))-2*fzero+
        np.asarray(internal_force(-h*d0)))/(h*h)
    rel_self=float(np.linalg.norm((fd_self-h_self)[ids])/max(np.linalg.norm(h_self[ids]),np.finfo(float).tiny))
    fd_checks.append(dict(term='self-uniform',step=h,relative_error=rel_self))
    fd_cross=(np.asarray(internal_force(h*(d0+d1)))-np.asarray(internal_force(h*(d0-d1)))-
        np.asarray(internal_force(h*(-d0+d1)))+np.asarray(internal_force(-h*(d0+d1))))/(4*h*h)
    h_cross=np.asarray(second(d0,d1))
    rel_cross=float(np.linalg.norm((fd_cross-h_cross)[ids])/max(np.linalg.norm(h_cross[ids]),np.finfo(float).tiny))
    fd_checks.append(dict(term='cross-uniform-central',step=h,relative_error=rel_cross))
    if max(rel_self,rel_cross)>.02:raise ValueError('Central-difference JVP check exceeded 2%')
    print(json.dumps(dict(stage='jvp-checked',linear_checks=linear_checks,finite_difference_checks=fd_checks)),flush=True)

    second_vectors=[];second_receipts=[]
    for i in range(3):
        for j in range(i,3):
            hvec=np.asarray(second(d_jax[i],d_jax[j]))
            rhs=hvec[ids].copy()
            sol,receipt=solve_static(rhs,'second-order-'+CASES[i]+'-'+CASES[j])
            full=np.zeros(len(nodes)*3);full[ids]=sol
            second_vectors.append(full);second_receipts.append(dict(pair=[CASES[i],CASES[j]],
                derivative_sign='K du_ij = D2(f_internal)(0)[d_i,d_j]',**receipt))
            print(json.dumps(dict(stage='second-order-solve',**second_receipts[-1])),flush=True)

    raw=np.column_stack(directions+second_vectors)
    rawfree=raw[ids]
    for _ in range(2):rawfree-=C@(C.T@(mf0[:,None]*rawfree))
    U,singular,_=np.linalg.svd(np.sqrt(mf0)[:,None]*rawfree,full_matrices=False)
    keep=singular>singular[0]*1e-8
    Z=np.column_stack((C,U[:,keep]/np.sqrt(mf0)[:,None]))
    gram=Z.T@(mf0[:,None]*Z)
    Z=Z@np.linalg.solve(np.linalg.cholesky(.5*(gram+gram.T)).T,np.eye(len(gram)))
    if np.max(np.abs(Z.T@(mf0[:,None]*Z)-np.eye(Z.shape[1])))>1e-9:raise ValueError('M0 orthogonality failed')
    lost=C-Z@(Z.T@(mf0[:,None]*C))
    if np.linalg.norm(np.sqrt(mf0)[:,None]*lost)>1e-8:raise ValueError('Parent span not retained')
    Kfull_free=Kfull[ids][:,ids].tocsr()
    density_mass=np.zeros(len(nodes));np.add.at(density_mass,support.indices.ravel(),
        (support.weights*(support.volume*density)[:,None]).ravel())
    mfq=np.repeat(density_mass,3)[ids]
    Mr=Z.T@(mfq[:,None]*Z);Kr=Z.T@(Kfull_free@Z);Kr=.5*(Kr+Kr.T)
    coarse_load=Z.T@load_free
    basis=np.zeros((len(nodes)*3,Z.shape[1]));basis[ids]=Z;basis=basis.reshape(len(nodes),3,-1)
    bottom_ids,bottom_weights=namespace['surface_weights'](support,geo['bottom'])
    back=np.einsum('pk,pkr->pr',bottom_weights,basis[bottom_ids,2,:])
    fine=integrate(Mr,Kr,coarse_load,back,1.25e-7)
    coarse=integrate(Mr,Kr,coarse_load,back,2.5e-7)
    sensitivity=[]
    for case in range(3):
        finer=np.interp(coarse['time'],fine['time'],fine['curves'][:,case]);older=coarse['curves'][:,case]
        err=dict(peak_relative_difference=float(abs(fine['curves'][:,case].max()/older.max()-1)),
            curve_relative_l2=float(np.linalg.norm(finer-older)/np.linalg.norm(finer)),
            maximum_difference_over_finer_peak=float(np.max(np.abs(finer-older))/fine['curves'][:,case].max()))
        if max(err.values())>=.01:raise ValueError('Time sensitivity exceeded 1%')
        sensitivity.append(dict(case=CASES[case],**err))
    if fine['energy_relative_error']>=1e-8:raise ValueError('Reduced energy balance failed')
    np.savez_compressed(OUT/'frozen-basis.npz',basis=basis,nodes=nodes,mass=Mr,stiffness=Kr,
        loads=coarse_load,back=back,singular_values=singular,
        static_directions=static_full.reshape(len(nodes),3,3),second_order_directions=np.column_stack(second_vectors).reshape(len(nodes),3,6))
    np.savez_compressed(OUT/'response.npz',**{k:v for k,v in fine.items() if k!='energy_relative_error'})
    policy=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),
        method='Retain the complete rank-184 parent span; add three q94934 unit-load static solutions and six self/cross second-order static corrections from the public Neo-Hookean internal-force operator at zero displacement.',
        no_mpm_displacement_directions=True,parent_freeze_sha256=PARENT_FREEZE_SHA,parent_rank=rank,
        controls_sha256=Q,predeclaration_sha256=sha(OUT/'predeclaration.json'),
        source_heads_before_construction=preheads,source_heads_before_freeze=source_heads(signature),
        fresh_steps=FRESH,response_gate=GATE,omitted_energy_gate=GATE,
        static_solve_receipts=static_receipts,direction_max_gradient_normalization=direction_scales,
        internal_force_zero_norm_N=f0_norm,linear_tangent_checks=linear_checks,
        finite_difference_checks=fd_checks,second_order_solve_receipts=second_receipts,
        basis_input_directions=9,unique_added_rank=int(keep.sum()),singular_cutoff_relative=1e-8,
        M0_mass_kg=float(mass0.sum()),selected_mass_kg=float(density_mass.sum()),
        JAX_x64=True,elapsed_s=time.monotonic()-started)
    save(OUT/'construction-policy.json',policy)
    postheads=source_heads(signature)
    if not all(row['step']<FRESH[0] for row in postheads):
        save(OUT/'no-freeze-window-consumed.json',dict(reason='At least one allocated validation source reached step 71000 before final freeze.',
            source_heads=postheads,predeclaration_sha256=sha(OUT/'predeclaration.json'),
            candidate_basis_sha256=sha(OUT/'frozen-basis.npz')))
        print(json.dumps(dict(stage='candidate-built-no-freeze-window-consumed',source_heads=postheads)),flush=True)
        return
    receipt=dict(schema_version=2,scope=policy['method'],created_utc=datetime.now(timezone.utc).isoformat(),
        operator_vertex='selected',basis=dict(path='frozen-basis.npz',sha256=sha(OUT/'frozen-basis.npz'),
            numeric_basis_sha256=hashlib.sha256(basis.tobytes()).hexdigest(),rank=basis.shape[-1]),
        parent_basis=freeze['basis'],parent_freeze_sha256=PARENT_FREEZE_SHA,
        response=dict(path='response.npz',sha256=sha(OUT/'response.npz')),selected_controls_sha256=Q,
        inputs={key:dict(path=name,sha256=sha(FROZEN/name)) for key,name in [
            ('selected_material','material.npz'),('selected_controls','controls.npz'),('selected_tangent','tangent.npz')]},
        material_fields_sha256=freeze['material_fields_sha256'],
        full_mpm_source_signature=signature,
        construction_configuration=dict(method='static plus public-operator second derivative',retained_parent_span=True,
            added_static_directions=3,added_second_order_directions=6,mpm_snapshot_directions=False,
            selected_controls_sha256=Q,nominal_dt_s=1.25e-7,horizon_s=.002,
            predeclaration_sha256=sha(OUT/'predeclaration.json'),construction_policy_sha256=sha(OUT/'construction-policy.json')),
        fresh_validation=dict(steps=FRESH,history_interval_steps=[FRESH[0],FRESH[-1]],response_gate=GATE,omitted_energy_gate=GATE),
        actual_mpm_heads_before_fresh_window=postheads,time_sensitivity=sensitivity,
        source_sha256=dict(freeze['source_sha256'],**{'tools/extend_sandwich_geometric_rational.py':candidate_source_sha}),
        linear_energy_relative_error=fine['energy_relative_error'],complete2ms_three_case_mpm_qualification=False,
        global_optimum_claim=False)
    (OUT/'model-freeze.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(dict(stage='frozen',rank=basis.shape[-1],unique_added_rank=int(keep.sum()),
        freeze_sha256=sha(OUT/'model-freeze.json'),source_heads=postheads)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('/work/run-config.json'))
    config=json.loads(parser.parse_args().config.read_text(encoding='utf-8'))
    run(**config)
