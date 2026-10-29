"""Reproduce the frozen physics-only rational Krylov sandwich recipe.

Run ``python -m tools.build_sandwich_rational CONFIG.json STAGE``.
Stages: material, tangent, selected, canonicalize, union, freeze. No MPM displacement
arrays are read during construction. Direction and SVD ordering are preserved.
"""
from pathlib import Path
import argparse,hashlib,json,os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_key,'4')

def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def input_path(config,key):
    return (Path(config['_directory'])/config[key]).resolve()

def source_receipt(config):
    root=Path(__file__).resolve().parents[1]
    names=('examples/blast_sandwich_design.py','mpm_jax/compact_tl.py','mpm_jax/reduced_tl.py','tools/build_sandwich_rational.py')
    return {name:digest(root/name) for name in names}

PHYSICAL_SOURCES=('examples/blast_sandwich_design.py','mpm_jax/compact_tl.py','mpm_jax/reduced_tl.py')

def verify_sources(receipt):
    root=Path(__file__).resolve().parents[1]
    if not (set(PHYSICAL_SOURCES)|{'tools/build_sandwich_rational.py'})<=set(receipt):raise ValueError('Incomplete physical/construction source binding')
    for name in PHYSICAL_SOURCES:
        if digest(root/name)!=receipt[name]:raise ValueError('Physical source hash mismatch: '+name)
    name='tools/build_sandwich_rational.py'
    if name in receipt and digest(root/name)!=receipt[name]:raise ValueError('Construction tool hash mismatch')

def material_fields_digest(path):
    import numpy as np
    result=hashlib.sha256()
    with np.load(path,allow_pickle=False) as saved:
        for name in ('phi','young','density','volume','skin'):
            values=np.ascontiguousarray(saved[name])
            if not np.isfinite(values).all():raise ValueError('Nonfinite material: '+name)
            result.update(values.tobytes())
    return result.hexdigest()

def verified_material(config,label):
    path=input_path(config,label+'_material')
    receipt=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if receipt['material_sha256']!=digest(path):raise ValueError('Material artifact hash mismatch: '+label)
    if receipt['controls_sha256']!=config[label+'_controls_sha256']:raise ValueError('Material controls mismatch: '+label)
    if receipt['material_fields_sha256']!=material_fields_digest(path):raise ValueError('Material field hash mismatch: '+label)
    verify_sources(receipt['sources'])
    if label=='selected' and receipt['control_artifact_sha256']!=digest(input_path(config,'selected_controls')):
        raise ValueError('Selected control artifact changed')
    return receipt

def verify_construction_inputs(config,stage):
    if stage=='material':return
    verified_material(config,'selected')
    if stage in ('canonicalize','union','freeze'):verified_material(config,'initial')
    if stage in ('selected','canonicalize','union','freeze'):
        path=input_path(config,'selected_tangent');receipt=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        if receipt['tangent_sha256']!=digest(path) or receipt['material_sha256']!=digest(input_path(config,'selected_material')):
            raise ValueError('Selected tangent binding mismatch')
        verify_sources(receipt['sources'])
    if stage in ('canonicalize','union','freeze'):
        source=input_path(config,'selected_basis');receipt=json.loads((source.parent/'construction-inputs.json').read_text(encoding='utf-8'))
        if receipt['artifact_sha256']['rational-basis.npz']!=digest(source):raise ValueError('Selected basis artifact changed')
        for key in ('selected_material','selected_tangent'):
            if receipt['input_sha256'][key]!=digest(input_path(config,key)):raise ValueError('Selected basis input changed: '+key)
        verify_sources(receipt['physical_source_sha256'])
    if stage in ('union','freeze'):
        source=input_path(config,'canonical_selected_basis');receipt=json.loads(source.with_suffix('.json').read_text(encoding='utf-8'))
        if receipt['artifact_sha256']!=digest(source):raise ValueError('Canonical selected basis changed')
        for key in ('selected_basis','initial_material','selected_material'):
            if receipt['input_sha256'][key]!=digest(input_path(config,key)):raise ValueError('Canonicalization input changed: '+key)
        verify_sources(receipt['sources'])
    if stage=='freeze':
        directory=input_path(config,'union_output');receipt=json.loads((directory/'construction-inputs.json').read_text(encoding='utf-8'))
        for name,value in receipt['artifact_sha256'].items():
            if digest(directory/name)!=value:raise ValueError('Union output changed: '+name)
        for key,value in receipt['input_sha256'].items():
            if digest(input_path(config,key))!=value:raise ValueError('Union input changed: '+key)
        verify_sources(receipt['physical_source_sha256'])

def check_recipe(config):
    expected=dict(frequencies_hz=[0.,2000.,8000.,32000.,128000.,512000.],powers_per_shift=3,
        gmres_relative_tolerance=1e-8,svd_relative_cutoff=1e-8,horizon_s=.002,
        analytic_order=2,target_mass_kg=1.8468,load_cases=['uniform','central','eccentric'],
        gmres_restart=50,gmres_maxiter=40,ilu_drop_tolerance=1e-3,ilu_fill_factor=4,
        ilu_permutation='MMD_AT_PLUS_A',residual_projection_passes=2)
    expected['full_mpm_source_signature']='62cf1ac98a9657c2012233af5421441ef35820013d63f7d567278ea07a49842b'
    for key,value in expected.items():
        if config.get(key)!=value:raise ValueError(f'Frozen recipe mismatch: {key}')
    root=Path(__file__).resolve().parents[1]
    pins=config.get('expected_source_sha256',{})
    if set(pins)!=set(PHYSICAL_SOURCES):raise ValueError('Complete physical source pins are required')
    for name,expected_sha in pins.items():
        if digest(root/name)!=expected_sha:raise ValueError('Physical source hash mismatch: '+name)

def build_selected(config):
    """Full-horizon, three-load rational Krylov experiment using only known K/M."""
    from pathlib import Path
    from itertools import product
    from typing import NamedTuple
    import ast,hashlib,json,math,os,time
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='4'
    import numpy as np
    from scipy import sparse
    from scipy.linalg import cho_factor,cho_solve
    from scipy.sparse.linalg import gmres,spilu,LinearOperator
    from examples.blast_sandwich_design import Settings,geometry

    root=Path(__file__).resolve().parents[1];out=input_path(config,'output');out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    def sha(path):
        with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    def save(name,value):
        temporary=out/(name+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(out/name)
    settings=Settings(xy_step=.01,z_step=.001,controls=(24,24,3));geo=geometry(settings)
    namespace=dict(np=np,product=product,NamedTuple=NamedTuple,math=math)
    for filename,names in [('mpm_jax/compact_tl.py',('Support','rectilinear_support','surface_weights')),
        ('mpm_jax/reduced_tl.py',('clamped_basis','normalize_basis'))]:
        tree=ast.parse((root/filename).read_text());parts=[item for item in tree.body if isinstance(item,(ast.ClassDef,ast.FunctionDef)) and item.name in names]
        exec(compile(ast.Module(body=parts,type_ignores=[]),filename,'exec'),namespace)
    support=namespace['rectilinear_support'](geo['points'],geo['volume'],geo['plan']['spacing_m'])
    nodes=support.nodes
    with np.load(input_path(config,'selected_material'),allow_pickle=False) as saved:
        density=saved['density'];young=saved['young'];skin=saved['skin'].astype(bool)
    mass=np.zeros(len(nodes));np.add.at(mass,support.indices.ravel(),(support.weights*(support.volume*density)[:,None]).ravel())
    assert abs(mass.sum()/1.8468-1)<1e-10
    analytic,labels,fixed=namespace['clamped_basis'](nodes,settings,2)
    free=np.repeat(~fixed,3);ids=np.flatnonzero(free);mf=np.repeat(mass,3)[ids];sqrt_mass=np.sqrt(mf)
    K=sparse.load_npz(input_path(config,'selected_tangent'))[ids][:,ids].tocsr();assert np.min(K.diagonal())>0
    top_ids,top_weights=namespace['surface_weights'](support,geo['top']);loads=np.zeros((len(nodes),3,3))
    for case in range(3):np.add.at(loads[:,2,case],top_ids.ravel(),(-geo['face_area'][:,None]*geo['footprints'][case,:,None]*top_weights).ravel())
    F=loads.reshape(-1,3)[ids]
    bottom_ids,bottom_weights=namespace['surface_weights'](support,geo['bottom'])
    C=analytic.reshape(-1,24)[ids];gram=C.T@(mf[:,None]*C)
    C=C@np.linalg.solve(np.linalg.cholesky(.5*(gram+gram.T)).T,np.eye(24))
    coarseK=C.T@(K@C)
    frequencies=[0.,2000.,8000.,32000.,128000.,512000.]
    directions=[];receipts=[]
    for frequency in frequencies:
        sigma=(2*np.pi*frequency)**2
        A=K+sparse.diags(sigma*mf)
        scale=1/np.sqrt(A.diagonal());scaled=sparse.diags(scale)@A@sparse.diags(scale)
        ilu=spilu(scaled.tocsc(),drop_tol=1e-3,fill_factor=4,permc_spec='MMD_AT_PLUS_A')
        coarse=cho_factor(coarseK+sigma*np.eye(24))
        def precondition(r):return scale*ilu.solve(scale*r)+C@cho_solve(coarse,C.T@r)
        preconditioner=LinearOperator(A.shape,matvec=precondition,dtype=np.float64)
        rhs=F.copy()
        for power in range(3):
            solutions=[]
            for case in range(3):
                counter=[0]
                def count(_):
                    counter[0]+=1
                    if time.monotonic()-started>1500:raise TimeoutError('Rational experiment solve1500s bound')
                value,info=gmres(A,rhs[:,case],M=preconditioner,rtol=1e-8,atol=0.,restart=50,maxiter=40,callback=count,callback_type='pr_norm')
                relative=float(np.linalg.norm(A@value-rhs[:,case])/np.linalg.norm(rhs[:,case]))
                record=dict(frequency_hz=frequency,power=power+1,case=case,iterations=counter[0],info=int(info),relative_residual=relative)
                receipts.append(record);save('solves.json',receipts);print(json.dumps(record),flush=True)
                if info!=0 or relative>=1e-8:raise RuntimeError('Rational solve failed unchanged1e-8 gate')
                value/=np.sqrt(np.dot(mf,value*value));solutions.append(value);directions.append(value)
                np.savez_compressed(out/f'direction-{len(directions):03d}.npz',direction=value,free_ids=ids,receipt=np.asarray(json.dumps(record)))
            rhs=mf[:,None]*np.column_stack(solutions)
        del ilu,scaled,A,preconditioner
    for case in range(3):
        value=F[:,case]/mf;value/=np.sqrt(np.dot(mf,value*value));directions.append(value)
    raw=np.column_stack(directions)
    for _ in range(2):raw-=C@(C.T@(mf[:,None]*raw))
    U,singular,_=np.linalg.svd(sqrt_mass[:,None]*raw,full_matrices=False);keep=singular>singular[0]*1e-8
    Z=np.column_stack((C,U[:,keep]/sqrt_mass[:,None]))
    M=Z.T@(mf[:,None]*Z);Kr=Z.T@(K@Z);Kr=.5*(Kr+Kr.T);L=Z.T@F
    assert np.max(np.abs(M-np.eye(len(M))))<1e-9
    cho_factor(Kr)
    basis=np.zeros((len(nodes)*3,Z.shape[-1]));basis[ids]=Z;basis=basis.reshape(len(nodes),3,-1)
    back=np.einsum('pk,pkr->pr',bottom_weights,basis[bottom_ids,2,:])
    rank=Z.shape[-1]
    def solve(dt):
        count=int(np.ceil(.002/dt));h=.002/count;t=np.arange(count+1)*h
        pressure=np.where(t<.0002,.5e6*np.sin(np.pi*t/.0002)**2,0.)
        factor=cho_factor(M+.25*h*h*Kr);u=np.zeros((rank,3));v=u.copy();a=u.copy();work=np.zeros(3)
        states=[u.copy()];velocities=[v.copy()];energy=[]
        for n in range(count):
            predicted=u+h*v+.25*h*h*a
            following_a=cho_solve(factor,pressure[n+1]*L-Kr@predicted,check_finite=False)
            following=predicted+.25*h*h*following_a;following_v=v+.5*h*(a+following_a)
            work+=.5*(pressure[n]+pressure[n+1])*np.sum(L*(following-u),axis=0)
            kinetic=.5*np.sum(following_v*(M@following_v),axis=0);elastic=.5*np.sum(following*(Kr@following),axis=0)
            energy.append(np.stack((kinetic,elastic,work),axis=-1));u,v,a=following,following_v,following_a
            states.append(u.copy());velocities.append(v.copy())
        states=np.asarray(states);velocities=np.asarray(velocities);energy=np.asarray(energy)
        error=float(np.max(np.abs(energy[:,:,0]+energy[:,:,1]-energy[:,:,2]))/np.max(np.abs(energy)))
        assert error<1e-8
        curves=np.concatenate([np.max(np.abs(np.einsum('pr,trc->tpc',back,states[start:start+32])),axis=1) for start in range(0,len(states),32)])
        return t,states,velocities,energy,curves,error
    summaries=[]
    for label,dt in [('coarse',1e-6),('fine',2.5e-7)]:
        t,states,velocities,energy,curves,error=solve(dt)
        np.savez_compressed(out/f'{label}-response.npz',time=t,displacement=states,velocity=velocities,energy=energy,curves=curves)
        summaries.append(dict(label=label,dt_s=float(t[1]),steps=len(t)-1,peak_back_m=curves.max(axis=0).tolist(),energy_relative_error=error))
    # Known-operator force residuals across the COMPLETE2ms. No MPM states used.
    samples=[]
    for index in np.linspace(0,len(t)-1,41,dtype=int):
        q=states[index];v=velocities[index]
        pressure=.5e6*np.sin(np.pi*t[index]/.0002)**2 if t[index]<.0002 else 0.
        acceleration=np.linalg.solve(M,pressure*L-Kr@q)
        residual=K@(Z@q)+mf[:,None]*(Z@acceleration)-pressure*F
        samples.append(dict(time_s=float(t[index]),mass_dual_force_norm=np.sqrt(np.sum(residual*residual/mf[:,None],axis=0)).tolist()))
    np.savez_compressed(out/'rational-basis.npz',basis=basis,nodes=nodes,mass=M,stiffness=Kr,loads=L,back=back,free_ids=ids,
        directions=basis[:,:,24:],singular_values=singular)
    save('review.json',dict(scope='Physics-only complete2ms/3-load rational Krylov experiment, not independently MPM-qualified.',
        controls_sha256=config['selected_controls_sha256'],frequencies_hz=frequencies,powers_per_shift=3,
        analytic_rank=24,raw_operator_direction_count=len(directions),rank=rank,basis_sha256=hashlib.sha256(basis.tobytes()).hexdigest(),
        linear_summaries=summaries,full_space_residual_samples=samples,
        gmres_maximum_relative_residual=max(r['relative_residual'] for r in receipts),
        mass_gram_error=float(np.max(np.abs(M-np.eye(rank)))),wall_seconds=time.monotonic()-started,
        mpm_snapshots_used_for_directions=False,independent_mpm_validation_passed=False,
        immutable_sources=source_receipt(config)))
    print(json.dumps(dict(state='complete',rank=rank,wall_seconds=time.monotonic()-started)),flush=True)

def build_union(config):
    """Full-horizon, three-load rational Krylov experiment using only known K/M."""
    from pathlib import Path
    from itertools import product
    from typing import NamedTuple
    import ast,hashlib,json,math,os,time
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='4'
    import numpy as np
    from scipy import sparse
    from scipy.linalg import cho_factor,cho_solve
    from scipy.sparse.linalg import gmres,spilu,LinearOperator
    from examples.blast_sandwich_design import Settings,geometry

    root=Path(__file__).resolve().parents[1];out=input_path(config,'output');out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    def sha(path):
        with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    def save(name,value):
        temporary=out/(name+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(out/name)
    settings=Settings(xy_step=.01,z_step=.001,controls=(24,24,3));geo=geometry(settings)
    namespace=dict(np=np,product=product,NamedTuple=NamedTuple,math=math)
    for filename,names in [('mpm_jax/compact_tl.py',('Support','rectilinear_support','surface_weights')),
        ('mpm_jax/reduced_tl.py',('clamped_basis','normalize_basis'))]:
        tree=ast.parse((root/filename).read_text());parts=[item for item in tree.body if isinstance(item,(ast.ClassDef,ast.FunctionDef)) and item.name in names]
        exec(compile(ast.Module(body=parts,type_ignores=[]),filename,'exec'),namespace)
    support=namespace['rectilinear_support'](geo['points'],geo['volume'],geo['plan']['spacing_m'])
    nodes=support.nodes
    with np.load(input_path(config,'initial_material'),allow_pickle=False) as saved:
        density=saved['density'];young=saved['young'];skin=saved['skin'].astype(bool)
    mass=np.zeros(len(nodes));np.add.at(mass,support.indices.ravel(),(support.weights*(support.volume*density)[:,None]).ravel())
    assert abs(mass.sum()/1.8468-1)<1e-10
    analytic,labels,fixed=namespace['clamped_basis'](nodes,settings,2)
    free=np.repeat(~fixed,3);ids=np.flatnonzero(free);mf=np.repeat(mass,3)[ids];sqrt_mass=np.sqrt(mf)
    Kfull=sparse.csr_matrix((len(nodes)*3,len(nodes)*3))
    nu=np.where(skin,.33,.25);mu=young/(2*(1+nu));lam=young*nu/((1+nu)*(1-2*nu))
    for start in range(0,len(support.indices),2048):
        stop=min(start+2048,len(support.indices));ix=support.indices[start:stop];width=ix.shape[1]
        ii=np.broadcast_to(ix[:,:,None],(len(ix),width,width)).reshape(-1).astype(np.int32)
        jj=np.broadcast_to(ix[:,None,:],(len(ix),width,width)).reshape(-1).astype(np.int32)
        gradient=support.gradients[start:stop]
        for first in range(3):
            for second in range(3):
                values=lam[start:stop,None,None]*gradient[:,:,first,None]*gradient[:,None,:,second]
                values+=mu[start:stop,None,None]*gradient[:,:,second,None]*gradient[:,None,:,first]
                if first==second:values+=mu[start:stop,None,None]*np.einsum('pai,pbi->pab',gradient,gradient)
                Kfull+=sparse.coo_matrix(((values*support.volume[start:stop,None,None]).reshape(-1),(3*ii+first,3*jj+second)),shape=Kfull.shape).tocsr()
        print(json.dumps(dict(event='uniform-tangent-chunk',stop=stop,total=len(support.indices))),flush=True)
    Kfull=sparse.diags(free.astype(float))@Kfull@sparse.diags(free.astype(float))+sparse.diags((~free).astype(float));Kfull=.5*(Kfull+Kfull.T)
    sparse.save_npz(out/'full-tangent-initial.npz',Kfull)
    K=Kfull[ids][:,ids].tocsr();del Kfull,ii,jj,values,gradient
    assert np.min(K.diagonal())>0
    top_ids,top_weights=namespace['surface_weights'](support,geo['top']);loads=np.zeros((len(nodes),3,3))
    for case in range(3):np.add.at(loads[:,2,case],top_ids.ravel(),(-geo['face_area'][:,None]*geo['footprints'][case,:,None]*top_weights).ravel())
    F=loads.reshape(-1,3)[ids]
    bottom_ids,bottom_weights=namespace['surface_weights'](support,geo['bottom'])
    with np.load(input_path(config,'canonical_selected_basis'),allow_pickle=False) as saved:
        np.testing.assert_array_equal(nodes,saved['nodes']);C=saved['basis'].reshape(len(nodes)*3,-1)[ids]
    seed_rank=C.shape[-1];assert seed_rank==78
    gram=C.T@(mf[:,None]*C)
    C=C@np.linalg.solve(np.linalg.cholesky(.5*(gram+gram.T)).T,np.eye(seed_rank))
    coarseK=C.T@(K@C)
    frequencies=[0.,2000.,8000.,32000.,128000.,512000.]
    directions=[];receipts=[]
    for frequency in frequencies:
        sigma=(2*np.pi*frequency)**2
        A=K+sparse.diags(sigma*mf)
        scale=1/np.sqrt(A.diagonal());scaled=sparse.diags(scale)@A@sparse.diags(scale)
        ilu=spilu(scaled.tocsc(),drop_tol=1e-3,fill_factor=4,permc_spec='MMD_AT_PLUS_A')
        coarse=cho_factor(coarseK+sigma*np.eye(seed_rank))
        def precondition(r):return scale*ilu.solve(scale*r)+C@cho_solve(coarse,C.T@r)
        preconditioner=LinearOperator(A.shape,matvec=precondition,dtype=np.float64)
        rhs=F.copy()
        for power in range(3):
            solutions=[]
            for case in range(3):
                counter=[0]
                def count(_):
                    counter[0]+=1
                    if time.monotonic()-started>1500:raise TimeoutError('Rational experiment solve1500s bound')
                value,info=gmres(A,rhs[:,case],M=preconditioner,rtol=1e-8,atol=0.,restart=50,maxiter=40,callback=count,callback_type='pr_norm')
                relative=float(np.linalg.norm(A@value-rhs[:,case])/np.linalg.norm(rhs[:,case]))
                record=dict(frequency_hz=frequency,power=power+1,case=case,iterations=counter[0],info=int(info),relative_residual=relative)
                receipts.append(record);save('solves.json',receipts);print(json.dumps(record),flush=True)
                if info!=0 or relative>=1e-8:raise RuntimeError('Rational solve failed unchanged1e-8 gate')
                value/=np.sqrt(np.dot(mf,value*value));solutions.append(value);directions.append(value)
                np.savez_compressed(out/f'direction-{len(directions):03d}.npz',direction=value,free_ids=ids,receipt=np.asarray(json.dumps(record)))
            rhs=mf[:,None]*np.column_stack(solutions)
        del ilu,scaled,A,preconditioner
    for case in range(3):
        value=F[:,case]/mf;value/=np.sqrt(np.dot(mf,value*value));directions.append(value)
    raw=np.column_stack(directions)
    for _ in range(2):raw-=C@(C.T@(mf[:,None]*raw))
    U,singular,_=np.linalg.svd(sqrt_mass[:,None]*raw,full_matrices=False);keep=singular>singular[0]*1e-8
    Z=np.column_stack((C,U[:,keep]/sqrt_mass[:,None]))
    M=Z.T@(mf[:,None]*Z)
    Z=Z@np.linalg.solve(np.linalg.cholesky(.5*(M+M.T)).T,np.eye(len(M)))
    M=Z.T@(mf[:,None]*Z);Kr=Z.T@(K@Z);Kr=.5*(Kr+Kr.T);L=Z.T@F
    assert np.max(np.abs(M-np.eye(len(M))))<1e-10
    assert np.max(np.abs(M-np.eye(len(M))))<1e-9
    cho_factor(Kr)
    basis=np.zeros((len(nodes)*3,Z.shape[-1]));basis[ids]=Z;basis=basis.reshape(len(nodes),3,-1)
    back=np.einsum('pk,pkr->pr',bottom_weights,basis[bottom_ids,2,:])
    rank=Z.shape[-1]
    def solve(dt):
        count=int(np.ceil(.002/dt));h=.002/count;t=np.arange(count+1)*h
        pressure=np.where(t<.0002,.5e6*np.sin(np.pi*t/.0002)**2,0.)
        factor=cho_factor(M+.25*h*h*Kr);u=np.zeros((rank,3));v=u.copy();a=u.copy();work=np.zeros(3)
        states=[u.copy()];velocities=[v.copy()];energy=[]
        for n in range(count):
            predicted=u+h*v+.25*h*h*a
            following_a=cho_solve(factor,pressure[n+1]*L-Kr@predicted,check_finite=False)
            following=predicted+.25*h*h*following_a;following_v=v+.5*h*(a+following_a)
            work+=.5*(pressure[n]+pressure[n+1])*np.sum(L*(following-u),axis=0)
            kinetic=.5*np.sum(following_v*(M@following_v),axis=0);elastic=.5*np.sum(following*(Kr@following),axis=0)
            energy.append(np.stack((kinetic,elastic,work),axis=-1));u,v,a=following,following_v,following_a
            states.append(u.copy());velocities.append(v.copy())
        states=np.asarray(states);velocities=np.asarray(velocities);energy=np.asarray(energy)
        error=float(np.max(np.abs(energy[:,:,0]+energy[:,:,1]-energy[:,:,2]))/np.max(np.abs(energy)))
        assert error<1e-8
        curves=np.concatenate([np.max(np.abs(np.einsum('pr,trc->tpc',back,states[start:start+32])),axis=1) for start in range(0,len(states),32)])
        return t,states,velocities,energy,curves,error
    summaries=[]
    for label,dt in [('coarse',2.5e-7),('fine',1.25e-7)]:
        t,states,velocities,energy,curves,error=solve(dt)
        np.savez_compressed(out/f'{label}-response.npz',time=t,displacement=states,velocity=velocities,energy=energy,curves=curves)
        summaries.append(dict(label=label,dt_s=float(t[1]),steps=len(t)-1,peak_back_m=curves.max(axis=0).tolist(),energy_relative_error=error))
    # Known-operator force residuals across the COMPLETE2ms. No MPM states used.
    samples=[]
    for index in np.linspace(0,len(t)-1,41,dtype=int):
        q=states[index];v=velocities[index]
        pressure=.5e6*np.sin(np.pi*t[index]/.0002)**2 if t[index]<.0002 else 0.
        acceleration=np.linalg.solve(M,pressure*L-Kr@q)
        residual=K@(Z@q)+mf[:,None]*(Z@acceleration)-pressure*F
        samples.append(dict(time_s=float(t[index]),mass_dual_force_norm=np.sqrt(np.sum(residual*residual/mf[:,None],axis=0)).tolist()))
    np.savez_compressed(out/'two-design-basis.npz',basis=basis,nodes=nodes,mass=M,stiffness=Kr,loads=L,back=back,free_ids=ids,
        directions=basis[:,:,24:],singular_values=singular)
    # The same union basis is reevaluated with the other material vertex.
    with np.load(input_path(config,'selected_material'),allow_pickle=False) as saved:rho_selected=saved['density']
    nodal_selected=np.zeros(len(nodes));np.add.at(nodal_selected,support.indices.ravel(),(support.weights*(support.volume*rho_selected)[:,None]).ravel())
    mass_selected_free=np.repeat(nodal_selected,3)[ids]
    selectedK=sparse.load_npz(input_path(config,'selected_tangent'))[ids][:,ids].tocsr()
    M_initial=M.copy();K_initial=Kr.copy()
    M=Z.T@(mass_selected_free[:,None]*Z);Kr=Z.T@(selectedK@Z);Kr=.5*(Kr+Kr.T)
    selected_summaries=[]
    for label,dt in [('coarse',2.5e-7),('fine',1.25e-7)]:
        st,su,sv,se,sc,error=solve(dt)
        np.savez_compressed(out/f'selected-{label}-response.npz',time=st,displacement=su,velocity=sv,energy=se,curves=sc)
        selected_summaries.append(dict(label=label,dt_s=float(st[1]),steps=len(st)-1,peak_back_m=sc.max(axis=0).tolist(),energy_relative_error=error))
    np.savez_compressed(out/'selected-operators.npz',mass=M,stiffness=Kr,loads=L,back=back)
    M=M_initial;Kr=K_initial

    save('review.json',dict(scope='Physics-only two-design complete2ms/3-load rational Krylov experiment, not independently MPM-qualified.',
        controls_sha256=config['initial_controls_sha256'],frequencies_hz=frequencies,powers_per_shift=3,
        analytic_rank=24,raw_operator_direction_count=len(directions),rank=rank,basis_sha256=hashlib.sha256(basis.tobytes()).hexdigest(),
        linear_summaries_initial=summaries,linear_summaries_selected=selected_summaries,full_space_residual_samples_initial=samples,seed_selected_rank=seed_rank,
        gmres_maximum_relative_residual=max(r['relative_residual'] for r in receipts),
        mass_gram_error=float(np.max(np.abs(M-np.eye(rank)))),wall_seconds=time.monotonic()-started,
        mpm_snapshots_used_for_directions=False,independent_mpm_validation_passed=False,
        immutable_sources=source_receipt(config)))
    print(json.dumps(dict(state='complete',rank=rank,wall_seconds=time.monotonic()-started)),flush=True)

def material(config):
    import numpy as np
    import jax
    jax.config.update('jax_enable_x64',True)
    from examples.blast_sandwich_design import Settings,build_model
    settings=Settings(xy_step=.01,z_step=.001,controls=(24,24,3),stage='fixed-mass')
    base=build_model(settings)
    for label in ('initial','selected'):
        if label=='initial':q=np.asarray(base['q0'])
        else:
            controls_path=input_path(config,'selected_controls')
            if controls_path.suffix=='.json':q=np.asarray(json.loads(controls_path.read_text(encoding='utf-8'))['q'],dtype=np.float64)
            else:
                with np.load(controls_path,allow_pickle=False) as saved:q=saved['q']
        controls_sha=hashlib.sha256(q.tobytes()).hexdigest()
        if controls_sha!=config[label+'_controls_sha256']:raise ValueError('Control hash mismatch: '+label)
        phi,density,young,_=base['properties'](q)
        path=input_path(config,label+'_material');path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,phi=np.asarray(phi),density=np.asarray(density),young=np.asarray(young),volume=base['support'].volume,skin=base['geometry']['skin'])
        path.with_suffix('.json').write_text(json.dumps(dict(controls_sha256=controls_sha,material_sha256=digest(path),material_fields_sha256=material_fields_digest(path),
            control_artifact_sha256=digest(input_path(config,'selected_controls')) if label=='selected' else None,sources=source_receipt(config)),indent=2)+'\n')

def physics_support():
    import ast,math
    from itertools import product
    from typing import NamedTuple
    import numpy as np
    from examples.blast_sandwich_design import Settings,geometry
    root=Path(__file__).resolve().parents[1];settings=Settings(xy_step=.01,z_step=.001,controls=(24,24,3));geo=geometry(settings)
    namespace=dict(np=np,product=product,NamedTuple=NamedTuple,math=math)
    for filename,names in [('mpm_jax/compact_tl.py',('Support','rectilinear_support','surface_weights')),('mpm_jax/reduced_tl.py',('clamped_basis','normalize_basis'))]:
        tree=ast.parse((root/filename).read_text(encoding='utf-8'));parts=[item for item in tree.body if isinstance(item,(ast.ClassDef,ast.FunctionDef)) and item.name in names]
        exec(compile(ast.Module(body=parts,type_ignores=[]),filename,'exec'),namespace)
    support=namespace['rectilinear_support'](geo['points'],geo['volume'],geo['plan']['spacing_m'])
    return settings,geo,support,namespace

def tangent(config):
    import numpy as np
    from scipy import sparse
    settings,geo,support,namespace=physics_support()
    nodes=support.nodes
    with np.load(input_path(config,'selected_material'),allow_pickle=False) as saved:young=saved['young'];skin=saved['skin'].astype(bool)
    _,_,fixed=namespace['clamped_basis'](nodes,settings,2);free=np.repeat(~fixed,3)
    Kfull=sparse.csr_matrix((len(nodes)*3,len(nodes)*3))
    nu=np.where(skin,.33,.25);mu=young/(2*(1+nu));lam=young*nu/((1+nu)*(1-2*nu))
    for start in range(0,len(support.indices),2048):
        stop=min(start+2048,len(support.indices));ix=support.indices[start:stop];width=ix.shape[1]
        ii=np.broadcast_to(ix[:,:,None],(len(ix),width,width)).reshape(-1).astype(np.int32)
        jj=np.broadcast_to(ix[:,None,:],(len(ix),width,width)).reshape(-1).astype(np.int32)
        gradient=support.gradients[start:stop]
        for first in range(3):
            for second in range(3):
                values=lam[start:stop,None,None]*gradient[:,:,first,None]*gradient[:,None,:,second]
                values+=mu[start:stop,None,None]*gradient[:,:,second,None]*gradient[:,None,:,first]
                if first==second:values+=mu[start:stop,None,None]*np.einsum('pai,pbi->pab',gradient,gradient)
                Kfull+=sparse.coo_matrix(((values*support.volume[start:stop,None,None]).reshape(-1),(3*ii+first,3*jj+second)),shape=Kfull.shape).tocsr()
    Kfull=sparse.diags(free.astype(float))@Kfull@sparse.diags(free.astype(float))+sparse.diags((~free).astype(float));Kfull=.5*(Kfull+Kfull.T)
    path=input_path(config,'selected_tangent');path.parent.mkdir(parents=True,exist_ok=True);sparse.save_npz(path,Kfull)
    path.with_suffix('.json').write_text(json.dumps(dict(tangent_sha256=digest(path),material_sha256=digest(input_path(config,'selected_material')),sources=source_receipt(config)),indent=2)+'\n')

def canonicalize(config):
    import numpy as np
    settings,geo,support,namespace=physics_support()
    with np.load(input_path(config,'selected_basis'),allow_pickle=False) as saved:
        V=saved['basis'];nodes=saved['nodes'];M=saved['mass'];K=saved['stiffness'];L=saved['loads'];back=saved['back']
    np.testing.assert_array_equal(nodes,support.nodes)
    with np.load(input_path(config,'initial_material'),allow_pickle=False) as saved:rho0=saved['density']
    mass0=np.zeros(len(nodes));np.add.at(mass0,support.indices.ravel(),(support.weights*(support.volume*rho0)[:,None]).ravel())
    rank=V.shape[-1];flat=V.reshape(-1,rank);gram=flat.T@(np.repeat(mass0,3)[:,None]*flat)
    transform=np.linalg.solve(np.linalg.cholesky(.5*(gram+gram.T)).T,np.eye(rank))
    V=(flat@transform).reshape(V.shape)
    M=transform.T@M@transform;K=transform.T@K@transform;L=transform.T@L;back=back@transform
    error=float(np.max(np.abs(V.reshape(-1,rank).T@(np.repeat(mass0,3)[:,None]*V.reshape(-1,rank))-np.eye(rank))))
    assert error<1e-10
    path=input_path(config,'canonical_selected_basis');path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,basis=V,nodes=nodes,mass=M,stiffness=K,loads=L,back=back,directions=V[:,:,24:],reference_nodal_mass=mass0)
    path.with_suffix('.json').write_text(json.dumps(dict(rank=rank,basis_sha256=hashlib.sha256(V.tobytes()).hexdigest(),artifact_sha256=digest(path),reference_mass_gram_error=error,
        input_sha256={key:digest(input_path(config,key)) for key in ('selected_basis','initial_material','selected_material')},sources=source_receipt(config)),indent=2)+'\n')

def freeze(config):
    import numpy as np
    output=input_path(config,'union_output')
    with np.load(output/'two-design-basis.npz',allow_pickle=False) as saved:
        V=saved['basis'];nodes=saved['nodes'];initial_operators={key:saved[key] for key in ('mass','stiffness','loads','back')}
    vertex=config.get('freeze_vertex','selected')
    if vertex not in ('initial','selected'):raise ValueError('Unknown freeze material vertex')
    with np.load(output/'selected-operators.npz',allow_pickle=False) as saved:
        M=saved['mass'];K=saved['stiffness'];L=saved['loads'];back=saved['back']
    if vertex=='initial':M,K,L,back=(initial_operators[key] for key in ('mass','stiffness','loads','back'))
    path=input_path(config,'frozen_basis');path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Never replace a frozen artifact')
    np.savez_compressed(path,basis=V,nodes=nodes,mass=M,stiffness=K,loads=L,back=back,directions=V[:,:,24:])
    receipt=dict(schema_version=2,scope='Frozen physics-only union before independent selected-prefix extraction.',operator_vertex=vertex,
        basis=dict(path=config['frozen_basis'],sha256=digest(path),numeric_basis_sha256=hashlib.sha256(V.tobytes()).hexdigest(),rank=V.shape[-1]),
        selected_controls_sha256=config['selected_controls_sha256'],no_mpm_displacement_direction_fitting=True,
        fresh_validation=dict(steps=list(range(31000,42001,1000)),history_interval_steps=[31000,42000],response_gate=.05,omitted_energy_gate=.05) if vertex=='selected' else None,
        inputs={key:dict(path=config[key],sha256=digest(input_path(config,key))) for key in ('initial_material','selected_material','selected_tangent','selected_controls','canonical_selected_basis')},
        material_fields_sha256={label:material_fields_digest(input_path(config,label+'_material')) for label in ('initial','selected')},
        full_mpm_source_signature=config['full_mpm_source_signature'],
        source_sha256=source_receipt(config),construction_configuration={key:value for key,value in config.items() if not key.startswith('_')},
        complete2ms_three_case_mpm_qualification=False,global_optimum_claim=False)
    target=input_path(config,'freeze_receipt')
    if target.exists():raise FileExistsError('Never replace a frozen receipt')
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configuration',type=Path)
    parser.add_argument('stage',choices=('material','tangent','selected','canonicalize','union','freeze'))
    parser.add_argument('--output',type=Path,help='Override the selected/union experiment output directory')
    args=parser.parse_args();path=args.configuration.resolve();config=json.loads(path.read_text(encoding='utf-8'));config['_directory']=str(path.parent)
    check_recipe(config)
    verify_construction_inputs(config,args.stage)
    config['output']=str(args.output.resolve()) if args.output else config[args.stage+'_output'] if args.stage in ('selected','union') else config.get('output','.')
    if args.stage in ('selected','union'):
        output=input_path(config,'output')
        if output.exists() and any(output.iterdir()):raise FileExistsError('Choose an empty experiment output directory')
    {'material':material,'tangent':tangent,'selected':build_selected,'canonicalize':canonicalize,'union':build_union,'freeze':freeze}[args.stage](config)
    if args.stage in ('selected','union'):
        import numpy,scipy
        keys=['selected_material','selected_tangent']
        if args.stage=='union':keys+=['initial_material','canonical_selected_basis']
        artifacts=['rational-basis.npz'] if args.stage=='selected' else ['two-design-basis.npz','selected-operators.npz','fine-response.npz','selected-fine-response.npz','coarse-response.npz','selected-coarse-response.npz']
        receipt=dict(configuration={key:value for key,value in config.items() if not key.startswith('_')},artifact_sha256={name:digest(input_path(config,'output')/name) for name in artifacts},
            input_sha256={key:digest(input_path(config,key)) for key in keys},physical_source_sha256=source_receipt(config),
            runtime=dict(numpy=numpy.__version__,scipy=scipy.__version__),stage=args.stage)
        (input_path(config,'output')/'construction-inputs.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
