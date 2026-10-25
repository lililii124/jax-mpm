"""One frozen actual-input JAX QP comparison; no material or trajectory replay.

The current migration boundary is explicit: reference sparse geometry and LDL
preconditioner construction remain SciPy. Every iterative solve below is JAX;
no restricted146-column Schur construction or SciPy callback is used in JIT.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from tools.metal_affine_projection_numpy import ConsistentAffineProjection
from tools.metal_affine_contact_numpy import CompatibleContact
from tools.metal_affine_dynamic_numpy import moments
from mpm_jax.fragment_v6_matrixfree_projection import sparse_apply,consistent_project,cone_project,intersect_project,quadratic_reconstruction,reconstructed_apply,reconstructed_transpose,polynomial_modes,physical_contact,physical_sample,physical_adjoint


def triangular_blocks(matrix,size=128):
    """Exact block triangular solve preparation; not an approximate objective."""
    n=matrix.shape[0];blocks=(n+size-1)//size;N=blocks*size
    dense=np.zeros((blocks,size,size))
    coo=matrix.tocoo()
    block_rows=coo.row//size;inside=block_rows==coo.col//size
    dense[block_rows[inside],coo.row[inside]%size,coo.col[inside]%size]=coo.data[inside]
    padding_rows=np.arange(n,N);dense[padding_rows//size,padding_rows%size,padding_rows%size]=1.
    outside=~inside;order=np.argsort(block_rows[outside],kind='stable')
    counts=np.bincount(block_rows[outside],minlength=blocks).astype(np.int32);cap=int(counts.max(initial=0));offsets=np.concatenate(([0],np.cumsum(counts[:-1]))).astype(np.int32)
    count=int(np.sum(outside));padding=1<<max(0,(max(cap,1)-1).bit_length())
    row=np.pad((coo.row[outside][order]%size).astype(np.int32),(0,padding));col=np.pad(coo.col[outside][order].astype(np.int32),(0,padding));data=np.pad(coo.data[outside][order],(0,padding))
    dense,row,col,data,offsets,counts=map(jnp.asarray,(dense,row,col,data,offsets,counts))
    lower=bool(np.any(coo.row>coo.col));assert not (lower and np.any(coo.row<coo.col))
    parameters=(dense,row,col,data,offsets,counts)
    def solve(rhs,parameters=parameters):
        dense,row,col,data,offsets,counts=parameters
        padded=jnp.pad(rhs,((0,N-n),(0,0)))
        def make_bucket(width):
            def accumulate(operand):
                x,start,count=operand
                r=jax.lax.dynamic_slice(row,(start,),(width,));c=jax.lax.dynamic_slice(col,(start,),(width,));v=jax.lax.dynamic_slice(data,(start,),(width,))
                v=jnp.where(jnp.arange(width)<count,v,0.)
                return jnp.zeros((size,rhs.shape[1]),rhs.dtype).at[r].add(v[:,None]*x[c])
            return accumulate
        buckets=tuple(make_bucket(1<<i) for i in range(padding.bit_length()))
        def body(i,x):
            b=i if lower else blocks-1-i;offset=b*size
            bucket=32-jax.lax.clz(jnp.maximum(counts[b]-1,0).astype(jnp.uint32))
            correction=jax.lax.switch(bucket,buckets,(x,offsets[b],counts[b]))
            value=jax.lax.dynamic_slice(padded,(offset,0),(size,rhs.shape[1]))-correction
            answer=solve_triangular(dense[b],value,lower=lower)
            return jax.lax.dynamic_update_slice(x,answer,(offset,0))
        return jax.lax.fori_loop(0,blocks,body,jnp.zeros_like(padded))[:n]
    return solve,dict(blocks=blocks,block_size=size,maximum_off_diagonal_entries=cap,actual_off_diagonal_entries=count,old_padded_entries=blocks*cap,exact_triangular_blocks=True),parameters


def main():
    started=time.monotonic();out=Path('/work/result');out.mkdir()
    def load(path):
        with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    parent=Path('/original/result/step-22/accepted.npz');gold_path=Path('/gold/result/midpoint-raw.npz')
    parent_sha=hashlib.file_digest(parent.open('rb'),'sha256').hexdigest();assert parent_sha=='3579600dd7d78cfa728b3478f071d416d15e6226c5799012d37d346aa0530809'
    old=load(parent)
    with np.load(gold_path,allow_pickle=False) as z:gold={k:z[k] for k in ('trial','following','physical','multiplier','normals','geometry_x','geometry_F')}
    args=(old['x'],old['mass'],old['body_id'],.00025,old['F'],old['reference_root'])
    allowed=ConsistentAffineProjection(*args,clamped_particles=old['candidate_fixed_particles'])
    config=dict(old,x=gold['geometry_x'],F=gold['geometry_F'])
    current=ConsistentAffineProjection(config['x'],config['mass'],config['body_id'],.00025,config['F'],config['reference_root'],build_solver=False)
    contact=CompatibleContact(current)
    assert np.array_equal(contact.normal,gold['normals'])
    assert np.array_equal(allowed.tri.perm_r,np.arange(allowed.nodes)) and np.array_equal(allowed.tri.perm_c,np.arange(allowed.nodes))
    upper,pc,upper_parameters=triangular_blocks(allowed.tri.U);lower,_,lower_parameters=triangular_blocks(allowed.tri.U.T)
    root=jnp.asarray(allowed.root)
    right=lambda z:upper(root[:,None]*z)
    rightT=lambda z:root[:,None]*lower(z)
    def sparse(matrix):
        m=matrix.tocoo();data,row,col=map(jnp.asarray,(m.data,m.row.astype(np.int32),m.col.astype(np.int32)))
        parameters=(data,row,col)
        return (lambda x,p=parameters:sparse_apply(p[0],p[1],p[2],x,m.shape[0]),lambda x,p=parameters:sparse_apply(p[0],p[2],p[1],x,m.shape[1]),parameters)
    native_started=time.monotonic()
    native=quadratic_reconstruction(*(jnp.asarray(old[k]) for k in ('x','mass','body_id','F','reference_root')),jnp.asarray(.00025),jnp.asarray(old['candidate_fixed_particles']),jnp.asarray(allowed.node_keys[allowed.order]))
    assert bool(native['valid']), 'Native node roster/finite storage coverage failed; do not solve'
    trial_np=gold['trial'];probe=allowed.Qp.T@trial_np
    q_actual=np.asarray(jax.jit(reconstructed_apply)(native,jnp.asarray(probe)))
    qt_actual=np.asarray(jax.jit(lambda p,y:reconstructed_transpose(p,y,allowed.nodes))(native,jnp.asarray(trial_np)))
    q_error=float(np.linalg.norm(q_actual-allowed.Qp@probe)/np.linalg.norm(allowed.Qp@probe));qt_error=float(np.linalg.norm(qt_actual-probe)/np.linalg.norm(probe))
    assert max(q_error,qt_error)<=1e-12
    native_report=dict(valid=True,free_nodes=int(native['free_node_count']),capacity=65536,mass_error=float(native['mass_error']),forward_relative_difference=q_error,transpose_relative_difference=qt_error,setup_and_probe_seconds=time.monotonic()-native_started,source='JAX quadratic stencil, node discovery, clamp coverage, mass/column metrics and tensor reconstruction; no assembled reconstruction COO in solver')
    T,_,mode_report=jax.jit(polynomial_modes)(native,jnp.asarray(.00025));assert bool(mode_report['valid'])
    full=quadratic_reconstruction(jnp.asarray(config['x']),jnp.asarray(config['mass']),jnp.asarray(config['body_id']),jnp.asarray(config['F']),jnp.asarray(config['reference_root']),jnp.asarray(.00025),jnp.zeros(len(old['mass']),bool))
    currentT,poly,full_mode_report=jax.jit(polynomial_modes)(full,jnp.asarray(.00025));assert bool(full_mode_report['valid'])
    native_contact=physical_contact(full,currentT,poly,jnp.asarray(.00025),native['fixed_grid_keys']);assert bool(native_contact['valid'])
    active_pairs=len(contact.pairs);assert int(native_contact['pair_count'])==active_pairs
    for side in range(2):
        found=np.asarray(full['node_keys'])[np.asarray(native_contact['first' if side==0 else 'second'])[:active_pairs]]
        expected_keys=np.column_stack((contact.node_body[contact.pairs[:,side]],contact.node_coordinates[contact.pairs[:,side]]))
        assert np.array_equal(found,expected_keys)
    sample_native=np.asarray(jax.jit(physical_sample)(full,native_contact,jnp.asarray(trial_np)))[:active_pairs]
    sample_reference=contact.sample(trial_np);sample_error=float(np.linalg.norm(sample_native-sample_reference)/np.linalg.norm(sample_reference))
    multiplier=jnp.pad(jnp.asarray(gold['multiplier']),(0,len(native_contact['first'])-active_pairs))
    adjoint_native=np.asarray(jax.jit(physical_adjoint)(full,native_contact,multiplier))
    adjoint_error=float(np.linalg.norm(adjoint_native+gold['physical'])/np.linalg.norm(gold['physical']))
    mobility_error=float(np.linalg.norm(np.asarray(native_contact['mobility'])[:active_pairs,:active_pairs]-contact.mobility)/np.linalg.norm(contact.mobility))
    assert max(sample_error,adjoint_error,mobility_error)<=1e-11
    contact_report=dict(valid=True,pairs=active_pairs,pair_storage_capacity=len(native_contact['first']),adjacent_particles=int(native_contact['adjacent_particle_count']),particle_storage_capacity=4096,sample_relative_difference=sample_error,adjoint_relative_difference=adjoint_error,mobility_relative_difference=mobility_error,allowed_mode_orthogonality=float(mode_report['orthogonality_error']),physical_mode_orthogonality=float(full_mode_report['orthogonality_error']),source='JAX node pairing, centroid normals, affine modes/correction, local physical mobility, sample and exact adjoint; no reference contact arrays used by solver')
    trial=jnp.asarray(gold['trial']);prepare_seconds=time.monotonic()-started
    nodes=allowed.nodes
    parameters=dict(upper=upper_parameters,lower=lower_parameters,reconstruction=native,full_reconstruction=full,contact=native_contact,root=root,T=T)
    def kernel(y,p):
        q=lambda x:reconstructed_apply(p['reconstruction'],x);qt=lambda x:reconstructed_transpose(p['reconstruction'],x,nodes)
        right=lambda x:upper(p['root'][:,None]*x,p['upper'])
        rightT=lambda x:p['root'][:,None]*lower(x,p['lower'])
        sample=lambda x:physical_sample(p['full_reconstruction'],p['contact'],x)
        adjoint=lambda lam:physical_adjoint(p['full_reconstruction'],p['contact'],lam)
        project_range=lambda x:consistent_project(x,q,qt,p['T'],right,rightT,iterations=64)
        project_cone=lambda x:cone_project(x,sample,adjoint,p['contact']['mobility'],sweeps=512,tolerance=1e-7,physical_rows=p['contact']['pair_count'])
        return intersect_project(y,project_range,project_cone,sample,iterations=64)
    solve=jax.jit(kernel)
    print(json.dumps(dict(phase='jax-compiling',prepare_seconds=prepare_seconds,particles=len(old['mass']),preconditioner=pc,scope='SciPy preparation; pure JAX iterative kernel, not all-JAX impact integration')),flush=True)
    # Large operator arrays are dynamic inputs: prevent XLA constant folding
    # from replicating the padded triangular workspaces during compilation.
    del allowed
    import gc;gc.collect()
    begun=time.monotonic();compiled=solve.lower(trial,parameters).compile();compile_seconds=time.monotonic()-begun
    begun=time.monotonic();following,report=compiled(trial,parameters);following.block_until_ready();solve_seconds=time.monotonic()-begun
    following=np.asarray(following);values={k:np.asarray(v) for k,v in report.items()};physical=values.pop('physical_impulse');reaction=values.pop('boundary_reaction');lam=values.pop('multiplier')
    report={k:v.item() for k,v in values.items()};relative=float(np.linalg.norm(following-gold['following'])/np.linalg.norm(gold['following']))
    native_values=contact.sample(following);assert np.all(lam[active_pairs:]==0)
    kkt=float(np.max(np.where(lam[:active_pairs]>0,abs(native_values),np.maximum(native_values,0)),initial=0))
    dualphysical=-contact.adjoint(lam[:active_pairs]);dualdifference=float(np.linalg.norm(physical-dualphysical))
    P,L=moments(config,2*physical);boundarywork=float(np.sum(following*2*reaction));contactwork=float(np.sum(following*2*physical))
    tol=1.414599609375e-8;torquetol=1.5782228774332438e-14
    passed=bool(report['converged'] and relative<=1e-10 and kkt<=.005 and abs(boundarywork)<=tol and contactwork<=tol and np.linalg.norm(L)<=torquetol)
    np.savez_compressed(out/'jax-raw.npz',trial=np.asarray(trial),following=following,physical=physical,reaction=reaction,multiplier=lam)
    result=dict(status='passed' if passed else 'failed',parent_checkpoint_sha256=parent_sha,golden_npz_sha256=hashlib.file_digest(gold_path.open('rb'),'sha256').hexdigest(),source_input='Original accepted22 and already-solved same frozen qmid golden; no replay',particles=len(old['mass']),prepare_seconds=prepare_seconds,compile_seconds=compile_seconds,solve_seconds=solve_seconds,relative_velocity_difference=relative,original_KKT_m_s=kkt,physical_dual_difference=dualdifference,physical_momentum=P.tolist(),physical_torque=L.tolist(),boundary_work_J=boundarywork,contact_work_J=contactwork,report=report,preconditioner=pc,native_reconstruction=native_report,native_contact=contact_report,migration_boundary='Reconstruction, centroid/affine contact/mobility preparation, polynomial modes and iterative solves JAX64. Exact sparse LDL preconditioner setup remains reference-prepared. No restricted146-column Schur or JIT callbacks. Not yet installed as an all-JAX impact step.')
    (out/'review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    assert passed,'Bounded frozen-input migration failed; preserve raw and diagnose, do not integrate'


if __name__=='__main__':main()
