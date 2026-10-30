"""Read-only force and projection decomposition on already-seen q949 frames."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from scipy import sparse

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
L=ROOT/'.local';D=L/'q949-extended-projection-diagnostic'
MODEL=L/'q949-vertex-extension-v1-20261030'/'output'
FRAMES={'uniform':39000,'central':38000}
Q='94934d1d5792131b22756f3e2634d25708de07ade1a0ff37610881f8bb1b6ec4'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def pair_energy(K,x):return float(.5*np.dot(x,K@x))

def force_decomposition(name,step):
    from tools.build_sandwich_rational import physics_support,material_fields_digest
    from tools.export_sandwich_fullmpm_face import recover_nodes
    from examples.blast_sandwich_design import Settings
    freeze=json.loads((MODEL/'model-freeze.json').read_text())
    for source,digest in freeze['source_sha256'].items():assert sha(ROOT/source)==digest,source
    frame_path=D/f'{name}-{step}.npz'
    with np.load(frame_path,allow_pickle=False) as f:
        reference=f['reference'].copy();disp=f['displacement'].copy();F=f['F'].copy()
        young=f['young'].copy();skin=f['skin'].astype(bool);volume=f['volume'].copy();time=float(f['time'])
    settings=Settings(xy_step=.01,z_step=.001,controls=(24,24,3),stage='fixed-mass')
    _,geo,support,namespace=physics_support()
    _,axes,u,recovery=recover_nodes(reference,disp,np.asarray((.01,.01,.001)))
    origin=np.floor(reference/np.asarray((.01,.01,.001))-.5).astype(np.int64).min(axis=0)
    canonical=[(np.arange(len(axis))+origin[a])*.001 if a==2 else (np.arange(len(axis))+origin[a])*.01 for a,axis in enumerate(axes)]
    nodes=np.stack(np.meshgrid(*canonical,indexing='ij'),axis=-1).reshape(-1,3)
    with np.load(MODEL/'frozen-basis.npz',allow_pickle=False) as s:
        V=s['basis'].reshape(-1,s['basis'].shape[-1]);frozen_nodes=s['nodes'].copy()
    np.testing.assert_array_equal(nodes,frozen_nodes);np.testing.assert_array_equal(support.nodes,frozen_nodes)
    np.testing.assert_array_equal(reference,support.positions)
    with np.load(MODEL/'material.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(young,f['young']);np.testing.assert_array_equal(skin,f['skin'])
        rhoq=f['density'].copy()
    Kfull=sparse.load_npz(MODEL/'tangent.npz').tocsr()
    _,_,fixed=namespace['clamped_basis'](support.nodes,settings,2)
    unodes=u.reshape(-1,3);ids=np.flatnonzero(np.repeat(~fixed,3));u_flat=unodes.reshape(-1)
    # Initial optimizer q0 is all zeros in fixed-mass mode, which gives the
    # uniform target core fraction; skin density is 2700 kg/m^3. Verify total
    # M0 against the frozen mass target before using it for projection.
    rho0=np.where(geo['skin'],2700.,2700.*settings.target_phi)
    mass0=np.zeros(len(nodes));np.add.at(mass0,support.indices.ravel(),
        (support.weights*(support.volume*rho0)[:,None]).ravel())
    assert abs(mass0.sum()/1.8468-1)<1e-10, mass0.sum()
    m0=np.repeat(mass0,3)
    assert np.max(np.abs(V.T@(m0[:,None]*V)-np.eye(V.shape[-1])))<1e-9
    a0=V.T@(m0*u_flat);r0=u_flat-V@a0
    KV=Kfull@V;Ku=np.asarray(Kfull@u_flat).reshape(-1)
    Kr=V.T@KV;g=V.T@Ku;aK=np.linalg.solve(Kr,g);rK=u_flat-V@aK
    Etotal=pair_energy(Kfull,u_flat);E0=pair_energy(Kfull,r0);EK=pair_energy(Kfull,rK)
    # The independent particle-strain expression is the same ledger used by
    # assess_sandwich_rational.omitted_energy.
    grad=np.einsum('pki,pkj->pij',unodes[support.indices],support.gradients)
    F_recovered=np.eye(3)[None,:,:]+grad
    eps=.5*(grad+grad.swapaxes(1,2));nu=np.where(skin,.33,.25)
    mu=young/(2*(1+nu));lam=young*nu/((1+nu)*(1-2*nu))
    trace=np.trace(eps,axis1=1,axis2=2)
    energy_density=mu*np.sum(eps*eps,axis=(1,2))+.5*lam*trace**2
    Eparticle=float(np.sum(support.volume*energy_density))
    particle_fraction=float(1-.5*g@np.linalg.solve(Kr,g)/Eparticle)
    # Nonlinear TL internal force from the exact saved F. This is a constitutive
    # postprocess at a saved state, with no new MPM step, tangent, or AD solve.
    J=np.linalg.det(F);assert np.all(J>0)
    eye=np.eye(3);B=F@F.swapaxes(1,2)
    tau=mu[:,None,None]*(B-eye)+lam[:,None,None]*np.log(J)[:,None,None]*eye
    cof=np.stack((np.cross(F[:,:,1],F[:,:,2]),np.cross(F[:,:,2],F[:,:,0]),np.cross(F[:,:,0],F[:,:,1])),axis=-1)
    P=(tau@cof)/J[:,None,None]
    local=-support.volume[:,None,None]*np.einsum('pij,pkj->pki',P,support.gradients)
    fnl=np.zeros((len(nodes),3));np.add.at(fnl,support.indices,local)
    sigma_linear=2*mu[:,None,None]*eps+lam[:,None,None]*trace[:,None,None]*eye
    local_linear=-support.volume[:,None,None]*np.einsum('pij,pkj->pki',sigma_linear,support.gradients)
    flin=np.zeros((len(nodes),3));np.add.at(flin,support.indices,local_linear)
    ku=np.asarray(Kfull@u.reshape(-1)).reshape(-1,3)
    fn=fnl.reshape(-1)[ids];flinear=flin.reshape(-1)[ids];linear=-ku.reshape(-1)[ids]
    force_error=fn-linear
    nonlinear_energy=float(np.sum(support.volume*(.5*mu*(np.sum(F*F,axis=(1,2))-3)
        -mu*np.log(J)+.5*lam*np.log(J)**2)))
    # Split projected residual by displacement component, with all K cross terms
    # reported explicitly so components are not incorrectly treated as additive.
    comps={}
    for label,resid in (('M0',r0),('K',rK)):
        xyz=[np.zeros_like(u_flat) for _ in range(3)]
        for node in range(len(nodes)):
            for c in range(3):xyz[c][3*node+c]=resid[3*node+c]
        energies=[pair_energy(Kfull,x) for x in xyz]
        crosses=[[float(xyz[i]@(Kfull@xyz[j])) for j in range(i+1,3)] for i in range(3)]
        comps[label]=dict(axis_residual_energy_J=dict(zip(('ux','uy','uz'),energies)),
            cross_energy_J={'ux_uy':crosses[0][0],'ux_uz':crosses[0][1],'uy_uz':crosses[1][0]},
            residual_M0_norm_relative=float(np.sqrt(np.dot(m0*resid,resid)/np.dot(m0*u_flat,u_flat))))
    # Exact nodal pairing under the thickness midplane; report coordinate-even/
    # odd content by component, without assuming either parity is a boundary condition.
    node_index={tuple(np.rint(x*1e10).astype(np.int64)):i for i,x in enumerate(nodes)}
    mirror=[];zmid=.012
    for x in nodes:
        key=tuple(np.rint(np.array((x[0],x[1],2*zmid-x[2]))*1e10).astype(np.int64))
        mirror.append(node_index.get(key,-1))
    mirror=np.asarray(mirror,dtype=int);parity=None
    if np.all(mirror>=0):
        parity={}
        for label,resid in (('displacement',unodes),('M0_residual',np.zeros((len(nodes),3))),('K_residual',np.zeros((len(nodes),3)))):
            if label=='M0_residual':resid.reshape(-1)[:]=r0
            if label=='K_residual':resid.reshape(-1)[:]=rK
            even=.5*(resid+resid[mirror]);odd=.5*(resid-resid[mirror])
            parity[label]={}
            for c,namec in enumerate(('x_in_plane','y_in_plane','z_thickness')):
                parity[label][namec]=dict(even_m0_norm=float(np.sqrt(np.sum(mass0*even[:,c]**2))),
                    odd_m0_norm=float(np.sqrt(np.sum(mass0*odd[:,c]**2))))
        residual=rK.reshape(-1,3).copy()
        parity_energy={}
        for parity_name,part in (('midplane_even',.5*(residual+residual[mirror])),
                                 ('midplane_odd',.5*(residual-residual[mirror]))):
            xyz=[]
            for c in range(3):
                field=np.zeros_like(part);field[:,c]=part[:,c]
                xyz.append(field.reshape(-1))
            axis_energy=[pair_energy(Kfull,x) for x in xyz]
            parity_energy[parity_name]=dict(axis_K_energy_J=dict(zip(('ux','uy','uz'),axis_energy)),
                in_plane_ux_uy_energy_J=pair_energy(Kfull,xyz[0]+xyz[1]),
                cross_energy_J={'ux_uy':float(xyz[0]@(Kfull@xyz[1])),
                    'ux_uz':float(xyz[0]@(Kfull@xyz[2])),'uy_uz':float(xyz[1]@(Kfull@xyz[2]))})
        parity['K_residual_energy']=parity_energy
        even=.5*(residual+residual[mirror]);odd=.5*(residual-residual[mirror])
        membrane=np.column_stack((even[:,0],even[:,1],odd[:,2])).reshape(-1)
        bending=np.column_stack((odd[:,0],odd[:,1],even[:,2])).reshape(-1)
        em=pair_energy(Kfull,membrane);eb=pair_energy(Kfull,bending)
        cross_mb=float(membrane@(Kfull@bending));closure=em+eb+cross_mb
        def independently_projected_part(part):
            gpart=V.T@(Kfull@part);coef=np.linalg.solve(Kr,gpart);rest=part-V@coef
            return dict(absolute_omitted_energy_J=pair_energy(Kfull,rest),
                omitted_energy_fraction=float(pair_energy(Kfull,rest)/Etotal),
                projected_coefficient_K_norm=float(np.sqrt(max(0.,gpart@np.linalg.solve(Kr,gpart)))))
        parity['membrane_bending_K_energy']=dict(
            definition='membrane=(ux even, uy even, uz odd); bending=(ux odd, uy odd, uz even)',
            membrane_energy_J=em,bending_energy_J=eb,membrane_bending_cross_energy_J=cross_mb,
            reconstructed_energy_J=closure,total_K_residual_energy_J=EK,
            energy_closure_abs_error_J=float(abs(closure-EK)),
            energy_closure_relative_error=float(abs(closure-EK)/max(EK,np.finfo(float).tiny)),
            vector_closure_l2=float(np.linalg.norm(membrane+bending-rK)),
            membrane_independent_K_projection_residual=independently_projected_part(membrane),
            bending_independent_K_projection_residual=independently_projected_part(bending))
    result=dict(case=name,step=step,time_s=time,frame_sha256=sha(frame_path),
        controls_sha256=freeze['selected_controls_sha256'],freeze_receipt_sha256=sha(MODEL/'model-freeze.json'),
        basis_rank=freeze['basis']['rank'],mpm_recovery=recovery,
        total_linear_strain_energy_J=Eparticle,tangent_quadratic_energy_J=Etotal,
        tangent_vs_particle_energy_relative_difference=float(abs(Etotal-Eparticle)/Eparticle),
        M0_projection=dict(residual_energy_J=E0,residual_energy_fraction=float(E0/Etotal),
            residual_relative_m0_norm=float(np.sqrt(np.dot(m0*r0,r0)/np.dot(m0*u_flat,u_flat)))),
        K_projection=dict(absolute_omitted_energy_J=EK,omitted_energy_fraction=float(EK/Etotal),
            assessor_style_omitted_fraction=particle_fraction),
        residual_components=comps,thickness_parity=parity,
        nonlinear=dict(actual_elastic_energy_J=nonlinear_energy,
            nonlinear_minus_linear_energy_J=float(nonlinear_energy-Eparticle),
            max_abs_F_minus_I=float(np.max(np.abs(F-eye),initial=0)),
            max_abs_F_minus_recovered_gradient=float(np.max(np.abs(F-F_recovered),initial=0)),
            relative_F_recovery_l2=float(np.linalg.norm(F-F_recovered)/max(np.linalg.norm(F-eye),np.finfo(float).tiny)),
            max_abs_J_minus_1=float(np.max(np.abs(J-1),initial=0)),
            internal_force_norm_N=float(np.linalg.norm(fn)),linear_Ku_force_norm_N=float(np.linalg.norm(linear)),
            internal_force_minus_linear_force_norm_N=float(np.linalg.norm(force_error)),
            relative_force_residual=float(np.linalg.norm(force_error)/max(np.linalg.norm(fn),np.finfo(float).tiny)),
            relative_to_linear_force=float(np.linalg.norm(force_error)/max(np.linalg.norm(linear),np.finfo(float).tiny)),
            direct_small_strain_force_norm_N=float(np.linalg.norm(flinear)),
            nonlinear_vs_direct_small_strain_force_relative=float(np.linalg.norm(fn-flinear)/max(np.linalg.norm(flinear),np.finfo(float).tiny)),
            tangent_vs_direct_small_strain_force_relative=float(np.linalg.norm(linear-flinear)/max(np.linalg.norm(flinear),np.finfo(float).tiny)),
            particle_local_nonlinear_vs_small_strain_force_relative=float(np.linalg.norm(local-local_linear)/max(np.linalg.norm(local_linear),np.finfo(float).tiny)),
            per_axis_force_residual_N=np.linalg.norm(force_error.reshape(-1,3),axis=0).tolist()))
    return result

def main():
    records=[force_decomposition(case,step) for case,step in FRAMES.items()]
    freeze=json.loads((MODEL/'model-freeze.json').read_text())
    out=dict(scope='Read-only postprocessing of previously viewed development frames; no MPM stepping, fitting, or model changes.',
        q949_controls_sha256=Q,freeze_receipt_sha256=sha(MODEL/'model-freeze.json'),
        diagnostic_script_sha256=sha(__file__),model_source_sha256=freeze['source_sha256'],
        M0_definition='Uniform initial fixed-mass q0=0: target core phi=0.18, skin density=2700 kg/m^3; nodal mass total checked against 1.8468 kg.',
        original_5_percent_gate_unchanged=True,frames=records)
    path=D/'vertex184-projection-nonlinear-force-diagnostic.json'
    path.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'path':str(path),'frames':[{k:r[k] for k in ('case','step','M0_projection','K_projection','nonlinear')} for r in records]},indent=2))

if __name__=='__main__':main()
