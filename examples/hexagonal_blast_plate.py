"""Circular clamped sixfold hyperelastic plate under prescribed blast pressure.

Finite deformation total-Lagrangian MPM, quadratic grid, nodal velocity Verlet.
Pressure is a prescribed reference-area vertical traction, not a fluid model.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='job')
    p.add_argument('--export', action='store_true', help='Export existing frames to native VTK/PVD without JAX')
    p.add_argument('--spacing', type=float, default=.006)
    p.add_argument('--horizon', type=float, default=.006)
    p.add_argument('--pressure', type=float, default=3.e5)
    p.add_argument('--angle', type=float, default=0.)
    p.add_argument('--kf', type=float, default=5.e6)
    p.add_argument('--dt', type=float, default=1.e-6)
    p.add_argument('--wall-seconds', type=float, default=780.)
    args=p.parse_args()
    if args.export:
        export_frames(Path(args.out))
        return
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as j
    from mpm_jax.compact_tl import rectilinear_support, reference_grid, surface_weights
    from mpm_jax.total_lagrangian import deformation, sample_nodal
    from mpm_jax.hexagonal_hyperelastic import energy_density, first_piola
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    h=args.spacing;hz=.0015;radius=.15;thickness=.006;rho=1100.
    xy=np.arange(-radius+h/4,radius,h/2)
    xx,yy=np.meshgrid(xy,xy,indexing='ij')
    face=np.column_stack((xx.ravel(),yy.ravel()))
    face=face[np.sum(face**2,axis=1)<radius**2]
    z=np.arange(-thickness/2+hz/4,thickness/2,hz/2)
    X=np.column_stack((np.repeat(face,len(z),axis=0),np.tile(z,len(face))))
    support=rectilinear_support(X,(h/2)**2*hz/2,(h,h,hz))
    ref=reference_grid(support,rho)
    fixed=j.asarray((np.linalg.norm(support.nodes[:,:2],axis=1)>radius-.015)|(np.asarray(ref.nodal_mass)<1.e-14))[:,None]
    face3=np.column_stack((face,np.full(len(face),thickness/2)))
    si,sw=surface_weights(support,face3)
    load=np.zeros_like(support.nodes)
    profile=np.exp(-np.sum(face**2,axis=1)/(2*.035**2))*(h/2)**2
    np.add.at(load[:,2],si.ravel(),(-profile[:,None]*sw).ravel())
    load=j.asarray(load);M=ref.nodal_mass[:,None];inv=1/j.where(M>1.e-14,M,1.)
    params=dict(kf=args.kf,angle=np.deg2rad(args.angle))
    energies=jax.vmap(lambda f:energy_density(f,**params))
    piolas=jax.vmap(lambda f:first_piola(f,**params))
    def pressure(t):
        return args.pressure*j.where(t<.0005,j.sin(j.pi*t/.0005)**2,0.)
    def force(u,t):
        F=deformation(u,ref);P=piolas(F)
        local=-ref.volume[:,None,None]*j.einsum('pij,pkj->pki',P,ref.gradients)
        return j.zeros_like(u).at[ref.indices].add(local)+pressure(t)*load
    dt=args.dt
    @jax.jit
    def block(state,n):
        def step(_,s):
            u,v,f,t,work=s
            half=j.where(fixed,0.,v+.5*dt*f*inv)
            un=j.where(fixed,0.,u+dt*half)
            tn=t+dt;fn=force(un,tn)
            vn=j.where(fixed,0.,half+.5*dt*fn*inv)
            work=work+.5*(pressure(t)+pressure(tn))*j.sum(load*(un-u))
            return un,vn,fn,tn,work
        return jax.lax.fori_loop(0,n,step,state)
    @jax.jit
    def fields(state):
        u,v,f,t,work=state;F=deformation(u,ref);P=piolas(F);J=j.linalg.det(F)
        sigma=P@j.swapaxes(F,1,2)/J[:,None,None]
        dev=sigma-j.trace(sigma,axis1=1,axis2=2)[:,None,None]*j.eye(3)/3
        vm=j.sqrt(1.5*j.sum(dev*dev,axis=(1,2)))
        E=.5*(j.swapaxes(F,1,2)@F-j.eye(3))
        return sample_nodal(u,ref),vm,sigma,J,j.max(j.abs(E)),j.sum(ref.volume*energies(F)),.5*j.sum(M*v*v)
    state=(j.zeros_like(ref.nodes),j.zeros_like(ref.nodes),j.zeros_like(ref.nodes),j.asarray(0.),j.asarray(0.))
    start_step=0;checkpoint=out/'checkpoint.npz'
    root=Path(__file__).resolve().parents[1]
    sources={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
             for path in [Path(__file__).resolve(),root/'mpm_jax/hexagonal_hyperelastic.py',root/'mpm_jax/compact_tl.py',root/'mpm_jax/total_lagrangian.py']}
    binding=dict(physics={k:getattr(args,k) for k in ('spacing','pressure','angle','kf','dt')},
                 geometry=dict(radius=radius,thickness=thickness,hz=hz,rho=rho),sources=sources,
                 runtime=dict(jax=jax.__version__,backend=jax.default_backend(),dtype=str(ref.nodes.dtype)))
    binding_json=json.dumps(binding,sort_keys=True)
    binding_sha256=hashlib.sha256(binding_json.encode()).hexdigest()
    if checkpoint.exists():
        previous=json.loads((out/'settings.json').read_text())
        for key in ('spacing','pressure','angle','kf','dt'):
            if previous[key]!=getattr(args,key):raise ValueError('Resume parameter mismatch: '+key)
        with np.load(checkpoint) as q:
            if 'binding_sha256' not in q or str(q['binding_sha256'])!=binding_sha256:
                raise ValueError('Checkpoint physics/source/runtime binding mismatch')
            state=tuple(j.asarray(q[k]) for k in ('u','v','f','t','work'));start_step=int(q['step'])
            if abs(float(state[3])-start_step*dt)>1.e-10:raise ValueError('Checkpoint time mismatch')
    settings=vars(args)|dict(radius_m=radius,thickness_m=thickness,particle_count=len(X),node_count=len(ref.nodes),rho=rho)
    (out/'binding.json').write_text(binding_json)
    (out/'settings.json').write_text(json.dumps(settings,indent=2))
    nsteps=int(np.ceil(args.horizon/dt));stride=max(1,int(.0001/dt));began=time.monotonic()
    rows=[]
    previous_step=start_step
    schedule=sorted(set(list(range(start_step,nsteps+1,stride))+[nsteps]))
    for step in schedule:
        if step>previous_step:state=block(state,j.asarray(step-previous_step))
        previous_step=step
        disp,vm,sigma,J,strain,U,K=map(np.asarray,fields(state));t=float(state[3]);work=float(state[4])
        residual=float(U+K-work)
        energy_limit=2.e-3*max(abs(work),float(U+K),1.e-3)
        valid=bool(np.isfinite(vm).all() and np.isfinite(sigma).all() and np.isfinite(disp).all()
                   and all(np.isfinite(np.asarray(v)).all() for v in state)
                   and np.isfinite([U,K,work]).all() and J.min()>0 and abs(residual)<energy_limit)
        row=dict(step=step,time_s=t,min_J=float(J.min()),max_strain=float(strain),peak_vm_Pa=float(vm.max()),max_deflection_m=float(np.max(abs(disp[:,2]))),elastic_J=float(U),kinetic_J=float(K),pressure_work_J=work,energy_residual_J=float(U+K-work),valid=valid)
        rows.append(row);print(json.dumps(row),flush=True)
        if not valid:
            (out/'failure.json').write_text(json.dumps(row,indent=2))
            raise RuntimeError('Invalid deformation or energy ledger; previous valid checkpoint retained')
        np.savez_compressed(out/f'frame_{step:07d}.npz',position=X+disp,reference=X,displacement=disp,von_mises=vm,cauchy=sigma,time=t)
        np.savez(out/'checkpoint.tmp.npz',u=state[0],v=state[1],f=state[2],t=state[3],work=state[4],step=step,binding_sha256=binding_sha256)
        (out/'checkpoint.tmp.npz').replace(checkpoint)
        (out/'solver_state.json').write_text(json.dumps(row|dict(complete=step>=nsteps,elapsed_s=time.monotonic()-began),indent=2))
        with (out/'history.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        if not valid:raise RuntimeError('Invalid deformation; retained last full state')
        if time.monotonic()-began>args.wall_seconds:break



def export_frames(out, destination=None):
    """Native ParaView triangular upper material surface; no synthetic fields."""
    import base64
    import struct
    frames=sorted(out.glob('frame_*.npz'))
    if not frames:raise ValueError('No computed frames to export')
    target=out if destination is None else destination
    target.mkdir(parents=True,exist_ok=True)
    dest=target/'vtk';dest.mkdir(exist_ok=True)
    with np.load(frames[0]) as q:
        X=q['reference'];mask=np.isclose(X[:,2],X[:,2].max(),atol=1.e-12)
        top=X[mask]
        xs=np.unique(top[:,0]);ys=np.unique(top[:,1]);lookup=np.full((len(xs),len(ys)),-1,dtype=np.int32)
        lookup[np.searchsorted(xs,top[:,0]),np.searchsorted(ys,top[:,1])]=np.arange(len(top))
        triangles=[]
        for i in range(len(xs)-1):
            for k in range(len(ys)-1):
                corners=[lookup[i,k],lookup[i+1,k],lookup[i+1,k+1],lookup[i,k+1]]
                present=[n for n in corners if n>=0]
                if len(present)==4:triangles.extend(([present[0],present[1],present[2]],[present[0],present[2],present[3]]))
                elif len(present)==3:triangles.append(present)
        tri=np.asarray(triangles,dtype='<i4')
    def array(name,value,kind='Float64',components=1):
        dtype={'Float64':'<f8','Int32':'<i4','UInt8':'u1'}[kind]
        raw=np.asarray(value,dtype=dtype).tobytes()
        encoded=base64.b64encode(struct.pack('<I',len(raw))+raw).decode('ascii')
        return f'<DataArray type="{kind}" Name="{name}" NumberOfComponents="{components}" format="binary">{encoded}</DataArray>'
    records=[];metrics=[]
    for path in frames:
        with np.load(path) as q:
            x=q['position'][mask];vm=q['von_mises'][mask];n=len(x);nc=len(tri)
            fields=array('von_mises_Pa',vm)+array('displacement_m',q['displacement'][mask],components=3)
            fields+=array('Cauchy_Pa',q['cauchy'][mask],components=9)
            text='<?xml version="1.0"?><VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian"><UnstructuredGrid>'
            text+=f'<Piece NumberOfPoints="{n}" NumberOfCells="{nc}"><PointData>'+fields+'</PointData><Points>'
            text+=array('position',x,components=3)+'</Points><Cells>'
            text+=array('connectivity',tri,'Int32')+array('offsets',np.arange(1,nc+1)*3,'Int32')+array('types',np.full(nc,5),'UInt8')
            text+='</Cells></Piece></UnstructuredGrid></VTKFile>'
            name=path.stem+'.vtu';(dest/name).write_text(text)
            t=float(q['time']);records.append(f'<DataSet timestep="{t:.16g}" group="" part="0" file="vtk/{name}"/>')
            r=np.linalg.norm(top[:,:2],axis=1);a=np.arctan2(top[:,1],top[:,0]);m=abs(r-.05)<.0025
            cols=[np.ones(m.sum()),r[m]-.05,(r[m]-.05)**2]
            for k in (2,4,6,8,12):cols.extend([np.cos(k*a[m]),np.sin(k*a[m])])
            b=np.linalg.lstsq(np.array(cols).T,vm[m],rcond=None)[0]
            metrics.append(dict(frame=path.name,surface_vm_sha256=hashlib.sha256(np.asarray(vm,dtype='<f8').tobytes()).hexdigest(),time_s=t,ring_radius_m=.05,ring_halfwidth_m=.0025,mean_Pa=float(b[0]),a4_relative=float(np.hypot(*b[5:7])/max(b[0],1.e-30)),a6_relative=float(np.hypot(*b[7:9])/max(b[0],1.e-30)),a6_phase_deg=float(np.rad2deg(np.arctan2(b[8],b[7]))/6),peak_surface_vm_Pa=float(vm.max())))
    (target/'surface.pvd').write_text('<?xml version="1.0"?><VTKFile type="Collection" version="0.1" byte_order="LittleEndian"><Collection>'+''.join(records)+'</Collection></VTKFile>')
    (target/'angular_metrics.json').write_text(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
