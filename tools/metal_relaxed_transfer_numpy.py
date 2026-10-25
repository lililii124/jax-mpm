"""NumPy complete-map relaxation proof/reference; no physical continuation."""
import numpy as np
from tools.metal_split_transfer_numpy import remap


def relaxed_remap(x,v,C,mass,dx,frozen,*,dt,relaxation_time,increment=None):
    if not np.isfinite(relaxation_time) or relaxation_time<=0 or not np.isfinite(dt) or dt<0: raise ValueError('Finite nonnegative dt and positive frozen tau required')
    av,aC,apic=remap(x,v,C,mass,dx,frozen,mode='APIC',increment=increment)
    fv,fC,_=remap(x,v,C,mass,dx,frozen,mode='DELTA',increment=increment)
    alpha=float(np.exp(-dt/relaxation_time));nv=(1-alpha)*av+alpha*fv;nC=(1-alpha)*aC+alpha*fC
    kinetic=lambda vv,cc:float(.5*np.sum(mass[:,None]*vv*vv)+dx**2/8*np.sum(mass[:,None,None]*cc*cc))
    ka,kf,kb,kold=kinetic(av,aC),kinetic(fv,fC),kinetic(nv,nC),kinetic(v,C)
    variance=alpha*(1-alpha)*kinetic(fv-av,fC-aC)
    w,r,ids,wm,gm=frozen
    old_particle_field=v[:,None,:]+np.einsum('pij,pkj->pki',C,r)
    gp=np.stack([np.bincount(ids.ravel(),weights=(wm*old_particle_field[:,:,a]).ravel(),minlength=len(gm)) for a in range(3)],axis=-1)
    old=gp/np.where(gm>0,gm,1)[:,None];final=old if increment is None else old+increment
    torque=np.sum(wm[:,:,None]*np.cross(x[:,None,:]+r,final[ids]-old[ids]),axis=(0,1))
    spin=lambda cc:np.stack((cc[:,2,1]-cc[:,1,2],cc[:,0,2]-cc[:,2,0],cc[:,1,0]-cc[:,0,1]),axis=-1)
    angular=lambda vv,cc:np.sum(mass[:,None]*(np.cross(x,vv)+dx**2/4*spin(cc)),axis=0)
    angular_defect=angular(nv,nC)-angular(v,C)-torque
    impulse=np.sum(gm[:,None]*(final-old),axis=0)
    momentum_defect=np.sum(mass[:,None]*(nv-v),axis=0)-impulse
    correction=apic['g2p_signed_J']+ka-kb
    return nv,nC,dict(alpha=alpha,p2g_J=apic['p2g_J'],g2p_signed_J=correction,particle_delta_J=kb-kold,grid_work_J=apic['grid_work_J'],
        blending_variance_J=variance,convex_identity_residual_J=kb-(1-alpha)*ka-alpha*kf+variance,
        ledger_identity_residual_J=kb-kold+apic['p2g_J']+correction-apic['grid_work_J'],angular_defect=angular_defect,momentum_defect=momentum_defect)
