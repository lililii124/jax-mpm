"""Unadmitted sym-APIC/skew-DELTA reference, including its energy hazard."""
import numpy as np
from tools.metal_split_transfer_numpy import remap


def sym(a):
    return .5*(a+np.swapaxes(a,-1,-2))


def sym_delta_remap(x,v,C,mass,dx,frozen,increment=None):
    _,aC,apic=remap(x,v,C,mass,dx,frozen,mode='APIC',increment=increment)
    nv,dC,delta=remap(x,v,C,mass,dx,frozen,mode='DELTA',increment=increment)
    nC=sym(aC)+.5*(dC-np.swapaxes(dC,-1,-2))
    kinetic=lambda vv,cc:float(.5*np.sum(mass[:,None]*vv*vv)+dx**2/8*np.sum(mass[:,None,None]*cc*cc))
    w,r,ids,wm,gm=frozen
    pv=v[:,None,:]+np.einsum('pij,pkj->pki',C,r)
    gp=np.stack([np.bincount(ids.ravel(),weights=(wm*pv[:,:,a]).ravel(),minlength=len(gm)) for a in range(3)],axis=-1)
    old=gp/np.where(gm>0,gm,1)[:,None]
    change=np.zeros_like(old) if increment is None else increment
    final=old+change
    spin=lambda cc:np.stack((cc[:,2,1]-cc[:,1,2],cc[:,0,2]-cc[:,2,0],cc[:,1,0]-cc[:,0,1]),axis=-1)
    angular=lambda vv,cc:np.sum(mass[:,None]*(np.cross(x,vv)+dx**2/4*spin(cc)),axis=0)
    torque=np.sum(wm[:,:,None]*np.cross(x[:,None,:]+r,change[ids]),axis=(0,1))
    impulse=np.sum(gm[:,None]*change,axis=0)
    kp0,kp1,kd=kinetic(v,C),kinetic(nv,nC),kinetic(nv,dC)
    kg0=float(.5*np.sum(gm[:,None]*old*old));kg1=float(.5*np.sum(gm[:,None]*final*final))
    specific_sym_change=dx**2/8*np.sum(mass[:,None,None]*(sym(aC)**2-sym(dC)**2))
    work=float(np.sum(gm[:,None]*change*.5*(old+final)))
    correction=kg1-kp1
    return nv,nC,dict(momentum_defect=np.sum(mass[:,None]*(nv-v),axis=0)-impulse,
        angular_defect=angular(nv,nC)-angular(v,C)-torque,
        p2g_J=kp0-kg0,g2p_signed_J=correction,particle_delta_J=kp1-kp0,
        grid_work_J=work,ledger_identity_residual_J=kp1-kp0+(kp0-kg0)+correction-work,
        symmetric_affine_change_from_delta_J=float(specific_sym_change),
        orthogonal_identity_residual_J=float(kp1-kd-specific_sym_change),
        correction_identity_residual_J=float(correction-(delta['g2p_signed_J']-specific_sym_change)))
