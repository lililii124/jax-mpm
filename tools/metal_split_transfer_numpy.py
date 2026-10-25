"""NumPy reference for frozen-support split-transfer identities only."""
import numpy as np


def remap(x,v,C,mass,dx,support,*,mode,increment=None):
    if mode not in ('APIC','DELTA','delta_v_apic_C','apic_v_delta_C'): raise ValueError(mode)
    w,r,ids,wm,gm=support
    pv=v[:,None,:]+np.einsum('pij,pkj->pki',C,r)
    gp=np.stack([np.bincount(ids.ravel(),weights=(wm*pv[:,:,a]).ravel(),minlength=len(gm)) for a in range(3)],axis=-1)
    old=gp/np.where(gm>0,gm,1)[:,None];final=old if increment is None else old+increment
    fit=lambda field:(np.sum(w[:,:,None]*field[ids],axis=1),4/dx**2*np.einsum('pk,pki,pkj->pij',w,field[ids],r))
    av,aC=fit(final);dv,dC=fit(final-old)
    nv=v+dv if mode in ('DELTA','delta_v_apic_C') else av
    nC=C+dC if mode in ('DELTA','apic_v_delta_C') else aC
    kinetic=lambda vv,cc:float(.5*np.sum(mass[:,None]*vv*vv)+dx**2/8*np.sum(mass[:,None,None]*cc*cc))
    spin=lambda cc:np.stack((cc[:,2,1]-cc[:,1,2],cc[:,0,2]-cc[:,2,0],cc[:,1,0]-cc[:,0,1]),axis=-1)
    before=mass[:,None]*(np.cross(x,v)+dx**2/4*spin(C));after=mass[:,None]*(np.cross(x,nv)+dx**2/4*spin(nC))
    torque=np.sum(wm[:,:,None]*np.cross(x[:,None,:]+r,final[ids]-old[ids]),axis=1)
    defect=(after-before-torque).sum(0)
    kg0=float(.5*np.sum(gm[:,None]*old*old));kg1=float(.5*np.sum(gm[:,None]*final*final));kp0=kinetic(v,C);kp1=kinetic(nv,nC)
    return nv,nC,dict(angular_defect=defect,angular_scale=float(np.linalg.norm(before,axis=-1).sum()+np.linalg.norm(after,axis=-1).sum()+np.linalg.norm(torque,axis=-1).sum()),p2g_J=kp0-kg0,g2p_signed_J=kg1-kp1,particle_delta_J=kp1-kp0,grid_work_J=kg1-kg0)
