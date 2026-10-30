"""Finite-dimensional sign/work check for compression-only viscosity."""
import numpy as np


def main():
    F0=np.eye(3); F1=np.diag((.9,1.,1.)); dF=F1-F0
    J=float(np.linalg.det(F1)); div=-.2
    coefficient=3.7
    q=coefficient*max(-div,0.)
    Pvisc=-J*q*np.linalg.inv(F1).T
    diss=float(np.sum(Pvisc*dF))
    opposite=float(-np.sum(Pvisc*(F0-F1)))
    mechanical=float(-np.sum(Pvisc*dF))
    assert q>0 and diss>0 and abs(diss-opposite)<1e-14
    assert abs(mechanical+diss)<1e-14
    theta=.07
    R=np.array(((np.cos(theta),-np.sin(theta),0.),
                (np.sin(theta),np.cos(theta),0.),(0.,0.,1.)))
    omega=np.array(((0.,-1.,0.),(1.,0.,0.),(0.,0.,0.)))
    dF_rotation=R-np.eye(3)
    rotation_rate=float(np.trace(omega))
    expansion_F=1.1*np.eye(3)
    expansion_rate=float(np.trace(expansion_F-np.eye(3)))
    for name,rate,dF in [('rigid_rotation',rotation_rate,dF_rotation),
                         ('expansion',expansion_rate,expansion_F-np.eye(3))]:
        qr=coefficient*max(-rate,0.)
        assert qr==0.
        assert float(np.sum(np.zeros((3,3))*dF))==0.
    print({'compression_q':q,'compression_loss_density':diss,
           'mechanical_plus_loss':mechanical+diss,
           'rigid_rotation_q':0.,'rigid_rotation_loss_density':0.,
           'expansion_q':0.,'expansion_loss_density':0.})

if __name__=='__main__': main()
