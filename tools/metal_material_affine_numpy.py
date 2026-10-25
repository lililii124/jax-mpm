"""Small dense compatible affine-particle reference, not an impact solver.

The kinetic metric uses material affine velocity W=Fdot and a fixed reference
moment D0. Grid interpolation G maps u to (Iu, A(u)F); its H-adjoint defines both
the consistent mass and stress force. A nullspace gauge reproduces rigid fields.
"""
import numpy as np
from scipy.optimize import nnls
from tools.metal_frozen_transfer import support


def pack(v,W):return np.concatenate((v,W.reshape(len(v),9)),axis=1).ravel()
def unpack(z):
    values=np.asarray(z).reshape(-1,12)
    return values[:,:3],values[:,3:].reshape(-1,3,3)


class AffineOperators:
    def __init__(self,x,F,mass,body,dx,D0):
        self.x,self.F,self.mass,self.body,self.dx,self.D0=map(np.asarray,(x,F,mass,body,dx,D0))
        self.count=len(x);w,r,ids,wm,gm=support(x,body,mass,float(dx))
        points=np.zeros((len(gm),3));points[ids.ravel()]=(x[:,None,:]+r).reshape(-1,3)
        node_body=np.zeros(len(gm),int);node_body[ids.ravel()]=np.broadcast_to(body[:,None],ids.shape).ravel()
        active=gm>0;lookup=np.full(len(gm),-1,int);lookup[active]=np.arange(active.sum())
        self.coordinates,self.node_body=points[active],node_body[active]
        G=np.zeros((12*len(x),3*active.sum()))
        for p in range(len(x)):
            for k in range(27):
                i=lookup[ids[p,k]]
                if i<0:continue
                G[12*p:12*p+3,3*i:3*i+3]+=w[p,k]*np.eye(3)
                material_gradient=4/float(dx)**2*w[p,k]*r[p,k]@F[p]
                for a in range(3):G[12*p+3+3*a:12*p+6+3*a,3*i+a]+=material_gradient
        Hroot=np.zeros((12*len(x),12*len(x)));Hinv=np.zeros_like(Hroot)
        for p in range(len(x)):
            eig,V=np.linalg.eigh(D0[p])
            if eig.min()<=0:raise ValueError('Positive reference moment required')
            root=(V*np.sqrt(eig))@V.T;inverse=(V/np.sqrt(eig))@V.T
            Hroot[12*p:12*p+3,12*p:12*p+3]=np.sqrt(mass[p])*np.eye(3)
            Hroot[12*p+3:12*p+12,12*p+3:12*p+12]=np.sqrt(mass[p])*np.kron(np.eye(3),root)
            Hinv[12*p:12*p+3,12*p:12*p+3]=np.eye(3)/np.sqrt(mass[p])
            Hinv[12*p+3:12*p+12,12*p+3:12*p+12]=np.kron(np.eye(3),inverse)/np.sqrt(mass[p])
        Wgrid=Hroot@G;U,s,Vt=np.linalg.svd(Wgrid,full_matrices=False)
        # Algebraic backward-error rank only; it is recorded, never a fitted
        # physical cutoff. Production must diagnose conditioning separately.
        self.rank_tolerance=max(Wgrid.shape)*np.finfo(float).eps*s[0]
        keep=s>self.rank_tolerance;U,s,Vt=U[:,keep],s[keep],Vt[keep]
        self.G,self.Hroot,self.Hinv,self.U=G,Hroot,Hinv,U
        self.singular_values=s;self.rank=len(s);self.nullity=G.shape[1]-self.rank
        pinverse=(Vt.T/s)@U.T@Hroot
        # A minimum-norm grid inverse alone can alter rigid velocities in a
        # massless grid null mode. Add only a null-mode gauge that reproduces
        # each body's translations/rotations exactly; G R remains unchanged.
        bodies=np.unique(body);Kgrid=np.zeros((G.shape[1],6*len(bodies)))
        for b in bodies:
            particle=body==b;node=self.node_body==b
            center=np.average(x[particle],axis=0,weights=mass[particle])
            for axis in range(3):
                for index in np.flatnonzero(node):
                    Kgrid[3*index+axis,6*b+axis]=1.
                    Kgrid[3*index:3*index+3,6*b+3+axis]=np.cross(np.eye(3)[axis],self.coordinates[index]-center)/float(dx)
        K=G@Kgrid;HK=Hroot@K;coeff=np.linalg.solve(HK.T@HK,HK.T@Hroot)
        self.R=pinverse+(Kgrid-pinverse@K)@coeff
        self.velocity_basis=Hinv@U
        self.consistent_mass=Wgrid.T@Wgrid
        self.moment_modes=K

    def energy(self,z):return float(.5*np.dot(self.Hroot@z,self.Hroot@z))
    def project(self,z):return self.velocity_basis@(self.U.T@self.Hroot@z)
    def coordinates_in_range(self,z):return self.U.T@self.Hroot@z
    def angular(self,z):
        v,W=unpack(z);B=W@self.D0@self.F.swapaxes(-1,-2)
        spin=np.stack((B[:,2,1]-B[:,1,2],B[:,0,2]-B[:,2,0],B[:,1,0]-B[:,0,1]),axis=-1)
        return np.sum(self.mass[:,None]*(np.cross(self.x,v)+spin),axis=0)
    def linear(self,z):return np.sum(self.mass[:,None]*unpack(z)[0],axis=0)
    def physical_gradient(self,z):return unpack(z)[1]@np.linalg.inv(self.F)
    def transfer_coefficient(self,z):return unpack(z)[1]@self.D0@self.F.swapaxes(-1,-2)*(4/float(self.dx)**2)
    def stress_covector(self,tau,volume):
        material=-volume[:,None,None]*tau@np.linalg.inv(self.F).swapaxes(-1,-2)
        return pack(np.zeros((self.count,3)),material)
    def stress_kick(self,z,tau,volume,dt):
        covector=self.stress_covector(tau,volume)
        force=self.U.T@self.Hinv@covector
        y=self.coordinates_in_range(z);following=self.velocity_basis@(y+dt*force)
        midpoint=.5*(z+following)
        physical_work=float(dt*np.sum(volume[:,None,None]*tau*self.physical_gradient(midpoint)))
        grid_force=self.G.T@covector;u=self.R@midpoint
        return following,dict(kinetic_change_J=self.energy(following)-self.energy(z),physical_stress_work_J=physical_work,
            grid_stress_work_J=float(-dt*np.dot(grid_force,u)),virtual_work_residual_J=float(np.dot(covector,midpoint)-np.dot(grid_force,u)))
    def matching_contact_rows(self,normal):
        rows=[];a=np.flatnonzero(self.node_body==0);b=np.flatnonzero(self.node_body==1)
        table={tuple(self.coordinates[i]):i for i in b}
        for i in a:
            k=table.get(tuple(self.coordinates[i]))
            if k is None:continue
            row=np.zeros(self.G.shape[1]);row[3*i:3*i+3]=normal;row[3*k:3*k+3]=-normal;rows.append(row)
        return np.array(rows)
    def contact(self,z,grid_rows,tolerance=.005):
        B=grid_rows@self.R@self.velocity_basis
        y0=self.coordinates_in_range(z)
        multipliers,_=nnls(B.T,y0,maxiter=1024)
        following_y=y0-B.T@multipliers
        closing=B@following_y
        residual=np.where(multipliers>0,closing,np.maximum(closing,0.))
        following=self.velocity_basis@following_y
        valid=bool(np.max(abs(residual),initial=0.)<=tolerance)
        delta=following_y-y0
        return following,dict(valid=valid,closing_m_s=float(np.max(closing,initial=0.)),
            complementarity_m_s=float(np.max(abs(residual),initial=0.)),
            kinetic_loss_J=self.energy(z)-self.energy(following),
            projection_identity_J=self.energy(following)-self.energy(z)+.5*np.dot(delta,delta)+np.dot(multipliers,closing),
            linear_change=self.linear(following)-self.linear(z),angular_change=self.angular(following)-self.angular(z))
