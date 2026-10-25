"""Bounded dynamic-stage mechanics with explicit physical loss channels."""
import gc,math
import numpy as np
from tools.metal_affine_projection_numpy import ConsistentAffineProjection
from tools.metal_affine_contact_numpy import CompatibleContact
from tools.metal_affine_dirichlet_numpy import DirichletContact
from tools.metal_affine_resource import require_commit_headroom
from tools.metal_frozen_transfer import support


def accurate_sum(values):return np.array([math.fsum(values[:,j]) for j in range(3)])
def kinetic(y):return float(.5*np.sum(y*y))
def whiten(state):
    values=np.concatenate((state['v'][:,:,None],state['W']@state['reference_root']),axis=2).transpose(0,2,1)
    return values.reshape(-1,3)*np.repeat(np.sqrt(state['mass']),4)[:,None]
def unpack(state,y):
    n=len(state['mass']);a=(y/np.repeat(np.sqrt(state['mass']),4)[:,None]).reshape(n,4,3).transpose(0,2,1)
    return a[:,:,0],a[:,:,1:]@np.linalg.inv(state['reference_root'])
def synchronize(state,y):
    state['v'],state['W']=unpack(state,y)
    state['C']=state['W']@state['D0']@state['F'].swapaxes(-1,-2)*(4/.00025**2)
    state['C_transfer']=state['C'];state['L_physical']=state['W']@np.linalg.inv(state['F'])
def moments(state,y):
    v,W=unpack(state,y);B=W@state['D0']@state['F'].swapaxes(-1,-2)
    spin=np.stack((B[:,2,1]-B[:,1,2],B[:,0,2]-B[:,2,0],B[:,1,0]-B[:,0,1]),-1)
    return accurate_sum(state['mass'][:,None]*v),accurate_sum(state['mass'][:,None]*(np.cross(state['x'],v)+spin))
def thermal(state,p):
    J=np.linalg.det(state['F']);elastic=np.sum(state['tau']**2,axis=(-2,-1))/(4*p['shear_modulus']*p['density']*np.maximum(1-state['damage'],1e-12))
    volume=np.where(J>1,1-state['damage'],1)*p['bulk_modulus']*(1-J)**2/(2*p['density'])
    heat=state['internal_energy']-state['cold_work']-state['fracture_energy']-elastic-volume
    T=p['reference_temperature']+heat/p['heat_capacity']
    return J,T,p['heat_capacity']*np.log(T/p['reference_temperature'])
def cfl_limit(state,p,cfl=.1):
    J,T,_=thermal(state,p);cs=np.sqrt(p['bulk_modulus']*J*J/p['density'])
    longitudinal=np.sqrt(cs*cs+4*p['shear_modulus']*np.maximum(J,1)/(3*p['density']))
    affine=1.5*np.sqrt(3)*.00025*np.linalg.norm(state['C'],axis=(-2,-1))
    bounds=[]
    for gradient in (state['C'],state['L_physical']):
        compression=np.maximum(-np.trace(gradient,axis1=-2,axis2=-1),0)
        signal=longitudinal+np.linalg.norm(state['v'],axis=1)+affine+p['viscosity_linear']*cs+2*p['viscosity_quadratic']*.00025*compression
        bounds.append(cfl*.00025/float(np.max(signal)))
    return min(bounds)
def stress_force(state,y,p):
    _,W=unpack(state,y);F=state['F'];L=W@np.linalg.inv(F);J=np.linalg.det(F)
    pressure=p['bulk_modulus']*(1-J);pressure=np.where(pressure<0,(1-state['damage'])*pressure,pressure)
    speed=.00025*np.maximum(-np.trace(L,axis1=-2,axis2=-1),0)
    q=p['density']/J*(p['viscosity_linear']*np.sqrt(p['bulk_modulus']*J*J/p['density'])*speed+p['viscosity_quadratic']*speed**2)
    viscous=-J[:,None,None]*q[:,None,None]*np.eye(3)
    tau=state['tau']-J[:,None,None]*pressure[:,None,None]*np.eye(3)+viscous
    covector=-state['volume'][:,None,None]/np.sqrt(state['mass'])[:,None,None]*tau@np.linalg.inv(F).swapaxes(-1,-2)@np.linalg.inv(state['reference_root']).swapaxes(-1,-2)
    force=np.concatenate((np.zeros((len(J),3,1)),covector),axis=2).transpose(0,2,1).reshape(-1,3)
    return force,tau,viscous,q


class DynamicFrame:
    def __init__(self,state,fixed,incoming_y,progress=lambda phase:None):
        require_commit_headroom()
        args=(state['x'],state['mass'],state['body_id'],.00025,state['F'],state['reference_root'])
        self.full=ConsistentAffineProjection(*args)
        progress('free-factor-ready')
        self.contact=CompatibleContact(self.full)
        self.incoming_y=incoming_y.copy()
        self.free_state,self.free_projection_rows=self.full.project(incoming_y)
        # Only the known affine modes and assembled contact sampler survive.
        # Do not hold two large SuperLU factors at once.
        self.full.tri=None;self.full.A=None;self.full.Q=None;self.full.Qp=None;self.full.Q0=None
        gc.collect();progress('free-state-projected-factor-released')
        self.allowed=ConsistentAffineProjection(*args,clamped_particles=fixed)
        progress('fixed-node-factor-ready')
        self.restricted=DirichletContact(self.contact,self.allowed)
        progress('constrained-contact-ready')
        w,r,ids,wm,gm=support(state['x'],state['body_id'],state['mass'],.00025)
        partition=float(np.max(abs(w.sum(axis=1)-1)))
        first_moment=float(np.max(abs(np.sum(w[:,:,None]*r,axis=1))))
        _,counts=np.unique(self.contact.node_coordinates,axis=0,return_counts=True)
        shared=int(np.sum(counts==2));eps=np.finfo(float).eps
        if partition>64*eps or first_moment>64*eps*.00025 or np.any(w<0):raise ValueError('Original stencil consistency failed')
        if np.any(counts>2) or shared!=len(self.contact.pairs):raise ValueError('Original complete contact coverage failed')
        if self.full.nodes>2_000_000:raise ValueError('Original node resource bound failed')
        self.report=dict(nodes=self.full.nodes,free_nodes=self.allowed.nodes,clamped_nodes=len(self.allowed.clamped_nodes),
            pairs=len(self.contact.pairs),mode_projection=self.restricted.mode_projection_diagnostics,
            original_clamped_particle_count=int(np.sum(fixed)),eliminated_grid_dofs=3*len(self.allowed.clamped_nodes),
            coverage_valid=True,occupied_shared_nodes=shared,stencil_partition_error=partition,stencil_first_moment_error=first_moment,
            free_factor=self.full.factor_report,constrained_factor=self.allowed.factor_report)
    def project(self,y):
        if not np.array_equal(y,self.incoming_y):raise ValueError('Stale free-state projection')
        free,free_rows=self.free_state,self.free_projection_rows;allowed,bc_rows=self.allowed.project(free)
        grid_loss=float(.5*np.sum((free-y)**2));boundary_loss=float(.5*np.sum((allowed-free)**2))
        return allowed,dict(grid_projection_loss_J=grid_loss,boundary_projection_loss_J=boundary_loss,
            grid_projection_identity_J=kinetic(free)-kinetic(y)+grid_loss,
            boundary_projection_identity_J=kinetic(allowed)-kinetic(free)+boundary_loss,
            free_projection=free_rows,constrained_projection=bc_rows),allowed-free
    def force(self,raw):
        allowed,bc_rows=self.allowed.project(raw)
        # This residual includes both the unconstrained Galerkin-orthogonal
        # force and the fixed-node reaction. The former has zero global
        # impulse/torque because the full space reproduces rigid fields.
        # Thus global impulse/torque of this residual is the boundary reaction;
        # both parts do zero admissible virtual work. No free inverse needed.
        return allowed,allowed-raw,dict(constrained=bc_rows,
            reaction_scope='fixed-node plus Galerkin-orthogonal residual; only fixed nodes contribute global impulse/torque')
    def release(self):
        for op in (self.full,self.allowed):op.tri=None;op.A=None
        self.restricted=None;self.contact=None;self.full=None;self.allowed=None;gc.collect()
