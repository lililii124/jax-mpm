"""Variational AT2 extensions: SPD crack-density metric and elastic inclusion.

No damage boundary conditions, path constraints, or changed initial damage.
The inclusion has explicit Young's-modulus and Gc ratios, shares ell and
Poisson ratio, and is perfectly bonded. Fracture may cross its interface.
"""
import numpy as np
from .at2_quasistatic import Model as BaseModel, split_reference_particles

class MaterialCaseModel(BaseModel):
    def __init__(self,*args,case='anisotropic',preferred_angle_degrees=25.,anisotropy_ratio=9.,inclusion_center=(.65,.53),inclusion_radius=.12,inclusion_modulus_ratio=8.,inclusion_gc_ratio=1.,**kwargs):
        data=kwargs.get('reference_data')
        if data is None:
            n=kwargs.get('n',args[0] if args else 64);crack=kwargs.get('crack_length',args[6] if len(args)>6 else .375)
            data=split_reference_particles(n,crack)
        physical_volume=np.asarray(data['volume']).copy();P=np.asarray(data['particles'])[:,:2]
        inclusion=np.linalg.norm(P-np.asarray(inclusion_center),axis=1)<=inclusion_radius
        is_inclusion=case in ['hard-inclusion','hard-tough-inclusion']
        modulus=np.where(inclusion,inclusion_modulus_ratio,1.) if is_inclusion else np.ones(len(P))
        gc_ratio=np.where(inclusion,inclusion_gc_ratio,1.) if is_inclusion else np.ones(len(P))
        if min(inclusion_modulus_ratio,inclusion_gc_ratio)<=0:raise ValueError('Positive inclusion E/Gc ratios are required')
        # The base mechanical JIT captures its quadrature volume. Multiplying
        # it by E(x)/E0 exactly scales BOTH Lamé constants, energy and tangent.
        # Restore physical integration for the AT2 crack-density and damage RHS.
        mechanical_data={**data,'volume':physical_volume*modulus};kwargs['reference_data']=mechanical_data
        super().__init__(*args,**kwargs)
        self.volume=physical_volume;self.area=physical_volume;self.v=self.jnp.asarray(physical_volume)
        self.mass=physical_volume[:,None,None]*self.weight[:,:,None]*self.weight[:,None,:]
        angle=np.deg2rad(preferred_angle_degrees);t=np.array([np.cos(angle),np.sin(angle)]);normal=np.array([-t[1],t[0]])
        if anisotropy_ratio<1:raise ValueError('Anisotropy ratio must be >=1')
        self.crack_metric=anisotropy_ratio*np.outer(t,t)+np.outer(normal,normal) if case=='anisotropic' else np.eye(2)
        local=gc_ratio[:,None,None]*(self.gc/self.ell*self.mass+self.gc*self.ell*physical_volume[:,None,None]*np.einsum('pki,ij,plj->pkl',self.grad[:,:,:2],self.crack_metric,self.grad[:,:,:2]))
        self.fracture_matrix=self.assemble_d(local)
        self.case_description={'case':case,'crack_density':'Gc(x)/2 * [d^2/ell + ell grad(d).A.grad(d)]','crack_metric':self.crack_metric.tolist(),'preferred_crack_tangent_degrees':preferred_angle_degrees if case=='anisotropic' else None,'metric_eigenvalues':np.linalg.eigvalsh(self.crack_metric).tolist(),'orientation_effective_toughness':'Gc*sqrt(n.A.n) for unit crack normal n','inclusion_center':list(inclusion_center) if is_inclusion else None,'inclusion_radius':inclusion_radius if is_inclusion else None,'inclusion_modulus_ratio':inclusion_modulus_ratio if is_inclusion else None,'inclusion_Gc_ratio':inclusion_gc_ratio if is_inclusion else None,'matrix_E_Pa':self.young,'matrix_Gc_J_per_m2':self.gc,'ell_m':self.ell,'interface':'Perfect displacement bonding; common ell/nu; piecewise Gc weights BOTH local and gradient crack-density terms; no interface weakening or impenetrable fracture constraint.','physical_initial_damage':0.,'fixed_damage_nodes':int(self.damage_fixed.sum()),'prescribed_propagation_path':False}
        self.particle_modulus_ratio=modulus;self.particle_gc_ratio=gc_ratio;self.particle_inclusion=inclusion if is_inclusion else np.zeros(len(P),bool)

    def mechanical(self,u,d,tangent=False):
        result=super().mechanical(u,d,tangent)
        # Mechanical integration already contains E(x)/E0. The returned
        # tensile density must carry that factor too, because the damage RHS
        # and reaction matrix integrate it using the restored physical volume.
        return result[:2]+(result[2]*self.particle_modulus_ratio,)+result[3:]

    def operator_equivalence(self,plus,d):
        # Independent direct quadrature action for the anisotropic weak form.
        dp=np.sum(self.weight*d[self.nodes],axis=1);gd=np.einsum('pk,pki->pi',d[self.nodes],self.grad[:,:,:2])
        local=self.volume[:,None]*((self.gc*self.particle_gc_ratio/self.ell+2*(1-self.residual)*plus)[:,None]*dp[:,None]*self.weight+self.gc*self.particle_gc_ratio[:,None]*self.ell*np.einsum('pi,ij,pkj->pk',gd,self.crack_metric,self.grad[:,:,:2]))
        applied=np.bincount(self.nodes.ravel(),weights=local.ravel(),minlength=len(d))
        A=self.fracture_matrix+self.assemble_d(2*(1-self.residual)*plus[:,None,None]*self.mass);expected=A@d
        return float(abs(applied-expected).max()/max(abs(expected).max(),1e-30))
