"""Restrict the established contact sampler to original fixed-node motion.

No boundary reaction is inferred from an energy residual. The constrained
impulse is P_BC times the established unconstrained contact impulse; their
difference is an orthogonal reaction with zero admissible virtual work.
"""
from types import SimpleNamespace
import numpy as np
from tools.metal_affine_contact_numpy import CompatibleContact


class DirichletContact(CompatibleContact):
    def __init__(self,full_contact,constrained_projection):
        self.full_contact=full_contact;self.constrained=constrained_projection
        if np.any(np.isin(full_contact.pairs,constrained_projection.clamped_nodes)):
            raise ValueError('Bounded candidate stops: contact reaches an eliminated fixed node')
        allowed_modes,self.mode_projection_diagnostics=constrained_projection.project(full_contact.op.T)
        self.op=SimpleNamespace(n=constrained_projection.n,T=allowed_modes)
        self.B0,self.C=full_contact.B0,full_contact.C
        self.pairs,self.normal=full_contact.pairs,full_contact.normal
        self.affine_reconstruction_error=full_contact.affine_reconstruction_error
        G=np.empty_like(self.C)
        for c in range(3):G[:,c::3]=self.B0[:,c::3]@allowed_modes
        gram=allowed_modes.T@allowed_modes
        mobility=(self.B0@self.B0.T).toarray()+G@self.C.T+self.C@G.T
        for c in range(3):mobility+=self.C[:,c::3]@gram@self.C[:,c::3].T
        self.mobility=.5*(mobility+mobility.T)

    def apply(self,y,tolerance=.005):
        following,report,multiplier=self.project(y,tolerance,return_impulse=True)
        self.last_multiplier=multiplier.copy()
        physical_impulse=-self.full_contact.adjoint(multiplier)
        boundary_reaction=(following-y)-physical_impulse
        midpoint=.5*(y+following)
        report.update(boundary_virtual_work_J=float(np.sum(midpoint*boundary_reaction)),
            physical_contact_work_J=float(np.sum(midpoint*physical_impulse)),
            original_sample_closing_m_s=float(np.max(self.full_contact.sample(following),initial=0)),
            restricted_sample_difference_m_s=float(np.max(abs(self.sample(following)-self.full_contact.sample(following)),initial=0)))
        return following,report,boundary_reaction,physical_impulse
