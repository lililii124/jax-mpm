"""Current-configuration contact with its entire adjoint restricted to fixed P0.

This is a NumPy/SciPy reference operator, not an all-JAX production solver.
"""
from types import SimpleNamespace
from time import monotonic
import numpy as np
from scipy.sparse import vstack

from tools.metal_affine_contact_numpy import CompatibleContact
from tools.metal_affine_projection_numpy import ConsistentAffineProjection


class WholeProjectedContact(CompatibleContact):
    def __init__(self, full_contact, allowed, fixed_node_keys, progress=lambda label: None):
        started=monotonic();self.full_contact=full_contact;self.normal=full_contact.normal.copy()
        self.pairs=full_contact.pairs;self.C=full_contact.C
        self.affine_reconstruction_error=full_contact.affine_reconstruction_error
        self.op=SimpleNamespace(n=allowed.n)
        for index in self.pairs.ravel():
            key=(int(full_contact.node_body[index]), *map(int,full_contact.node_coordinates[index]))
            if key in fixed_node_keys:raise ValueError('Current physical contact reaches an original eliminated fixed node')
        # J has a scalar reconstructed relative velocity H followed by the
        # normal dot product. P0 acts identically on each Cartesian component,
        # so project m scalar adjoints rather than3m vector components.
        count=len(self.pairs);width=4*allowed.n
        if count:
            component=np.argmax(abs(self.normal),axis=1);amplitude=self.normal[np.arange(count),component]
            active=amplitude!=0;rows=[];correction=np.zeros((count,full_contact.op.T.shape[1]))
            for index in range(count):
                c=int(component[index]);row=full_contact.B0[index,c::3]
                if active[index]:
                    rows.append(row/amplitude[index]);correction[index]=full_contact.C[index,c::3]/amplitude[index]
                else:rows.append(row)
            physical=vstack(rows,format='csr').toarray()+correction@full_contact.op.T.T
            assert physical.shape==(count,width)
            progress('whole-contact-adjoint-projection-'+str(count))
            following,rows=allowed.project(physical.T)
            self.restricted_H=following.T
        else:
            physical=np.empty((0,width));self.restricted_H=physical.copy();rows=[]
        self.physical_H=physical
        gram=self.restricted_H@self.restricted_H.T
        self.mobility=gram*(self.normal@self.normal.T)
        self.mobility=.5*(self.mobility+self.mobility.T)
        self.construction=dict(
            restriction='Entire current physical scalar sampler adjoint projected to original fixed P0',
            scalar_projection_rhs=count,vector_projection_rhs_avoided=3*count,
            maximum_original_normal_residual=max((r['original_normal_residual'] for r in rows),default=0.),
            maximum_projection_iterations=max((r['iterations'] for r in rows),default=0),
            seconds=monotonic()-started,particles=allowed.n,current_physical_nodes=full_contact.op.nodes,
            scalar_factorization='J y=sum_c n_c H y_c; mobility=(H0 H0^T) hadamard (n n^T)',
            objective_regularization=0.,original_final_gates_unchanged=True)
        progress('whole-contact-adjoint-ready')

    def sample(self,y):return np.sum((self.restricted_H@y)*self.normal,axis=1)

    def adjoint(self,multiplier):return self.restricted_H.T@(multiplier[:,None]*self.normal)

    def apply(self,y,tolerance=.005):
        following,report,multiplier=self.project(y,tolerance,return_impulse=True)
        self.last_multiplier=multiplier.copy()
        # Physical contact belongs to the current geometry. Only the allowed
        # reaction is its orthogonal restriction; never add angular correction.
        physical=-self.full_contact.adjoint(multiplier)
        reaction=(following-y)-physical;midpoint=.5*(y+following)
        original=self.full_contact.sample(following)
        report.update(boundary_virtual_work_J=float(np.sum(midpoint*reaction)),
            physical_contact_work_J=float(np.sum(midpoint*physical)),
            original_sample_closing_m_s=float(np.max(original,initial=0)),
            restricted_sample_difference_m_s=float(np.max(abs(self.sample(following)-original),initial=0)),
            physical_scalar_factorization_difference_m_s=float(np.max(abs(np.sum((self.physical_H@following)*self.normal,axis=1)-original),initial=0)),
            current_configuration_whole_projection=self.construction)
        return following,report,reaction,physical


def update_contact_geometry(frame,configuration,progress=lambda label: None):
    if not hasattr(frame,'original_fixed_node_keys'):
        frame.original_fixed_node_keys={
            (int(frame.contact.node_body[i]), *map(int,frame.contact.node_coordinates[i]))
            for i in frame.allowed.clamped_nodes}
    args=(configuration['x'],configuration['mass'],configuration['body_id'],.00025,
          configuration['F'],configuration['reference_root'])
    current=ConsistentAffineProjection(*args,build_solver=False)
    if current.nodes>2_000_000:raise ValueError('Original node resource bound failed at contact geometry')
    full=CompatibleContact(current)
    _,counts=np.unique(full.node_coordinates,axis=0,return_counts=True)
    if np.any(counts>2) or int(np.sum(counts==2))!=len(full.pairs):raise ValueError('Original complete contact coverage failed at current geometry')
    projected=WholeProjectedContact(full,frame.allowed,frame.original_fixed_node_keys,progress)
    frame.contact=full;frame.restricted=projected
    return projected.construction
