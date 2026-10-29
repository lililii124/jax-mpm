"""Sharp bottom-notch beam with genuine three-point-bending constraints."""
import numpy as np
from .at2_quasistatic import Model
from .at2_cut_reference import odd_cut_reference_particles

class BendingModel(Model):
    def __init__(self,length=1.,height=.25,notch=.05,vertical_cells=40,horizontal_cells=129,ell=.015,young=1e6,poisson=.2,gc=.001,residual=1e-7,head_width=.04,support_span=.8):
        source=odd_cut_reference_particles(vertical_cells,horizontal_cells,notch/height)
        data=dict(source)
        def transform(X):return np.column_stack((length*X[:,1],height*X[:,0],np.zeros(len(X))))
        for key in ['coordinates','particles','display_coordinates']:data[key]=transform(source[key])
        G=source['gradient'];data['gradient']=np.stack((G[:,:,1]/length,G[:,:,0]/height,np.zeros_like(G[:,:,0])),axis=-1)
        data['volume']=source['volume']*length*height
        data['display_quads']=source['display_quads'][:,[0,3,2,1]]
        super().__init__(n=vertical_cells,ell=ell,young=young,poisson=poisson,gc=gc,residual=residual,crack_length=notch/height,reference_data=data,initial_damage_mode='geometry-only')
        self.length=length;self.height=height;self.crack_length=notch;self.notch=notch
        self.hx=length/horizontal_cells;self.hy=height/vertical_cells
        if ell/max(self.hx,self.hy)<1.8:raise ValueError('Resolve ell with at least1.8 cells in each physical direction')
        bottom=np.flatnonzero(abs(self.xy[:,1])<1e-12)
        targets=[.5*(length-support_span),.5*(length+support_span)]
        rollers=np.array([bottom[np.argmin(abs(self.xy[bottom,0]-x))] for x in targets])
        head=np.flatnonzero((abs(self.xy[:,1]-height)<1e-12)&(abs(self.xy[:,0]-.5*length)<=head_width/2+1e-12))
        if len(head)<2:raise ValueError('The finite head must cover at least two physical nodes')
        self.top=2*head+1
        self.fixed=np.unique(np.r_[self.top,2*rollers+1,2*rollers[0]])
        self.free=np.setdiff1d(np.arange(2*len(self.xy)),self.fixed)
        self.geometric_crack_trace=source['seed'].copy()
        self.reference_data=data
        reproduced=np.einsum('pk,pki->pi',self.weight,self.coordinates[self.nodes])
        affine=np.einsum('pki,pkj->pij',self.coordinates[self.nodes],self.grad)
        self.geometry_verification={'particle_coordinate_error':float(abs(reproduced-self.particles).max()),'affine_gradient_error':float(abs(affine-np.diag([1.,1.,0.])).max()),'physical_area_error':float(abs(self.volume.sum()-length*height)),'ell_over_max_h':ell/max(self.hx,self.hy),'zero_width_bottom_notch':True}
        assert max(self.geometry_verification[k] for k in ['particle_coordinate_error','affine_gradient_error','physical_area_error'])<1e-11
        self.boundary_description={'case':'three-point-bending','length_m':length,'height_m':height,'bottom_central_notch_m':notch,'ell_m':ell,'vertical_cells':vertical_cells,'horizontal_cells':horizontal_cells,'requested_support_span_m':support_span,'actual_roller_positions':self.xy[rollers].tolist(),'actual_support_span_m':float(np.diff(self.xy[rollers,0])[0]),'head_nodes':head.tolist(),'head_reference_x_interval_m':[float(self.xy[head,0].min()),float(self.xy[head,0].max())],'head_width_requested_m':head_width,'head_prescription':'Same downward uy at all finite-width head nodes; ux is free.','roller_prescription':'uy=0 at two bottom roller nodes; ONLY left roller ux=0 removes rigid translation.','all_other_boundaries':'traction-free, including explicit notch faces','initial_damage':0.,'damage_fixed_nodes':int(self.damage_fixed.sum()),'imposed_crack_path':False,'matrix_E_Pa':young,'matrix_Gc_J_per_m2':gc,'poisson':poisson,'energy_ledger':'Positive downward head reaction magnitude times positive downward travel; zero work at stationary rollers/pin.'}
