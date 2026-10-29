"""A genuine void: conforming triangular solid mesh around a circular hole."""
import numpy as np
from .at2_quasistatic import Model,split_reference_particles

def hole_reference(n=64,crack_length=.375,center=(.65,.58),radius=.10):
    old=split_reference_particles(n,crack_length);X=old['coordinates'].copy();quads=old['nodes'][::4];centroid=X[quads,:2].mean(axis=1);center=np.asarray(center,float)
    removed=np.linalg.norm(centroid-center,axis=1)<radius
    boundary=np.intersect1d(np.unique(quads[removed]),np.unique(quads[~removed]))
    offset=X[boundary,:2]-center;X[boundary,:2]=center+radius*offset/np.linalg.norm(offset,axis=1)[:,None]
    triangles=[]
    def area(t):
        e=X[t[1],:2]-X[t[0],:2];f=X[t[2],:2]-X[t[0],:2];return .5*(e[0]*f[1]-e[1]*f[0])
    for cell,q in enumerate(quads):
        if removed[cell]:continue
        candidates=[q[[0,1,2]],q[[0,2,3]]],[q[[0,1,3]],q[[1,2,3]]]
        scores=[min(area(t) for t in option) for option in candidates]
        valid=[j for j,s in enumerate(scores) if s>1e-14]
        if not valid:raise ValueError('Hole projection creates an inverted cell: '+str(cell))
        best=max(valid,key=lambda j:scores[j])
        if len(valid)==2 and abs(scores[0]-scores[1])<1e-14:best=(cell//n+cell%n)%2
        triangles.extend(candidates[best])
    triangles=np.asarray(triangles,np.int32);used=np.unique(triangles);mapping=np.full(len(X),-1,int);mapping[used]=np.arange(len(used));X=X[used];triangles=mapping[triangles];boundary=mapping[boundary]
    angle=np.arctan2(X[boundary,1]-center[1],X[boundary,0]-center[0]);boundary=boundary[np.argsort(angle)];polygon=X[boundary,:2]
    corners=X[triangles,:2];J=np.stack((corners[:,1]-corners[:,0],corners[:,2]-corners[:,0]),axis=-1);det=np.linalg.det(J)
    assert det.min()>0
    grad=np.einsum('ki,eij->ekj',np.array([[-1.,-1.],[1.,0.],[0.,1.]]),np.linalg.inv(J))
    bary=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
    nodes=np.repeat(triangles,3,axis=0);weight=np.tile(bary,(len(triangles),1));gradient=np.repeat(np.pad(grad,((0,0),(0,0),(0,1))),3,axis=0);volume=np.repeat(det/6,3);particles=np.einsum('pk,pki->pi',weight,X[nodes])
    inside=np.zeros(len(particles),bool);px=particles[:,0];py=particles[:,1]
    for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
        crossing=(a[1]>py)!=(b[1]>py)
        if a[1]!=b[1]:inside^=crossing&(px<(b[0]-a[0])*(py-a[1])/(b[1]-a[1])+a[0])
    void_area=.5*abs(np.sum(polygon[:,0]*np.roll(polygon[:,1],-1)-polygon[:,1]*np.roll(polygon[:,0],-1)))
    affine=np.einsum('pki,pkj->pij',X[nodes],gradient);mass=np.bincount(nodes.ravel(),weights=(volume[:,None]*weight).ravel(),minlength=len(X))
    verification={'solid_area_m2':float(volume.sum()),'void_polygon_area_m2':float(void_area),'solid_plus_void_area_error':float(abs(volume.sum()+void_area-1)),'minimum_triangle_area_m2':float(det.min()/2),'affine_gradient_error':float(abs(affine-np.diag([1.,1.,0.])).max()),'particles_inside_actual_void':int(inside.sum()),'unused_background_nodes_removed':int(len(old['coordinates'])-len(X)),'minimum_active_nodal_quadrature_mass':float(mass.min()),'hole_boundary_nodes':len(boundary),'removed_background_cells':int(removed.sum()),'geometry':'Circular hole approximated by an inscribed conforming polygon; no material, quadrature or orphan DOFs inside the actual void.'}
    assert not inside.any() and mass.min()>0 and verification['solid_plus_void_area_error']<1e-10 and verification['affine_gradient_error']<1e-10,verification
    data={'coordinates':X,'particles':particles,'nodes':nodes,'weight':weight,'gradient':gradient,'volume':volume,'triangles':triangles,'display_coordinates':X,'display_nodes':np.arange(len(X))[:,None],'display_weights':np.ones((len(X),1)),'display_triangles':triangles,'hole_boundary_nodes':boundary,'hole_center':center,'hole_radius':radius,'validation':verification}
    return data

class HoleModel(Model):
    def __init__(self,n=64,ell=.04,young=1e6,poisson=.2,gc=.001,residual=1e-7,crack_length=.375,center=(.65,.58),radius=.10):
        data=hole_reference(n,crack_length,center,radius)
        super().__init__(n,ell,young,poisson,gc,residual,crack_length,reference_data=data,initial_damage_mode='geometry-only')
        self.reference_data=data;self.geometry_verification=data['validation']
        assert not np.intersect1d(np.r_[2*data['hole_boundary_nodes'],2*data['hole_boundary_nodes']+1],self.fixed).size
        self.boundary_description={'case':'offset-hole-tension','hole_center_m':list(center),'hole_radius_m':radius,'actual_void_boundary':'Conforming polygon with vertices on the stated circle','hole_boundary_mechanics':'Natural traction-free boundary; no mechanical hole pins','hole_boundary_damage':'Natural AT2 zero flux; NO prescribed damage','integration':'Only solid P1 triangles, three-point exact Gaussian quadrature; void cells and unused nodes removed','initial_damage':0.,'damage_fixed_nodes':0,'imposed_crack_path':False,'young_Pa':young,'Gc_J_per_m2':gc,'ell_m':ell,'notch_length_m':crack_length,'geometry_verification':self.geometry_verification}
