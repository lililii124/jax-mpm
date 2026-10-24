"""Hertz/JKR local-region coupling for three separate helical fibre fields.

Area-weighted section kinematics preserve circular centreline coordinates despite
curved-tube mass quadrature. Extended line contact is explicitly unsupported.
"""
import jax.numpy as jnp
import numpy as np
from .contact_patches import contact_patches
from .fibre_coupling import closest_segments,section_centres
from .fibre_patch_contact import evaluate_hertz_patches


def evaluate_rope_contact(state,history,data,law,neighbours,*,capacity=4096,max_patches=32,max_iterations=256,model='hertz',tracking_gap=1e-5):
    # Equal-area cross sections; the physical APIC state/masses are untouched.
    kinematic=state._replace(mass=jnp.ones_like(state.mass))
    if model=='hertz':
        force,updated,metrics=evaluate_hertz_patches(kinematic,history,data,law,neighbours,capacity=capacity,max_patches=max_patches,max_iterations=max_iterations)
    elif model=='jkr':
        from .fibre_jkr_regions import evaluate_jkr_regions
        force,updated,result=evaluate_jkr_regions(kinematic,history,data,law,neighbours,capacity=capacity,max_patches=max_patches,max_iterations=max_iterations,tracking_gap=tracking_gap)
        metrics=result.contact
    else:raise ValueError('Only Hertz or persistent local JKR supported')
    centres,_=section_centres(kinematic.x,kinematic.mass,data)
    near=data.coverage_pairs;ends=data.segments[near]
    p,q=centres[ends[:,0,0]],centres[ends[:,0,1]];r,s=centres[ends[:,1,0]],centres[ends[:,1,1]]
    first,second,_,_,distance=closest_segments(p,q,r,s)
    gap=distance-jnp.sum(data.fibre_radius[data.segment_fibre[near]],axis=1)
    active_cells=gap<0
    if model=='jkr':
        slots=updated.cell_slots
        active_cells=(slots>=0)&updated.law.active[jnp.maximum(slots,0)]
    selection=contact_patches(gap,active_cells,neighbours,capacity=capacity,max_patches=max_patches,max_iterations=max_iterations)
    n=len(near);cells=selection.cells;live=cells<n
    centre=(first+second)/2
    padded=jnp.concatenate((centre,jnp.zeros((1,3),dtype=centre.dtype)))
    positions=padded[cells]
    low=jnp.full((capacity+1,3),jnp.inf).at[selection.labels].min(jnp.where(live[:,None],positions,jnp.inf))
    high=jnp.full((capacity+1,3),-jnp.inf).at[selection.labels].max(jnp.where(live[:,None],positions,-jnp.inf))
    roots=jnp.nonzero(live&(selection.labels==jnp.arange(capacity)),size=max_patches,fill_value=capacity)[0]
    extent=jnp.linalg.norm(jnp.where((roots<capacity)[:,None],high[roots]-low[roots],0),axis=1)
    spacing=jnp.max(jnp.linalg.norm(centres[data.segments[:,1]]-centres[data.segments[:,0]],axis=1))
    # Circular-equivalent Hertz is local. A connected ribbon longer than two
    # centreline cells or four local footprint radii requires a line/surface law.
    limit=jnp.maximum(2*spacing,4*jnp.max(metrics.contact_radius,initial=0.))
    local=jnp.all(extent<=limit)&selection.valid
    valid=metrics.valid&local
    metrics=metrics._replace(valid=valid)
    updated=updated._replace(valid=updated.valid&local)
    return force,updated,metrics,jnp.max(extent,initial=0.),~local
