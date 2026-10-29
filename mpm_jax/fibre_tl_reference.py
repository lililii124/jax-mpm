"""Exact active-stencil packing for a fixed TL fibre reference grid.

Only nodes outside every reference stencil are omitted. No mass threshold is
used; even zero-weight entries of the original 27-node stencils are retained.
This layout is unsuitable for a moving world-grid/APIC simulation.
"""
from itertools import product
import numpy as np
import jax.numpy as jnp
from .fibre_mpm import PackedLayout,build_layout


def build_tl_reference_layout(x,fibre_id,*,dx,padding_cells=6,max_nodes=40_000_000,origin=(0.,0.,0.)):
    """Preserve the original global lattice and all 27 reference entries."""
    dense=build_layout(x,fibre_id,dx=dx,padding_cells=padding_cells,max_nodes=max_nodes,origin=origin)
    x=np.asarray(x,dtype=np.float64);ids=np.asarray(fibre_id)
    base=np.floor((x-np.asarray(origin))/dx-.5).astype(np.int64)
    offsets=np.array(list(product(range(3),repeat=3)),dtype=np.int64)
    local=base[:,None,:]+offsets-np.asarray(dense.lower)[ids,None,:]
    if np.any(local<0) or np.any(local>=np.asarray(dense.shape)[ids,None,:]):
        raise ValueError('Reference stencil lies outside the original padded grid')
    dense_ids=np.asarray(dense.offsets)[ids,None]+np.sum(local*np.asarray(dense.strides)[ids,None,:],axis=-1)
    active=np.unique(dense_ids.reshape(-1)).astype(np.int32)
    node_map=np.full(dense.node_count,-1,dtype=np.int32)
    node_map[active]=np.arange(len(active),dtype=np.int32)
    return PackedLayout(dense.lower,dense.shape,dense.strides,dense.offsets,dense.origin,dense.dx,
        len(active),dense.fibre_count,jnp.asarray(node_map),jnp.asarray(active))


def reference_layout_maps(old,new):
    """Return explicit old-to-new and new-to-old indices on the same lattice.

    The inverse mapping is lossless for states whose omitted DOFs are zero.
    Checkpoint consumers must record the layout change rather than silently
    treating these index spaces as compatible.
    """
    for name in ('lower','shape','strides','offsets','origin'):
        if not np.array_equal(np.asarray(getattr(old,name)),np.asarray(getattr(new,name))):
            raise ValueError('Reference grids differ in their original global lattice')
    if old.dx!=new.dx or old.fibre_count!=new.fibre_count:
        raise ValueError('Reference lattice spacing or fibre ownership differs')
    old_dense=np.asarray(old.dense_indices);new_dense=np.asarray(new.dense_indices)
    new_to_old=np.searchsorted(old_dense,new_dense)
    if np.any(new_to_old>=len(old_dense)) or not np.array_equal(old_dense[new_to_old],new_dense):
        raise ValueError('New layout is not a subset of the old reference nodes')
    old_to_new=np.full(old.node_count,-1,dtype=np.int32)
    old_to_new[new_to_old]=np.arange(new.node_count,dtype=np.int32)
    return old_to_new,new_to_old.astype(np.int32)


def compact_tl_state(state,old,new):
    """Explicitly map an existing state, rejecting any nonzero omitted DOF."""
    old_to_new,new_to_old=reference_layout_maps(old,new)
    omitted=old_to_new<0;changes={}
    for name in ('displacement','velocity','force'):
        values=np.asarray(getattr(state,name))
        if values.shape!=(old.node_count,3) or np.any(values[omitted]!=0):
            raise ValueError('Cannot discard a nonzero or incorrectly shaped '+name+' DOF')
        changes[name]=jnp.asarray(values[new_to_old])
    return state._replace(**changes)
