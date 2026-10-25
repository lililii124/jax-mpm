"""Sparse global-node address lookup; quadratic/APIC arithmetic stays unchanged.

Only packed-index allocation and lookup differ from the dense-map tube. Empty
hash probes and any incomplete physical stencil are rejected by the same MPM
validity path. No node is silently dropped to satisfy a memory budget.
"""
from itertools import product
import numpy as np

MAX_PROBES=128


def _hash32(values,xp):
    x=xp.asarray(values,dtype=xp.uint32)
    x=(x^(x>>xp.uint32(16)))*xp.uint32(0x7feb352d)
    x=(x^(x>>xp.uint32(15)))*xp.uint32(0x846ca68b)
    return x^(x>>xp.uint32(16))


def build_tables(dense_indices,max_table_bytes=1024**3):
    ids=np.asarray(dense_indices,dtype=np.int64)
    if ids.ndim!=1 or not len(ids) or ids[0]<0 or ids[-1]>=2**31 or np.any(np.diff(ids)<=0):
        raise ValueError('Sorted unique nonnegative int32 node addresses required')
    capacity=1<<int((2*len(ids)-1).bit_length())
    if capacity*8>max_table_bytes:raise ValueError('Hash-key/value budget exceeded')
    keys=np.full(capacity,-1,np.int32);values=np.full(capacity,-1,np.int32)
    slots=(_hash32(ids,np)&np.uint32(capacity-1)).astype(np.int64)
    remaining=np.arange(len(ids));rounds=0
    while len(remaining):
        if rounds>=MAX_PROBES:raise ValueError('Hash probe bound exceeded; enlarge the table')
        unique,first=np.unique(slots[remaining],return_index=True)
        writable=keys[unique]==-1;winning_offsets=first[writable];winners=remaining[winning_offsets]
        keys[slots[winners]]=ids[winners];values[slots[winners]]=winners
        live=np.ones(len(remaining),bool);live[winning_offsets]=False;remaining=remaining[live]
        slots[remaining]=(slots[remaining]+1)&(capacity-1);rounds+=1
    assert np.array_equal(np.sort(keys[keys>=0].astype(np.int64)),ids)
    assert np.array_equal(keys[slots],ids.astype(np.int32)) and np.array_equal(values[slots],np.arange(len(ids)))
    return keys,values,rounds


def lookup_numpy(indices,keys,values):
    wanted=np.asarray(indices,dtype=np.int32);slots=(_hash32(wanted,np)&np.uint32(len(keys)-1)).astype(np.int64)
    unresolved=np.ones(wanted.shape,bool);result=np.full(wanted.shape,-1,np.int32)
    for _ in range(MAX_PROBES):
        if not unresolved.any():break
        observed=keys[slots];hit=unresolved&(observed==wanted)&(observed>=0)
        result[hit]=values[slots[hit]];unresolved&=~(hit|(observed<0));slots=np.where(unresolved,(slots+1)&(len(keys)-1),slots)
    return result


def lookup(indices,keys,values):
    import jax
    import jax.numpy as jnp
    wanted=jnp.asarray(indices,dtype=jnp.int32)
    slots=(_hash32(wanted,jnp)&jnp.uint32(len(keys)-1)).astype(jnp.int32)
    def condition(carry):
        count,_,_,unresolved=carry
        return (count<MAX_PROBES)&jnp.any(unresolved)
    def body(carry):
        count,slots,result,unresolved=carry;observed=keys[slots]
        hit=unresolved&(observed==wanted)&(observed>=0)
        result=jnp.where(hit,values[slots],result);unresolved=unresolved&~(hit|(observed<0))
        slots=jnp.where(unresolved,(slots+1)&(len(keys)-1),slots)
        return count+1,slots,result,unresolved
    _,_,result,_=jax.lax.while_loop(condition,body,(jnp.asarray(0),slots,jnp.full(wanted.shape,-1,jnp.int32),jnp.ones(wanted.shape,bool)))
    return result


def build_hash_tube_layout(x,fibre_id,*,dx,padding_cells=6,reference_bounds=None,
                           origin=(0.,0.,0.),max_active_nodes=30_000_000,
                           max_dense_addresses=2**31-1,max_hash_bytes=1024**3,capacity=None,capacity_factor=None,direct_map_capacity=None):
    """Same padded occupancy as build_tube_layout without its full dense map.

    One local per-fibre Boolean AABB is materialised at a time. Its active
    addresses are sorted exactly as the original builder; only address lookup
    changes. Source-only until actual-JAX stencil/state parity is established.
    """
    import jax.numpy as jnp
    from scipy.ndimage import maximum_filter1d
    from .fibre_mpm import build_layout,PackedLayout
    dense=build_layout(x,fibre_id,dx=dx,padding_cells=padding_cells,
        reference_bounds=reference_bounds,origin=origin,max_nodes=max_dense_addresses)
    points=np.asarray(x,dtype=float);ids=np.asarray(fibre_id);lower=np.asarray(dense.lower);shape=np.asarray(dense.shape);offsets=np.asarray(dense.offsets)
    order=np.argsort(ids,kind='stable');starts=np.concatenate(([0],np.cumsum(np.bincount(ids,minlength=dense.fibre_count))))
    neighbours=np.array(list(product(range(3),repeat=3)),np.int64);active=[];total=0
    for fibre in range(dense.fibre_count):
        selected=order[starts[fibre]:starts[fibre+1]]
        base=np.floor((points[selected]-np.asarray(origin))/dx-.5).astype(np.int64)-lower[fibre]
        local=(base[:,None,:]+neighbours).reshape(-1,3)
        if np.any(local<0) or np.any(local>=shape[fibre]):raise ValueError('Seed stencil exceeds its local AABB')
        occupied=np.zeros(tuple(shape[fibre]),bool);occupied[tuple(local.T)]=True
        for axis in range(3):occupied=maximum_filter1d(occupied,size=2*padding_cells+1,axis=axis,mode='constant',cval=False)
        found=np.flatnonzero(occupied.ravel())+int(offsets[fibre]);active.append(found.astype(np.int32));total+=len(found)
        if total>max_active_nodes:raise ValueError('Active-node budget exceeded')
        del occupied,local,base
    inverse=np.concatenate(active);occupied_count=total
    if capacity is None and capacity_factor is not None:
        capacity=1<<int((int(np.ceil(total*capacity_factor))-1).bit_length())
    if capacity is not None:
        capacity=int(capacity)
        if capacity<total or capacity>max_active_nodes:raise ValueError('Fixed packed capacity exceeded; preserve the last valid checkpoint')
        # Make every padding key a distinct, real but currently unused node.
        shape_copy=np.asarray(dense.shape).copy();dense_total=dense.node_count
        if dense_total<capacity:
            extra=int(np.ceil((capacity-dense_total)/int(np.asarray(dense.strides)[-1,0])))
            shape_copy[-1,0]+=extra;dense_total+=extra*int(np.asarray(dense.strides)[-1,0])
            if dense_total>=2**31:raise ValueError('Padded address range exceeds int32')
            dense=PackedLayout(dense.lower,jnp.asarray(shape_copy),dense.strides,dense.offsets,dense.origin,dense.dx,dense_total,dense.fibre_count)
        if capacity>total:
            remaining=capacity-total;unused=[]
            for first in range(0,dense.node_count,262144):
                candidates=np.arange(first,min(first+262144,dense.node_count),dtype=np.int32)
                selected=candidates[~np.isin(candidates,inverse,assume_unique=True)][:remaining]
                unused.append(selected);remaining-=len(selected)
                if not remaining:break
            if remaining:raise RuntimeError('Not enough distinct empty padding nodes')
            inverse=np.sort(np.concatenate([inverse,*unused]));total=capacity
    if direct_map_capacity is not None:
        map_capacity=(1<<int((4*dense.node_count-1).bit_length())) if direct_map_capacity=='auto' else int(direct_map_capacity)
        if map_capacity<dense.node_count or map_capacity*4>1024**3:raise ValueError('Fixed direct-map address budget exceeded')
        mapping=np.full(map_capacity,-1,np.int32);mapping[inverse]=np.arange(total,dtype=np.int32)
        assert np.array_equal(mapping[inverse],np.arange(total,dtype=np.int32))
        return PackedLayout(dense.lower,dense.shape,dense.strides,dense.offsets,dense.origin,dense.dx,total,dense.fibre_count,jnp.asarray(mapping),jnp.asarray(inverse)),dict(active_nodes=total,occupied_nodes=occupied_count,dense_address_count=dense.node_count,direct_map_capacity=map_capacity,direct_map_bytes=mapping.nbytes,hash_capacity=0,hash_bytes=0,maximum_insert_probe_rounds=0,complete_occupied_lookup_exact=True,lookup='direct')
    keys,values,rounds=build_tables(inverse,max_hash_bytes)
    if not np.array_equal(lookup_numpy(inverse,keys,values),np.arange(total,dtype=np.int32)):
        raise RuntimeError('Hash fails its independent complete occupied-node lookup')
    layout=PackedLayout(dense.lower,dense.shape,dense.strides,dense.offsets,dense.origin,dense.dx,total,dense.fibre_count,
        None,jnp.asarray(inverse),jnp.asarray(keys),jnp.asarray(values))
    return layout,dict(lookup='hash',active_nodes=total,occupied_nodes=occupied_count,dense_address_count=dense.node_count,hash_capacity=len(keys),
        hash_bytes=keys.nbytes+values.nbytes,maximum_insert_probe_rounds=rounds,complete_occupied_lookup_exact=True)
