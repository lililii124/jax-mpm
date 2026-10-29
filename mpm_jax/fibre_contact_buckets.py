"""Shape buckets for an unchanged real fibre-contact coverage graph.

Construct and validate the true graph before calling ``bucket_context``.
True pair/seed/graph rows retain their original order; only JIT input storage
is padded. The contact evaluators must apply the returned boolean mask.
"""
from dataclasses import replace
import os
import numpy as np

_contexts_built=0

def _pad(data,graph,valid,capacity):
    import jax.numpy as jnp
    n=int(np.asarray(valid).sum())
    if n>capacity:raise ValueError('Bucket smaller than true coverage')
    pairs=np.asarray(data.coverage_pairs)[:n].copy();seeds=np.asarray(data.coverage_seed)[:n].copy()
    real_graph=np.asarray(graph)[:n].copy()
    assert real_graph.shape==(n,8) and np.all((real_graph>=-1)&(real_graph<n))
    padded=replace(data,coverage_pairs=jnp.asarray(np.pad(pairs,((0,capacity-n),(0,0)))),
        coverage_seed=jnp.asarray(np.pad(seeds,(0,capacity-n),constant_values=-1)))
    mask=jnp.arange(capacity)<n
    padded_graph=jnp.asarray(np.pad(real_graph,((0,capacity-n),(0,0)),constant_values=-1))
    # Real rows preserve order, true graph and seed identities exactly.
    np.testing.assert_array_equal(np.asarray(padded.coverage_pairs)[:n],pairs)
    np.testing.assert_array_equal(np.asarray(padded.coverage_seed)[:n],seeds)
    np.testing.assert_array_equal(np.asarray(padded_graph)[:n],real_graph)
    return padded,padded_graph,mask

def bucket_context(cls,data,graph,history,failed,time):
    global _contexts_built
    import jax.numpy as jnp
    n=len(data.coverage_pairs);capacity=max(64,1<<max(0,n-1).bit_length())
    if _contexts_built==0 and os.environ.get('WEAVING_BUCKET_FIRST_TIGHT')=='1':capacity=max(1,n)
    _contexts_built+=1
    padded,graph,valid=_pad(data,graph,jnp.ones(n,dtype=bool),capacity)
    return cls(padded,graph,history,failed,time,valid)

def rebucket_context(context,capacity):
    data,graph,valid=_pad(context.data,context.graph,context.coverage_valid,capacity)
    return context._replace(data=data,graph=graph,coverage_valid=valid)
