"""Prove/deny disjoint finite-guide supports on the actual unchanged F7 mesh.

This is a setup-only algebra check: it builds the real reference stencil and
never advances the physical state.  Duplicate stencil entries belonging to
the same guide/node are coalesced before testing ownership and Gram diagonality.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


EXPECTED={
    'reference_x':'cb4f1e40c02c60b2e67758007de4c4f908596ea3e7d20880b2986749ac2eda65',
    'volume':'0f100b1b11bde952e30812ef9f96bd33137f59fd2e38e3837bb73108a4947951',
    'fibre_id':'1fa382c20e0304872c4a2a52f12ee9db033fb2f5b48e9f8fbdc2f2c52f959b15',
    'guide_particle_ids':'11839e814d646365c618a2128f1cd0b8f90e69040dcde025cba3eb2e66851456',
}


def digest(value,dtype):
    return hashlib.sha256(np.ascontiguousarray(value,dtype=dtype).tobytes()).hexdigest()


def run(output):
    import jax
    import jax.numpy as jnp
    from examples.triaxial_weaving_guided_geometry import plan_guided
    from mpm_jax.fibre_mpm import build_tube_layout,stencil
    from mpm_jax.fibre_tl_guides import make_guide_operator,project

    start=time.perf_counter();geometry,plan=plan_guided(per_family=8,nominal_cfl=.08);meta=geometry['metadata']
    x=np.asarray(geometry['x']);volume=np.asarray(geometry['volume']);fids=np.asarray(geometry['fibre_id'])
    guides=np.asarray([g['particle_ids'] for g in meta['placement_guides']],np.int32)
    identities=dict(reference_x=digest(x,'<f8'),volume=digest(volume,'<f8'),fibre_id=digest(fids,'<i4'),
        guide_particle_ids=digest(guides,'<i4'))
    if identities!=EXPECTED:raise RuntimeError('F7 support proof input does not match the admitted R9 geometry')
    rho=float(plan['density_kg_m3']);layout=build_tube_layout(x,fids,dx=plan['dx'],padding_cells=6,max_nodes=40_000_000)
    support=stencil(jnp.asarray(x),jnp.asarray(fids),layout);jax.block_until_ready(support.valid)
    if not bool(support.valid):raise RuntimeError('Actual F7 reference stencil is invalid')
    node_count=layout.node_count;idx=np.asarray(support.indices);weights=np.asarray(support.weights)
    if np.any(idx<0) or np.any(idx>=node_count):raise RuntimeError('Out-of-range packed reference node in guide proof')
    mass=rho*volume
    nodal_mass=np.zeros(node_count,np.float64)
    np.add.at(nodal_mass,idx.reshape(-1),(mass[:,None]*weights).reshape(-1))

    guide_labels=np.repeat(np.arange(len(guides),dtype=np.int32),guides.shape[1]*idx.shape[1])
    pids=guides.reshape(-1);particle_mass=mass[pids]
    band_mass=particle_mass.reshape(guides.shape).sum(axis=1)
    coefficient=(particle_mass[:,None]*weights[pids]/np.repeat(band_mass,guides.shape[1])[:,None]).reshape(-1)
    guide_nodes=idx[pids].reshape(-1)
    pair_key=guide_labels.astype(np.int64)*node_count+guide_nodes
    unique_key,inverse=np.unique(pair_key,return_inverse=True)
    pair_coefficient=np.bincount(inverse,weights=coefficient)
    pair_guide=(unique_key//node_count).astype(np.int32);pair_node=(unique_key%node_count).astype(np.int32)
    if np.any(nodal_mass[pair_node]<=0):raise RuntimeError('A guide constraint uses a zero-mass reference node')
    node_order=np.argsort(pair_node,kind='stable');sorted_nodes=pair_node[node_order]
    _,node_start,node_counts=np.unique(sorted_nodes,return_index=True,return_counts=True)
    conflicts=np.flatnonzero(node_counts>1)
    diagonal=np.bincount(pair_guide,weights=pair_coefficient**2/nodal_mass[pair_node],minlength=len(guides))
    if np.any(diagonal<=0):raise RuntimeError('Guide Gram diagonal is not positive')

    operator=make_guide_operator(jnp.asarray(x),jnp.asarray(volume),rho,support,jnp.asarray(nodal_mass),jnp.asarray(guides))
    rng=np.random.default_rng(90731);rhs=rng.normal(size=len(guides)).astype(np.float64)
    values=jnp.zeros((node_count,3),dtype=jnp.float64)
    projected,multipliers,residual=project(operator,values,jnp.asarray(rhs),iterations=96)
    cg_lambda=np.asarray(multipliers);cg_residual=np.asarray(residual)
    direct_lambda=rhs/diagonal
    lambda_relative=float(np.linalg.norm(cg_lambda-direct_lambda)/max(np.linalg.norm(direct_lambda),1e-300))
    residual_relative=float(np.linalg.norm(cg_residual)/max(np.linalg.norm(rhs),1e-300))
    result=dict(case='F7 real reference stencil: finite centroid constraint Gram structure',
        state='strictly-disjoint' if not len(conflicts) else 'overlap-detected',
        input_identity_sha256=identities,particles=len(x),fibres=int(fids.max()+1),guides=len(guides),
        guide_particles_per_section=int(guides.shape[1]),packed_node_count=int(node_count),
        sparse_stencil_entries=int(len(guide_nodes)),coalesced_guide_node_entries=int(len(pair_node)),
        duplicate_entries_merged=int(len(guide_nodes)-len(pair_node)),nodes_owned_by_multiple_guides=int(len(conflicts)),
        maximum_guides_per_node=int(node_counts.max()),conflicting_node_examples=[int(pair_node[node_order[node_start[k]]]) for k in conflicts[:12]],
        gram_diagonal_min=float(diagonal.min()),gram_diagonal_max=float(diagonal.max()),
        test_rhs_seed=90731,cg_iterations=96,direct_vs_original_cg_lambda_relative_error=lambda_relative,
        original_G_adjoint_residual_relative=residual_relative,compile_and_check_seconds=time.perf_counter()-start,
        mathematical_conclusion=('G M^-1 G^T is exactly diagonal after same-guide duplicate coalescing; exact Jacobi division is the exact solve'
            if not len(conflicts) else 'guide supports intersect; retain a sparse coupled solve with diagonal preconditioner and residual stopping'),
        geometry_mutated=False,physical_steps=0)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'guide-diagonality.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(output/'algebra-comparison.npz',rhs=rhs,direct_lambda=direct_lambda,cg_lambda=cg_lambda,
        gram_diagonal=diagonal,residual=cg_residual)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='output');run(parser.parse_args().output)
