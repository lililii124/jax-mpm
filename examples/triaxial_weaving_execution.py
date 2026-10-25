"""Bounded moving execution support; no material or fixture state is prescribed here."""
import time
from dataclasses import replace
import numpy as np
from .triaxial_weaving_geometry import fixture_position
from .triaxial_fibre_geometry import _segment_distance


def allocate(state,geometry,dx,dt,window_steps,packed_capacity=None,lookup='hash',direct_capacity=None):
    """Current free body + ballistic support + actual next guide sweep, not a body target.

    Three cells cover unpredicted transverse deformation in this short window.
    Every actual future quadratic stencil and nonrigid coverage bound still
    rejects an insufficient allocation. Packed capacity stays fixed after its
    first conservative allocation; overflow stops at the preceding checkpoint.
    """
    import jax.numpy as jnp
    from mpm_jax.fibre_hash_layout import build_hash_tube_layout
    from mpm_jax.fibre_coupling import build_contact_data
    from mpm_jax.contact_patches import patch_neighbours
    start=time.perf_counter();x=np.asarray(state.x);v=np.asarray(state.v);ids=geometry['fibre_id'];span=dt*window_steps
    seeds=[x,x+span*v];seed_ids=[ids,ids]
    t=float(state.time);times=geometry['metadata']['fixture_times'];positions=geometry['metadata']['fixture_positions'];base=fixture_position(t,times,positions)
    for fraction in [.5,1.]:
        target=fixture_position(t+span*fraction,times,positions)
        for mask,displacement in zip(geometry['masks'],target-base):
            seeds.append(x[mask]+displacement);seed_ids.append(ids[mask])
    layout,grid=build_hash_tube_layout(np.concatenate(seeds),np.concatenate(seed_ids),dx=dx,padding_cells=3,
        max_active_nodes=8_000_000,capacity=packed_capacity,capacity_factor=(2. if packed_capacity is None else None),direct_map_capacity=(('auto' if direct_capacity is None else direct_capacity) if lookup=='direct' else None))
    grid_seconds=time.perf_counter()-start;contact_start=time.perf_counter();skin=.0003
    data=build_contact_data(geometry,search_window=8,coverage_skin=skin,coverage_positions=x,max_coverage_pairs=2_000_000)
    actual=np.asarray(data.coverage_pairs);count=len(actual)
    # Fixed per-station capacity; all real entries are retained or the job fails.
    capacity=1<<int((max(4096,256*len(geometry['contact_candidates']))-1).bit_length())
    if count>capacity:raise RuntimeError('Real contact coverage exceeds fixed capacity; no pairs were discarded')
    graph=np.full((capacity,8),-1,np.int32);graph[:count]=patch_neighbours(actual,np.asarray(data.segment_fibre))
    fibre=geometry['metadata']['fibres'][0];a,b=fibre['particle_slice'];centre=x[a:b].reshape(fibre['longitudinal_sections'],-1,3).mean(1)
    far=np.array([0,len(centre)-2],np.int32);distance=_segment_distance(centre[:2],centre[-2:])
    radius=fibre['radius_m'];far_gap=float(distance-2*radius)
    if far_gap<=skin+1e-9:raise RuntimeError('No certified-separated padding pair for this station')
    padded=np.concatenate([actual,np.tile(far,(capacity-count,1))]);seed=np.pad(np.asarray(data.coverage_seed),(0,capacity-count),constant_values=-1)
    data=replace(data,coverage_pairs=jnp.asarray(padded),coverage_seed=jnp.asarray(seed))
    resource=dict(grid,grid_seconds=grid_seconds,contact_seconds=time.perf_counter()-contact_start,
        real_coverage_pairs=count,coverage_capacity=capacity,padding_far_reference_gap_m=far_gap,skin_m=skin,
        future_window_steps=window_steps,free_deformation_padding_cells=3,complete_stencil_fail_closed=True,
        padding_proof='far reference gap>skin and2*nonrigid displacement<=skin imply padding remains strictly inactive')
    return layout,data,jnp.asarray(graph),resource
