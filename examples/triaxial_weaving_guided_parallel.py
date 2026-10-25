"""Same-world-grid per-cord APIC dispatch in certified contact-free blocks.

A real coverage pair switches the caller back to the original globally coupled
per-dt contact operator. No force is omitted inside a certified block: empty
coverage plus every particle's cumulative path<skin/2 proves zero contact.
"""
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import numpy as np


class GuidedParallelFibres:
    def __init__(self,geometry,plan,material,config,workers=16):
        import jax
        import jax.numpy as jnp
        from mpm_jax.fibre_mpm import _step
        from mpm_jax.fibre_normal_guides import normal_guides
        from .triaxial_weaving_geometry import fixture_position
        self.geometry=geometry;self.plan=plan;self.config=config;self.pool=ThreadPoolExecutor(max_workers=workers)
        self.capacities={};self.map_capacities={};self.workers=workers;self.last_resource={}
        meta=geometry['metadata'];times=jnp.asarray(meta['fixture_times']);cap=3*meta['cross_section_particles']
        @partial(jax.jit,static_argnames=('count',))
        def block(s,layout,positions,guide_ids,release_times,count):
            index=np.arange(len(s.x))
            def speed(group):return lambda t:(fixture_position(t,times,positions,jnp)[group]-fixture_position(t-config.dt,times,positions,jnp)[group])/config.dt
            def guide_speed(i):
                return lambda t:(fixture_position(jnp.minimum(t,release_times[i]),times,positions,jnp)[0,1]-fixture_position(jnp.minimum(t-config.dt,release_times[i]),times,positions,jnp)[0,1])/config.dt
            boundary=normal_guides((index<cap,index>=len(index)-cap),(speed(0),speed(1)),guide_ids,tuple(guide_speed(i) for i in range(len(guide_ids))),release_times,mass=s.mass,wave_speed=plan['elastic_wave_speed_m_s'])
            def body(carry,_):
                old,path,ledger=carry
                def attempt(_):
                    new,diag=_step(old,material,layout,config,None,None,boundary)
                    maximum=jnp.max(jnp.abs(jnp.log(jnp.linalg.svd(new.F,compute_uv=False))))
                    courant=config.dt*(plan['elastic_wave_speed_m_s']+jnp.max(jnp.linalg.norm(new.v,axis=1)))/plan['dx']
                    updated_path=path+jnp.linalg.norm(new.x-old.x,axis=1)
                    flags=(~new.valid).astype(jnp.int32)+8*(maximum>.12).astype(jnp.int32)+16*jnp.any(new.history.damage>1e-10).astype(jnp.int32)+32*(courant>.25).astype(jnp.int32)+64*(jnp.max(updated_path)>=.0003/2).astype(jnp.int32)
                    new=new._replace(valid=new.valid&(flags==0))
                    targets=boundary.targets(new.time)
                    group=jnp.sum(diag.group_impulses*targets,axis=1)
                    metrics=jnp.array([diag.boundary_work,jnp.sum(group),jnp.sum(jnp.abs(group)),courant,maximum,flags])
                    increment=jnp.concatenate((diag.group_impulses,group[:,None],diag.group_work[:,None],jnp.abs(group)[:,None]),axis=1)
                    return (new,updated_path,ledger+increment),metrics
                return jax.lax.cond(old.valid,attempt,lambda _:((old,path,ledger),jnp.zeros(6)),None)
            (result,path,ledger),metrics=jax.lax.scan(body,(s,jnp.zeros(len(s.x)),jnp.zeros((boundary.group_count,6))),None,length=count)
            return result,path,metrics,ledger
        self.block=block

    def advance(self,state,accumulator,count):
        import time,jax
        import jax.numpy as jnp
        from mpm_jax.fibre_mpm import FibreState,validate_state
        from mpm_jax.materials.isotropic_brittle import BrittleHistory
        from mpm_jax.fibre_hash_layout import build_hash_tube_layout
        from .triaxial_weaving_geometry import fixture_position
        geometry=self.geometry;meta=geometry['metadata'];x=np.asarray(state.x);v=np.asarray(state.v);ids=np.asarray(state.fibre_id)
        span=200*self.config.dt;t=float(state.time);times=meta['fixture_times'];positions=meta['fixture_positions']
        seeds=[x,x+span*v];seed_ids=[ids,ids];base=fixture_position(t,times,positions)
        for fraction in [.5,1.]:
            target=fixture_position(t+span*fraction,times,positions)
            for mask,displacement in zip(geometry['masks'],target-base):seeds.append(x[mask]+displacement);seed_ids.append(ids[mask])
        seeds=np.concatenate(seeds);seed_ids=np.concatenate(seed_ids);arguments=[];resources=[];begin=time.perf_counter()
        for fibre in meta['fibres']:
            i=fibre['id'];first,last=fibre['particle_slice'];sl=slice(first,last)
            fields={k:getattr(state,k)[sl] for k in ['x','v','F','C','mass','volume','tau']}
            fields.update(fibre_id=jnp.zeros(last-first,jnp.int32),history=BrittleHistory(*(getattr(state.history,k)[sl] for k in BrittleHistory._fields)),time=state.time,valid=state.valid,boundary_impulse=jnp.zeros(3),external_impulse=jnp.zeros(3),boundary_work=jnp.asarray(0.))
            local=FibreState(**fields);selected=seeds[seed_ids==i]
            layout,r=build_hash_tube_layout(selected,np.zeros(len(selected),np.int32),dx=self.plan['dx'],padding_cells=3,capacity=self.capacities.get(i),capacity_factor=(1.25 if i not in self.capacities else None),direct_map_capacity=self.map_capacities.get(i,'auto'),max_active_nodes=8_000_000)
            self.capacities[i]=layout.node_count;self.map_capacities[i]=r['direct_map_capacity'];validate_state(local,layout)
            selected_guides=[g for g in meta['placement_guides'] if g['fibre_id']==i]
            arguments.append((local,layout,jnp.asarray(positions)[:,2*i:2*i+2],jnp.asarray([np.asarray(g['particle_ids'])-first for g in selected_guides]),jnp.asarray([g['release_time_s'] for g in selected_guides]),count));resources.append(r)
        rebuild_seconds=time.perf_counter()-begin;cpu=time.process_time();wall=time.perf_counter()
        def execute(arguments):result=self.block(*arguments);jax.block_until_ready(result);return result
        records=list(self.pool.map(execute,arguments)) # map preserves fixed fibre-ID order
        elapsed=time.perf_counter()-wall;cpu_seconds=time.process_time()-cpu
        fields={k:jnp.concatenate([getattr(row[0],k) for row in records]) for k in ['x','v','F','C','mass','volume','tau']}
        fields.update(fibre_id=state.fibre_id,history=BrittleHistory(*(jnp.concatenate([getattr(row[0].history,k) for row in records]) for k in BrittleHistory._fields)),time=records[0][0].time,valid=jnp.all(jnp.stack([row[0].valid for row in records])),boundary_impulse=state.boundary_impulse+jnp.sum(jnp.stack([row[0].boundary_impulse for row in records]),axis=0),external_impulse=state.external_impulse+jnp.sum(jnp.stack([row[0].external_impulse for row in records]),axis=0),boundary_work=state.boundary_work+jnp.sum(jnp.stack([row[0].boundary_work for row in records])))
        result=FibreState(**fields);per_step=jnp.sum(jnp.stack([row[2][:,:3] for row in records]),axis=0)
        grid=per_step[:,0];machine=per_step[:,1];absolute=per_step[:,2]
        increment=jnp.array([0.,jnp.sum(jnp.abs(grid)),jnp.sum(machine),jnp.sum(machine-grid),jnp.sum(absolute),0.,0.,0.])
        updated=accumulator+increment;metrics=jnp.stack([row[2] for row in records]);updated=updated.at[7].set(jnp.maximum(accumulator[7],jnp.max(metrics[:,:,3])))
        flags=jnp.max(metrics[:,:,5]).astype(jnp.int32);jax.block_until_ready((result,updated,flags))
        self.last_ledger=jnp.zeros((len(self.geometry['masks'])+len(meta['placement_guides']),6))
        for fibre,row in zip(meta['fibres'],records):
            i=fibre['id'];groups=[2*i,2*i+1]+[len(self.geometry['masks'])+g['id'] for g in meta['placement_guides'] if g['fibre_id']==i]
            self.last_ledger=self.last_ledger.at[jnp.asarray(groups)].add(row[3])
        self.last_resource=dict(workers=self.workers,local_rebuild_seconds=rebuild_seconds,dispatch_wall_seconds=elapsed,dispatch_cpu_seconds=cpu_seconds,cpu_wall_ratio=cpu_seconds/elapsed,occupied_nodes=sum(r['occupied_nodes'] for r in resources),capacity=sum(r['active_nodes'] for r in resources),map_bytes=sum(r['direct_map_bytes'] for r in resources),absolute_path_bound_m=max(float(row[1].max()) for row in records),max_log_stretch=float(jnp.max(metrics[:,:,4])),source_operator='Original _step, same material/SVD and world grid',contact_mode='Certified empty real coverage and cumulative absolute path<skin/2; otherwise globally coupled original operator',fixed_fibre_id_reassembly=True)
        return result,updated,flags
