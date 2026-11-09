"""Transactional time-step retries for the moving cohesive solver."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .extrinsic_crack import Work, insert_tip, step


class RetryReport(NamedTuple):
    accepted: jax.Array
    attempts: jax.Array
    rejected: jax.Array
    accepted_dt: jax.Array
    last_attempt_dt: jax.Array
    last_residual: jax.Array


@partial(jax.jit,static_argnames=('config','law','max_halvings','consistent_gradients','couple_tip_nodes','face_mass_fraction'))
def retry_step(state,path,material,config,law,dt,max_halvings=8,consistent_gradients=False,couple_tip_nodes=False,face_mass_fraction=0.):
    """Return one accepted substep or the unchanged input and zero work.

    Each rejected attempt starts again from the original state, including the
    irreversible interface history. Callers must check report.accepted and must
    insert new crack segments only after acceptance. This is a nonlinear-solver
    recovery mechanism, not a time-discretization error estimator.
    """
    if not isinstance(max_halvings,int) or max_halvings<0:
        raise ValueError('max_halvings must be a nonnegative integer')
    dtype=state.x.dtype
    dt=jnp.asarray(dt,dtype=dtype)
    zero=jnp.asarray(0.,dtype=dtype)
    count=jnp.asarray(0,dtype=jnp.int32)
    empty=Work(zero,zero,zero,zero,zero,zero,jnp.asarray(0),zero)
    report=RetryReport(jnp.asarray(False),count,count,zero,zero,zero)
    can_try=state.valid & path.valid & jnp.isfinite(dt) & (dt>0)

    def pending(carry):
        report=carry[3]
        return can_try & ~report.accepted & (report.attempts<=max_halvings)

    def attempt(carry):
        previous=carry[3]
        trial_dt=dt*2.**(-previous.attempts)
        following,newpath,work=step(state,path,material,config,law,
                                    consistent_gradients=consistent_gradients,dt=trial_dt,couple_tip_nodes=couple_tip_nodes,
                                    face_mass_fraction=face_mass_fraction)
        accepted=following.valid & newpath.valid & jnp.all(jnp.isfinite(jnp.stack(work)))
        returned=jax.lax.cond(accepted,lambda _:(following,newpath,work),lambda _:(state,path,empty),None)
        report=RetryReport(accepted,previous.attempts+1,previous.rejected+(~accepted).astype(jnp.int32),
                           jnp.where(accepted,trial_dt,zero),trial_dt,work.residual)
        return (*returned,report)

    return jax.lax.while_loop(pending,attempt,(state,path,empty,report))


class Controller(NamedTuple):
    proposed_dt: jax.Array
    accepted: jax.Array
    rejected: jax.Array
    quiet: jax.Array
    minimum_used: jax.Array
    maximum_residual: jax.Array
    stop_code: jax.Array


def controller(dt,dtype=jnp.float64):
    zero=jnp.asarray(0,dtype=jnp.int32)
    duration=jnp.asarray(dt,dtype=dtype)
    return Controller(duration,zero,zero,zero,duration,jnp.asarray(0.,dtype=dtype),zero)


@partial(jax.jit,static_argnames=('config','law','consistent_gradients','couple_tip_nodes','max_steps','face_mass_fraction'))
def integrate_to(state,path,material,config,law,ledger,control,target,spacing,radius,max_extension,
                 consistent_gradients=True,couple_tip_nodes=True,max_steps=100000,face_mass_fraction=0.):
    """Advance through accepted steps; ledger holds eight work fields and insertion energy.

    stop_code: 1 exhausted retries, 2 minimum duration, 3 invalid insertion,
    4 accepted-step budget. A stopped advance retains its last accepted state.
    """
    tolerance=64*jnp.finfo(state.x.dtype).eps*jnp.maximum(jnp.abs(target),config.dt)
    minimum=config.dt/256

    def pending(carry):
        s,p,w,c=carry
        return (s.time<target-tolerance)&(c.accepted<max_steps)&(c.stop_code==0)

    def advance(carry):
        s,p,w,c=carry
        requested=jnp.minimum(c.proposed_dt,target-s.time)
        ns,npth,increment,report=retry_step(s,p,material,config,law,requested,
                                           consistent_gradients=consistent_gradients,couple_tip_nodes=couple_tip_nodes,
                                           face_mass_fraction=face_mass_fraction)
        small=report.accepted & (report.accepted_dt<minimum)&(requested>=minimum)
        accepted=report.accepted & ~small

        def commit(_):
            inserted,event=insert_tip(ns,npth,material,config,law,spacing,radius,max_extension)
            update=jnp.pad(jnp.stack(increment),(0,1)).at[8].set(event.energy_jump)
            return ns,inserted,w+update

        ns,npth,nw=jax.lax.cond(accepted,commit,lambda _:(s,p,w),None)
        insertion_ok=ns.valid & npth.valid
        accepted &= insertion_ok
        ns,npth,nw=jax.lax.cond(accepted,lambda _:(ns,npth,nw),lambda _:(s,p,w),None)
        quiet=jnp.where(report.rejected>0,0,c.quiet+1)
        proposed=jnp.where(report.rejected>0,report.accepted_dt,c.proposed_dt)
        proposed=jnp.where(quiet>=32,jnp.minimum(2*proposed,config.dt),proposed)
        quiet=jnp.where(quiet>=32,0,quiet)
        code=jnp.where(~report.accepted,1,jnp.where(small,2,jnp.where(~insertion_ok,3,0)))
        nc=Controller(proposed,c.accepted+accepted.astype(jnp.int32),c.rejected+report.rejected,quiet,
                      jnp.minimum(c.minimum_used,jnp.where(accepted,report.accepted_dt,c.minimum_used)),
                      jnp.maximum(c.maximum_residual,jnp.where(accepted,increment.residual,0.)),code.astype(jnp.int32))
        return ns,npth,nw,nc

    following,newpath,work,result=jax.lax.while_loop(pending,advance,(state,path,ledger,control))
    exhausted=(following.time<target-tolerance)&(result.accepted>=max_steps)&(result.stop_code==0)
    result=result._replace(stop_code=jnp.where(exhausted,4,result.stop_code))
    return following,newpath,work,result
