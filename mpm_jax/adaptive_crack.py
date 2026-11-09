"""Transactional time-step retries for the moving cohesive solver."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .extrinsic_crack import Work, step


class RetryReport(NamedTuple):
    accepted: jax.Array
    attempts: jax.Array
    rejected: jax.Array
    accepted_dt: jax.Array
    last_attempt_dt: jax.Array
    last_residual: jax.Array


@partial(jax.jit,static_argnames=('config','law','max_halvings','consistent_gradients'))
def retry_step(state,path,material,config,law,dt,max_halvings=8,consistent_gradients=False):
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
                                    consistent_gradients=consistent_gradients,dt=trial_dt)
        accepted=following.valid & newpath.valid & jnp.all(jnp.isfinite(jnp.stack(work)))
        returned=jax.lax.cond(accepted,lambda _:(following,newpath,work),lambda _:(state,path,empty),None)
        report=RetryReport(accepted,previous.attempts+1,previous.rejected+(~accepted).astype(jnp.int32),
                           jnp.where(accepted,trial_dt,zero),trial_dt,work.residual)
        return (*returned,report)

    return jax.lax.while_loop(pending,attempt,(state,path,empty,report))
