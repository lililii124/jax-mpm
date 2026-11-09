import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import Config,block,neo_hookean
from mpm_jax.adaptive_crack import retry_step
from mpm_jax.cohesive_crack import CohesiveLaw
from mpm_jax.extrinsic_crack import initialize,step

jax.config.update('jax_enable_x64',True)


def loaded_interface():
    config=Config(dim=2,n_grid=16,dt=.01,gravity=(0.,0.),boundary_cells=0)
    state=block(config,(.25,.375),(.75,.625),dtype=jnp.float64)
    state=state._replace(F=state.F.at[:,1,1].set(1.03))
    path=initialize([[.25,.5],[.5,.5],[.75,.5]],capacity=6)
    law=CohesiveLaw(100.,1.,1e5)
    path=path._replace(birth_traction=path.birth_traction.at[:2,1].set(100.),
                      maximum=path.maximum.at[:2].set(.001),bonded=path.bonded.at[:2].set(True))
    return state,path,neo_hookean(1e6,.2),config,law


def test_rejected_attempts_leave_no_damage_or_work_in_accepted_substep():
    state,path,material,config,law=loaded_interface()
    following,newpath,work,report=retry_step(state,path,material,config,law,config.dt,consistent_gradients=True)
    assert bool(report.accepted) and int(report.rejected)>0
    direct=step(state,path,material,config,law,consistent_gradients=True,dt=report.accepted_dt)
    for actual,expected in zip(jax.tree.leaves((following,newpath,work)),jax.tree.leaves(direct)):
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(following.time-state.time,report.accepted_dt,rtol=1e-14)


def test_exhausted_retries_return_original_state_history_and_zero_ledger():
    state,path,material,config,law=loaded_interface()
    following,newpath,work,report=retry_step(state,path,material,config,law,config.dt,
                                           max_halvings=0,consistent_gradients=True)
    assert not bool(report.accepted) and int(report.rejected)==1
    for actual,expected in zip(jax.tree.leaves((following,newpath)),jax.tree.leaves((state,path))):
        np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(jnp.stack(work),jnp.zeros(8))
    assert float(report.accepted_dt)==0.
