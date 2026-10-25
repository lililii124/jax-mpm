"""One translating free blob checks the newly connected resin step only."""
import json
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from mpm_jax.resin_apic import MACLayout,component_shape
from mpm_jax.resin_fill import FlowState,make_step


def main():
    h=.00025;dt=.00004;layout=MACLayout((10,10,10),(0.,0.,0.),h)
    a=(np.arange(3,7)+.5)*h;x=np.stack(np.meshgrid(a,a,a,indexing='ij'),axis=-1).reshape(-1,3)
    velocity=np.array([.01,-.007,.003]);n=len(x)
    fixed=tuple(jnp.zeros(component_shape(layout,c),dtype=bool) for c in range(3));prescribed=tuple(jnp.zeros(component_shape(layout,c)) for c in range(3))
    state=FlowState(jnp.asarray(x),jnp.broadcast_to(velocity,(n,3)),jnp.zeros((n,3,3)),jnp.full(n,1100*h**3),jnp.ones(n,dtype=bool),jnp.asarray(0.),jnp.asarray(True))
    step=make_step(layout,fixed,prescribed,jnp.ones(layout.shape,dtype=bool),dt=dt)
    after,m=step(state);jax.block_until_ready(after.x)
    errors=dict(position=float(jnp.max(jnp.abs(after.x-(x+dt*velocity)))),velocity=float(jnp.max(jnp.abs(after.v-velocity))),affine=float(jnp.max(jnp.abs(after.C))))
    assert bool(after.valid) and errors['position']<1e-15 and errors['velocity']<1e-14 and errors['affine']<1e-11
    assert abs(float(m['energy_identity_J']))<1e-20
    result=dict(passed=True,physical_fill_steps=0,operator_steps=1,errors=errors,metrics={k:float(v) for k,v in m.items()},scope='Connected step translating blob only; no actual inlet or obstacles')
    Path('resin-step-review.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
