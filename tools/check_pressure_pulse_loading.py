"""Focused area, time-impulse and rotated follower-direction check; no material update."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from mpm_jax.pressure_pulse import pressure,pressure_integral,surface_force


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();jax.config.update('jax_enable_x64',True)
    # Independent rectangular cell arithmetic at the engineering defaults.
    dimensions=np.array([.18,.012,.18]);counts=np.ceil(dimensions/(.24/128)*2).astype(int)
    volume=float(np.prod(dimensions)/np.prod(counts));layer_thickness=dimensions[1]/counts[1]
    area=float(counts[0]*counts[2]*volume/layer_thickness);assert abs(area-.18**2)<1e-15
    F=np.array([[[1.2,.17,.04],[.03,.8,.09],[.02,.05,.9]],[[1.1,-.02,.01],[0.,.95,.04],[.03,.07,1.02]]])
    weights=np.array([.013,.0194]);peak=5e6;duration=80e-6;time=duration/2
    cofactor=np.linalg.det(F)[:,None,None]*np.linalg.inv(F).transpose(0,2,1)
    expected=-peak*weights[:,None]*cofactor[:,:,1]
    actual=np.asarray(jax.jit(surface_force)(jnp.asarray(F),jnp.asarray(weights),time,peak,duration))
    np.testing.assert_allclose(actual,expected,rtol=2e-14,atol=1e-9)
    angle=.71;Q=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
    rotated=np.asarray(surface_force(jnp.asarray(Q@F),jnp.asarray(weights),time,peak,duration))
    np.testing.assert_allclose(rotated,actual@Q.T,rtol=2e-14,atol=1e-9)
    flat=np.asarray(surface_force(jnp.asarray(np.eye(3)[None]),jnp.array([area]),time,peak,duration))[0]
    np.testing.assert_allclose(flat,[0.,-peak*area,0.],rtol=1e-14,atol=1e-9)
    samples=jnp.arange(10000,dtype=jnp.float64)*duration/10000
    integral=float(np.asarray(pressure(samples,peak,duration)).sum()*duration/10000)
    exact=peak*duration/2;assert abs(integral-exact)<1e-10
    assert float(pressure(-duration,peak,duration))==float(pressure(2*duration,peak,duration))==0.
    assert abs(float(pressure_integral(2*duration,peak,duration))-exact)<1e-12
    result=dict(passed=True,physical_steps_executed=0,material_updates_executed=0,reference_area_m2=area,default_counts=counts.tolist(),default_particles=int(np.prod(counts)),
                flat_peak_force_N=flat.tolist(),pressure_impulse_Pa_s=exact,flat_reference_impulse_Ns=area*exact,
                discrete_time_integral_error_Pa_s=integral-exact,max_cofactor_force_error_N=float(abs(actual-expected).max()),max_rotation_force_error_N=float(abs(rotated-actual@Q.T).max()),
                scope='Only surface quadrature units, prescribed time integral and objective follower direction; not plate dynamics/material/energy qualification')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
