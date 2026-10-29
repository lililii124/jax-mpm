"""Independent symmetry and normalization checks for the new directional cases."""
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as j
import numpy as np
from mpm_jax.directional_hyperelastic import energy_density as W, first_piola as P
from mpm_jax.hexagonal_hyperelastic import energy_density as original


def R(a):
    c,s=np.cos(a),np.sin(a)
    return j.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


def test_new_families():
    F=j.array([[1.09,.017,0.],[.002,.99,.004],[0.,0.,1.01]])
    periods={'square':np.pi/2,'rectangular':np.pi,'octagonal':np.pi/4,'hexagonal':np.pi/3,'isotropic':.137}
    for family,period in periods.items():
        args=dict(symmetry=family)
        np.testing.assert_allclose(W(j.eye(3),**args),0.,atol=1.e-8)
        np.testing.assert_allclose(P(j.eye(3),**args),0.,atol=1.e-8)
        np.testing.assert_allclose(W(R(.37)@F,**args),W(F,**args),rtol=2.e-12)
        np.testing.assert_allclose(W(F@R(period),**args),W(F,**args),rtol=2.e-12)
        if family!='isotropic':assert abs(float(W(F@R(period/2),**args)-W(F,**args)))>1.
    np.testing.assert_allclose(W(F,symmetry='hexagonal'),original(F),rtol=2.e-12)
    stretch=1.08;e=(stretch**2-1)/2;expected=5.e6/(4*.03**2)*e**4*105/128
    for family in ('square','rectangular','octagonal','hexagonal'):
        values=[]
        for angle in np.arange(16)*np.pi/16:
            n=j.array([np.cos(angle),np.sin(angle),0.]);U=j.eye(3)+(stretch-1)*j.outer(n,n)
            values.append(W(U,symmetry=family)-W(U,symmetry='isotropic'))
        np.testing.assert_allclose(np.mean(values),expected,rtol=3.e-12)
    print('PASS new-family stress-free/objective energy, C2/C4/C6/C8 symmetries and contrasts, exact hex agreement, equal analytic angular-mean reinforcement',flush=True)

if __name__=='__main__':test_new_families()
