"""Focused material verification independent of the plate calculation."""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as j
import numpy as np
from mpm_jax.hexagonal_hyperelastic import energy_density as W, first_piola as P


def rotation(a):
    c,s=np.cos(a),np.sin(a)
    return j.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


def test_material():
    I=j.eye(3);F=j.array([[1.05,.013,0.],[.002,.99,.008],[0.,0.,1.01]])
    Q=rotation(.71);R=rotation(np.pi/3)
    np.testing.assert_allclose(W(I),0.,atol=1.e-9)
    np.testing.assert_allclose(P(I),0.,atol=1.e-8)
    np.testing.assert_allclose(W(Q@F),W(F),rtol=1.e-12)
    np.testing.assert_allclose(P(Q@F),Q@P(F),rtol=1.e-12,atol=1.e-7)
    np.testing.assert_allclose(W(F@R),W(F),rtol=1.e-12)
    assert abs(float(W(F,angle=np.pi/6)-W(F)))>100.
    D=j.array([[.3,.1,.2],[-.2,.4,.1],[0.,-.1,.2]])
    eps=1.e-6
    np.testing.assert_allclose((W(F+eps*D)-W(F-eps*D))/(2*eps),j.sum(P(F)*D),rtol=1.e-8)
    stretch=1.05;s=(stretch**2-1)/2
    for a in (0.,.13,np.pi/6):
        n=j.array([np.cos(a),np.sin(a),0.]);U=I+(stretch-1)*j.outer(n,n)
        expected=1.e6*(stretch-1/stretch)+2.333333333e6*np.log(stretch)/stretch+5.e6/.03**2*s**3*stretch*(105+24*np.cos(6*a))/128
        np.testing.assert_allclose(n@P(U)@n,expected,rtol=2.e-12)
    print('PASS stress-free, objectivity, 60degree symmetry, 30degree contrast, AD derivative, analytic uniaxial stress')

if __name__=='__main__':test_material()
