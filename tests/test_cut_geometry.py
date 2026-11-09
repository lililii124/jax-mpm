import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import Config
from mpm_jax.core import stencil
from mpm_jax.cut_geometry import polygon,clip,split,bulk_quadrature,boundary_quadrature

jax.config.update('jax_enable_x64',True)


def moments(poly):
    x,w=bulk_quadrature(poly)
    return jnp.concatenate((jnp.array([w.sum()]),jnp.sum(w[:,None]*x,axis=0)))


def test_batched_splits_preserve_area_and_centroid_and_ad():
    p=polygon([[.1,.2],[.9,.2],[.9,.8],[.1,.8]])
    angles=jnp.linspace(-1.1,1.3,31);normals=jnp.stack((jnp.cos(angles),jnp.sin(angles)),axis=1)
    offsets=normals@jnp.array([.5,.5])+jnp.linspace(-.28,.28,31)
    @jax.jit
    @jax.vmap
    def cut(n,o):
        a,b=split(p,n,o)
        return moments(a)+moments(b),a.valid&b.valid
    combined,valid=cut(normals,offsets)
    assert bool(jnp.all(valid))
    np.testing.assert_allclose(combined,jnp.tile(moments(p),(31,1)),atol=4e-16,rtol=1e-14)
    area=lambda offset:moments(clip(p,jnp.array([1.,0.]),offset))[0]
    np.testing.assert_allclose(jax.jit(jax.value_and_grad(area))(.37),[.318,-.6],atol=2e-15)
    for offset in (.1,.9,1.1):
        q=clip(p,jnp.array([1.,0.]),offset)
        assert bool(q.valid)
        assert np.isfinite(np.asarray(moments(q))).all()
        assert int(q.count)<=4
        if offset>=.9:
            np.testing.assert_array_equal(boundary_quadrature(q)[1],0.)


def test_actual_spline_bulk_boundary_balance_inside_knot_piece():
    config=Config(dim=2,n_grid=16,dt=1e-5,gravity=(0.,0.),boundary_cells=0)
    # This whole polygon lies in one tensor-polynomial spline interval.
    p=polygon([[.476,.477],[.522,.480],[.520,.524],[.478,.520]])
    size=17**2
    @jax.jit
    def residual(normal,offset):
        q=clip(p,normal,offset)
        x,w=bulk_quadrature(q);b=stencil(x,config,active=w>0)
        bulk=jnp.zeros((size,2)).at[b.indices].add(w[:,None,None]*b.gradients)
        y,z,n=boundary_quadrature(q);f=stencil(y,config,active=z>0)
        boundary=jnp.zeros((size,2)).at[f.indices].add(z[:,None,None]*f.weights[:,:,None]*n[:,None,:])
        return jnp.linalg.norm(bulk-boundary)/jnp.linalg.norm(bulk),q.valid&b.valid&f.valid
    for angle,phase in ((0.,0.),(.31,.004),(.68,-.008)):
        n=jnp.array([np.cos(angle),np.sin(angle)])
        error,valid=residual(n,n@jnp.array([.5,.5])+phase)
        assert bool(valid)
        assert float(error)<5e-13


def test_invalid_geometry_and_capacity_overflow_are_reported():
    assert not bool(polygon([[0.,0.],[0.,1.],[1.,1.],[1.,0.]]).valid)
    assert not bool(polygon([[0.,0.],[1.,0.],[.2,.2],[0.,1.]]).valid)
    p=polygon([[0.,0.],[1.,0.],[1.,1.],[0.,1.]],capacity=4)
    q=clip(p,jnp.array([1.,1.]),.25)
    assert not bool(q.valid)  # five true vertices cannot fit in capacity four
