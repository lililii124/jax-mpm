import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax.cut_geometry import polygon
from mpm_jax.cut_fragments import fragment,split_fragment,invariants,current_domain,moments

jax.config.update('jax_enable_x64',True)


def specimen():
    return fragment(polygon([[-.02,-.015],[.02,-.015],[.02,.015],[-.02,.015]]),1200.,
                    x=[.501,.487],v=[.6,-.4],F=[[1.07,.21],[-.11,.95]],C=[[.7,-.3],[1.1,.2]],
                    history={'plastic_strain':jnp.array(.037),'direction':jnp.array([.3,.4])})


def test_arbitrary_spatial_cuts_conserve_affine_invariants_and_history():
    parent=specimen();dx=1/16
    angles=jnp.linspace(-1.2,1.1,17);normals=jnp.stack((jnp.cos(angles),jnp.sin(angles)),axis=1)
    offsets=normals@parent.x+jnp.linspace(-.012,.012,17)
    @jax.jit
    @jax.vmap
    def cut(n,o):
        children=split_fragment(parent,n,o)
        values=jax.vmap(lambda c:invariants(c,dx))(children)
        return children,jax.tree.map(lambda x:x.sum(axis=0),values)
    children,totals=cut(normals,offsets)
    assert bool(jnp.all(children.valid))
    expected=invariants(parent,dx)
    for actual,target in zip(totals,expected):
        np.testing.assert_allclose(actual,jnp.broadcast_to(target,actual.shape),atol=3e-15,rtol=3e-13)
    np.testing.assert_array_equal(children.history['plastic_strain'],np.full((17,2),.037))
    np.testing.assert_array_equal(children.F,jnp.broadcast_to(parent.F,children.F.shape))
    # F and intensive history are identical; the area factor is the only change in V0 W.
    J=jnp.linalg.det(parent.F)
    density=.5*(jnp.sum(parent.F**2)-2)-jnp.log(J)+.4*jnp.log(J)**2
    np.testing.assert_allclose(children.reference_area.sum(axis=1)*density,parent.reference_area*density,atol=1e-17)
    physical=jax.vmap(jax.vmap(lambda c:moments(current_domain(c))))(children)
    np.testing.assert_allclose(physical.area,children.reference_area*J,atol=1e-17,rtol=1e-12)


def test_empty_and_small_fragments_are_not_silently_deleted():
    parent=specimen();normal=jnp.array([1.,0.])
    empty=split_fragment(parent,normal,2.)
    assert bool(jnp.all(empty.valid)) and int(jnp.sum(empty.active))==1
    np.testing.assert_allclose(empty.mass.sum(),parent.mass,atol=1e-15)
    edge=jnp.max(current_domain(parent).vertices[:4,0])
    tiny=split_fragment(parent,normal,edge-1e-5)
    assert bool(jnp.all(tiny.valid&tiny.active))
    assert 0<float(tiny.mass[0]/parent.mass)<1e-5
    value=lambda o:jax.vmap(lambda c:invariants(c,1/16).kinetic_energy)(split_fragment(parent,normal,o)).sum()
    np.testing.assert_allclose(jax.jit(jax.grad(value))(.504),0.,atol=1e-12)
def test_world_coordinate_pullback_does_not_create_unresolved_boundary_slivers():
    h=.06/96/2
    reference=polygon(h*jnp.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]))
    parent=fragment(reference,1200.,x=[.01015625,.02+(31+.5)*h])
    pieces=split_fragment(parent,jnp.array([0.,1.]),.03)
    assert int(pieces.active.sum())==1
    np.testing.assert_allclose(pieces.mass.sum(),parent.mass,rtol=2e-14)
    np.testing.assert_allclose(pieces.reference_area.sum(),parent.reference_area,rtol=2e-14)
    # A resolved offset still creates both positive pieces.
    resolved=split_fragment(parent,jnp.array([0.,1.]),.03-1e-10)
    assert int(resolved.active.sum())==2

