import jax
import numpy as np
import pytest

from examples.hierarchical_wave import sample_bar
from mpm_jax.boundaries import fixed_box
from mpm_jax.hierarchical_dynamics import advance, initial_state
from mpm_jax.hierarchical_grid import uniform_grid
from mpm_jax.materials import neo_hookean

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("resolution,cells,offset", [("coarse",80,0.), ("fine",160,.5)])
def test_uniform_plane_strain_wave_is_independent_of_transverse_sampling(resolution, cells, offset):
    h = .2 / cells
    grid = uniform_grid(cells=cells,domain_size=.2,offset=offset,
                        node_bounds=((0,round(.08/h),round(.08/h)),(cells,round(.12/h),round(.12/h))))
    material = neo_hookean(20e6,.3)
    c = np.sqrt((material.lam+2*material.mu)/1200.)
    boundaries = (fixed_box((0.,)*3,(.2,)*3,components=(False,True,True)),)
    fields = []
    for ppc in (1,2):
        x,volume = sample_bar(resolution,transverse_ppc=ppc)
        phase=(x[:,0]-.045)/.006
        v=np.zeros_like(x)
        v[:,0]=.02*np.exp(-phase**2)
        F=np.broadcast_to(np.eye(3),(len(x),3,3)).copy()
        F[:,0,0]-=v[:,0]/c
        C=np.zeros_like(F)
        C[:,0,0]=-2*phase/.006*v[:,0]/F[:,0,0]
        state=initial_state(x,volume,1200.,grid,velocity=v,deformation=F,affine=C)
        final,record=advance(state,material,grid,1e-6,steps=20,boundaries=boundaries)
        assert np.asarray(record['valid']).all()
        coordinates,ids=np.unique(x[:,0],return_inverse=True)
        mass=np.bincount(ids,weights=np.asarray(final.mass))
        vx=np.bincount(ids,weights=np.asarray(final.mass*final.v[:,0]))/mass
        Fxx=np.bincount(ids,weights=np.asarray(final.mass*final.F[:,0,0]))/mass
        np.testing.assert_allclose(final.v[:,0],vx[ids],rtol=0,atol=2e-11)
        fields.append((coordinates,vx,Fxx))
    np.testing.assert_array_equal(fields[0][0],fields[1][0])
    np.testing.assert_allclose(fields[0][1],fields[1][1],rtol=0,atol=2e-11)
    np.testing.assert_allclose(fields[0][2],fields[1][2],rtol=0,atol=2e-12)
