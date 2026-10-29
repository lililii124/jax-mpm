"""Explicit Ti64 cutting parameter groups (SI), with one displacement law.

Seo changes the five flow-stress constants only. Shared thermal, elastic and
failure assumptions are retained; this is not a full reproduction of the paper.
"""
from mpm_jax.materials import johnson_cook_displacement

FLOW_PARAMETERS = {
    'johnson': dict(A=862.5e6, B=331.2e6, n=.34, C=.012, m=.8),
    'seo': dict(A=997.9e6, B=653.1e6, n=.45, C=.0198, m=.7),
}
SHARED_PARAMETERS = dict(
    young=109e9, poisson=.34, density=4430., reference_rate=1.,
    reference_temperature=298.15, melt_temperature=1878.15,
    heat_capacity=611., heat_fraction=.9,
)
FAILURE_PARAMETERS = (-.09, .25, -.5, .014, 3.87)

def cutting_material(preset='johnson', *, damage=False, dx=None,
                     failure_displacement_cells=2., fast_return=False):
    if preset not in FLOW_PARAMETERS:
        raise ValueError(f'unknown cutting material preset: {preset}')
    length=1. if dx is None else dx
    return johnson_cook_displacement(
        **SHARED_PARAMETERS, **FLOW_PARAMETERS[preset],
        failure=FAILURE_PARAMETERS if damage else None,
        damage_length=length,
        failure_displacement=failure_displacement_cells*length,
        fast_return=fast_return,
    )
