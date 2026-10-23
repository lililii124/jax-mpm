"""Material point method in JAX."""

from .core import Config, Material, State, advance, elastic, fluid, step
from .initialization import block, validate_state
from .materials import linear_elastic, neo_hookean, orthotropic, transverse_isotropic
from .materials import hjc, jh2, johnson_cook, rht
from .materials import bilinear_plastic, thermal_plastic, blatz_ko
from .materials import steinberg_guinan
from .contact import coupled_advance, coupled_step
from .boundaries import (
    CyclicVelocityBox, MovingVelocityBox, PlaneWall, RotatingVelocityBox, VelocityBox,
    fixed_box,
)
from .contact_laws import (
    RigidSphereBoundary,
    hertz_contact,
    jkr_contact,
    jkr_history,
    reduced_modulus,
)

__all__ = [
    "Config",
    "Material",
    "State",
    "advance",
    "block",
    "elastic",
    "fluid",
    "hjc",
    "jh2",
    "johnson_cook",
    "rht",
    "bilinear_plastic",
    "thermal_plastic",
    "blatz_ko",
    "steinberg_guinan",
    "coupled_advance",
    "coupled_step",
    "PlaneWall",
    "VelocityBox",
    "MovingVelocityBox",
    "CyclicVelocityBox",
    "RotatingVelocityBox",
    "RigidSphereBoundary",
    "hertz_contact",
    "jkr_contact",
    "jkr_history",
    "reduced_modulus",
    "fixed_box",
    "linear_elastic",
    "neo_hookean",
    "orthotropic",
    "transverse_isotropic",
    "step",
    "validate_state",
]
