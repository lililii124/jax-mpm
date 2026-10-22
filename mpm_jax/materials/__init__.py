"""Material models and material-point updates."""

from .elastic import (
    LinearMaterial,
    Material,
    elastic,
    fluid,
    kirchhoff_stress,
    linear_elastic,
    linear_stress,
    neo_hookean,
    orthotropic,
    transverse_isotropic,
    blatz_ko,
)
from .history import History, corotated_increment, initial_history, rotate_history
from .johnson_cook import JohnsonCook, johnson_cook, update_johnson_cook
from .jh2 import JH2, jh2, update_jh2
from .hjc import HJC, hjc, update_hjc
from .rht import RHT, rht, update_rht
from .plastic import (
    Bilinear,
    ThermalPlastic,
    bilinear_plastic,
    thermal_plastic,
    update_bilinear,
    update_thermal_plastic,
)
from .steinberg import SteinbergGuinan, steinberg_guinan, update_steinberg


INCREMENTAL_MODELS = (
    "johnson-cook",
    "jh2",
    "hjc",
    "rht",
    "bilinear",
    "thermal-plastic",
    "steinberg-guinan",
)


def update_history(history, F, previous_F, dt, material, kind):
    strain, rotation, valid = corotated_increment(F, previous_F)
    import jax.numpy as jnp

    updates = {
        "johnson-cook": update_johnson_cook,
        "jh2": update_jh2,
        "hjc": update_hjc,
        "rht": update_rht,
        "bilinear": update_bilinear,
        "thermal-plastic": update_thermal_plastic,
        "steinberg-guinan": update_steinberg,
    }
    result = updates[kind](history, strain, jnp.linalg.det(F), dt, material)
    result = rotate_history(result, rotation)
    return result._replace(valid=result.valid & valid)
