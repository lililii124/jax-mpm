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
from .johnson_cook_displacement import (
    JohnsonCookDisplacement, johnson_cook_displacement, update_johnson_cook_displacement,
)
from .plane_strain_kinematics import corotated_increment_plane_strain
from .jh2 import JH2, jh2, update_jh2
from .hjc import HJC, hjc, update_hjc
from .mat173 import MAT173, mat173, update_mat173
from .mat173_softening import (
    MAT173Softening, mat173_softening, softening_fraction, softening_properties,
    update_mat173_softening,
)
from .mat173_tensile_cutoff import (
    TensileCutoff, with_tensile_cutoff, update_mat173_tensile_cutoff,
    update_mat173_softening_tensile_cutoff,
)
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
from .woven import WovenElastic, woven_elastic, woven_energy, woven_kirchhoff_stress


INCREMENTAL_MODELS = (
    "johnson-cook",
    "johnson-cook-displacement",
    "jh2",
    "hjc",
    "rht",
    "bilinear",
    "thermal-plastic",
    "steinberg-guinan",
    "mat173",
    "mat173-softening",
    "mat173-tensile-cutoff",
    "mat173-softening-tensile-cutoff",
)


def update_history(history, F, previous_F, dt, material, kind, *, plane_strain=False):
    if plane_strain:
        if kind not in ("johnson-cook", "johnson-cook-displacement"):
            raise ValueError("Analytic plane-strain kinematics requires a Johnson-Cook material")
        strain, rotation, valid = corotated_increment_plane_strain(F, previous_F)
    else:
        strain, rotation, valid = corotated_increment(F, previous_F)
    import jax.numpy as jnp

    updates = {
        "johnson-cook": update_johnson_cook,
        "johnson-cook-displacement": update_johnson_cook_displacement,
        "jh2": update_jh2,
        "hjc": update_hjc,
        "mat173": update_mat173,
        "mat173-softening": update_mat173_softening,
        "mat173-tensile-cutoff": update_mat173_tensile_cutoff,
        "mat173-softening-tensile-cutoff": update_mat173_softening_tensile_cutoff,
        "rht": update_rht,
        "bilinear": update_bilinear,
        "thermal-plastic": update_thermal_plastic,
        "steinberg-guinan": update_steinberg,
    }
    result = updates[kind](history, strain, jnp.linalg.det(F), dt, material)
    result = rotate_history(result, rotation)
    return result._replace(valid=result.valid & valid)
