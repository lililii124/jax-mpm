"""Midpoint material work for the linear-EOS metal branch."""
from typing import NamedTuple

import jax
import jax.numpy as j

from .fragment_multifield_v6_mpm import (
    ImpactConfig, damage_volume_energy, elastic_energy,
)
from .fragment_v6_midpoint_strength import midpoint_return


def cofactor(a):
    return j.stack((j.cross(a[..., :, 1], a[..., :, 2]),
                    j.cross(a[..., :, 2], a[..., :, 0]),
                    j.cross(a[..., :, 0], a[..., :, 1])), axis=-1)


def volume_piola(F0, F1, damage, bulk):
    """Objective discrete gradient of the fixed-damage volume potential."""
    J0, J1 = j.linalg.det(F0), j.linalg.det(F1)
    C0, C1 = F0.swapaxes(-1, -2) @ F0, F1.swapaxes(-1, -2) @ F1
    dC = C1 - C0
    G = cofactor(.5 * (C0 + C1)) + cofactor(dC) / 12
    G = .5 * (G + G.swapaxes(-1, -2))
    gradJ = (F0 + F1) @ G / (J0 + J1)[..., None, None]
    z0, z1 = J0 - 1, J1 - 1
    positive0, positive1 = j.maximum(z0, 0), j.maximum(z1, 0)
    negative0, negative1 = j.minimum(z0, 0), j.minimum(z1, 0)
    delta = J1 - J0
    crossing = ((1 - damage) * (positive1 - positive0) * (positive1 + positive0)
                + (negative1 - negative0) * (negative1 + negative0))
    secant = .5 * bulk * crossing / j.where(delta != 0, delta, 1)
    secant = j.where((z0 <= 0) & (z1 <= 0), .5 * bulk * (z0 + z1), secant)
    secant = j.where((z0 >= 0) & (z1 >= 0), .5 * bulk * (1 - damage) * (z0 + z1), secant)
    return secant[..., None, None] * gradJ


class WorkMaterial(NamedTuple):
    state: tuple
    piola: object
    work_density: object
    plastic_heat: object
    volume_identity: object
    deviatoric_identity: object
    valid: object


@jax.jit
def update(F, tau, ep, cold, fracture, damage, internal, W, dt, material):
    J = j.linalg.det(F)
    newF = F + dt * W
    newJ = j.linalg.det(newF)
    midpoint = .5 * (F + newF)
    inverse = j.linalg.inv(midpoint)
    L = W @ inverse
    skew = .5 * (L - L.swapaxes(-1, -2))
    rotation = j.linalg.solve(j.eye(3) - .5 * dt * skew, j.eye(3) + .5 * dt * skew)
    strain = .5 * dt * (L + L.swapaxes(-1, -2))
    heat = internal - cold - fracture - elastic_energy(tau, material, damage) - damage_volume_energy(J, damage, material)
    temperature = material.reference_temperature + heat / material.heat_capacity
    factor = 1 - damage
    returned = midpoint_return(tau / j.maximum(factor, 1e-12)[..., None, None], ep,
                               strain, rotation, j.where(factor > 0, newJ, 1), dt, temperature, material)
    nexttau = returned.kirchhoff_stress * factor[..., None, None]
    nextep = j.where(factor > 0, returned.plastic_strain, ep)
    nextcold = cold + factor * returned.specific_cold_work
    plasticheat = factor * returned.specific_plastic_heat
    config = ImpactConfig()
    nextdamage = j.maximum(damage, j.clip((nextep - config.damage_onset_strain) /
                                        (config.damage_failure_strain - config.damage_onset_strain), 0, 1))
    before = elastic_energy(nexttau, material, damage)
    nexttau = nexttau * ((1 - nextdamage) / j.maximum(1 - damage, 1e-12))[..., None, None]
    elastic_release = j.maximum(before - elastic_energy(nexttau, material, nextdamage), 0)
    volume_release = j.maximum(damage_volume_energy(newJ, damage, material) -
                               damage_volume_energy(newJ, nextdamage, material), 0)
    nextfracture = fracture + elastic_release + volume_release
    nextinternal = (damage_volume_energy(newJ, nextdamage, material) + heat + plasticheat +
                    nextcold + nextfracture + elastic_energy(nexttau, material, nextdamage))
    bulk_piola = volume_piola(F, newF, damage, material.bulk_modulus)
    dev_piola = factor[..., None, None] * returned.midpoint_stress @ inverse.swapaxes(-1, -2)
    piola = bulk_piola + dev_piola
    volume_work = j.sum(bulk_piola * (dt * W), axis=(-2, -1))
    dev_work = j.sum(dev_piola * (dt * W), axis=(-2, -1))
    volume_change = material.density * (damage_volume_energy(newJ, damage, material) -
                                         damage_volume_energy(J, damage, material))
    dev_change = material.density * (elastic_energy(nexttau, material, nextdamage) -
        elastic_energy(tau, material, damage) + elastic_release + plasticheat + nextcold - cold)
    valid = (j.all(returned.valid | (factor == 0)) & j.all(j.isfinite(nextinternal)) &
             j.all(nextdamage >= damage) & j.all(nextep >= ep) & j.all(j.isfinite(piola)))
    return WorkMaterial((newF, nexttau, nextep, nextcold, nextfracture, nextdamage, nextinternal),
                        piola, volume_work + dev_work, plasticheat,
                        volume_work - volume_change, dev_work - dev_change, valid)
