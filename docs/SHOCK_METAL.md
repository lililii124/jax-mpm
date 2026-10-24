# Shock metal material

`mpm_jax.materials.shock_metal` provides an energy-dependent EOS and a separate
corotated strength return. It does not change the existing Johnson–Cook model or
register a new spatial solver. The intended example is a deformable metal debris
sphere striking a deformable metal plate at **5000 m/s**. No impact calculation or
experimental validation is claimed by these material-level checks.

## Pressure, energy and temperature

All quantities use SI. Pressure is positive in compression, `J=det(F)=rho0/rho`,
and `e` is specific EOS internal energy in J/kg, measured from the initial state.
It excludes separately stored deviatoric elastic energy and cold plastic work.
Writing `eta=1-J`, the compressed reference curve is

```
Us = c0 + s up
pH = rho0 c0² eta / (1-s eta)²
eH = pH eta / (2 rho0)
p  = pH + Gamma0 rho0 (e-eH)
```

The constant product `Gamma*rho=Gamma0*rho0` is explicit. Temperature follows the
Hugoniot-temperature integral, evaluated with 32-point Gauss–Legendre quadrature,
then `T=TH+(e-eH)/Cv`. It is not `T0+e/Cv`: reversible compression stores energy
and changes temperature. Entropy relative to the initial state is
`Cv*(log(T/T0)-Gamma0*eta)`. These expressions follow the laboratory
[MGUsup thermodynamic formulation](https://lanl.github.io/singularity-eos/main/src/models.html#mie-gruneisen-linear-u-s-u-p-eos).

For expansion, the reference isentrope is `pI=rho0*c0²*eta`,
`eI=c0²*eta²/2`, `TI=T0*exp(Gamma0*eta)`. This is an explicit continuation,
not tensile fracture, cavitation or a gas EOS. Changing the reference curve in
expansion follows the approach described in the
[Sandia Mie–Gruneisen report](https://www.osti.gov/servlets/purl/1762624).

`eos_state(J,e,p)` returns pressure, temperature, density, relative entropy,
isentropic bulk modulus, bulk sound speed and an elastic longitudinal speed.
Negative pressure is explicitly marked by `tension_extrapolated`; no spall or
cavitation limit has been fitted.
The derivative uses `cs²=(dp/drho)|e + (dp/de)|rho * p/rho²`; a fixed-energy
derivative alone is insufficient. The longitudinal value retains the larger of
the reference-Kirchhoff and constant-Cauchy shear estimates, adding
`4*mu*max(J,1)/(3*rho0)` to `cs²`, including after softening. Using
`4*mu/(3*rho)` alone would underestimate the reference-Kirchhoff stiffness
contribution during compression.
The spatial solver must also account for velocity and artificial-viscosity
restrictions. There is no single fixed sound-speed bound for arbitrary energy.

The default numerical domain is `0.4 <= J <= 2`, separated from the Hugoniot
pole. Invalid states are flagged, not repaired by a pressure clamp. This is a
numerical domain, not a measured validity range. Linear shock-speed fits and
constant `Gamma*rho` eventually fail under extreme compression, as noted in
the [official shock-EOS description](https://ansyshelp.ansys.com/public/Views/Secured/corp/v252/en/exd_ag/ds_ex_mat_shockeos.html).

## Update contract

`volume_energy_step(J0,J1,e0,q,heat,p)` solves the implicit midpoint equation
`e1-e0 = -(p_mid+q)*(J1-J0)/rho0 + heat` algebraically. It returns the pressure
work, viscous heat and discrete residual separately. `heat` is a separately
supplied nonnegative specific dissipative input. The same shock pressure must
appear in momentum and energy. `artificial_viscosity` supplies a linear plus
quadratic compression-only pressure; it vanishes in expansion and pure shear.
It is a numerical shock regularization, not molecular viscosity or a complete
multidimensional shock detector. The need to couple viscosity to energy is
also visible in the [LLNL LULESH hydrodynamics implementation](https://github.com/LLNL/LULESH/blob/master/lulesh.cc).

`deviatoric_return(tau,ep,strain,R,J,dt,T,p)` accepts an old-frame corotated
strain increment and a proper incremental rotation. The returned stress is
deviatoric Kirchhoff stress; Cauchy stress is `tau/J - p*I`. Johnson–Cook strength
uses `max(rate/reference_rate,1)` and a frozen temperature during each return.
Plastic work, its thermal/cold-work partition, recoverable deviatoric energy
and elastic-predictor energy loss are returned separately. The excess return
loss over plastic work is backward-Euler numerical dissipation and must be
reported when selecting a time discretization.

There are two distinct integration choices. A compatible total-energy solver
updates total internal energy once and subtracts the separately stored elastic
and cold-work parts to obtain EOS energy. A split material update supplies only
its independently accounted heat to `volume_energy_step`. **Do not add plastic
or viscous heat a second time to an energy already updated with their stress
work.** The module does not certify global energy conservation of an MPM
transfer or spatial time integrator. That requires planar shock/release and
global energy tests of the eventual solver.

## Scope and verification

Defaults are an illustrative Al-like set: rho0=2700 kg/m³, c0=5350 m/s, s=1.34,
Gamma0=2, Cv=900 J/(kg K), shear modulus26 GPa; JC A=100 MPa, B=300 MPa,
n=.34, C=.01, m=1, reference rate1/s, heat fraction.9, T0=300 K and nominal
melting threshold933.15 K. They are assumptions, not an alloy calibration.

Above the nominal melt temperature the deviatoric strength vanishes while EOS
pressure remains. `phase_extrapolated` reports this regime. There is no latent
heat, pressure-dependent melting curve, liquid/vapour coexistence, ionization,
thermal conduction or resolved vaporization. A 5 km/s impact can enter regimes
where those omissions matter; a smooth animation cannot validate them.

Focused tests check planar Rankine–Hugoniot identities, thermodynamic
derivatives, temperature/energy inversion, homogeneous release convergence,
compression-only viscosity, discrete volume-energy work, proper rigid
rotation, plastic dissipation and melt-softening behavior. These are material
and algebra checks; no planar shock spatial solution or full HVI example has
yet been validated.
