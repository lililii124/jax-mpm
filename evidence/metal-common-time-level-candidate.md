# Common time level for the metal impact step

The candidate holds the starting Galerkin space fixed during a complete step.
Endpoint velocities determine one midpoint velocity for particle motion,
deformation, material work and force work. Geometry projection follows the
complete coupled step. Contact tolerances remain unchanged.

This follows the fixed-mass, within-step grid approach discussed by
[Love and Sulsky](https://www.math.unm.edu/~sulsky/papers/LoveSulskyHPC.pdf).
The affine kinetic variables and contact sampler used here require their own
checks; that reference does not validate this implementation.

For the linear volume potential, use the old damage during mechanical work,
then account for damage release in the existing fracture-energy history.
With `Ck = Fk.T Fk`, the determinant discrete gradient is

```
G = cof((C0 + C1)/2) + cof(C1 - C0)/12
dJ/dF = (F0 + F1) G / (J0 + J1)
```

Multiplying this expression by the volume-potential secant gives exact volume
work and a symmetric `P Fmid.T`. No energy remainder enters the stress, heat
or history update. Deviatoric work uses the existing midpoint JC return.

The bounded driver starts from the authenticated post-contact 125 ns state,
uses at most eight nonlinear evaluations and initially admits only a 2 ps
window. It retains the original material, CFL, contact, energy, linear-momentum
and angular-momentum checks. Contact torque at the midpoint is measured
explicitly; its general conservation has not been established. The driver is
a NumPy/SciPy projection reference with a JAX material worker, not the final
production solver.

The seven returned material fields match the prior saved evaluation exactly.
The volume-work checks cover compression, tension, crossing `J=1`, zero
increment, torque symmetry and a common rigid rotation.

The first 2 ps run completed in 120.18 s with 1.03 GB peak USS and three
nonlinear evaluations. Its reported total energy residual is `4.26e-13 J`,
and angular residual norm is below the original `2.47e-18` tolerance.
Independent raw-state review passed, with a recomputed energy residual of
`4.46e-13 J` and plastic-heating identity error of `1.45e-15 J`.
The next continuation uses the accepted state, retains all tolerances, and
advances in steps up to 200 ps with cumulative balances. The in-process JAX
material evaluator retains the compiled update between nonlinear evaluations.

The first 200 ps continuation was accepted at 125.202 ns. The next attempted
step failed the contact-power check: endpoint contact produced `4.20e-7 J`
of positive work despite a material-work defect below `2e-16 J`.

The optional midpoint contact split projects the half-step trial velocity
onto the fixed starting contact cone, then reflects it to obtain the velocity
used for midpoint motion. An instantaneous inelastic projection onto that same
cone follows the motion, before geometry remapping and the new-cone impact.
The initial velocity is admissible in the starting cone. This last old-cone
projection prevents an unprocessed midpoint reflection from surviving a
change of contact geometry. Each projection keeps its actual kinetic loss
separate from material work; no loss is converted into heat. Impulses and
torques after motion are evaluated at the advanced configuration. The stored
pre-impact velocity is the endpoint used for the position/deformation
midpoint identities. Two exact closed-contact histories check zero restitution
and separating motion reversed by an applied force. General collision geometry
still requires the original coverage, power and momentum gates.
