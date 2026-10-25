# Elastic models and three-dimensional examples

This source inventory covers isotropic linear, orthotropic, transversely
isotropic and hyperelastic solids at main revision `3eed727`. All four have
JAX implementations and runnable 3D examples. No new solver or duplicate
example is needed to supply these model categories.

| Category | JAX implementation in [`elastic.py`](../mpm_jax/materials/elastic.py) | 3D example model flag | Existing focused verification |
|---|---|---|---|
| Isotropic linear | `linear_elastic`, `linear_stress` | `linear` | Uniaxial stress, engineering shear and parameter derivative in [`test_elastic_materials.py`](../tests/test_elastic_materials.py); analytical longitudinal rod study in [`VALIDATION.md`](../VALIDATION.md) |
| Orthotropic linear | `orthotropic`, `linear_stress`; 9 engineering constants and rotated material axes | `orthotropic` | Compliance recovery, three uniaxial directions, positive definiteness and rotated response in the same focused file |
| Transversely isotropic linear | `transverse_isotropic`; 5 constants, longitudinal local axis 3 | `transverse-isotropic` | In-plane rotational invariance and isotropic limit in the same focused file |
| Isotropic hyperelastic | `neo_hookean` (`elastic` alias), selected `blatz_ko` branch | `neo-hookean`, `blatz-ko` | 3D Neo-Hookean stress as energy derivative, stress-free rotation and dynamic parameter gradient in [`test_3d.py`](../tests/test_3d.py); Blatz–Ko energy derivative and reference stress in [`test_additional_materials.py`](../tests/test_additional_materials.py) |

The linear laws use infinitesimal strain `sym(F)-I` with Cauchy stress
`C:strain`, converted to Kirchhoff stress `J*sigma` for transfer. They assume
small strain **and small rotation**. Their orientation matrix rotates material
axes; it does not make these laws objective finite-rotation formulations.
Neo-Hookean and the selected Blatz–Ko branch use finite-strain stresses.
The Blatz–Ko implementation is the documented MAT007 REF=0 branch, not every
model bearing that name. Conventions are in [MATERIALS.md](MATERIALS.md).

## Runnable examples

[`examples.elastic_models`](../examples/elastic_models.py) releases a
prestrained 3D block with zero gravity and no imposed boundary velocity:

```bash
python -m examples.elastic_models --model all --grid 32 --frames 30 --substeps 20 --output outputs/elastic_models
```

[`examples.suite`](../examples/suite.py) provides a clamped cantilever,
torsion, prescribed shear and compression for each of the same five flags:

```bash
python -m examples.suite --case cantilever --model all-elastic --grid 48 --ppc 2 --frames 20 --no-render --output outputs/elastic_cantilever
python -m examples.suite --case shear --model transverse-isotropic --grid 48 --ppc 2 --frames 20 --no-render
```

Choose a single flag to avoid running all models. These are existing entry
points, not newly executed results. The suite's `wave` case explicitly
replaces the requested material with isotropic linear elasticity; it is not
an anisotropic-wave verification command. Some historical wave directory
names retain `neo-hookean`; their actual configuration identifies the law.

## Evidence and remaining work

The [inventory receipt](results/elastic-model-inventory.json) binds source
hashes, the existing historical XML test entries and local summary hashes.
The XML records the six focused linear/anisotropic tests and the 3D
hyperelastic tests without failure. This inventory did not rerun CI or
revalidate raw trajectory arrays.

Local historical records include all five free-release models at grid 20,
2,197 particles and 600 steps, plus linear/orthotropic/TI/Blatz–Ko cantilevers
at grid 48 with 36,540 particles. These summary records establish that examples
were executed, not source-bound accuracy or convergence. In particular, the
old free-release TI orientation rotates only inside its isotropic plane.
Current source tilts the longitudinal axis by 30 degrees; that change needs
its own source-bound spatial result before the old run can be cited for it.

The public catalogue already contains Neo-Hookean structural and contact
demonstrations, and the isotropic linear rod has an analytical refinement
study. No separately published spatial refinement study was found for the
generic orthotropic/TI laws or Blatz–Ko branch. A useful next validation is
an analytical small-amplitude anisotropic mode or manufactured 3D field
with fixed material axes, followed by grid and time refinement. Repeating
an arbitrary cantilever animation would not fill that accuracy gap.

The common example energy columns omit affine APIC kinetic energy and a
complete boundary/contact work budget. Finite states and positive `det(F)`
do not certify discrete energy conservation or experimental agreement.
The woven finite-strain material implementation is separate; woven result
admission must not be transferred to the generic small-strain anisotropic laws.
