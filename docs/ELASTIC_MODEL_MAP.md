# Elastic models and 3D examples

| Model | JAX implementation in [`elastic.py`](../mpm_jax/materials/elastic.py) | Example flag | Existing verification |
|---|---|---|---|
| Isotropic linear | `linear_elastic`, `linear_stress` | `linear` | Uniaxial stress, engineering shear, parameter derivative; analytical longitudinal rod refinement |
| Orthotropic linear | `orthotropic`, `linear_stress`; nine constants and rotated axes | `orthotropic` | Engineering constants, three uniaxial directions, positive compliance and rotated response |
| Transversely isotropic linear | `transverse_isotropic`; five constants, longitudinal local axis 3 | `transverse-isotropic` | Transverse rotational invariance and isotropic limit |
| Isotropic hyperelastic | `neo_hookean` (`elastic` alias), `blatz_ko` | `neo-hookean`, `blatz-ko` | Stress as energy derivative; Neo-Hookean 3D stress-free rotation and dynamic parameter gradient |

The three linear laws use infinitesimal strain `sym(F)-I` and assume small
strain and small rotation. Rotating material axes does not supply finite-rotation
objectivity. Blatz–Ko implements the selected MAT007 REF=0 branch with reference
Poisson ratio 0.463. Neo-Hookean and this Blatz–Ko branch use finite-strain stresses.

[`elastic_models`](../examples/elastic_models.py) releases a prestrained 3D block;
[`suite`](../examples/suite.py) supports each flag for cantilever, torsion, shear
and compression:

```bash
python -m examples.elastic_models --model all --grid 32 --frames 30 --substeps 20 --output outputs/elastic_models
python -m examples.suite --case cantilever --model all-elastic --grid 48 --ppc 2 --frames 20 --no-render --output outputs/elastic_cantilever
```

Focused checks are in [test_elastic_materials.py](../tests/test_elastic_materials.py),
[test_3d.py](../tests/test_3d.py) and
[test_additional_materials.py](../tests/test_additional_materials.py).
[VALIDATION.md](../VALIDATION.md) records the isotropic rod refinement;
the [inventory receipt](results/elastic-model-inventory.json) binds source,
historical test entries and local run metadata.

A [source-bound 3D force refinement study](ELASTIC_FORCE_REFINEMENT.md) covers
rotated orthotropic axes, the TI longitudinal axis tilted by 30 degrees and
Blatz–Ko. Relative acceleration errors decrease at nearly second order and reach
3.768%, 3.771% and 3.844% on the finest grid. This frozen operator comparison
does not certify complete dynamic time integration or structural convergence.
The old TI free-release result rotates only inside its isotropic plane and cannot
validate the tilted longitudinal axis. Woven finite-strain results use a separate law.
Example energy columns omit affine APIC kinetic energy and the full boundary/
contact work budget, so they do not certify discrete energy conservation.
