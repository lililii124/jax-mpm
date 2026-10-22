# Validation

The reference runs use Python 3.12, JAX 0.12.1.dev20261022, and jaxlib 0.12.0.1
on a Windows CPU. All 21 tests pass.

A clean installation with the PyPI release JAX 0.12.0.1 also passes all 21
tests, the dependency check, and a CLI run with GIF output.

## Transfer and constitutive tests

The suite checks B-spline partition of unity and moments, mass and linear/APIC
angular momentum conservation, affine velocity reproduction, rigid-rotation
stress, the elastic energy derivative, free fall, free-slip boundaries,
and invalid-state detection.

## Material gradient

For the deformed block in `examples/gradient.py`, after 60 steps:

| Method | Derivative with respect to Young's modulus |
|---|---:|
| Automatic differentiation | −4.13625828355 × 10⁻⁹ |
| Central difference, h = 2 | −4.13625828477 × 10⁻⁹ |
| Relative error | 2.96 × 10⁻¹⁰ |

## Example runs

Both cases use a 48 × 48 grid, dt = 2 × 10⁻⁴, 4,000 steps, and float32.

| Case | Particles | Minimum J over saved frames | Final J range |
|---|---:|---:|---|
| Elastic block | 841 | 0.28074 | 0.7823–1.3634 |
| Dam break, K = 2 × 10⁵ | 2,112 | 0.69980 | 0.9294–1.0642 |

Both runs retain positive deformation determinants and finite particle states.
The fluid case allows about 30% local compression; these results are numerical
checks, not a comparison against an incompressible-flow benchmark or experiment.
GPU execution has not been validated.
