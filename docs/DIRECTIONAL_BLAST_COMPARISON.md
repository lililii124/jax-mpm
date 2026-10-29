# Directional hyperelastic blast comparison

This extends the constitutive demonstration in [Hexagonal blast](HEXAGONAL_BLAST.md). Every case uses the same circular plate, clamped rim, density, Neo-Hookean matrix, Gaussian vertical reference-pressure pulse, 1 microsecond time step, 6 ms observation interval and material-point quadrature. All five cases use 6 mm in-plane grid spacing.

With E=(F^T F-I)/2, the directional reference energy is

Wdir = kf / (p es^(p-2)) sum_i w_i max(a_i.E.a_i,0)^p,

where a_i are in-plane unit vectors. Matrix parameters are mu=1 MPa and lambda=2.333333333 MPa; kf=5 MPa and es=0.03. Every reinforced family uses exponent p=4 and sum(w_i)=3, giving equal angular means of their uniaxial directional energy. The isotropic control contains only the Neo-Hookean matrix.

| Family | Directions (degrees) | Weights | Material rotation period |
| --- | --- | --- | --- |
| Square | 0, 90 | 1.5, 1.5 | 90 degrees |
| Rectangular | 0, 90 | 2.25, 0.75 | 180 degrees |
| Hexagonal | 0, 120, 240 | 1, 1, 1 | 60 degrees |
| Octagonal | 0, 45, 90, 135 | 0.75 each | 45 degrees |
| Isotropic | No directional reinforcement | 0 | Any angle |

The threefold vector set in the hexagonal case consists of unoriented material axes: a and -a give the same strain contraction, producing sixfold symmetry. The rectangular case denotes two unequal orthogonal material families, not a rectangular specimen. The matrix supplies the common isotropic infinitesimal tangent; the directional anisotropy activates at finite tensile Green strain.

Eightfold contrast is intrinsically weaker for this common quartic law. Under uniaxial extension its directional angular coefficient is 1/35 for the octagonal family, versus 8/35 for the hexagonal family.

Run a case with `python -m examples.directional_blast_plate --symmetry square --exponent 4 --out results/square --horizon .006`. The same command accepts rectangular, hexagonal, octagonal, or isotropic. All stress fields are computed as P=dW/dF by JAX automatic differentiation and sigma=P F^T/J. Full-state checkpoints bind the parameters, geometry, source hashes and runtime; work-energy and positive-J checks protect the accepted state.

`python tests/test_directional_hyperelastic.py` verifies the new material periods and half-period contrasts, stress-free identity, objectivity, agreement with the original hexagonal energy, and an independent analytic check of equal angular-mean reinforcement. The independent hexagonal 30-degree rotation and isotropic control are documented in [Hexagonal blast](HEXAGONAL_BLAST.md).

For output, use `--export` and the common ParaView renderer `tools/render_hexagonal_blast.py`. Every 400x400 animation uses the same camera, fixed 0 to 483285.864127 Pa stress range and deformation scale of 1. Values above the display range are counted in the receipt and remain present in the actual stress array. The exported hash is checked against the array read by ParaView.

The angular diagnostic uses the same radius 50 +/- 2.5 mm annulus for all cases, with a quadratic radial trend and m=2,4,6,8,12 harmonics. It reports every harmonic through time, rather than only the expected family harmonic. This permits the fourfold Cartesian-grid contribution and any weak octagonal response to be assessed against the isotropic calculation.
