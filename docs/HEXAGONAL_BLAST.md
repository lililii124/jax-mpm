# Sixfold hyperelastic circular plate under a blast pressure pulse

This is an assumed constitutive demonstration, not a reproduction of a published material calibration. SI units are used. The plate is circular (radius 0.15 m, thickness 0.006 m), with its outer 0.015 m clamped. The load and boundary are axisymmetric; no hexagonal geometry or angular stress multiplier is used.

With F the full 3D deformation gradient, E=(F^T F-I)/2, J=det(F), and three in-plane unit vectors a_i at theta+{0,120,240} degrees, the reference energy density is

W = mu/2 (F:F-3) - mu log(J) + lambda/2 log(J)^2
    + kf/(4 es^2) sum_i max(a_i.E.a_i,0)^4.

mu=1 MPa, lambda=2.333333333 MPa, kf=5 MPa, es=0.03, rho=1100 kg/m^3. The directional contribution is a C3 tensile activation, stress-free at identity. Each unoriented direction also represents its negative, producing 60-degree material symmetry. The matrix supplies the isotropic linear tangent. Finite directional extension activates anisotropy; a zero-strain isotropic response is expected for a hexagonal material. The first Piola stress is the JAX derivative dW/dF, and the exported Cauchy stress is P F^T/J.

The prescribed vertical reference-area traction is -p(t) exp(-r^2/(2*0.035^2)) e_z on the upper reference face, with p(t)=300 kPa sin(pi*t/0.0005)^2 for 0<t<0.0005 s and zero afterward. This is an imposed blast-like pressure history, not a coupled detonation or fluid calculation; the force direction remains vertical rather than following the deformed normal.

The solver uses a quadratic reference-grid MPM basis, full 3D particle quadrature at two points per cell per direction, lumped nodal inertia, and explicit velocity Verlet without artificial damping. A Cartesian grid can introduce fourfold error; resolution and an isotropic kf=0 control are required before attributing every contour detail to the material.

Run `python -m examples.hexagonal_blast_plate --out results/hexagonal --spacing .006 --dt 1e-6 --horizon .006`. Each checkpoint stores the complete nodal displacement, velocity, force, time, accumulated pressure work, and step. Re-running with the same output path resumes it. Frames contain current/reference particle positions, displacement, actual Cauchy stress, von Mises stress, and time. The pressure work and elastic plus kinetic energy are recorded independently in history.jsonl. A nonpositive determinant or nonfinite stress stops the run with its checkpoint retained.

Focused material verification: `python tests/test_hexagonal_hyperelastic.py` checks stress-free identity, frame indifference of energy and stress, 60-degree invariance, 30-degree contrast, a centered directional derivative against AD stress, and the independent analytic uniaxial traction formula.

References informing the symmetry distinction: Maalouf and Vel, Physical Review B 112, 174120 (2025), https://doi.org/10.1103/35fw-813c ; Markert, Ehlers and Karajan (2005), https://doi.org/10.1002/pamm.200510099 . The quartic tensile coefficients above are assumed, not taken from those papers.

The completed 6 mm demonstration has 62,880 particles and 19,663 nodes, 6,000 steps at 1 microsecond, and 61 output frames through 6 ms. All saved states have positive J; the minimum over the history is 0.942575. The maximum displacement is 24.6174 mm. The maximum absolute work-energy residual is 5.6482e-6 J; at 6 ms it is 1.3371e-8 J, against 1.628326 J of pressure work. These are coarse demonstration results; a finer calculation is required to assess spatial convergence, particularly near the clamped rim.

At 2.5 ms, a least-squares angular fit to upper quadrature-layer von Mises stress in the radius 50 +/- 2.5 mm annulus includes a quadratic radial trend and harmonics m=2,4,6,8,12. The normalized sixth harmonic is 16.07224% for both material angles 0 and 30 degrees; its phase rotates by 30 degrees modulo 60. The fourth harmonic is 0.009671%. The independently integrated isotropic kf=0 control has sixth harmonic below 1e-12 and fourth harmonic 0.004575%. Thus the observed sixfold pattern follows the constitutive directions rather than the Cartesian grid. A phase is not meaningful for the zero-amplitude isotropic sixth harmonic.

ParaView export and rendering commands:

```
python -m examples.hexagonal_blast_plate --out results/hexagonal --export
pvpython --force-offscreen-rendering tools/render_hexagonal_blast.py --input results/hexagonal --out results/hexagonal/png --color-max 483285.86412711476
python tools/render_hexagonal_blast.py --assemble-gif --input results/hexagonal/png --out results/hexagonal/stress.gif
python tools/gif_review.py results/hexagonal/stress.gif --open
```

The renderer uses the actual upper particle quadrature layer (0.375 mm below the reference top face), without angular multiplication or deformation scaling. It checks the ParaView-read stress array SHA256 against the exported NPZ source. The fixed display range is 0 to 483,285.864 Pa, as requested: reducing the original 724,928.796 Pa maximum by one third. This changes color saturation, not the computed stress. In the full coarse 6 ms animation, 18 of 61 frames contain samples above this color upper bound; the largest saturated fraction is 13.0789%. The physical stress maximum remains recorded separately. The animation is 400x400 pixels on white with no axes or text.

New checkpoints bind the physics, geometry, source hashes, JAX version, backend and floating dtype. A mismatch is rejected. They retain the last valid full nodal state; failed deformation or an excessive independently computed work-energy residual is written separately before any checkpoint replacement. The initial completed coarse demonstration predates this binding and is retained as a complete archived run, not offered as a resumable checkpoint.
