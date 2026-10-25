# Single fibre tension

An isotropic elastic-brittle circular fibre is pulled through two prescribed end collars. A smooth central waist localizes failure. The finite-strain corotated elastic law uses an equivalent tensile history and linear crack-band softening, with characteristic length 0.2 mm, tensile strength 0.1 MPa and fracture energy 1 J/m². These are demonstration parameters.

The total-Lagrangian quadratic MPM trajectory uses fixed reference quadrature, lumped nodal inertia and velocity-Verlet integration. Run the saved loading segment in bounded windows:

```sh
python -m examples.single_fibre_tlmpm --output outputs/fibre-1000 --stop-after-steps 1000
python -m examples.single_fibre_tlmpm --output outputs/fibre-3000 --resume-from outputs/fibre-1000 --stop-after-steps 3000
python -m examples.single_fibre_tlmpm --output outputs/fibre-5000 --resume-from outputs/fibre-3000 --stop-after-steps 5000
```

The [displayed segment](results/fibres--single-fibre-tension/summary.json) reaches complete axial-section failure and a 1.058 mm gap between intact fragments at 3.062 ms. All failed particles remain visible and carry their original mass. Its maximum energy remainder is 0.280% of peak fixture work. Saved-state checks independently recover the damage envelope and elastic energy; spatial and temporal convergence are unqualified.

The original 12 ms loading-and-hold plan is incomplete. Its continuation first develops a nonpositive deformation Jacobian at step 5643, within fully failed material. The fixed reference interpolation is not validated for subsequent fragment motion. The [rejected continuation](results/fibres--single-fibre-tension/post-failure-review.json) is retained separately from the valid segment.

[Full view](images/single-fibre-tension-full.gif) · [Detail](images/home/fibres--single-fibre-tension.gif)
