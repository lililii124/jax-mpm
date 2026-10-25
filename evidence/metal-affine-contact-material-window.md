# Compatible contact and material work: bounded operator windows

These diagnostics use the original authenticated 125 ns metal state and
unchanged material parameters. They do not advance or admit a full impact
trajectory. No homepage or public media changes are made.

For the same mass-whitened reconstruction Q0, let T contain known per-body
affine velocity modes and Tgrid their exact grid representatives. Define the
contact velocity sample by

    S0 = diag(grid_mass)^-1 Q0^T
    S  = S0 + (Tgrid - S0 T) T^T.

This reproduces every affine field, including rigid velocity, at each common
node. Its adjoint is a sum of Q0 and T terms, hence lies in range(Q0).
Contact impulses therefore preserve the consistent velocity range. The
contact mobility is assembled from a sparse term and a small affine-mode
correction; no arbitrary grid-null velocity or preconditioner-dependent grid
inverse enters the physical constraint. The initial reference moment equals
the original grid moment, so the affine correction at the seed is roundoff
only. Original common-node geometry, centroid normals, frictionless law and
0.005 m/s contact tolerance are retained. Future moving geometry must rebuild
and check the pair coverage; the current 114 occupied pairs are a subset of
the original swept allocation's 786 pairs.

Analytic contact fixtures cover both the seed and a transported anisotropic
affine metric, with seven redundant contact modes. The exact kinetic loss
29.12 J is recovered, common-point affine velocities agree to 1.8e-15, and
the sampling/impulse adjoint identity closes to 7.2e-15. Prescribed normals
occur ONLY in these analytic fixtures; the actual seed uses its original
centroid law.

In the original full field, initial closing speed 3665.36 m/s falls below
6e-11 m/s. Instantaneous contact loses 0.229187513 J, with a work identity
residual below 1e-14 J. This loss is explicit and is never added to heat.
Frozen-geometry contact/stress/contact diagnostics include the original
pressure, damage traction and artificial viscosity. At impulse durations
2e-11, 1e-11 and 5e-12 s, total work closes within 5.2e-14 J. Geometry and
material state do not advance in this diagnostic; it is not a dynamic run.

The material test then prescribes the same contact-corrected ballistic affine
path over 0.04 ns, with 2/4/8 substeps. All original material histories start
unchanged. E0=Fseed^-1 h/2, D0=E0 E0^T and Wseed=Cseed Fseed initialize the
new variables exactly in the original kinetic metric. Projection and contact
are separate instantaneous kinetic losses; neither changes F or material
histories. Final prescribed-path raw files explicitly carry
accepted_impact_state=False and save W, E0, D0, Lphysical and Ctransfer.

The original backward-Euler return has a first-order deviatoric work deficit
of -3.474e-8/-1.706e-8/-8.289e-9 J. The first two exceed the original global
1.414599609375e-8 J energy tolerance. This failure is retained, not hidden by
a reservoir or a finer-step success.

The separate candidate midpoint return preserves the original endpoint
Johnson-Cook yield equation, material parameters, endpoint residual tolerance
and bounded root solve. In the corotated frame write

    A = tau_old + G strain
    qbar = q(A) - 1.5 G dep
    taubar = A qbar/q(A)
    tau_end = 2 taubar - tau_old.

Solve the original endpoint yield equation for dep on the monotone first
branch. This gives the algebraic power identity

    taubar:strain/rho0 = (|tau_end|^2-|tau_old|^2)/(4 G rho0)
                        + qbar dep/rho0.

Plastic heat is exactly 0.9 times this nonnegative physical plastic work;
cold work is the other 0.1. No energy difference is used to adjust either.
Half-rotation places strain and midpoint stress in the same frame, with
proper-rotation and square-root checks using the original rotation tolerance.
Damage release remains in the original fracture-energy channel. The original
material source is unchanged.

Elastic loading, plastic loading, nonproportional loading, rigid rotation and
unloading fixtures pass. The same full-state prescribed window has deviatoric
work errors around 1e-18 J; remaining bulk work errors are approximately
2.89e-11/7.22e-12/1.81e-12 J and converge quadratically. This does NOT assert
second-order constitutive state convergence: successive mass-weighted errors
contract by about 0.42 for stress/plastic strain and 0.51 for internal energy.

Before an actual bounded dynamic continuation, integrate compatible force,
drift and material stages, original clamped boundary constraints and their
impulse/work, changing-grid coverage, CFL and every original validity gate.
Verify the complete accepted-step energy and momentum ledger. The present
operator windows do not authorize bypassing any of these requirements or
starting a long impact/fragment run.

Local entry points inspect Windows system (CommitLimit-CommitTotal)*PageSize
through GetPerformanceInfo before large imports, and the projection checks
headroom again before factor allocations. Below 1.5 GiB they exit without a
numerical fallback. LDL diagnostic Frobenius norms use bounded column chunks;
the resulting fixed-state projection is byte-identical to the earlier
qualified result. This diagnostic-memory change does not alter the solver.
