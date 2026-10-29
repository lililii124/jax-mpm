# Sandwich-panel design under transient pressure

Status: planned; the coupled optimization has not been implemented or run.

## Problem

Minimize core mass subject to limits on back-face displacement and face-sheet
stress under three prescribed pressure histories. A preliminary fixed-mass
optimization establishes whether redistribution improves the transient response
before reducing the mass budget. Compare every design with a uniform core at
the same mass, external impulse, boundary conditions and observation time.

Initial assumed geometry is a 300 × 300 mm panel with two 2 mm aluminium face
sheets and a 20 mm core. All four 15 mm edge strips are fixed through the
thickness; there is no backing. The interfaces are initially perfectly bonded.
The face sheets and clamped perimeter are not removable design regions.

The target design is a filtered core relative-density field with 24 × 24 × 3 controls,
initially 0.18 and bounded by 0.08 and 0.35. This first stage designs a graded
effective material, not explicit cell walls. A physical filter radius stays
fixed during grid refinement. Any later conversion to explicit cells requires
a separate forward analysis.

The initial face material has E = 70 GPa, nu = 0.33 and density = 2700 kg/m³.
For the elastic differentiation benchmark, the core uses
rho = 2700 phi, E = 0.3 × 70 GPa × phi² and nu = 0.25. These are assumed
effective-property laws, not measured foam calibration. Density enters inertia;
the modulus enters stress. The reference quadrature volume remains geometric.
The initial whole-panel mass is 1.8468 kg before any explicit edge reinforcement.

## Loading and constraints

Use p(x,y,t) = p0 s(x,y) sin²(pi t/T) for 0 < t < T, and zero thereafter.
Initially p0 = 0.5 MPa and T = 0.2 ms. The three footprints are uniform,
a central Gaussian, and an eccentric Gaussian. Normalize each footprint over
the loaded surface so its area average is one: all three have 50 Pa s mean
impulse and 4.5 N s resultant impulse. The first benchmark uses fixed reference
area and normal, not follower pressure; the prescribed force is independent of
the material design. Divide that force by the actual design-dependent mass
when supplying particle acceleration. These are equivalent mechanical loads;
no explosive-source model or pressure-test calibration is implied.

The first observation interval is 2 ms and must extend if the peak response or
rebound is still outside the saved history. Provisional constraints are 5 mm
peak back-face displacement and 200 MPa face-sheet equivalent stress. Evaluate
both across the complete time history and all three footprints. A smooth upper
bound may drive optimization; acceptance uses the actual discrete maxima.
Record pressure work, support reaction/impulse, kinetic and internal energy,
mass, and both face-sheet responses.

Elastic gradients qualify the implementation only. The design is not accepted
as a blast-resistant result until nonlinear forward checks include face-sheet
plasticity, core crushing, and adequate spatial/time resolution. Residual
deflection and crushing absorption are not outputs of the elastic benchmark.

## Implementation sequence

1. Implement a compact reference grid with sufficient through-thickness
   resolution and explicit per-particle material coefficients. Start with at
   least two grid cells across each face sheet and independently check finer
   resolution; more particles alone do not resolve a thin layer. Check bonded
   load transfer and surface-force integration.
   The current cubic dense TL-MPM grid is not an adequate production layout
   for these thin face sheets.
2. Run the uniform panel over the full pulse and rebound. Establish time-step,
   resolution and energy-error tolerances before design updates.
3. Differentiate the transient MPM trajectory using `jit`, `value_and_grad`
   and `lax.scan`; use nested checkpointed time blocks. Process load cases
   sequentially until measured memory permits batching. Check a directional
   derivative at a feasible design with a reduced control set before expanding
   to the target field. Use one time step bounded by the stiffest admissible
   material across all design iterations; verify it with a smaller-step replay.
   Sequential JVPs are an alternative for the initial small control set.
4. Optimize the fixed-mass core distribution, then reduce the mass budget under
   the response limits. Check refined forward solutions independently and
   add the nonlinear constitutive qualification above.
5. Export uniform/optimized density and deformation comparisons through
   ParaView, with identical camera, time, scale and image dimensions. Publish
   only actual accepted trajectories.

Checkpointing trades recomputation for saved intermediate memory; it does not
make an arbitrary full-history reverse pass fit in memory. See the
[JAX checkpoint documentation](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html).
The distinction between graded effective cores and explicit cellular topology
also appears in [dynamic multiscale sandwich optimization](https://www.sciencedirect.com/science/article/abs/pii/S0045782523004917).

## Completion criteria

Report the mass/response trade-off, per-load initial and optimized responses,
gradient check, mesh/time-step sensitivity, actual peak memory and wall time.
A changed density image alone does not establish an improved design.
