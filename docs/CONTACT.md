# Contact and adhesion

The implementation has four separate uses:

| Interface | Solves | Example |
|---|---|---|
| `hertz_contact`, `jkr_contact`, `jkr_history` | Reduced, axisymmetric elastic contact laws | `jkr_law_cycle.csv` |
| `RigidSphereBoundary` | Frictionless unilateral contact between a prescribed rigid sphere and an MPM body | `sphere-contact` |
| `spherical_surface_load` | A prescribed Hertz/JKR pressure footprint applied as particle forces | `hertz-footprint`, `jkr-footprint` |
| `DugdaleLaw`, `sphere_interaction` | Finite-range adhesive surface forces coupled to the deforming MPM body | `dugdale-array` |

The geometric sphere example uses the MPM body's stiffness to determine its response. The footprint examples prescribe a load from a reduced law; their nominal law displacement is independent of the computed MPM surface displacement. They are not a self-consistent three-dimensional adhesive-contact solution.

## Units and signs

Lengths are in m, forces in N, elastic moduli in Pa, and work of adhesion `work` in J/m². Positive indentation and force mean compression. Parameters require positive radius and modulus and nonnegative work of adhesion.

For two isotropic bodies,

\[
\frac{1}{E^*}=\frac{1-\nu_1^2}{E_1}+\frac{1-\nu_2^2}{E_2}.
\]

`reduced_modulus(E, nu)` uses a rigid second body. The sphere radius is the effective curvature radius; for two convex spheres use \(1/R=1/R_1+1/R_2\).

Hertz contact is

\[
a=\sqrt{R\delta},\qquad P=\frac{4E^*a^3}{3R},\qquad \delta>0.
\]

The JKR relations are

\[
\delta=\frac{a^2}{R}-\sqrt{\frac{2\pi wa}{E^*}},\qquad
P=\frac{4E^*a^3}{3R}-\sqrt{8\pi wE^*a^3}.
\]

These use `work = w`, the work to separate the interface, rather than one surface's energy. See [Johnson, Kendall and Roberts (1971)](https://doi.org/10.1098/rspa.1971.0141) and the dimensional conventions in [ContactEngineering/Adhesion's reference solution](https://github.com/ContactEngineering/Adhesion/blob/d4590102abef8e7f392ab7456bf53a6b036d5090/Adhesion/ReferenceSolutions/JKR.py). The JAX solver is implemented independently from the published equations.

## JKR stability and history

`jkr_contact` selects the largest contact-radius root on the stable, displacement-controlled branch using a bracketed solve. The displacement instability is

\[
a_c=\left(\frac{\pi wR^2}{8E^*}\right)^{1/3},\qquad
\delta_c=-\frac{3a_c^2}{R},\qquad P_c=-\frac56\pi wR.
\]

This differs from the load-controlled pull-off force \(-3\pi wR/2\). Contact opens at or below \(\delta_c\). A previously separated pair first attaches at nonnegative indentation; there is no finite-range attraction before contact. `jkr_history` carries this state through a displacement sequence.

```python
import jax.numpy as jnp
from mpm_jax.contact_laws import jkr_history, reduced_modulus

delta = jnp.r_[jnp.linspace(-0.006, 0.025, 200),
               jnp.linspace(0.025, -0.006, 200)]
contact = jkr_history(delta, radius=0.4,
                      modulus=reduced_modulus(2e4, 0.25), work=3.0)
```

The functions support JIT and `vmap`. The stable JKR radius uses an implicit custom JVP, checked against finite differences in indentation, radius, modulus and adhesion. Derivatives at attachment, detachment and the branch instability are not smooth sensitivities. Zero adhesion selects Hertz directly.

The full axisymmetric Maugis–Dugdale sphere solution is not implemented. It resolves the contact and adhesive zones and their transition between JKR and DMT; adding a constant attractive force to Hertz would not implement that model. The finite-range interface below uses the Dugdale traction–separation assumption from [Maugis (1992)](https://doi.org/10.1016/0021-9797(92)90285-T), with the deformable body's response computed by MPM.

## Actual MPM sphere contact

`RigidSphereBoundary` follows the existing `Config.boundaries` interface. Its parameters are an immutable dataclass and are included in saved configurations. For center \(c(t)\), radius \(R\), normal \(n=(x-c)/|x-c|\), the grid velocity inside the sphere satisfies

\[
v'=v+\max[-(v-\dot c)\cdot n,0]n.
\]

Tangential velocity is unchanged. A separating node receives no tensile force. The smooth loading/unloading cycle is

\[
c(t)=c_0+\tfrac12[1-\cos(2\pi s)]d,\qquad
s=\operatorname{clip}(t/T,0,1).
\]

`boundary_reaction` replays the next grid update and returns minus the selected boundary's impulse on the occupied grid divided by the step duration. Earlier boundaries are applied in their configured order. It does not advance the particles. Optional `force=...` supplies the same particle forces used by `step`. The reported value is a force sample at each output frame, not an integral over all intervening steps.

```powershell
python -m examples.adhesive_contact --case sphere-contact --grid 48 --ppc 3 --frames 40 --no-render --resume
python -m mpm_jax.visualization outputs/contact-models/sphere-contact --gif
```

The default body has **386,019 particles**, 27 particles per fully filled grid cell, and occupies 0.6 × 0.35 × 0.6 m. Its base is fixed; the other faces are free. A 0.3 m radius sphere moves through a 0.04 m stroke and returns in 0.3 s. Gravity is zero. The solid is Neo-Hookean, with \(E=20\) kPa, \(\nu=0.25\), and \(\rho=1000\) kg/m³. The maximum Hertz reference contact radius is about 5.3 grid cells.

`observables.csv` contains nominal indentation, sampled sphere reaction, Hertz reference force/radius, central surface displacement, and the force difference. The initial small contact patch is under-resolved. The finite block, fixed base, finite strain, grid-node contact and inertia also differ from Hertz's static half-space assumptions. A force difference is recorded as a diagnostic, not an established accuracy tolerance. A half-space verification requires slower loading and spatial/domain refinement.

The renderer may add the analytic rigid sphere as a separate CAD surface. That surface is not a simulated particle body and is excluded from particle counts and diagnostics.

## Prescribed pressure footprint

```powershell
python -m examples.adhesive_contact --case jkr-footprint --grid 48 --ppc 3 --frames 40 --no-render --resume
python -m mpm_jax.visualization outputs/contact-models/jkr-footprint --gif
```

The same constrained elastic block receives compressive Hertz-shaped pressure plus a tensile JKR edge distribution. The two parts are normalized separately, preserving their discrete resultant even near zero total force. The singular tensile edge is regularized over one surface-particle spacing. Loads use the reference top-face coordinates and fixed surface normal.

`spherical_surface_load` returns particle forces, equal and opposite indenter reaction, a resolution flag, and the number of loaded particles. Fewer than 12 footprint particles marks an unresolved sample. A sub-particle footprint is assigned to the nearest surface particle; this preserves the force but is not a resolved pressure field. Empty surface selections apply no load and remain unresolved while contact is active.

The particle-count flag checks surface-load sampling. It does not establish stress convergence on the MPM grid. The review also reports contact radius in grid cells, including active frames below two cells near separation.

The footprint example provides `force_assignment_error_N`, the attachment state, imposed contact radius and computed surface displacement. Its 801-sample `jkr_law_cycle.csv` is the reduced-law loading/unloading curve. No second elastic compliance is added to the MPM equations, and nominal law indentation is not enforced as a geometric constraint.

## Finite-range Dugdale interface

`mpm_jax.adhesion` evaluates contact against the **current** advected surface points. For a sphere with center \(c\), radius \(R\), and a sampling point \(x_i\),

\[
g_i=|x_i-c|-R,\qquad n_i=\frac{x_i-c}{|x_i-c|},\qquad
\sigma_0=\frac{w}{h}.
\]

The potential per reference area and its normal traction are

\[
\psi(g)=\begin{cases}
\tfrac12 k g^2-w,&g<0,\\
\sigma_0 g-w,&0\le g<h,\\
0,&g\ge h,
\end{cases}
\qquad
t(g)=\begin{cases}
-kg,&g<0,\\
-\sigma_0,&0\le g<h,\\
0,&g\ge h.
\end{cases}
\]

Here \(h\) is the interaction range in m, \(k\) is the penalty stiffness in Pa/m, and \(w\) is in J/m². This constant attractive traction over a finite opening is the Dugdale approximation; see [Johnson (1997), §2 and Fig. 1](https://bdml.stanford.edu/twiki/pub/Rise/AdhesionModels/JohnsonAdhesionModel.pdf). The finite penalty permits a measured overlap in compression.

With fixed reference quadrature weights \(A_i^0\), the discrete energy is \(U=\sum_i A_i^0\psi(g_i)\), and \(f_i=A_i^0t(g_i)n_i=-\partial U/\partial x_i\) within each smooth branch. The force on the sphere is \(-\sum_i f_i\). The potential is not differentiable at \(g=0,h\); the implementation selects the right branch there. JAX position and center derivatives are supported. The immutable law is a static JIT argument; its parameters are not advertised as differentiable inputs.

This is a reversible interface with frictionless tangential motion. Opening a reference area from zero gap beyond \(h\) costs \(wA^0\); closing restores that potential energy. There is no irreversible bond damage, cohesive fracture dissipation, or molecular dynamics. The dynamic finite-layer calculation is not an analytical Maugis solution, a JKR/DMT limit verification, or an experimental calibration.

### Three moving indenters

The [adhesion case suite](ADHESION_CASES.md) adds a matched nonadhesive/adhesive pull-off pair and a lateral sweep with oblique release. These use the same gap-based interface and expose an undeformed-surface force diagnostic; the original array entry point below is retained.

```powershell
python -m examples.adhesive_array --grid 48 --ppc 3 --frames 50 --no-render --resume
python -m mpm_jax.visualization outputs/adhesion/dugdale-array --gif
```

The default soft layer occupies 0.72 × 0.23 × 0.72 m, with **367,744 particles** and a fixed base. Its Neo-Hookean parameters are \(E=12\) kPa, \(\nu=0.25\), and \(\rho=1000\) kg/m³; gravity is zero and the other faces are free. Three 0.16 m radius rigid spheres sequentially indent, translate laterally and retract during 0.85 s. Their prescribed cosine waypoint paths are serialized in `definition.rigid_spheres`; ParaView evaluates those same paths for the auxiliary CAD surfaces. The spheres are excluded from the MPM particle count.

The 10,816 top-layer particle centers form the material surface quadrature. Each carries its fixed reference patch area. The initial sampling plane lies half a vertical particle spacing below the nominal top face; this offset is recorded. No deformation-dependent face offset or current-area multiplier is introduced. Surface deformation therefore changes the gap and resulting load without omitted derivatives of those additional geometric quantities. The example covers a top-face contact problem; it does not search all exposed surfaces or support folded-surface self-contact.

The illustrative interface uses \(w=12\) J/m², \(h=0.025\) m and \(k=1.2\times10^6\) Pa/m. Its range spans 3.61 surface-particle spacings at the default resolution. This deliberately resolved, macroscopic range is not a fitted molecular adhesion distance. The time step meets both the bulk wave-speed restriction and \(kA^0\Delta t^2/m\le0.09\); the latter is a local penalty indicator, not a global nonlinear stability proof. Runs with a range below three surface spacings are only smoke tests.

`observables.csv` records every sphere's position, velocity, reaction, reaction moment, compressed and adhesive reference areas, interacting point counts, minimum gap, and potential energy. Maximum overlap is measured both over the sampled surface and over all body particles, so unmodelled contact with an interior or side face can be detected. The base reaction includes the same contact force used in the next P2G step. Force samples use time \(t\); the base boundary is evaluated at \(t+\Delta t\), as recorded in separate columns.

Actuator power is \(-R\cdot\dot c\), where \(R\) is force on the sphere. Together with surface-point power, it satisfies the instantaneous interface potential identity. Sampled powers and the common runner's trapezoidal particle work are diagnostics; they do not establish global MPM energy conservation. A complete pull-off requires tensile reaction during retraction followed by zero interacting points and a final minimum gap beyond the finite interaction range.

## Checks

`tests/test_contact_laws.py` checks Hertz scaling and stiffness; JKR roots against an independent NumPy quartic eigensolve across three dimensional parameter sets; the two different instability criteria; zero-load contact; separation/reattachment history; the zero-adhesion limit; implicit derivatives; the tensile pressure ring and force/moment balance; the moving sphere's unilateral constraint; and agreement between its grid reaction and the actual particle momentum change.

These checks establish the law and coupling identities. They do not certify the three-dimensional MPM solution against a converged JKR or Maugis reference.

`tests/test_adhesion.py` checks traction against the potential gradient, separation work, the moving-sphere power identity, equal and opposite force and moment, disappearance of interaction beyond the cutoff, MPM force transfer, and the base reaction with external forces. A separate geometric check integrates an undeformed plane with radial Gauss quadrature: for a sphere center height \(c=R+g_0\), \(0<g_0<h\), the exact normal reaction is \(-2\pi\sigma_0c(h-g_0)\), and the adhesive reference area is \(\pi[(R+h)^2-c^2]\). This checks the discrete geometric integration independently of the MPM response. The NumPy motion used by ParaView is compared with the JAX sphere trajectory.

After both full examples finish, summarize their saved CSV files without loading particle trajectories:

```powershell
python tools/review_contact.py outputs/contact-models
```

This writes `contact-review.json` and a Hertz/JKR `law_comparison.csv`. The sphere review separates loading from unloading and reports the interval where the Hertz reference radius spans at least four grid cells. This selection describes spatial resolution; it does not establish quasi-static or half-space conditions.

## Recorded runs: 23 October 2026

Both default cases completed **386,019 particles, 480 steps and 0.3 s** on the CPU in float64, using a 48³ grid and three particles per cell per axis. There are 41 saved frames per case.

| Case | Minimum det(F) | Peak normal resultant | Maximum force-assignment error |
|---|---:|---:|---:|
| Rigid sphere | 0.657805 | 146.957 N reaction | — |
| JKR pressure footprint | 0.835082 | 67.135 N prescribed load | 1.42 × 10⁻¹⁴ N |

The sphere reaction remained compressive in all saved samples. At maximum nominal indentation, its reaction was 109.200 N against the Hertz reference of 124.629 N, a −12.38% difference. Across the nine loading samples with reference radius at least four grid cells, the median relative difference was +48.32% and the maximum absolute difference was 72.95%. The kinetic/strain-energy ratio at maximum indentation was 0.162. These results describe dynamic finite-block contact and do **not** establish Hertz half-space accuracy. The block rebounded during the return stroke; the final reaction was still 14.555 N, so this run did not sample complete separation.

The JKR footprint had 32 attached output frames, including seven with tensile resultant. Load assignment preserved the resultant to 2.12 × 10⁻¹⁶ relative to the peak load, with zero assigned force while detached. All attached footprints contained at least 37 surface particles, but two frames near release had a contact radius below two grid cells. Central surface displacement ranged from −23.54 to +4.78 mm. These are responses to prescribed traction, not a solved geometric adhesion condition.

The exported 801-point reduced-law cycle had one attachment and one detachment. It bracketed the displacement instability at −3.20545 mm between the last attached sample at −3.16873 mm and the first detached sample at −3.24325 mm. Maximum equation residuals were 1.39 × 10⁻¹⁷ m and 2.84 × 10⁻¹⁴ N. Full [review metrics](results/contact-review.json) and [law comparison](results/contact-models--jkr-footprint/law_comparison.csv) are saved with the run records and can be regenerated with the command above.
