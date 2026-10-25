# Fixed length-scale spatial comparison candidate

New source-bound case, not an interpolation of the coarse evolving crack.
E=1000, nu=.2, rho=1, Gc=1.25e-5, ell=.0375, h=.01 (ell/h=3.75), particle spacing=.005,72000 points.
Top normal speed=.005 with ramp=.06 and travel=.00045; same lateral freedom and Ax/Az/Bz minimal pins.
Undamped dynamic loading remains unchanged. First pilot is64 actual steps only, then resource admission.

The fixed seed cutoff is |y-.3|<=.0125, x<=.3 in physical model coordinates.
It is a discrete coefficient band; h=.01 realises rows .29/.30/.31, whereas coarse h=.0125 has .2875/.30/.3125.
Thus the cutoff is fixed but grid quantisation still changes its realised native profile. Report both initial fields/threshold widths and each initial fracture energy; subtract each initial seed from growth/energy. Do not describe this as exact equal zero-thickness seeds or definitive mesh convergence. A second seed-width study may be needed if the quantisation effect is material.

Compare at physical t=.072 and .078 (existing coarse4800/5200), top travel=.00021/.00024. Use complete native d profiles, full-band .5/.8/.9/.95/.99 fronts and lateral shift, new seed-excluded threshold volumes, reaction history, K/U, fracture increment, external absolute work and energy residual, KKT, CFL, strain/rotation and all immutable seed policies. Grid h=.01 dt=.000012 atCFL.04; aligned times are6000/6500.
Do not transfer a coarse checkpoint to the fine grid and call it full spatial convergence. Evolving irreversible fine history starts from its own minimum-energy seed and zero mechanical state.

Resource guard remains8 CPU/16GiB, outer2400s/USS12GiB and inner2100s/14GiB. This spatial64 pilot waits for the half-dt task to finish usingCPU32–39. No long spatial job is admitted by this source-only preparation.

## Local time-step check (predeclared before endpoint comparison)

Start from exactly the original4400 physical state, t=.066. Rebind only dt/CFL/required step count, global step8800 and history step indices. Original physical state/history/work arrays remain exact. Advance800 fine steps to9600, the same t=.072 as original4800; conditionally another800 fine steps to10400, same t=.078 as original5200. Fine dt7.5us/CFL.02. This tests only evolution after4400; prior accumulated coarse integration error remains.

Provisional case checks: native full-band d>=.95 front difference<=.00625; normal-profile d>=.5 and .95 widths differ<=10% at genuinely populated sections; reaction curve normalized RMS and peak differences<=5%; K, U, K/U and seed-excluded fracture increment differences<=5%; energy-window integration defect should not worsen and cumulative physical residual<=5% of absolute work. Tiny near-zero values are reported with their dimensional differences rather than divided by zero. Sampling error is measured explicitly. Failure means sensitivity, not permission to relax limits. One successful pair is local agreement, not full time or mesh convergence.
