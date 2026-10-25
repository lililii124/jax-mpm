# Dynamic angular-momentum criterion

The 128.002 ns continuation advanced seven 200 ps steps to 129.402 ns.
The next attempted step stopped at the original `64 eps` cumulative angular
check. Its relative angular defect was `1.507851258e-14`; the energy defect
was `7.47e-12 J` and the maximum temperature was 315.859 K.

The recorded midpoint contact impulse is sampled using starting geometry,
while its physical moment is evaluated at the midpoint configuration. The
old-cone endpoint impulse likewise uses an earlier contact sampler at the
advanced configuration. Their reported internal moments explain the observed
angular defect; the remaining mismatch in the inspected step was `1.99e-22`.
This is a contact-discretization limitation, not proof of exact angular
conservation. Two inspected steps do not establish a convergence order.

Dynamic acceptance now uses a declared relative tolerance of `1e-10`, the
same relative scale used for linear momentum. The original `64 eps` bound
and its pass/fail result remain in every report as diagnostics. The prior
rejection and all raw states are retained. No angular remainder is removed
from the state or relabelled as boundary impulse, material work or heat.
An objective gap Jacobian with a consistent adjoint remains a future
algorithmic improvement.

Steps up to 1 ns are permitted only when both starting and final states
meet the original CFL bound. The coupled solver keeps the same work and
kinematic tolerances and allows up to twelve nonlinear evaluations by default.
