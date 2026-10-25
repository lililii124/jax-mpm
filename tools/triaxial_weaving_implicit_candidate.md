# Independent implicitMPM forming candidate (not implemented or dispatched)

The current dynamic batch case remains authoritative and continues unchanged.
Measured contact-free cost~.16s/step implies~30hours for671000steps even
before the per-dt global contact fallback. Thus contact-rich total wall time
can still reach several days. This is an execution estimate, not a result.

A separate quasistatic updated-LagrangianMPM version is a reasonable next
algorithm if that cost remains unacceptable. Keep the24real initially straight
cords, materialE/nu/rho, same world grid and actual2batch guide path. Introduce
an equilibrium load parameter, explicitly dropping the dynamic-time claim;
do not splice its results into the current dynamic trajectory.

For each increment, actual nodal displacements produce
x_p(new)=x_p(old)+sum_i w_ip u_i and
F_p(new)=(I+sum_i u_i outer gradw_ip)F_p(old).
Assemble internal forces from exactly the existing corotated stretch material
and particle quadrature. Use residual-based Newton-Krylov, positive-J and12%
logstretch line search, with the actual essential end-guide constraints.
Avoid differentiating arbitrary eigenvectors at repeated principal stretches:
a stable polar differential/Sylvester tangent or verified Jacobian-vector
implementation is necessary at the initially stress-freeF=I state.

Contact is solved globally at every nonlinear iteration. The existing Hertz
patch limits remain applicable; freeze topology only within a Newton substep,
refresh/reject on active-set or coverage changes. Do not assume the current
patch force distribution is an exact gradient of the reported patch potential;
verify that before using an energy-minimization formulation. A residual-based
solve can instead retain the actual force with a consistent verified tangent.
Extended-contact failure would require an explicitly separate physically
appropriate distributed contact model, not a larger Hertz validity limit.

Start with maximum guide increment.2h, adaptive backtracking. The.038m largest
feed then requires~2500increments per family rather than acoustic microsteps.
This is only an increment estimate; actual Newton/linear-solve cost is unknown.
First necessary tests would be one purely elastic load increment against the
explicit small-step endpoint and a real contacting increment with force/work/
gap complementarity. Report nonlinear residuals, globalcontact residuals,
energy/work,12%strain, positiveJ, mass, actual guide errors, and final crossing
order. Refine guide increments at a meaningful forming endpoint. A successful
implicit solve is still not proof of dynamic equivalence or weaving topology.

No particle is driven toward a final weave. No hidden/revealed completed
geometry, interpolated body path or resin is part of this proposal.
