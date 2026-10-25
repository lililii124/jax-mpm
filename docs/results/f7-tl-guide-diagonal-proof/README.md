# F7 guide Gram structure and no-guide compatibility

On the exact R9 input geometry, 600 guide rows use 194,400 raw stencil entries. Coalescing duplicates by `(guide,node)` leaves 47,546 coefficients; no node belongs to multiple guide rows. Thus `G M⁻¹ Gᵀ` is exactly diagonal, and direct division matches the prior sparse CG solve to 8.74e-17 relative multiplier error (adjoint residual 3.31e-16). This proof advances zero physical steps.

The same source passed a one-step TL smoke with the guide operator omitted and both guide callbacks absent; no F1 long run was repeated.
