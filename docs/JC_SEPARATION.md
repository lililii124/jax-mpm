# Current grid separation

```powershell
python tools/jc_grid_separation.py outputs/material-engineering/jc-necking/johnson-cook
```

For a completed result, this read-only diagnostic compares the positive quadratic grid-node supports
of active particles on either side of the specimen centre. It uses the actual
saved reference coordinates and current checkpoint, reports shared 3D nodes
and reference-grip membership, and hashes its raw inputs. The complete endpoint,
state clock, time index and failure labels must agree. No earlier tensor history
is reconstructed. The reported sides are partitions, not a claim that each
side is internally connected.

For the single-body ballistic solver with local material updates and prescribed
node velocities, disjoint supports imply no current grid-mediated transmission
across that partition. Failed debris has no grid participation. The inference
does not extend to added interparticle forces or global feedback constraints.

Particles from an original material layer may survive on both separated pieces.
Their mean reference-layer Piola force is not the traction across a current
empty plane. Neither requiring every original-layer particle to reach D=1 nor
requiring its mean force to vanish is necessary to demonstrate this numerical
separation. Conversely, a zero mean force can result from cancellation while
the supports remain connected.

The report is a current-state diagnostic. It does not establish the failure
cause, the earlier separation history, resolved crack surfaces, fragment
recontact or mesh-independent material fracture. It changes no original
acceptance verdict and grants no rendering or publication eligibility.
