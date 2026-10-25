# Force and material time coupling

The saved 125–125.002 ns step permits a decomposition without another solve.
For each kick, insert the drift affine velocity and then the midpoint
deformation gradient into its stress-work contraction. Their differences,
plus the endpoint stress quadrature difference, recover the recorded work
remainder to `4.69e-20 J`.

| Contribution | Work difference (J) |
| --- | ---: |
| Kick velocity versus drift velocity | 7.66407e-10 |
| Endpoint versus midpoint geometry | 2.40292e-14 |
| Endpoint versus material stress quadrature | -9.24040e-12 |
| Total | 7.57190e-10 |

The velocity contribution splits into `5.94314e-10 J` from the first kick,
`-6.00175e-10 J` from the last kick, and `7.72267e-10 J` from the intervening
geometry projection and contact. The endpoint damage reduction accounts for
nearly all of the smaller stress-quadrature difference.

The next coupling must use a common time level for force work, contact and
material kinematics. Updating only the material return or reducing the step
does not address the dominant term. Any midpoint construction must also retain
the moving-grid constraints, physical boundary torque and contact passivity;
an energy correction applied afterward would not establish those properties.

The diagnostic does not modify states, accept a longer impact window, or
assign the remainder to heat. Data and input hashes are in
[the decomposition record](metal-force-time-decomposition.json).
