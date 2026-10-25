# Three-strand tension

Three independent circular fibres form a stress-free helical reference with a
24 mm length, 2 mm diameter and 8 mm pitch. Axial end collars apply tension;
each fibre retains its own MPM velocity field and colour.

```bash
python -m examples.three_strand_tension --output outputs/three-strand-tension --stop-after-frame 12
```

The published animation contains saved frames 0–12: 3,500 accepted steps and
2.1433 ms of early loading. The displayed grip stroke reaches 0.369 mm. It does
not cover the full loading protocol or a simulated twisting process. The
material remains undamaged in this segment.

The local Hertz model uses reference circular sections. A later contact patch
exceeds its extent limit, stopping the original run; a distributed contact
formulation is needed before continuing. The saved-state check verifies
material response, mass, momentum and the energy ledger, with a maximum
residual of 0.363% of the maximum fixture work. It does not establish exact
contact between independently reconstructed deformed surfaces.

[Saved results](results/fibres--three-strand-tension/summary.json) ·
[Source and scope](results/fibres--three-strand-tension/provenance.json)

To render the archived raw segment with ParaView 6.2:

```bash
python tools/render_three_strand_prefix.py RAW_DIRECTORY OUTPUT_DIRECTORY --pvpython /path/to/pvpython
```
