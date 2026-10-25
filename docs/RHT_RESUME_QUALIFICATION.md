RHT isolated resource/parity qualification (source only)
======================================================

Files: tools/rht_resume_qualification.py, tools/rht_qualification_child.py.
Original tools/rht_impact_continue.py and workflows are unchanged.

Current conservative admission remains >=8 logical CPUs/>=12 GiB available.
The explicit measured-experimental policy requires >=2 effective cgroup/affinity
CPUs and >=4 GiB effective available RAM, with a STRICTER 3 GiB aggregate child
USS kill limit (original 6 GiB guard remains too). This is an experimental
admission revision motivated by the historical ~2 GiB peak, not established
feasibility and not a silent bypass. Both retain original runtime, archive,
controller Git/workflow identity, exact 48-frame plan and 35-step f3->f4 restore,
finite/positive-J/material/signature/contact and endpoint validation, 10800 s,
output/free disk guards and all final penetration claims/gates. Raw cgroup membership/mountinfo, resolved v1/v2/hybrid ancestor limits and
effective admission are saved. Unknown/unreadable layouts fail closed.

Publisher sequence, on original registered engineering-previews workflow only:
  python -m tools.rht_resume_qualification resources
  python -m tools.rht_resume_qualification arm --parent PARENT --destination ORIGINAL --arm original --policy measured-experimental --weekly-used FRESH_COUNT
  # Recheck quota before each arm; stop at 90, not after a multi-arm batch.
  python -m tools.rht_resume_qualification arm --parent PARENT --destination DERIVED --arm no-warmup --policy measured-experimental --weekly-used FRESH_COUNT
  python -m tools.rht_resume_qualification compare --destination ORIGINAL --other DERIVED

Use pinned original Python/Linux/packages installed through existing requirements
command. Both arms start from the SAME immutable original f3 transport, never
from the first arm's f4. Separate processes, frozen environment and no persistent
compilation cache introduced. Controller Git binding remains the exact original
registered workflow; a standalone rht-impact-continue workflow fails that gate.
Publisher must register/version these new wrapper blobs as well before launch;
the wrapper requires both its own and child bytes to equal registered Git blobs
before it launches an arm. No source-only result
here constitutes numerical parity.

Instrumentation phase-timings.jsonl records outer import-call time, imports plus
case construction, make_case, _load_state, and frozen common.run line durations:
original warm batch lines112/113, restoration line150 and resume-loop lines,
retained batch204 + readiness205. Original common.py source bytes/line numbers
are fixed by SHA. The no-warmup arm retains every other source byte and proves
AST equality after exactly three deleted discarded statements. First retained
batch combines compile+execute; do NOT label this pure compile or compare it to
original post-warmup solver time. Trace overhead is included. The original
status compile_and_warmup_seconds provides synchronized warmup. No fine-grained
XLA compile-only timing is claimed. Receipt and resource snapshots survive failure.

Comparison requires all NPZ leaves dtype/shape/bytes, full checkpoint accounting
except named timing fields, and observables CSV bytes. It independently checks
every prior raw frame byte unchanged. Run original transport package/verify on
each arm and independently recover official artifacts before admitting parity.
The comparison is one-frame qualification only; no automatic f4->f5 chain and
no perforation/fragments/publication claim. Retain original evidence once before
implementing any later flat chain. Source-only controller still uses original
parent copies and is intentionally NOT a production multi-segment controller.

Quota reported by parent:74, stop90,16 headroom; not refreshed by this package.
No local heavy JAX/ParaView, Git, push or cloud dispatch performed.

Control hardening revision (quota75, stop90)
------------------------------------------
- Linux subreaper required. Every numerical validation/solver child gets its
  own new session. Stable owned-session inventories count orphan descendants
  after leader exit and include the controller exactly once across sessions.
- Two matching PID/create-time inventories are required around USS collection;
  process churn retries three times and then rejects. AccessDenied and every
  observer exception latch failure, kill owned process groups and prohibit any
  later numerical launch. No observer error permits an accepted arm.
- Finally stops observer, unconditionally kills/reaps every owned session using
  bounded cleanup, restores hooks and writes qualification.json. Cleanup or
  final resource-snapshot errors fail admission. Original run receipt is kept.
- Compare requires clean successful original/no-warmup qualification receipts
  plus admitted original-controller receipts and actual complete f4/35-step
  endpoints before reading NPZ leaves. Failed f3 prefixes cannot claim parity.
- No original numerical/controller source or numerical gates were changed.

Publisher dispatch contract (register in engineering-previews workflow):

```yaml
on:
  workflow_dispatch:
    inputs:
      arm:
        type: choice
        required: true
        options: [original, no-warmup, compare]
      weekly_used:
        type: string
        required: true
# Route ONE selected arm only; compare retrieves/verifies BOTH official
# transports first and runs no numerical child. No needs/matrix auto-arm chain.
```

```sh
case "$ARM" in
  original|no-warmup)
    python -m tools.rht_resume_qualification arm \
      --arm "$ARM" --parent .local/rht-parent-transport \
      --destination ".local/rht-qualification-$ARM" \
      --policy measured-experimental --weekly-used "$WEEKLY_USED"
    ;;
  compare)
    python -m tools.rht_resume_qualification compare \
      --destination .local/recovered-original \
      --other .local/recovered-no-warmup
    ;;
  *) exit 2 ;;
esac
```

Publisher must check fresh quota before each numerical-arm dispatch (currently
75;15 headroom to90). One arm invocation has no code path to launch the other.
The compare dispatch is read-only and must have no fallback to either arm.
Separate official transport recovery remains mandatory before comparison.


Cgroup resolution correction after official original-arm run40154998527
---------------------------------------------------------------------
That run failed at preflight before JAX/integration; its official log did not
contain membership/mountinfo, so the actual layout is unknown. The old lookup
assumed /sys/fs/cgroup/memory.max exists. New parsing resolves /proc/self/cgroup
against cgroup/cgroup2 entries of /proc/self/mountinfo, including separate v1
memory/CPU mounts, hybrid controllers, namespace roots and ancestor limits.
Effective available memory is min(host available, every finite limit-usage);
effective CPU is min(affinity, all finite quotas). Missing required memory
accounting fails closed. A verified true unified root (mount root '/', member
'/', resolved leaf equals mountpoint) legitimately has no memory.max, so uses
host available memory. No guessed unlimited limit for unresolved subgroups.
Bounded reads cap mountinfo at256KiB, membership64KiB and controller files16KiB.

Any arm preflight failure now creates only real diagnostic evidence at the
previously absent destination: resource-diagnostic.json, qualification.json and
receipt.json. No raw state or numerical success is fabricated. Existing package
can preserve this small diagnostic transport; qualification/compare still reject
it. Previously existing destinations remain untouched by this diagnostic path.
Policy >=4GiB effective available RAM and owned-session3GiB ceiling remain.
Twelve lightweight tests pass; no JAX or cloud execution used for this fix.
