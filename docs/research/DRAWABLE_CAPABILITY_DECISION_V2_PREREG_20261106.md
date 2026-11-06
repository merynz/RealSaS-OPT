# Drawable capability decision V2 — preregistered finite experiment

Base: PR351 `e07d7dc6dd77e01f88e400c85e51b1e986923852`.
Independent review branch: `research/drawable-capability-decision-v2`.
This is external research evidence, not a Platform Attempt or a qualification.
All owner, training and materialization authority remains false.

## Question and mathematical objects

Source evidence E consists of immutable source identities, contexts (pose,
camera, sample), interventions and independently observed consequences, with
explicit validity. A capability contract C specifies required operations,
their address/state semantics, observations and error bounds. A representation
policy P ranks feasible implementations; it cannot create source facts.

A drawable is a tuple (surface addresses, allowed state/operators,
deformation binding, compositing rule). A puppet representation R additionally
has mechanical carriers, possibly overlapping graphic controls, and an
operation compiler. Its surface partition alone does not determine capability.
For operation u and context x, the compiled output is O(R,u,x). Feasibility is
universal over the finite required test domain, with UNKNOWN for unmeasured
required domains. One counterexample refutes a candidate. PASS means only this
finite contract, not all possible poses or edits.

Separation is necessary relative to a declared representation class only when
every allowed merged implementation fails a required operation. Failure of one
merged implementation is not proof that every merged implementation fails.
Keep equivalent alternatives. With numerical tolerances, compatibility is NOT
transitive: output a pairwise compatibility graph, never union-find its edges
into a claimed equivalence class. Exact operation-state equality does define an
equivalence relation, distinct from empirical approximate compatibility.

Constructive non-identifiability: if a class permits arbitrary source-addressed
operators, identical per-face deformation and a common sample-depth compositor,
splitting/repacking a drawable while preserving its fragment stream changes its
partition but not any allowed observation. Conversely an addressed merge can
simulate the original partition. Artist ownership cannot be uniquely recovered
from these observations in that class. This does not assert that arbitrary
implementations or globally sorted flat sprites are equivalent.

## Frozen retrospective physical experiment A — Cat controls and reveal

Reuse the exact independently rendered PR351 Cat V2 array (SHA
`d422b7d7d082fba5deac5dd1239b8c6aec078f4d635a7bd5b1f10e514100b114`),
manifest `8a56e47a09f5c1216de2c97e6f13d5e04385f7fe7d2e580d0085b2ab69dc2da1`,
court `90b85b9c1a963b23f7c3037ca443975bf2d5c5d4123568d8aa8e54557e433c08`.
This is a NEW decision analysis of ALREADY observed physical trials, not new
Blender renders and not prospective/unseen validation. Parent headline counts
were known. New candidate-comparison measurements have not been inspected.

Contract: independently toggle each of the three source-defined RGB atlas edits
without altering alpha or other edit bits; support all eight combinations and
all four recorded source-rest cameras, at matching Cycles sample index. Repeat
the observation analysis on V3/V5 alone and on V1/V3/V5/V7. This tests camera
reveal observability, not animation or physical detachment.

Enumerate ALL five partitions of three edit addresses. A control toggles every
address in its group simultaneously; controls may be set independently.
For every required state, enumerate every reachable state. A candidate is
source-address feasible only if the exact required bit pattern is reachable.
Independently compare the actual physical render for each reachable state with
the actual requested-state render. Optimistic best-match enumeration is an
impossibility test, NOT a compiler that may inspect its target. Report best
mismatch counts, max error, and all compatible reachable alternatives.

Evaluate distinct representation descriptions sharing three independent
address controls: one addressed drawable; several drawables sharing a carrier;
per-face packaging with retained subface UV masks; hybrid carrier/control
representation. Exact operation compiler identity certifies control-semantic
equivalence only; actual repackaged runtime performance is NOT measured here.
No face-level UV-centroid partition substitutes for exact texel addresses.

## Frozen retrospective physical experiment B — Triceratops compositing

Use PR351 final K3 original-polygon source run42248465757 only. Pin its manifest,
array and court SHA from the source court before numerical analysis.
Contract: source FULL plus each of three independent RGB edits, V1/V3/V5/V7,
all eight same-sample offsets; source alpha must be binary under this policy.

Hypotheses: (1) three source palette drawables with one global total order;
(2) same partition with an order chosen per camera; (3) same partition with
sample-local depth; (4) a logical addressed aggregate retaining the same depth
and control access. Enumerate ALL six flat orders, not an order tuned to targets.
Hypothesis3 reads isolated RGBA/depth, never original FULL targets. FULL/edited
FULL appear only in the independent evaluator. Alternative4's equivalence is a
representation theorem, not a newly benchmarked draw-call implementation.

Keep parent's .025 max premultiplied RGBA error, .0002 depth-tie threshold,
depth interval [.019,.981] and all two unknown samples. No threshold tuning.
Record all order/arm/camera errors, both directions of qualified local precedence,
any cycle witnesses, and per-camera best order as an optimistic lower bound.
If any flat order satisfies the contract, preserve it. If every order fails,
reject only that flat-order representation class, never mint a different-owner
edge. A depth-capable merge or finer partition can remain alternatives.

## Independent court, failure accounting, and decision output

Producer and court use separate implementations: candidate front-to-back versus
court back-to-front OVER, vector sorting versus independent nearest-fragment
selection (binary-alpha scope), Boolean generator versus scalar enumeration.
The court recomputes every comparison from pinned physical arrays; it must not
merely compare a producer report to itself. Test tampered source SHA, wrong arm,
wrong tolerance, unknown samples, nonbinary alpha, forced labels and policy drift.

Generic mechanism returns per-capability SATISFIED / VIOLATED / UNKNOWN;
counterexample witnesses; all surviving alternatives; missing observations;
pairwise compatibility; structural cost vector and optional explicit Pareto
policy. No weighted-cost default and no runtime latency estimate. Costs include
logical drawable count, independent graphic control count, addressing state and
depth-decision requirement; geometry/skin sharing is explicit. Zero extra
drawables never means zero addressing cost.

Pose deformation, attachment/release, secondary motion, arbitrary draw-order
override and nonbinary alpha are separately REQUIRED_UNKNOWN in the full puppet
contract until independently measured. A finite graphics PASS cannot promote the
full contract. Synthetic pose/attachment counterexamples test UNKNOWN handling,
not real-source attachment truth. The checkpoint must say which of these domains
remain open and give the next discriminating intervention.

No changes to existing source producers, parent evidence, model heads, corpus
family states or thresholds. New code lives in isolated tools/corpus modules,
tests, docs, and a branch-scoped hosted CPU workflow. Do not merge to main.
