# Operational drawable decision V1 — preregistration

Date: 2026-11-06. Parent source: PR343 `635f985fb888ff17862520afd9f5f07813385778`.
Scope: teacher research. No model training, materialization or intrinsic owner promotion.

## Question and decision rule

A partition is meaningful only relative to a representation class and the edits
that class must express. In the class tested here, an atomic handle applies a
uniform Boolean hide/tint selector to a fixed set of source faces. It retains
the original per-face appearance, per-vertex deformation and ray-local order.
Composite handles can address unions of atoms, and can overlap. Their nesting
is not assumed to be a tree. A single drawable with arbitrary per-face masks is
a *different representation class*: its K=1 does not eliminate the complexity
of its internal selectors.

For required operations t, let Q[t,f] be the source face support and V[t,f]
its measurement validity. Q is NOT a visibility map. For complete evidence:

    f ~ g  iff  for every required t, Q[t,f] = Q[t,g].

The equal-column quotient is the unique coarsest partition that can realize
all required selectors as unions of atoms. Proof: a block containing different
columns cannot realize the row that distinguishes them; equal columns can
always share a block. Every sufficient partition refines this quotient.
This is a finite Boolean edit-algebra result, not general dynamic bisimulation.
Uniform selector composition is preserved; arbitrary geometric edits, physical
contacts and noncommuting operations are outside this theorem.

`MUST_SEPARATE_FOR_WORKLOAD` has a distinguishing task witness. Equal complete
columns give `OPERATION_EQUIVALENT_FOR_WORKLOAD`, not a fact that two surfaces
were originally owned together. The explicit minimum-sufficient policy merges
them. Refinements remain admissible. Missing compatible columns are not joined:
compatibility under partial observation is non-transitive. Unknown addresses
remain unresolved singleton placeholders, not qualified individual drawables.

Thus this mechanism can prove sufficiency and decide merge/split for a declared
workload, but cannot choose a meaningful workload from unlabelled observations
by logical necessity. That missing product prior is exposed, not hidden in graph
weights. All task-conditional labels stay out of the existing unconditional
IRIS `same_drawable_affinity` target unless a later versioned contract admits
the exact policy and observability domain.

## Frozen real-source workload probes

Cat and Triceratops; exact original V18a and V13b arrays and source hashes.
Fit: original authored pose ordinals 0/1. Heldout: ordinal 2. Do not change these
policies after inspecting source measurements or heldout renders.

1. **Palette edit policy:** permit replacement of each distinct *qualified
   constant source shader color value* independently. Equal color is a shared
   palette handle, NOT same drawable. Source material/object/bone names and slot
   IDs are not grouping keys. Only simple constant shader outputs are admitted;
   linked texture/procedural/mixed shader paths remain unavailable in this V1
   probe. Colors are compared as exact source float values, without tuning bins.
   Original appearance/UV is always retained. Different shaders with equal
   measured colors share a palette control under this explicit policy only.
2. **Affected-region tint policy:** permit a tint of each measured anonymous
   native actuator's affected surface support. This is an explicit graphics
   selection operation, not identification of a mechanical or artist part.
   Use max mean face vertex displacement across XYZ and +/- probes, normalized
   by first fit pose source extent. The inherited V18a floor `2e-4` is a fixed
   instrument support threshold, NOT an ownership similarity threshold.
   Both fit poses must agree on active/inactive; disagreement is UNKNOWN.
   Keep all source controls, including empty/global/duplicate supports, and
   record the private action-basis dependency. Do not select useful controls
   from heldout data. Native action basis invariance is NOT claimed.
3. **Joint policy:** compile the union of palette and affected-region tasks.
   Measure crossing handle supports, atom complexity and policy sensitivity.
   Do not promote the joint result simply because it can reproduce the tasks
   used to define it. Its exact support preservation is an algebraic property;
   heldout stability and actual source rendering are separate observations.

UV/material relations are new factual evidence, with separate validity, and
are not made into must-join or must-cut labels. Explicit source rest topology
and face order must agree with the same SHA-bound V18a arrays. The source
renderer must replay admitted material policy, not substitute synthetic color
as evidence for original artwork. Controlled tint is a named intervention.

## Comparison and independent evaluation

Compare source topology, V18a .045/.10/.18 and V18q multi-axis stress against
the exact workload quotient, all on the same workload/source domain. Report
source-area minimum edit mismatch and unknown denominator separately from
render error. Report K, effective K, singleton/dominant mass as diagnostics.
No K target or winner is selected from heldout. Mutex/correlation clustering
are algorithmic checks only when supplied genuinely qualified signed task
constraints; they cannot invent the missing task prior. Vanilla lifted
multicut preserves connectivity in its *base* graph: lifted edges alone do
not join disconnected base components. Cat requires an explicitly augmented
base connectivity model or unrestricted correlation clustering.

The independent algebra court imports no producer code. It checks every
pairwise necessity/equivalence and reconstructs every task by scalar membership.
An exhaustive small-partition oracle verifies minimum K/unique coarsest blocks.
The source court must separately verify file hashes, original Blender rest
identity, source UV/material extraction, heldout support stability and source
edit render consequences. Reusing the producer in an allegedly independent
court is insufficient to detect a shared mathematical implementation error.

Evaluate palette and affected-region task ablations and their common refinement.
Reject original-source appearance mismatch, source identity drift, heldout fit
leakage, unsupported shader-as-constant, unobserved-as-negative, and promotion
of policy-conditioned same/different into unconditional owner labels.
Source material color extraction without independent shader verification is
only a candidate, not `SOURCE_RGBA PASS`.

## Outcomes permitted

- Engineering: exact finite-workload compiler, source identity bindings,
  independently checked candidate outputs and measured source consequences.
- Scientific partial: quantify how a stated operation family reduces the
  admissible partition set and which additional operation changes it.
- Failure: task-specific output offers no coherent useful editor controls,
  heldout support is unstable, artifacts misbind, or raw/source-render parity
  fails. Retain failures, do not tune the floor to pass.
- Still open: a source-grounded product workload that yields useful generic
  character organization, attachment/secondary physics, nonconstant/partial
  alpha coverage, unseen morphologies and canonical student target admission.

This experiment cannot by itself finish the doctoral mandate. It is intended
to replace unexplained feature thresholds with a falsifiable decision mechanism
and to expose exactly what extra prior is doing the work.
