# Controlled component interventions

Canonical source continues on `main`. A short PR is source review, not an
experiment identity. Use the persistent Go deployment and an `Attempt` for a
causal component replacement; hosted CI databases/CAS are disposable engineering
fixtures, not the durable research authority.

## Open the candidate with an explicit scope

Seal baseline/candidate EngineReleases with their versioned DAGs and transitive
implementation closures. Complete/admit the required baseline stages first.
Pin a parent Attempt from the same subject and baseline release. It must have
one unambiguous SubjectInput identity in its Go compile commands.

Add this object to the existing `realsasctl research-start` JSON request:

```json
{
  "intervention": {
    "direct_changed_stage_ids": ["21_CAA_COMPILE"],
    "preserved_stage_ids": [
      "19_STATIC_CANONICAL_MESH_QUALIFIED",
      "28_SKELETON_QUALIFIED",
      "32_SKIN_QUALIFIED",
      "35_DYNAMIC_MECHANICAL_MESH_QUALIFIED"
    ]
  }
}
```

The normal subject/release/parent IDs and `created_by` remain required. The stage
list above is an example of a Stage21-only parameter intervention, not a promise
that editing a shared appearance module changes only Stage21. Inspect actual
release impact and declare every intended direct implementation, policy,
parameter, graph, added or removed stage. Do not widen the declaration just to
make an unexpected change pass; investigate it first.

Go rejects undeclared direct changes and preserved IDs that are missing,
duplicated, removed or invalidated. It captures the parent's available stage
artifact IDs in the hashed attempt spec and `CODE_CHANGE_IMPACT_RESOLVED` event.
The baseline input role/order/artifact IDs are also captured. By default all
inputs stay identical. For an intended input intervention, declare the exact
`changed_input_roles` (for example `["manifest:appearance"]`, without the
`subject:` prefix). Go adds those consumers and their descendants to invalidation
using the released read-set. Actual added/removed/changed/reordered input roles
must match the declaration exactly; unrelated source or motion drift is rejected.
An unsegmented source role conservatively affects every consumer, so it cannot
be declared while asserting that mechanics are preserved.

Every unchanged candidate stage is frozen, including stages not explicitly
listed in `preserved_stage_ids`. `preserved_stage_ids` adds an assertion that
particular boundaries must remain unchanged.

## Resolve before scientific work

Submit `research-run` with an explicit target and immutable SubjectInput.
The normal semantic resolver computes the target ancestor closure. Before any
Engine stage starts, Go verifies that every frozen stage in that closure:

1. has a captured parent artifact;
2. resolves to qualified REUSE;
3. reuses that exact artifact ID;
4. receives only the declared input-role changes from the pinned baseline.

An independent cache miss, qualification loss, undeclared input drift, or
different artifact fails closed. It must not become an unrelated reinference.
Guards also reject execution/commit of frozen stages and replacement of their
bindings, even if an activity request supplies a wider allowed execution list.

`INTERVENTION_REUSE_VERIFIED` is an engineering receipt with exact preserved IDs
and semantic SHA256s, actual EXECUTE IDs, and unchanged IDs outside the target.
Outside-target stages are not counted as measured reuse. Input identities and
declared consumer roots are recorded too. `INTERVENTION_REUSE_REJECTED` preserves
scope rejection diagnostics; deterministic contract failures do not retry. Inspect the receipt and
final diagnostics through the existing Attempt endpoint/CLI. A reused stage
still needs its bytes in the configured persistent CAS; this does not change
artifact-store verification or make nested external files recursively portable.

Legacy requests without `intervention` keep normal cache/recompute behavior for
compatibility. **A causal “only component X changed” claim requires this contract**
and a matched measurement apparatus: source bytes, camera/direction, motion,
probe policy, metrics and renderer identities. A scope receipt is not a visual,
mechanical, product or generalization PASS.

## Geometry and RGB must have distinct dependency identities

The current IRIS stage produces geometry authority. A future appearance head
must publish an independently versioned producer/output contract and read-set.
Putting geometry and RGB into one stage-result manifest will invalidate both
consumers when that stage identity changes. Shared implementation/trunk changes
must remain explicit dependencies, not be hidden by suppressing hashes.

The current product DAG also routes CAA through the puppet seal and motion
proof. An appearance-only change can preserve model fit and M/G/W artifacts while
renewing downstream motion artifacts. Separating that dependency requires a
reviewed DAG change with equivalent proofs. See the
[visual dependency map](../presentation/VISUAL_DEPENDENCY_MAP_20261009.md).

## Source review and archived diagnostics

PR #65 was merged into main at `1888d91252fdce23843772f3ec0d0e3ddd91b9ae`.
PR #66's 74-commit presentation candidate is retained as diagnostic history,
not the active continuation or an accepted appearance solution. The completed
V6 evidence run `37933851760` failed both Attempts. Its successor runner changed
37/42/43/45, rather than Stage42 alone. The fast relation replay's green workflow
included an exposure failure tolerated by `continue-on-error`.

Future work extracts reviewed, bounded source changes onto main and runs matched
Attempts on durable services. Do not reopen that branch as a scientific ledger
or promote its direct PNG binding as qualified complete appearance.

Main currently has no required-check branch protection/ruleset. The connected
GitHub app cannot administer that setting. CI and this intervention contract do
not claim to replace repository protection; merges must verify the current PR
head's applicable checks. Enabling server-side required checks remains an
administrator operation.
