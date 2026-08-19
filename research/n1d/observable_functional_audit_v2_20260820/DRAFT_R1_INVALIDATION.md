# Observable Functional Audit V2 — Draft R1 Invalidation

**Status:** `INVALID_PRE_FREEZE_DRAFT__DO_NOT_EXECUTE`

The source draft identified by ZIP SHA-256 `3bd133e805880ccca2e6a177778e3e931064eb0178da5a2ac5b02c07f1595278` never passed `SOURCE_THREE_STORE_PASS`, never reached preregistration, and never opened evaluation truth.

It is permanently invalidated for two independent reasons detected during pre-freeze audit:

1. **GitHub transport integrity:** the attempted `source_bundle_b64/part03` persistence did not preserve the intended local part boundary, so the GitHub textual bundle was not verified to reconstruct the frozen ZIP.
2. **Metric/source-lineage audit:** state-vs-state NRMS was directional rather than symmetric, and a descriptive candidate rank used `.5/.5` although the frozen V8/V5 lineage uses `OBS_ALPHA=.75`.

No scientific output from this draft exists and no threshold/result was inspected. The replacement source is a new append-only R2 source authority; R1 files remain only as forensic history and must not be used as experiment dependencies.
