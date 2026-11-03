# Post139 additive corpus audit

Truth139 remains immutable. Post139 is an additive source-audit lane; these files do not alter
an active Truth139 training run and do not mint Platform/Attempt/ModelRelease/ProductRevision
authority.

## Admission policy

The mechanical policy is:

- the same rig alone is **not** a duplicate;
- GLTF/GLB/FBX format pairs are one subject when canonical geometry/topology/rig/skin match;
- material/color-only variants collapse when geometry/topology/skin weights are identical;
- full static admission requires Geometry + Appearance + Rig + Skin;
- animation truth is recorded separately;
- AI-generated or uncertain-authorship assets must remain rejected/quarantined by provenance.

The structural subject key therefore excludes appearance and combines three independent hashes:
`geometry_topology`, `rig_rest`, and `skin_weights`. Appearance and animation inventory are retained
as separate hashes.

## Commands

For a final, cross-format-capable pack audit use Blender for every candidate:

```bash
python tools/corpus/post139_audit_v1.py audit \
  --source-root /data/Quaternius/UltimateAnimatedCharacter \
  --out /data/Post139/Quaternius/UltimateAnimatedCharacter \
  --family Quaternius \
  --pack UltimateAnimatedCharacter \
  --provenance CLEAN_CC0_HUMAN_AUTHORED \
  --license-name CC0 \
  --blender-exe blender \
  --reference-root /data/Truth139/source_audit \
  --reference-root /data/Post139/already_audited
```

Without `--blender-exe`, GLTF/GLB can still be censused and technically inspected, but their
admission remains `AUDIT_READY_PENDING_CANONICAL_DEDUP`. FBX fails closed until Blender is supplied.

For packs where filenames are not one-subject-per-file, provide an explicit candidate manifest:

```json
{
  "schema": "RealSaS.Post139.CandidateManifest.v1",
  "source_family": "Quaternius",
  "pack": "ExamplePack",
  "provenance": "CLEAN_CC0_HUMAN_AUTHORED",
  "license": "CC0",
  "candidates": [
    {"subject_id": "actor-a", "path": "Actors/ActorA.fbx"},
    {"subject_id": "actor-b", "path": "Actors/ActorB.fbx"}
  ]
}
```

## Output

Each candidate receives:

```text
subjects/<subject_id>/
  source_structure.json
  technical_audit.json
  admission.json
```

The pack root also receives `POST139_AUDIT_INDEX.json`. `source_structure.json` carries exact source
SHA-256, capability census, metrics and independent structural/appearance/animation fingerprints.
`technical_audit.json` records whether the required truth is present and whether canonical dedup is
ready. `admission.json` applies provenance, Base139/cross-pack duplicate references and the Post139
collapse policy.

Reference roots are ordinary prior audit output roots. This keeps Base139 immutable: Post139 reads
its fingerprints but never writes into the Base139 corpus.

## Canonical Blender fingerprint

The Blender worker imports GLTF/GLB/FBX under `--factory-startup`, switches armatures to rest pose,
and computes scale/rigid-transform-invariant geometry distributions, topology distributions,
normalized bone lengths/hierarchy and a per-vertex skin-influence multiset. This intentionally
avoids source-format byte identity as the duplicate key. Exact source SHA remains recorded separately.

This fingerprint is a deterministic corpus-admission heuristic, not a proof of semantic identity.
Any unexpected collision or disagreement should be quarantined for manual source comparison rather
than silently merged.
