# RealSaS Source License Admission Policy V1

Status: CANONICAL_DATA_ADMISSION_POLICY
Date: 2026-08-22
Branch: `g0-g1/single-pose-geometry`

## 1. Purpose

Technical availability is not legal admission.

No source asset may enter a RealSaS model-training/export view merely because it is present in Articulation-XL2.0, Rig-XL, Objaverse, VRoid, Hugging Face, or another downloadable corpus.

Every source record has two independent gates:

```text
TECHNICALLY_ADMISSIBLE
LEGAL_ADMISSIBLE
```

Training/render export requires both to PASS.

L0 may retain provenance/index metadata for a non-admitted record, but restricted source bytes must not be redistributed through RealSaS artifacts.

## 2. Repository/dataset license is not automatically the asset license

The UniRig repository is MIT-licensed software. That does not automatically relicense upstream 3D assets represented by Rig-XL.

Rig-XL documentation states that, aside from VRoid, its models are selected from Objaverse and `mapping.json` points back to original source identifiers/URLs. Therefore source-asset licensing must be resolved at object level.

Objaverse 1.0 is ODC-By as a dataset, while individual objects carry their own Creative Commons license metadata. VRoid records carry model-specific conditions of use.

## 3. Per-record license record

For every source record the builder must resolve and store, when available:

- `license_source` (e.g. Objaverse metadata, VRoid metadata, upstream dataset card);
- `license_id` / SPDX-like normalized identifier where possible;
- original license text or canonical URL/reference;
- original author/creator attribution fields;
- original source URL / object UID;
- commercial-use permission state;
- modification/adaptation permission state;
- redistribution permission state;
- attribution requirement;
- share-alike requirement;
- source-specific restrictions;
- resolution timestamp;
- resolver version;
- evidence SHA-256 / metadata snapshot hash.

Absence or ambiguity is not treated as permission.

## 4. Conservative automated admission classes

These are corpus-engineering defaults, not legal advice.

### AUTO_ALLOW_CANDIDATE

- CC0 1.0
- CC-BY 4.0, provided attribution/provenance obligations are preserved

These may proceed to technical admission, while attribution metadata remains mandatory.

### LEGAL_REVIEW_REQUIRED

- CC-BY-SA 4.0
- licenses with unusual additional terms
- source records whose license applies differently to model data, annotations, textures or geometry
- any case where training/derived-model implications are unclear

These are not included in the default commercial-training export until explicitly approved.

### EXCLUDE_FROM_COMMERCIAL_TRAINING

- CC-BY-NC 4.0
- CC-BY-NC-SA 4.0
- explicitly non-commercial source terms
- commercial/corporate use explicitly disallowed
- modification/adaptation disallowed when our normalization/rendering requires adaptation

### EXCLUDE_UNRESOLVED

- unknown license
- missing source identity
- missing license metadata
- conflicting license metadata that cannot be resolved

## 5. VRoid policy

VRoid records must be evaluated per model. At minimum the resolver must inspect the applicable model conditions for:

- third-party use;
- corporate/commercial use;
- individual commercial use where relevant;
- modification;
- redistribution;
- attribution;
- any other model-specific restrictions carried by the VRM/VRoid metadata.

Default policy: VRoid records are **not** admitted to the commercial-training export unless the record's terms affirmatively allow the uses required by RealSaS and provenance is preserved.

## 6. Objaverse/Rig-XL policy

For Rig-XL records sourced from Objaverse:

1. resolve the Rig-XL `mapping.json` record to the original Objaverse UID/source identity;
2. load the official Objaverse annotation/metadata for that UID;
3. record the individual object's license;
4. apply the conservative class above;
5. preserve creator/source attribution metadata where required;
6. fail closed if the mapping cannot be resolved exactly.

The MIT license of UniRig code/repository is never substituted for the object-level license.

## 7. Articulation-XL2.0 policy

The upstream Articulation-XL2.0 dataset card declares CC-BY-4.0. The builder shall preserve this dataset-level license and citation provenance.

Because Articulation-XL2.0 is derived from external 3D sources, source identity/provenance should still be retained when available. If an individual source record exposes a conflicting or more restrictive source term, the stricter unresolved state wins until reviewed.

## 8. Training/export behavior

Each model-specific export manifest must state a `license_policy_id` and include only records whose legal state satisfies that policy.

Default commercial-product research/training view:

```text
AUTO_ALLOW_CANDIDATE + technical PASS
```

`LEGAL_REVIEW_REQUIRED`, `EXCLUDE_FROM_COMMERCIAL_TRAINING`, and `EXCLUDE_UNRESOLVED` records are physically absent from that export.

A research-only view, if ever created, must be a separately named/exported artifact and must never be silently mixed into product-training data.

## 9. Generated render derivatives

Rendering an asset into 1024x1024 NPR views does not erase the upstream license. Every rendered family retains a provenance edge to the source record and its license state.

The master manifest must make it possible to regenerate an attribution ledger for every admitted source asset.

## 10. Redistribution

RealSaS corpus artifacts must not redistribute original source assets, textures, or source-derived data beyond what their licenses permit.

Internal training artifacts and externally distributable artifacts are separate policy classes. Public release requires an additional redistribution audit.

## 11. Audit outputs

The corpus builder must emit at least:

- total source records;
- license-resolved count;
- count per normalized license;
- AUTO_ALLOW_CANDIDATE count;
- LEGAL_REVIEW_REQUIRED count;
- excluded-noncommercial count;
- excluded-unresolved count;
- source/author attribution ledger for admitted records;
- unresolved/conflict report with exact record IDs.

No count may be inferred from the parent dataset's aggregate license distribution; it must be computed over the actual RealSaS-selected source identities.

## 12. Fail-closed rule

```text
NO PROVABLE LICENSE PERMISSION
        => NO COMMERCIAL-TRAINING EXPORT
```

This policy is intentionally conservative. Final legal interpretation for commercial launch should be reviewed by qualified counsel; the engineering pipeline's job is to preserve enough exact provenance and evidence that such review is possible without reconstructing the corpus history later.
