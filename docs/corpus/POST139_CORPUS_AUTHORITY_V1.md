# Post139 Corpus Authority V1

Post139 corpus state is not authoritative because a source was downloaded or because an audit JSON exists. Authority is a closed chain:

`publisher source -> acquired byte ledger -> structural audit -> subject identity policy -> global admission -> lineage -> training roster -> materialization -> sealed corpus authority`

## Authority boundary

`POST139_CORPUS_AUTHORITY_V1.json` and the ledgers hash-bound by it are the only Post139 consumer authority. Per-source receipts, publisher trees, Blender audit files, and V2 compiler outputs remain reviewable evidence/staging inputs. A training/materialization consumer must not independently reinterpret those staging files.

The root binds six ledgers:

- `POST139_SOURCE_EVIDENCE_LEDGER_V1.json`: exact publisher/acquired byte closure and stable audit evidence bindings.
- `POST139_SUBJECT_IDENTITY_LEDGER_V1.json`: every audited subject row, including excluded, alias, reference-variant, duplicate and admitted identities.
- `POST139_ADMISSION_LEDGER_V1.json`: the explicit global admission decision for every identity row.
- `POST139_LINEAGE_LEDGER_V1.json`: leakage-connected components that are atomic for fit/unseen splits.
- `POST139_TRAINING_ROSTER_LEDGER_V1.json`: immutable Base139 rows plus admitted Post139 rows.
- `POST139_MATERIALIZATION_LEDGER_V1.json`: exact source model hashes and source-ledger bindings needed to materialize admitted rows.

Every ledger is bound by both file SHA-256 and canonical semantic digest. The root itself has an `authority_digest`. Verification is fail-closed.

## Determinism

Machine-local paths are evidence transport details and are excluded from semantic authority. The sealer reconstructs stable per-subject evidence digests after removing runtime paths such as `/tmp/...`. Raw source files are rehashed from bytes and must reproduce `RAW_FILE_LEDGER.ledger_sha256` exactly.

The materialization closure also reconstructs each source ledger including file extension, count, bytes and extension counts. Its ZIP uses fixed timestamps/permissions and sorted paths, so identical inputs produce identical archive bytes.

## Semantic identity policy

`canonical/POST139_SUBJECT_POLICY_V1.json` is applied before global admission. Policy never deletes source/audit evidence. It can only classify an audited row as a selected immutable Base139 row, a non-selected reference representation variant, an authored alias, an authored state/loadout variant excluded from the subject denominator, or an eligible row for structural admission.

Current explicit resolutions include the canonical Textured lane for the four Animated Mech Base139 rows, Cute Fish `BlueGoldfish -> Goldfish`, Cyberpunk `*Gun` loadout variants, `Snake_angry`, and Zombie `*_SingleWeapon` variants.

Structural duplicate identity remains `canonical geometry/topology + rest rig + spatial skin`. Appearance is recorded independently and does not silently collapse immutable Base139 training rows.

## Recovery and sealing

`.github/workflows/post139_recover_and_seal_v1.yml` is the sole recovery/materialization orchestration owner. The authority-contract workflow only validates tooling contracts; it does not invoke recovery.

Recovery reuses the 21 immutable successful source workspaces from source run `41192533150` and operates on the eight missing sources as isolated matrix jobs with bounded parallelism. Each successful recovered source is uploaded as its own immutable workspace before aggregation. One source failure therefore cannot hide the state of the other recovery sources or force already-successful sources to be reacquired.

Mixed publisher packs may narrow byte acquisition to a mechanically relevant source superset while preserving the complete publisher tree. Acquisition lane size is never a training denominator: subject count is decided only after canonical Blender audit, semantic policy, global identity/admission and lineage compilation. Exact glTF buffer/image dependencies are derived from the selected model documents and included in the acquired byte ledger.

After all eight recovery jobs pass, the aggregate job combines them with the 21 immutable source workspaces, verifies the exact 29-source non-quarantine workspace closure, runs corrected V2 semantic compilation, corpus authority sealing plus standalone verification, emits the sealed training-consumer manifest, builds the deterministic materialization source closure, and applies the final fail-closed readiness gate.

Recovery/seal executions are head-specific. A cancelled run or a run from an earlier orchestration head is staging evidence only and must never be interpreted as a sealed corpus result. After recovery or acquisition tooling changes, the canonical recovery workflow must execute from the current branch head before any final corpus count is consumed.

No Post139 training consumer should proceed unless the root status is `SEALED`, the V2 report is `PASS_MATERIALIZATION_READY`, and standalone authority verification passes.
