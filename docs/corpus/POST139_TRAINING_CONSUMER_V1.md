# Post139 Training Consumer V1

Post139 training code is not allowed to infer corpus membership from source folders, audit indexes, the V2 finalizer report, filenames, pack names, or local availability. It consumes only a verified `RealSaS.Post139.CorpusAuthority.v1` bundle.

## Consumer contract

`tools/corpus/post139_training_consumer_v1.py emit` verifies the authority root and every hash-bound ledger before producing `RealSaS.Post139.TrainingConsumerManifest.v1`.

The full animated training roster is exactly:

- every immutable Base139 row with admission `REFERENCE_BASE139_ROW`;
- every Post139 row with admission `ADMIT_PRIMARY`.

Rows classified as aliases, authored state/loadout variants, publisher representation variants, structural duplicates, static auxiliaries, policy auxiliaries, partial rows, or quarantine rows remain visible in corpus authority but are not silently promoted into the full animated training roster.

Base139 materialization remains owned by the existing immutable Truth139 authority. Post139 primary rows are bound to `POST139_MATERIALIZATION_LEDGER_V1`; the consumer verifies source SHA-256 and lineage bindings before emitting them.

## Split safety

`validate-split` accepts a `RealSaS.Post139.TrainingSplitPlan.v1` containing one assignment for every emitted training subject. The subject set must match exactly: no omitted row and no undeclared row is accepted.

The invariant is strict:

`LINEAGE_COMPONENT_ID_IS_ATOMIC_FOR_FIT_UNSEEN_SPLITS`

If any sealed lineage component appears in more than one split, validation fails with `POST139_TRAINING_SPLIT_LINEAGE_LEAKAGE`. This covers Base139/Post139 cross-links as well as Post139/Post139 links; an exact rig-hierarchy or canonical-surface relation therefore cannot leak across FIT/UNSEEN merely because the rows came from different packs.

Example invocation:

```bash
python tools/corpus/post139_training_consumer_v1.py emit \
  --authority-dir /path/to/post139-authority-v1 \
  --out /path/to/POST139_TRAINING_CONSUMER_MANIFEST_V1.json

python tools/corpus/post139_training_consumer_v1.py validate-split \
  --authority-dir /path/to/post139-authority-v1 \
  --split-plan /path/to/split-plan.json \
  --out /path/to/POST139_VALIDATED_TRAINING_SPLIT_V1.json
```

The output manifest and validated split each carry deterministic digests bound to the sealed corpus authority digest.
