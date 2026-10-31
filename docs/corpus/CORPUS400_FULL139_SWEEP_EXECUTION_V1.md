# Corpus400 Full139 Truth-Family Sweep Execution V1

Status: execution protocol only. This document does not mint corpus authority.

## Canonical entrypoint

Run on a machine where the RealSaS Google Drive is mounted as a normal filesystem.
The launcher discovers the exact historical readiness ledger and source-cache
manifest by schema/status/139-subject identity; it refuses ambiguous compatible
pairs rather than selecting a newest file.

```bash
python tools/corpus/launch_full139_truth_family_sweep_drive_v1.py \
  --drive-root /content/drive/MyDrive \
  --execute
```

For a one-subject smoke test before the full execution:

```bash
python tools/corpus/launch_full139_truth_family_sweep_drive_v1.py \
  --drive-root /content/drive/MyDrive \
  --execute \
  --subject-id truth139_00b08f9cf0e6f661213d
```

The launcher always runs the V3 plan gate before execution. Output defaults beside
the selected historical readiness ledger under:

`CORPUS400_FULL139_TRUTH_FAMILY_SWEEP_V1/`

Execution is resumable. Existing produced artifacts may skip recomputation, but all
downstream courts reopen/hash their parents; resume is not an authority bypass.

## Historical authority firewall

The historical Truth139 V3.2 readiness ledger is only an immutable artifact index.
The following historical fields have no Corpus400 authority:

- historical TRAIN/UNSEEN split;
- historical `training_authorized=true`;
- historical global PASS labels;
- historical subject count as a retention requirement.

The source-cache manifest is likewise byte inventory only.

## Per-subject court chain

V3 runs and binds:

1. `SOURCE_IMPORT` + `CAMERA_OBSERVATION` independent court;
2. component-scoped surface geometry / sign / normal court;
3. qualified mechanical surface topology court;
4. raw-source rig/skin court plus importer-equivalence court;
5. raw-source animation court;
6. independent canonical animation replay court;
7. causal-response court on mechanical surface topology;
8. Stage48-bound graphics measured-domain court;
9. TESSA V5 mechanical-surface consumer court.

Canonical `SURFACE_TOPOLOGY` comes only from the mechanical-surface court. The
owner-preserving geometry partition remains local SDF/sign addressing evidence.

## Truth that deliberately remains non-final

`APPEARANCE_CAA`, `VISIBILITY_OCCLUSION`, and `RAY_COMPOSITING` measured rows remain
pending a corpus-level student-observability/leakage admission. Complete ray order
and intervention evidence may remain compiler oracle even when they are not valid
student supervision.

No sweep result directly authorizes training.

## Required closure after sweep

After the 139-subject sweep:

1. classify each subject as `ADMIT_CORE`, `ADMIT_MASKED`,
   `QUARANTINE_REPAIRABLE`, or `DROP_UNSUITABLE`;
2. rebuild duplicate/lineage families from the final admitted set;
3. run corpus-level graphics student-observability admission;
4. create a new lineage-safe TRAIN/UNSEEN split;
5. bind consumer manifests to the final family matrix;
6. emit a single new `TRAINING_READINESS_RECEIPT`.

Until step 6 passes, optimizer authorization is false.
