# Shared-Scene Stage A Result V1

**Run:** `40316053020`  
**Head SHA:** `0fa3d3104526c40e54c9a204e73f1f2c0d6f3e2c`  
**Artifact digest:** `sha256:fb340ee5bad0e808c400ccc2abef8d28da35472d2a925db62e2ab75b514ac8c3`  
**Scientific status:** `SHARED_SCENE_STAGE_A_VALID__S2_SHARED_TRUNK_PRIVATE`

## Apparatus validity

- parameter budget matched: PASS;
- random/firewall normalized error: `1.00393`, PASS;
- five fixed seeds;
- Stage A is synthetic pre-Truth evidence only.

## Summary

| Arm | IID GM | IID GP | shift GM | shift GP | worst joint interference | default gate |
|---|---:|---:|---:|---:|---:|---|
| S0 independent | 0.00660 | 0.01273 | 0.03996 | 0.06517 | -0.28425 | eligible |
| S1 late fusion | 0.01215 | 0.01718 | 0.04776 | 0.03819 | +0.11043 | eligible |
| **S2 shared trunk + private subspaces** | **0.00757** | **0.01505** | **0.04545** | **0.03865** | **-0.07139** | **selected provisional default** |
| S3 full shared Z | 0.00935 | 0.01820 | 0.04792 | 0.03696 | +0.19226 | reject as default |
| S4 shared Z + family adapters | 0.00813 | 0.01694 | 0.04465 | 0.04112 | +0.18088 | reject as default |

S3 and S4 exceed the preregistered 15% required-family negative-transfer threshold on GP. S2 does not and retains useful shared substrate while preserving family-specific capacity.

## Verdict

Carry **S2 shared trunk + private GM/GP subspaces** into the V3.1 pre-Truth assembly.

This does **not** establish the real shared-scene hypothesis. Stage B with authored causal truth remains mandatory and can overturn the provisional choice. The result only says that, among the tested matched synthetic mechanisms, forcing all families through one undifferentiated shared state is a worse default than sharing a trunk and protecting private subspaces.
